"""Score a frozen NIH CXR checkpoint on the CheXpert expert-labeled test set.

The release contains 500 studies from 500 patients and 668 radiographs. Scores
are max-aggregated across all images in each study, as described for the
CheXpert single-view baseline when a study has multiple views. This script is
evaluation-only: it never tunes thresholds or updates model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.modeling import build_classifier, image_to_tensor  # noqa: E402


LABEL_MAP = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Pleural Effusion": "Effusion",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def patient_id(study: str) -> str:
    match = re.search(r"patient\d+", study)
    if not match:
        raise ValueError(f"Could not extract patient ID from study path: {study}")
    return match.group(0)


def study_directory(image_root: Path, raw_study: str) -> Path:
    """Resolve the study under common archive-root layouts without escaping root."""
    root = image_root.resolve()
    parts = Path(raw_study.replace("\\", "/")).parts
    marker = next((i for i, part in enumerate(parts) if part.lower() == "test"), None)
    if marker is None:
        raise ValueError(f"Study path has no test component: {raw_study}")
    suffix = parts[marker + 1 :]
    candidates = [root.joinpath(*suffix), root.joinpath(*parts[marker:]), root.joinpath(*parts)]
    chexpert_marker = next((i for i, part in enumerate(parts)
                            if part.lower() == "chexpert-v1.0"), None)
    if chexpert_marker is not None:
        candidates.insert(1, root.joinpath(*parts[chexpert_marker + 1 :]))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_relative_to(root) and resolved.is_dir():
            return resolved
    raise FileNotFoundError(f"Study directory not found under {root}: {raw_study}")


def patient_bootstrap(y: np.ndarray, p: np.ndarray, patients: np.ndarray,
                      *, seed: int, samples: int) -> dict[str, list[float] | None]:
    unique = np.unique(patients)
    groups = {pid: np.flatnonzero(patients == pid) for pid in unique}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"auroc": [], "average_precision": [], "f1_at_0_5": []}
    for _ in range(samples):
        selected = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([groups[pid] for pid in selected])
        yb, pb = y[indices], p[indices]
        if len(np.unique(yb)) == 2:
            values["auroc"].append(float(roc_auc_score(yb, pb)))
        if yb.sum() > 0:
            values["average_precision"].append(float(average_precision_score(yb, pb)))
        values["f1_at_0_5"].append(float(f1_score(yb, pb >= 0.5, zero_division=0)))
    return {
        key: [float(v) for v in np.quantile(scores, [0.025, 0.975])] if scores else None
        for key, scores in values.items()
    }


def macro_patient_bootstrap(y: np.ndarray, p: np.ndarray, patients: np.ndarray,
                            *, seed: int, samples: int) -> dict[str, list[float] | None]:
    unique = np.unique(patients)
    groups = {pid: np.flatnonzero(patients == pid) for pid in unique}
    rng = np.random.default_rng(seed)
    aurocs: list[float] = []
    aps: list[float] = []
    for _ in range(samples):
        selected = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([groups[pid] for pid in selected])
        boot_auroc: list[float] = []
        boot_ap: list[float] = []
        for col_i in range(y.shape[1]):
            target, score = y[indices, col_i], p[indices, col_i]
            if len(np.unique(target)) == 2:
                boot_auroc.append(float(roc_auc_score(target, score)))
            if target.sum() > 0:
                boot_ap.append(float(average_precision_score(target, score)))
        if boot_auroc:
            aurocs.append(float(np.mean(boot_auroc)))
        if boot_ap:
            aps.append(float(np.mean(boot_ap)))
    return {
        "macro_auroc": [float(v) for v in np.quantile(aurocs, [0.025, 0.975])] if aurocs else None,
        "macro_average_precision": [float(v) for v in np.quantile(aps, [0.025, 0.975])] if aps else None,
    }


def score_checkpoint(checkpoint_path: Path, image_paths: list[Path], study_indices: list[int],
                     study_count: int, batch_size: int, device: torch.device
                     ) -> tuple[np.ndarray, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_labels = checkpoint.get("labels")
    if not isinstance(model_labels, list) or any(name not in model_labels for name in LABEL_MAP.values()):
        raise ValueError(f"Checkpoint does not contain all five matching NIH labels: {checkpoint_path}")
    if checkpoint.get("architecture") not in {"densenet121", "efficientnet_b0"}:
        raise ValueError(f"Unsupported external-evaluation architecture: {checkpoint.get('architecture')}")
    model = build_classifier(checkpoint["architecture"], len(model_labels), pretrained=False,
                             label_adjacency=checkpoint.get("label_adjacency"))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    image_size = int(checkpoint.get("image_size", 224))
    image_scores: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(image_paths), batch_size):
            tensors = []
            for path in image_paths[start : start + batch_size]:
                with Image.open(path) as image:
                    tensors.append(image_to_tensor(image.convert("RGB"), image_size))
            logits = model(torch.stack(tensors).to(device))
            image_scores.append(torch.sigmoid(logits).cpu().numpy())
    all_image_scores = np.concatenate(image_scores, axis=0)
    study_scores = np.zeros((study_count, len(LABEL_MAP)), dtype=np.float64)
    row_groups = np.asarray(study_indices)
    for study_i in range(study_count):
        rows = np.flatnonzero(row_groups == study_i)
        for col_i, nih_label in enumerate(LABEL_MAP.values()):
            study_scores[study_i, col_i] = all_image_scores[rows, model_labels.index(nih_label)].max()
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "architecture": checkpoint["architecture"],
        "training_images": checkpoint.get("training_images"),
        "image_size": image_size,
    }
    del model, checkpoint, all_image_scores
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return study_scores, metadata


def summarize_scores(y: np.ndarray, scores: np.ndarray, patients: np.ndarray,
                     bootstrap_samples: int, seed: int) -> dict:
    per_label: dict[str, dict] = {}
    for col_i, name in enumerate(LABEL_MAP):
        target, probability = y[:, col_i], scores[:, col_i]
        prediction = probability >= 0.5
        per_label[name] = {
            "studies": int(len(target)),
            "positive": int(target.sum()),
            "negative": int(len(target) - target.sum()),
            "auroc": float(roc_auc_score(target, probability)) if len(np.unique(target)) == 2 else None,
            "average_precision": float(average_precision_score(target, probability)) if target.sum() else None,
            "f1_at_0_5": float(f1_score(target, prediction, zero_division=0)),
            "precision_at_0_5": float(precision_score(target, prediction, zero_division=0)),
            "recall_at_0_5": float(recall_score(target, prediction, zero_division=0)),
            "patient_bootstrap_95_ci": patient_bootstrap(
                target, probability, patients, seed=seed + col_i, samples=bootstrap_samples),
        }
    macro_auroc = [entry["auroc"] for entry in per_label.values() if entry["auroc"] is not None]
    macro_ap = [entry["average_precision"] for entry in per_label.values()
                if entry["average_precision"] is not None]
    return {
        "macro_auroc": float(np.mean(macro_auroc)) if macro_auroc else None,
        "macro_average_precision": float(np.mean(macro_ap)) if macro_ap else None,
        "macro_patient_bootstrap_95_ci": macro_patient_bootstrap(
            y, scores, patients, seed=seed + 100, samples=bootstrap_samples),
        "labels": per_label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True,
                        help="Directory containing test/patient*/study* or patient*/study* folders")
    parser.add_argument("--labels-csv", type=Path, required=True,
                        help="CheXpert test-set groundtruth.csv from the official label release")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="primary",
                        help="Name used in result columns for --checkpoint")
    parser.add_argument("--second-checkpoint", type=Path,
                        help="Optional second frozen candidate; equal-average probabilities are also reported")
    parser.add_argument("--second-checkpoint-name", default="secondary")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--expected-studies", type=int, default=500)
    parser.add_argument("--expected-images", type=int, default=668)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if (args.batch_size < 1 or args.expected_studies < 1 or args.expected_images < 1
            or args.bootstrap_samples < 100):
        parser.error("batch size and expected counts must be positive; use at least 100 bootstrap samples")
    paths_to_check = [args.image_root, args.labels_csv, args.checkpoint]
    if args.second_checkpoint is not None:
        paths_to_check.append(args.second_checkpoint)
        if args.checkpoint_name == args.second_checkpoint_name:
            parser.error("Primary and secondary checkpoint names must differ")
    for path in paths_to_check:
        if not path.exists():
            raise FileNotFoundError(path)

    frame = pd.read_csv(args.labels_csv)
    required = {"Study", *LABEL_MAP}
    if missing := required - set(frame.columns):
        raise ValueError(f"groundtruth.csv is missing columns: {sorted(missing)}")
    frame["Study"] = frame["Study"].astype(str)
    if frame["Study"].duplicated().any() or len(frame) != args.expected_studies:
        raise ValueError(f"Expected {args.expected_studies} unique studies; found {len(frame)} rows")
    frame["patient_id"] = frame["Study"].map(patient_id)
    if frame["patient_id"].nunique() != len(frame):
        raise ValueError("Expected one study per patient in this test release")
    labels = frame[list(LABEL_MAP)].apply(pd.to_numeric, errors="coerce")
    if labels.isna().any().any() or not labels.isin([0, 1]).all().all():
        raise ValueError("Expected binary 0/1 majority-vote labels for all five findings")

    image_paths: list[Path] = []
    study_indices: list[int] = []
    for row_i, raw_study in enumerate(frame["Study"]):
        directory = study_directory(args.image_root, raw_study)
        paths = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not paths:
            raise FileNotFoundError(f"No readable image files found for study {raw_study} in {directory}")
        image_paths.extend(paths)
        study_indices.extend([row_i] * len(paths))
    if len(image_paths) != args.expected_images:
        raise ValueError(
            f"Expected {args.expected_images} images across the complete test set; found {len(image_paths)}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = labels.to_numpy(dtype=np.int32)
    patients = frame["patient_id"].to_numpy()
    candidate_definitions = [(args.checkpoint_name, args.checkpoint)]
    if args.second_checkpoint is not None:
        candidate_definitions.append((args.second_checkpoint_name, args.second_checkpoint))
    candidate_scores: dict[str, np.ndarray] = {}
    candidate_metadata: dict[str, dict] = {}
    for candidate_name, checkpoint_path in candidate_definitions:
        score_matrix, metadata = score_checkpoint(
            checkpoint_path, image_paths, study_indices, len(frame), args.batch_size, device)
        candidate_scores[candidate_name] = score_matrix
        candidate_metadata[candidate_name] = metadata
    if len(candidate_scores) == 2:
        ensemble_name = f"equal_average_{args.checkpoint_name}_{args.second_checkpoint_name}"
        candidate_scores[ensemble_name] = (
            candidate_scores[args.checkpoint_name] + candidate_scores[args.second_checkpoint_name]
        ) / 2.0

    summaries = {
        name: summarize_scores(y, scores, patients, args.bootstrap_samples, 20261101 + i * 1000)
        for i, (name, scores) in enumerate(candidate_scores.items())
    }
    predictions = pd.DataFrame({"study": frame["Study"], "patient_id": patients})
    for col_i, name in enumerate(LABEL_MAP):
        predictions[f"target_{name}"] = y[:, col_i]
        for candidate_name, scores in candidate_scores.items():
            predictions[f"score_{candidate_name}_{name}"] = scores[:, col_i]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "chexpert_expert_test_predictions.csv", index=False)
    result = {
        "dataset": "CheXpert radiologist-labeled test set",
        "scope": "external evaluation; 500 studies / unique patients; five shared findings",
        "study_aggregation": "maximum finding score across all images in each study",
        "images_scored": int(len(image_paths)),
        "studies_scored": int(len(frame)),
        "patients_scored": int(frame["patient_id"].nunique()),
        "candidates": candidate_metadata,
        "results": summaries,
        "ensemble_policy": ("arithmetic mean of the two frozen candidates' study-level probabilities"
                            if len(candidate_scores) == 3 else None),
        "device": str(device),
        "threshold": 0.5,
        "threshold_tuned_on_test": False,
        "official_NIH_test_used": False,
        "primary_metric": "macro AUROC across the five CheXpert competition findings",
        "clinical_use": "not suitable for diagnosis or care decisions",
    }
    (args.output_dir / "chexpert_expert_test_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
