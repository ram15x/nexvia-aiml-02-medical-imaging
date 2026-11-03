"""One-time held-out check on NIH train patients excluded from the 10k model.

This creates a small development holdout from patients never used by the saved
8k/10k checkpoints or prior prediction files, then scores both frozen models.
It is not the official NIH test split and is underpowered for rare findings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_audit import NIH_LABELS
from train_classifier import NIHChestXrayDataset, evaluate, make_model, seed_everything


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_candidate_manifest(project_root: Path, output_dir: Path) -> tuple[Path, dict]:
    nih = project_root / "data" / "NexviaProjectData" / "nih"
    source = pd.read_csv(
        nih / "metadata" / "fresh_15k_patient_split_20261029" / "train_manifest.csv"
    )
    candidate_manifest = pd.read_csv(
        nih / "learning_curve_10000_train_1000_val_20261031" /
        "manifest_10000_train_1000_validation.csv"
    )
    if set(source["split"].astype(str)) != {"train"}:
        raise ValueError("Source cohort is not exclusively from the development training split")
    candidate_train = candidate_manifest[candidate_manifest["split"] == "train"]
    if len(candidate_train) != 10000 or candidate_train["Patient ID"].nunique() != 10000:
        raise ValueError("Expected exactly 10,000 patient-disjoint candidate training rows")

    candidate_train_ids = set(candidate_train["Image Index"].astype(str))
    remainder = source[~source["Image Index"].astype(str).isin(candidate_train_ids)].copy()

    # Exclude every patient already represented in saved prediction outputs.
    prior_prediction_ids: set[str] = set()
    current_output = output_dir.resolve()
    for path in nih.rglob("*predictions*.csv"):
        if current_output in path.resolve().parents:
            continue
        try:
            columns = pd.read_csv(path, nrows=0).columns
            if "Image Index" in columns:
                prior_prediction_ids.update(
                    pd.read_csv(path, usecols=["Image Index"])["Image Index"].astype(str)
                )
        except (OSError, pd.errors.ParserError, ValueError):
            continue

    # Exclude rows used in the earlier resolution-screen manifest as well.
    resolution_manifest = nih / "metadata" / "resolution_screen_2500_train_1779_val_seed20261030.csv"
    prior_resolution_ids: set[str] = set()
    if resolution_manifest.is_file():
        resolution = pd.read_csv(resolution_manifest)
        if "Image Index" in resolution:
            prior_resolution_ids = set(resolution["Image Index"].astype(str))

    original_remainder_ids = set(remainder["Image Index"].astype(str))
    prediction_overlap_removed = original_remainder_ids & prior_prediction_ids
    resolution_overlap_before = original_remainder_ids & prior_resolution_ids
    resolution_only_removed = resolution_overlap_before - prior_prediction_ids
    blocked_ids = prior_prediction_ids | prior_resolution_ids
    remainder = remainder[~remainder["Image Index"].astype(str).isin(blocked_ids)].copy()
    if len(remainder) != 99 or remainder["Patient ID"].nunique() != 99:
        raise ValueError(
            f"Unexpected unused remainder size: {len(remainder)} rows, "
            f"{remainder['Patient ID'].nunique()} patients; stop and audit before scoring"
        )

    # Candidate weights must not have seen any holdout patient.
    holdout_ids = set(remainder["Image Index"].astype(str))
    if holdout_ids & candidate_train_ids:
        raise ValueError("Holdout overlaps the 10k training cohort")
    holdout_patient_ids = set(remainder["Patient ID"].astype(str))
    if holdout_patient_ids & set(candidate_train["Patient ID"].astype(str)):
        raise ValueError("Holdout patient overlap with candidate training cohort")

    remainder["split"] = "validation"
    manifest_path = output_dir / "manifest_99_train_remainder_validation.csv"
    remainder.to_csv(manifest_path, index=False)
    supports = {}
    finding_labels = remainder["Finding Labels"].fillna("").astype(str)
    for label in NIH_LABELS:
        supports[label] = int(finding_labels.map(lambda value: label in value.split("|")).sum())
    audit = {
        "purpose": "single exploratory holdout comparison of frozen 8k/10k candidates",
        "source_split": "fresh_15k development train split only",
        "source_train_rows": int(len(source)),
        "candidate_10k_training_rows": int(len(candidate_train)),
        "holdout_rows": int(len(remainder)),
        "holdout_patients": int(remainder["Patient ID"].nunique()),
        "candidate_training_overlap": False,
        "prior_prediction_rows_excluded": len(prediction_overlap_removed),
        "prior_resolution_screen_rows_found": len(resolution_overlap_before),
        "new_rows_excluded_by_resolution_screen": len(resolution_only_removed),
        "remaining_holdout_overlap_with_prior_predictions": 0,
        "official_test_used": False,
        "new_downloads": 0,
        "label_support_positive": supports,
        "holdout_manifest_sha256": sha256(manifest_path),
        "caveat": "Only 99 patients; rare-label estimates are not decision-grade and the cohort is not a clinical test.",
    }
    return manifest_path, audit


def patient_bootstrap_micro(y: np.ndarray, p: np.ndarray, patients: np.ndarray,
                            *, seed: int = 20261031, samples: int = 1000) -> dict:
    unique = np.unique(patients)
    groups = {pid: np.flatnonzero(patients == pid) for pid in unique}
    rng = np.random.default_rng(seed)
    auc_values, ap_values = [], []
    for _ in range(samples):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([groups[pid] for pid in sampled])
        target, score = y[rows].reshape(-1), p[rows].reshape(-1)
        if len(np.unique(target)) == 2:
            auc_values.append(float(roc_auc_score(target, score)))
        if target.sum() > 0:
            ap_values.append(float(average_precision_score(target, score)))

    def ci(values: list[float]) -> list[float] | None:
        return [float(x) for x in np.quantile(values, [0.025, 0.975])] if values else None

    return {"micro_auroc_95_ci": ci(auc_values), "micro_average_precision_95_ci": ci(ap_values)}


def paired_micro_deltas(y: np.ndarray, p8: np.ndarray, p10: np.ndarray,
                        patients: np.ndarray, *, seed: int = 20261101,
                        samples: int = 1000) -> dict:
    """Paired patient bootstrap for 10k-minus-8k micro metric differences."""
    unique = np.unique(patients)
    groups = {pid: np.flatnonzero(patients == pid) for pid in unique}

    def point(target: np.ndarray, score: np.ndarray) -> dict[str, float]:
        target, score = target.reshape(-1), score.reshape(-1)
        return {
            "micro_auroc": float(roc_auc_score(target, score)),
            "micro_average_precision": float(average_precision_score(target, score)),
            "micro_f1_at_0_5": float(
                f1_score(target.astype(np.int32), score >= 0.5, zero_division=0)
            ),
        }

    point8, point10 = point(y, p8), point(y, p10)
    delta = {key: point10[key] - point8[key] for key in point8}
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {key: [] for key in delta}
    for _ in range(samples):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([groups[pid] for pid in sampled])
        target = y[rows].reshape(-1)
        score8, score10 = p8[rows].reshape(-1), p10[rows].reshape(-1)
        if len(np.unique(target)) == 2:
            draws["micro_auroc"].append(
                float(roc_auc_score(target, score10) - roc_auc_score(target, score8))
            )
        if target.sum() > 0:
            draws["micro_average_precision"].append(
                float(average_precision_score(target, score10) -
                      average_precision_score(target, score8))
            )
        draws["micro_f1_at_0_5"].append(
            point(target, score10)["micro_f1_at_0_5"] -
            point(target, score8)["micro_f1_at_0_5"]
        )
    intervals = {
        key: [float(x) for x in np.quantile(values, [0.025, 0.975])] if values else None
        for key, values in draws.items()
    }
    return {"train8000": point8, "train10000": point10,
            "delta_10000_minus_8000": delta, "paired_patient_bootstrap_95_ci": intervals,
            "bootstrap_samples": samples, "bootstrap_seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    manifest, audit = get_candidate_manifest(args.project_root, args.output_dir)
    if not args.image_root.is_dir():
        raise FileNotFoundError(f"Image root not found: {args.image_root}")

    dataset = NIHChestXrayDataset(
        manifest, args.image_root, "validation", None, args.seed,
        image_size=224, allow_available_subset=False,
    )
    if len(dataset) != 99 or dataset.missing_images:
        raise ValueError(f"Holdout incomplete: {len(dataset)} rows, {dataset.missing_images} missing images")
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0,
                        pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patient_ids = dataset.rows["Patient ID"].astype(str).to_numpy()

    nih = args.project_root / "data" / "NexviaProjectData" / "nih"
    candidate_paths = {
        "train8000": nih / "learning_curve_8000_train_1000_val_20261031" /
                    "densenet121_train8000" / "best_densenet121.pt",
        "train10000": nih / "learning_curve_10000_train_1000_val_20261031" /
                     "densenet121_train10000" / "best_densenet121.pt",
    }
    outputs: dict[str, np.ndarray] = {}
    results = {}
    for name, checkpoint_path in candidate_paths.items():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("labels") != list(NIH_LABELS):
            raise ValueError(f"Label order mismatch in {checkpoint_path}")
        if checkpoint.get("training_images") not in (8000, 10000):
            raise ValueError(f"Unexpected training cohort in {checkpoint_path}")
        model = make_model(pretrained=False, architecture=checkpoint["architecture"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device).eval()
        metrics, y, probabilities = evaluate(model, loader, device)
        outputs[name] = probabilities
        metrics["checkpoint"] = str(checkpoint_path)
        metrics["checkpoint_sha256"] = sha256(checkpoint_path)
        metrics["training_images"] = checkpoint.get("training_images")
        metrics["selected_epoch"] = checkpoint.get("best_epoch")
        metrics["patient_bootstrap_95_ci"] = patient_bootstrap_micro(y, probabilities, patient_ids)
        results[name] = metrics

    predictions = pd.DataFrame({"Image Index": dataset.rows["Image Index"].astype(str),
                                "Patient ID": patient_ids})
    for col, label in enumerate(NIH_LABELS):
        predictions[f"target_{label}"] = y[:, col].astype(int)
        for name, probabilities in outputs.items():
            predictions[f"score_{name}_{label}"] = probabilities[:, col]
    predictions.to_csv(args.output_dir / "predictions_99.csv", index=False)
    summary = {
        "dataset": "NIH ChestX-ray14 fresh development train remainder",
        "purpose": "exploratory independent-patient check on frozen 8k and 10k checkpoints",
        "device": str(device),
        "threshold": 0.5,
        "threshold_tuned": False,
        "official_test_used": False,
        "audit": audit,
        "results": results,
        "paired_micro_deltas_10000_minus_8000": paired_micro_deltas(
            y, outputs["train8000"], outputs["train10000"], patient_ids
        ),
        "warning": "This 99-patient subset is too small for dependable rare-label estimates; not clinical performance.",
    }
    (args.output_dir / "metrics_99.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"audit": audit, "metrics": results}, indent=2))


if __name__ == "__main__":
    main()
