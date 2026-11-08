"""Compare the frozen 8k/10k NIH checkpoints on a disjoint 5k-patient cohort.

The cohort is from the NIH official train/validation pool, not the official test
split. It was not used to train these two checkpoints or in saved prediction
files. A separate 15.5k model did train on it, so that model is not evaluated.
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


def prepare_holdout(project_root: Path, output_dir: Path) -> tuple[Path, dict]:
    nih = project_root / "data" / "NexviaProjectData" / "nih"
    metadata = nih / "metadata"
    additional_root = metadata / "additional_5k_patient_split_20261030"
    cohort = pd.read_csv(additional_root / "additional_train_manifest.csv")
    audit_source = json.loads((additional_root / "selection_audit.json").read_text(encoding="utf-8"))
    integrity = json.loads((additional_root / "integrity_verification.json").read_text(encoding="utf-8"))
    if len(cohort) != 5000 or cohort["Patient ID"].nunique() != 5000:
        raise ValueError("Expected 5,000 unique patients in the additional cohort")
    if audit_source.get("source") != "NIH ChestX-ray14 official train_val_list and Data_Entry_2017_v2020.csv":
        raise ValueError("Unexpected source for additional cohort")
    if audit_source.get("holdout_patients_added_to_training") is not False:
        raise ValueError("The source audit does not confirm official holdout exclusion")
    if integrity.get("additional_images_verified") != 5000 or integrity.get("train_missing") != 0:
        raise ValueError("The saved integrity audit does not confirm all 5,000 images")

    current_train_ids: set[str] = set()
    current_train_patients: set[str] = set()
    for size in (8000, 10000):
        stage = nih / f"learning_curve_{size}_train_1000_val_20261031"
        manifest = pd.read_csv(stage / f"manifest_{size}_train_1000_validation.csv")
        train = manifest[manifest["split"] == "train"]
        if len(train) != size or train["Patient ID"].nunique() != size:
            raise ValueError(f"Unexpected {size}-patient training manifest")
        current_train_ids.update(train["Image Index"].astype(str))
        current_train_patients.update(train["Patient ID"].astype(str))
    demo_manifest = pd.read_csv(nih / "trainval_available_20261028.csv")
    demo_train = demo_manifest[demo_manifest["split"] == "train"]
    if len(demo_train) != 10761:
        raise ValueError("Unexpected current demo checkpoint training manifest")
    current_train_ids.update(demo_train["Image Index"].astype(str))
    current_train_patients.update(demo_train["Patient ID"].astype(str))
    cohort_ids = set(cohort["Image Index"].astype(str))
    cohort_patients = set(cohort["Patient ID"].astype(str))
    if cohort_ids & current_train_ids or cohort_patients & current_train_patients:
        raise ValueError("Additional cohort overlaps a candidate checkpoint's training data")

    # Confirm that the additional cohort is outside the local NIH official test list.
    official_test_names = set((metadata / "test_list.txt").read_text(encoding="utf-8").splitlines())
    official_test_overlap = cohort_ids & official_test_names
    if official_test_overlap:
        raise ValueError(f"Additional cohort overlaps {len(official_test_overlap)} official-test images")

    # Prediction outputs from prior development must not already contain these cases.
    prior_predictions: set[str] = set()
    for path in nih.rglob("*predictions*.csv"):
        if output_dir.resolve() in path.resolve().parents:
            continue
        try:
            columns = pd.read_csv(path, nrows=0).columns
            if "Image Index" in columns:
                prior_predictions.update(
                    pd.read_csv(path, usecols=["Image Index"])["Image Index"].astype(str)
                )
        except (OSError, pd.errors.ParserError, ValueError):
            continue
    prediction_overlap = cohort_ids & prior_predictions
    if prediction_overlap:
        raise ValueError(f"The additional cohort has {len(prediction_overlap)} prior prediction rows")

    # Store an evaluation-only copy; split='validation' disables training augmentation.
    evaluation = cohort.copy()
    evaluation["split"] = "validation"
    manifest_path = output_dir / "manifest_additional_5000_validation.csv"
    evaluation.to_csv(manifest_path, index=False)
    supports = {}
    findings = evaluation["Finding Labels"].fillna("").astype(str)
    for label in NIH_LABELS:
        supports[label] = int(findings.map(lambda value: label in value.split("|")).sum())
    audit = {
        "cohort": "additional 5,000-patient NIH official train/validation cohort",
        "source_rows": len(cohort),
        "source_unique_patients": cohort["Patient ID"].nunique(),
        "candidate_checkpoints": ["current demo (10,761 train images)",
                                  "DenseNet-121 train8000", "DenseNet-121 train10000"],
        "candidate_training_overlap_rows": 0,
        "candidate_training_overlap_patients": 0,
        "prior_prediction_overlap": 0,
        "official_test_overlap_rows": 0,
        "official_test_images_loaded": False,
        "official_test_labels_loaded": False,
        "additional_cohort_previously_used_to_train_separate_15500_model": True,
        "that_15500_model_included_in_this_comparison": False,
        "new_downloads": 0,
        "label_support_positive": supports,
        "images_verified": integrity.get("additional_images_verified"),
        "image_total_bytes": audit_source.get("additional_training_images"),
        "evaluation_manifest_sha256": sha256(manifest_path),
        "warning": "NIH report-mined labels; patient-disjoint development evaluation, not external or clinical validation.",
    }
    audit["image_total_bytes"] = json.loads(
        (additional_root / "download_summary.json").read_text(encoding="utf-8")
    ).get("total_output_bytes")
    return manifest_path, audit


def paired_bootstrap(y: np.ndarray, predictions: dict[str, np.ndarray],
                     patients: np.ndarray, *, seed: int = 20261101,
                     samples: int = 1000) -> dict:
    unique = np.unique(patients)
    groups = {pid: np.flatnonzero(patients == pid) for pid in unique}
    rng = np.random.default_rng(seed)

    def fast_ap(target: np.ndarray, scores: np.ndarray) -> float | None:
        """Average precision with score ties grouped, matching sklearn's definition."""
        target = target.astype(np.int8, copy=False)
        positives = int(target.sum())
        if positives == 0:
            return None
        order = np.argsort(-scores, kind="mergesort")
        sorted_scores = scores[order]
        sorted_target = target[order]
        group_end = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(target) - 1]
        true_positives = np.cumsum(sorted_target, dtype=np.int64)[group_end]
        precision = true_positives / (group_end + 1)
        increments = np.diff(np.r_[0, true_positives]) / positives
        return float(np.sum(increments * precision))

    def point_metrics(target: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
        macro_aps = [average_precision_score(target[:, col], scores[:, col])
                     for col in range(target.shape[1]) if target[:, col].sum() > 0]
        aucs = [roc_auc_score(target[:, col], scores[:, col])
                for col in range(target.shape[1]) if len(np.unique(target[:, col])) == 2]
        yf, pf = target.reshape(-1), scores.reshape(-1)
        return {
            "macro_average_precision": float(np.mean(macro_aps)) if macro_aps else None,
            "macro_auroc": float(np.mean(aucs)) if aucs else None,
            "micro_auroc": float(roc_auc_score(yf, pf)) if len(np.unique(yf)) == 2 else None,
            "micro_average_precision": float(average_precision_score(yf, pf)) if yf.sum() else None,
            "micro_f1_at_0_5": float(f1_score(yf, pf >= 0.5, zero_division=0)),
        }

    point_scores = {name: point_metrics(y, p) for name, p in predictions.items()}
    pairs = [("demo_current", "train8000"), ("demo_current", "train10000"),
             ("train8000", "train10000")]
    pair_results = {}
    pair_draws = {}
    metric_names = ["macro_average_precision", "micro_auroc",
                    "micro_average_precision", "micro_f1_at_0_5"]
    for left_name, right_name in pairs:
        key = f"{right_name}_minus_{left_name}"
        left, right = point_scores[left_name], point_scores[right_name]
        pair_results[key] = {
            "left_candidate": left_name,
            "right_candidate": right_name,
            "left_metrics": left,
            "right_metrics": right,
            "delta_right_minus_left": {
                name: (right[name] - left[name]
                       if right[name] is not None and left[name] is not None else None)
                for name in point_scores[left_name]
            },
        }
        pair_draws[key] = {name: [] for name in metric_names}
    for _ in range(samples):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([groups[pid] for pid in sampled])
        target = y[rows]
        sampled_metrics = {}
        for candidate_name, candidate_scores in predictions.items():
            scores = candidate_scores[rows]
            macro_aps = [fast_ap(target[:, col], scores[:, col])
                         for col in range(target.shape[1]) if target[:, col].sum() > 0]
            yf, pf = target.reshape(-1), scores.reshape(-1)
            sampled_metrics[candidate_name] = {
                "macro_average_precision": float(np.mean(macro_aps)) if macro_aps else None,
                "micro_auroc": float(roc_auc_score(yf, pf)) if len(np.unique(yf)) == 2 else None,
                "micro_average_precision": fast_ap(yf, pf),
                "micro_f1_at_0_5": float(f1_score(yf, pf >= 0.5, zero_division=0)),
            }
        for left_name, right_name in pairs:
            pair_key = f"{right_name}_minus_{left_name}"
            for metric_name in metric_names:
                left, right = sampled_metrics[left_name][metric_name], sampled_metrics[right_name][metric_name]
                if left is not None and right is not None:
                    pair_draws[pair_key][metric_name].append(float(right - left))
    for pair_key, metric_draws in pair_draws.items():
        pair_results[pair_key]["paired_patient_bootstrap_95_ci"] = {
            metric: ([float(x) for x in np.quantile(values, [0.025, 0.975])]
                     if values else None)
            for metric, values in metric_draws.items()
        }
        pair_results[pair_key]["metrics_without_paired_ci"] = ["macro_auroc"]
    return {
        "point_metrics": point_scores,
        "pairwise_comparisons": pair_results,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    nih = args.project_root / "data" / "NexviaProjectData" / "nih"
    image_root = nih / "images_224_additional_5k_20261030"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_root}")
    manifest_path, audit = prepare_holdout(args.project_root, args.output_dir)
    dataset = NIHChestXrayDataset(
        manifest_path, image_root, "validation", None, args.seed,
        image_size=224, allow_available_subset=False,
    )
    if len(dataset) != 5000 or dataset.missing_images:
        raise ValueError(f"Expected 5,000 available images; found {len(dataset)} with {dataset.missing_images} missing")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_paths = {
        "demo_current": args.project_root / "artifacts" / "best_densenet121.pt",
        "train8000": nih / "learning_curve_8000_train_1000_val_20261031" /
                    "densenet121_train8000" / "best_densenet121.pt",
        "train10000": nih / "learning_curve_10000_train_1000_val_20261031" /
                     "densenet121_train10000" / "best_densenet121.pt",
    }
    predictions: dict[str, np.ndarray] = {}
    results = {}
    y_reference = None
    for name, checkpoint_path in candidate_paths.items():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected_training_images = {"demo_current": 10761, "train8000": 8000, "train10000": 10000}[name]
        if checkpoint.get("labels") != list(NIH_LABELS) or checkpoint.get("training_images") != expected_training_images:
            raise ValueError(f"Unexpected checkpoint schema or training size: {checkpoint_path}")
        model = make_model(pretrained=False, architecture=checkpoint["architecture"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device).eval()
        metric, targets, probabilities = evaluate(model, loader, device)
        if y_reference is not None and not np.array_equal(y_reference, targets):
            raise ValueError("Candidate checkpoints were not evaluated on identical labels")
        y_reference = targets
        predictions[name] = probabilities
        metric.update({
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "training_images": checkpoint.get("training_images"),
            "selected_epoch": checkpoint.get("best_epoch"),
        })
        results[name] = metric

    patient_ids = dataset.rows["Patient ID"].astype(str).to_numpy()
    prediction_frame = pd.DataFrame({
        "Image Index": dataset.rows["Image Index"].astype(str),
        "Patient ID": patient_ids,
    })
    for col, label in enumerate(NIH_LABELS):
        prediction_frame[f"target_{label}"] = y_reference[:, col].astype(int)
        for name, probability in predictions.items():
            prediction_frame[f"score_{name}_{label}"] = probability[:, col]
    prediction_frame.to_csv(args.output_dir / "predictions_additional_5k.csv", index=False)

    summary = {
        "dataset": "NIH ChestX-ray14 additional 5,000-patient development cohort",
        "device": str(device),
        "threshold": 0.5,
        "threshold_tuned": False,
        "audit": audit,
        "results": results,
        "paired_comparison": paired_bootstrap(y_reference, predictions, patient_ids),
        "interpretation": "Patient-disjoint comparison on report-mined NIH labels. Use macro AP as primary, AUROC/AP/F1 as secondary; do not call this external clinical validation.",
    }
    (args.output_dir / "metrics_additional_5k.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"audit": audit, "results": results,
                      "paired_comparison": summary["paired_comparison"]}, indent=2))


if __name__ == "__main__":
    main()
