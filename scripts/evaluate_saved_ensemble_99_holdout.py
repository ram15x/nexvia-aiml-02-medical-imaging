"""Compare frozen 8k, 10k, and equal-average predictions on a 99-patient holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)


LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
CANDIDATES = ["train8000", "train10000", "equal_average_8k_10k"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    defined_auc, defined_ap = [], []
    per_label = {}
    for col, label in enumerate(LABELS):
        target, probability = y[:, col], scores[:, col]
        auc = float(roc_auc_score(target, probability)) if np.unique(target).size == 2 else None
        ap = float(average_precision_score(target, probability)) if target.sum() else None
        if auc is not None:
            defined_auc.append(auc)
        if ap is not None:
            defined_ap.append(ap)
        prediction = probability >= 0.5
        per_label[label] = {
            "positive": int(target.sum()), "negative": int(len(target) - target.sum()),
            "auroc": auc, "average_precision": ap,
            "precision_at_0_5": float(precision_score(target, prediction, zero_division=0)),
            "recall_at_0_5": float(recall_score(target, prediction, zero_division=0)),
            "f1_at_0_5": float(f1_score(target, prediction, zero_division=0)),
        }
    flat_y, flat_p = y.ravel(), scores.ravel()
    return {
        "macro_auroc_defined_labels": float(np.mean(defined_auc)) if defined_auc else None,
        "macro_auroc_label_count": len(defined_auc),
        "macro_average_precision_defined_labels": float(np.mean(defined_ap)) if defined_ap else None,
        "macro_average_precision_label_count": len(defined_ap),
        "micro_auroc": float(roc_auc_score(flat_y, flat_p)) if np.unique(flat_y).size == 2 else None,
        "micro_average_precision": float(average_precision_score(flat_y, flat_p)) if flat_y.sum() else None,
        "micro_precision_at_0_5": float(precision_score(flat_y, flat_p >= 0.5, zero_division=0)),
        "micro_recall_at_0_5": float(recall_score(flat_y, flat_p >= 0.5, zero_division=0)),
        "micro_f1_at_0_5": float(f1_score(flat_y, flat_p >= 0.5, zero_division=0)),
        "labels": per_label,
    }


def bootstrap_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    aucs, aps = [], []
    for col in range(y.shape[1]):
        target, probability = y[:, col], scores[:, col]
        if np.unique(target).size == 2:
            aucs.append(float(roc_auc_score(target, probability)))
        if target.sum():
            aps.append(float(average_precision_score(target, probability)))
    flat_y, flat_p = y.ravel(), scores.ravel()
    return {
        "macro_auroc_defined_labels": float(np.mean(aucs)) if aucs else None,
        "macro_average_precision_defined_labels": float(np.mean(aps)) if aps else None,
        "micro_f1_at_0_5": float(f1_score(flat_y, flat_p >= 0.5, zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("use at least 100 bootstrap samples")

    pred = pd.read_csv(args.predictions)
    manifest = pd.read_csv(args.manifest)
    if len(pred) != 99 or pred["Patient ID"].nunique() != 99:
        raise ValueError("Expected exactly 99 image rows from 99 unique patients")
    if len(manifest) != 99 or manifest["Patient ID"].nunique() != 99:
        raise ValueError("Expected exactly 99 manifest rows from 99 unique patients")
    if set(pred["Image Index"]) != set(manifest["Image Index"]):
        raise ValueError("Prediction image IDs do not match the holdout manifest")
    manifest_patients = manifest.set_index("Image Index")["Patient ID"].astype(str)
    pred_patients = pred.set_index("Image Index")["Patient ID"].astype(str)
    if not manifest_patients.sort_index().equals(pred_patients.sort_index()):
        raise ValueError("Prediction patient IDs do not match the manifest")
    truth_cols = [f"target_{name}" for name in LABELS]
    if not set(truth_cols).issubset(pred.columns):
        raise ValueError("Prediction table is missing target columns")
    y = pred[truth_cols].to_numpy(dtype=np.int8)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Expected binary 0/1 labels")

    score_matrices = {
        "train8000": pred[[f"score_train8000_{name}" for name in LABELS]].to_numpy(dtype=float),
        "train10000": pred[[f"score_train10000_{name}" for name in LABELS]].to_numpy(dtype=float),
    }
    score_matrices["equal_average_8k_10k"] = (
        score_matrices["train8000"] + score_matrices["train10000"]
    ) / 2
    if not all(np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
               for scores in score_matrices.values()):
        raise ValueError("Scores must be finite probabilities in [0, 1]")

    summaries = {name: metrics(y, scores) for name, scores in score_matrices.items()}
    patients = pred["Patient ID"].astype(str).to_numpy()
    unique_patients = np.unique(patients)
    patient_rows = {pid: np.flatnonzero(patients == pid) for pid in unique_patients}
    rng = np.random.default_rng(args.seed)
    delta_samples = {"macro_auroc_defined_labels": [],
                     "macro_average_precision_defined_labels": [],
                     "micro_f1_at_0_5": []}
    for _ in range(args.bootstrap_samples):
        chosen = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        ix = np.concatenate([patient_rows[pid] for pid in chosen])
        metric_a = bootstrap_metrics(y[ix], score_matrices["equal_average_8k_10k"][ix])
        metric_b = bootstrap_metrics(y[ix], score_matrices["train8000"][ix])
        for key, values in delta_samples.items():
            va, vb = metric_a[key], metric_b[key]
            if va is not None and vb is not None and np.isfinite(va - vb):
                values.append(float(va - vb))
    paired = {}
    for key, values in delta_samples.items():
        paired[key] = {
            "observed_delta_ensemble_minus_8k": (
                summaries["equal_average_8k_10k"][key] - summaries["train8000"][key]
                if summaries["equal_average_8k_10k"][key] is not None
                and summaries["train8000"][key] is not None else None
            ),
            "paired_patient_bootstrap_95_ci": (
                [float(v) for v in np.quantile(values, [0.025, 0.975])] if values else None
            ),
            "valid_resamples": len(values),
        }

    per_label_rows = []
    for label in LABELS:
        for name in CANDIDATES:
            per_label_rows.append({"candidate": name, "label": label,
                                   **summaries[name]["labels"][label]})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_label_rows).to_csv(args.output_dir / "per_label_metrics.csv", index=False)
    result = {
        "dataset": "NIH ChestX-ray14 train remainder patient holdout",
        "rows": int(len(pred)), "unique_patients": int(len(unique_patients)),
        "threshold": 0.5, "threshold_tuned": False,
        "candidates": summaries, "paired_ensemble_minus_8k": paired,
        "positive_support_by_label": {name: int(y[:, i].sum()) for i, name in enumerate(LABELS)},
        "prediction_file_sha256": sha256(args.predictions),
        "manifest_sha256": sha256(args.manifest),
        "bootstrap_samples_requested": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "caveat": ("Exploratory only: 99 patients, very sparse labels, same-source report-mined NIH labels. "
                   "Do not treat this as clinical or external validation."),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
