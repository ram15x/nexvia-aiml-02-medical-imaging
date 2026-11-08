"""Compare an equal-probability average of the frozen 8k/10k CXR models.

This is an exploratory analysis of the already-used 5,000-patient NIH
development cohort. It does not train or promote a model. The paired bootstrap
quantifies row sampling variation but cannot remove model-selection bias.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def metric_summary(targets: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    aucs = [roc_auc_score(targets[:, j], scores[:, j])
            for j in range(targets.shape[1]) if np.unique(targets[:, j]).size == 2]
    aps = [average_precision_score(targets[:, j], scores[:, j])
           for j in range(targets.shape[1]) if targets[:, j].sum() > 0]
    hard = scores >= 0.5
    return {
        "macro_auroc": float(np.mean(aucs)) if aucs else float("nan"),
        "macro_average_precision": float(np.mean(aps)) if aps else float("nan"),
        "micro_precision_at_0_5": float(precision_score(targets.ravel(), hard.ravel(), zero_division=0)),
        "micro_recall_at_0_5": float(recall_score(targets.ravel(), hard.ravel(), zero_division=0)),
        "micro_f1_at_0_5": float(f1_score(targets.ravel(), hard.ravel(), zero_division=0)),
    }


def paired_bootstrap_deltas(targets: np.ndarray, reference: np.ndarray,
                            candidate: np.ndarray, *, samples: int,
                            seed: int) -> dict[str, list[float] | float]:
    rng = np.random.default_rng(seed)
    deltas = {"macro_auroc": [], "macro_average_precision": [], "micro_f1_at_0_5": []}
    reference_point = metric_summary(targets, reference)
    candidate_point = metric_summary(targets, candidate)
    for _ in range(samples):
        rows = rng.integers(0, len(targets), size=len(targets))
        ref = metric_summary(targets[rows], reference[rows])
        cand = metric_summary(targets[rows], candidate[rows])
        for key in deltas:
            delta = cand[key] - ref[key]
            if np.isfinite(delta):
                deltas[key].append(delta)
    result: dict[str, list[float] | float] = {}
    for key, values in deltas.items():
        result[f"{key}_valid_bootstrap_resamples"] = len(values)
        result[f"{key}_difference"] = candidate_point[key] - reference_point[key]
        result[f"{key}_paired_patient_bootstrap_95_ci"] = [
            float(x) for x in np.quantile(values, [0.025, 0.975])
        ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("Use at least 100 bootstrap samples")

    data = pd.read_csv(args.predictions)
    manifest = pd.read_csv(args.manifest)
    required_manifest = {"Image Index", "Patient ID"}
    if missing := required_manifest - set(manifest.columns):
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if "Image Index" not in data or "Patient ID" not in data:
        raise ValueError("Prediction table must contain Image Index and Patient ID")
    if data["Image Index"].duplicated().any() or manifest["Image Index"].duplicated().any():
        raise ValueError("Duplicate image IDs found")
    if set(data["Image Index"].astype(str)) != set(manifest["Image Index"].astype(str)):
        raise ValueError("Prediction image IDs do not exactly match the frozen cohort manifest")
    audit = data[["Image Index", "Patient ID"]].merge(
        manifest[["Image Index", "Patient ID"]], on="Image Index", suffixes=("_pred", "_manifest"),
        validate="one_to_one")
    if not (audit["Patient ID_pred"].astype(str) == audit["Patient ID_manifest"].astype(str)).all():
        raise ValueError("Patient IDs in predictions do not match the cohort manifest")
    if data["Patient ID"].nunique() != len(data):
        raise ValueError("Expected one prediction row per patient for this cohort")

    labels = [column.removeprefix("target_") for column in data.columns
              if column.startswith("target_")]
    targets = data[[f"target_{label}" for label in labels]].to_numpy(dtype=np.int8)
    p8 = data[[f"score_train8000_{label}" for label in labels]].to_numpy(dtype=np.float64)
    p10 = data[[f"score_train10000_{label}" for label in labels]].to_numpy(dtype=np.float64)
    ensemble = (p8 + p10) / 2.0
    if not np.isin(targets, [0, 1]).all() or not np.isfinite(np.concatenate([p8, p10], axis=1)).all():
        raise ValueError("Targets must be binary and scores finite")
    if ((p8 < 0) | (p8 > 1) | (p10 < 0) | (p10 > 1)).any():
        raise ValueError("Expected sigmoid probabilities in [0, 1]")

    summaries = {
        "densenet_8k": metric_summary(targets, p8),
        "densenet_10k": metric_summary(targets, p10),
        "equal_average_8k_10k": metric_summary(targets, ensemble),
    }
    paired = paired_bootstrap_deltas(
        targets, p8, ensemble, samples=args.bootstrap_samples, seed=args.seed)

    per_label = []
    for j, label in enumerate(labels):
        y = targets[:, j]
        row = {"label": label, "positive_support": int(y.sum())}
        for name, scores in (("8k", p8[:, j]), ("10k", p10[:, j]), ("ensemble", ensemble[:, j])):
            row[f"{name}_auroc"] = float(roc_auc_score(y, scores)) if np.unique(y).size == 2 else None
            row[f"{name}_ap"] = float(average_precision_score(y, scores)) if y.sum() else None
            row[f"{name}_f1_at_0_5"] = float(f1_score(y, scores >= 0.5, zero_division=0))
        per_label.append(row)
    per_label_frame = pd.DataFrame(per_label)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_label_frame.to_csv(args.output_dir / "per_label_metrics.csv", index=False)
    payload = {
        "dataset": "NIH ChestX-ray14 additional 5,000-patient development cohort",
        "scope": "exploratory post-hoc comparison; previously used for checkpoint promotion",
        "rows": int(len(data)),
        "unique_patients": int(data["Patient ID"].nunique()),
        "primary_metric": "macro AUROC; macro average precision is co-primary for this exploratory decision",
        "score_ensemble": "arithmetic mean of sigmoid probabilities from frozen 8k and 10k DenseNet-121 checkpoints",
        "threshold": 0.5,
        "threshold_tuned": False,
        "model_weights_changed": False,
        "official_test_used": False,
        "prediction_file_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "cohort_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "results": summaries,
        "paired_ensemble_minus_8k": paired,
        "interpretation": (
            "The ensemble's development-cohort ranking metrics improved, but this cohort already informed model selection. "
            "The bootstrap describes row sampling variation only and does not remove selection bias. "
            "Do not promote or claim independent generalization from this result."
        ),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, metric in zip(axes, ("auroc", "ap")):
        ax.barh(x - 0.18, per_label_frame[f"8k_{metric}"], height=0.35, label="8k DenseNet")
        ax.barh(x + 0.18, per_label_frame[f"ensemble_{metric}"], height=0.35, label="8k + 10k mean")
        ax.set_title("AUROC" if metric == "auroc" else "Average precision")
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(x, labels)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("NIH finding")
    axes[1].legend(loc="lower right")
    fig.suptitle("Exploratory score ensemble on the reused 5,000-patient cohort")
    fig.tight_layout()
    fig.savefig(args.output_dir / "per_label_metrics.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"output_dir": str(args.output_dir), "results": summaries,
                      "paired_ensemble_minus_8k": paired}, indent=2))


if __name__ == "__main__":
    main()
