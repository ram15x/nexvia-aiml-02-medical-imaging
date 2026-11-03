"""Compare EfficientNet-B0 with frozen DenseNet OOF scores on the same 500 patients."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_audit import NIH_LABELS  # noqa: E402
from scripts.run_500_image_oof_architecture_comparison import (  # noqa: E402
    bootstrap_intervals,
    paired_delta_intervals,
    summarize,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    return np.column_stack([
        frame[f"score_{prefix}_{label}"].to_numpy(dtype=np.float32)
        for label in NIH_LABELS
    ])


def main() -> None:
    nih = PROJECT_ROOT / "data" / "NexviaProjectData" / "nih"
    baseline_dir = nih / "oof_500_architecture_seed20261031"
    candidate_dir = nih / "oof_500_efficientnet_b0_20261031"
    baseline_csv = baseline_dir / "oof_predictions.csv"
    candidate_csv = candidate_dir / "oof_predictions.csv"
    baseline_manifest = baseline_dir / "cohort_manifest.csv"
    candidate_manifest = candidate_dir / "cohort_manifest.csv"
    baseline = pd.read_csv(baseline_csv)
    candidate = pd.read_csv(candidate_csv)
    if baseline["Image Index"].tolist() != candidate["Image Index"].tolist():
        raise ValueError("Candidate and baseline OOF rows do not match exactly")
    if len(candidate) != 500 or candidate["Image Index"].nunique() != 500:
        raise ValueError("Expected exactly 500 distinct image rows")
    if digest(baseline_manifest) != digest(candidate_manifest):
        raise ValueError("Candidate and baseline cohort manifests differ")

    fold_hashes = {}
    for fold in range(1, 6):
        a = baseline_dir / "fold_manifests" / f"fold_{fold:02d}.csv"
        b = candidate_dir / "fold_manifests" / f"fold_{fold:02d}.csv"
        if digest(a) != digest(b):
            raise ValueError(f"Fold {fold} assignment differs from reference screen")
        fold_hashes[str(fold)] = digest(b)

    y = np.column_stack([
        candidate[f"target_{label}"].to_numpy(dtype=np.float32)
        for label in NIH_LABELS
    ])
    dense = scores(baseline, "densenet121")
    efficientnet = scores(candidate, "efficientnet_b0")
    result = {
        "purpose": "matched exploratory 500-patient OOF architecture comparison",
        "cohort_images": 500,
        "unique_patients": int(pd.read_csv(candidate_manifest)["Patient ID"].nunique()),
        "folds": 5,
        "epochs_per_fold": 2,
        "threshold": 0.5,
        "official_test_rows_used": 0,
        "cohort_manifest_sha256": digest(candidate_manifest),
        "fold_manifest_sha256": fold_hashes,
        "models": {
            "DenseNet-121": {
                "metrics": summarize(y, dense),
                "bootstrap_95_percent_intervals": bootstrap_intervals(y, dense, seed=20261031),
            },
            "EfficientNet-B0": {
                "metrics": summarize(y, efficientnet),
                "bootstrap_95_percent_intervals": bootstrap_intervals(y, efficientnet, seed=20261031),
            },
        },
        "paired_delta_efficientnet_minus_densenet": paired_delta_intervals(
            y, dense, efficientnet, seed=20261031
        ),
        "limitations": [
            "The 500 patients were already used in a prior pilot; this is exploratory, not a pristine benchmark.",
            "This was a fixed two-epoch architecture screen, not a converged best-effort comparison.",
            "Several findings have sparse support and Hernia has no positives in this cohort.",
            "Threshold 0.5 was fixed; no threshold fitting or official test rows were used.",
            "Selecting a model from these OOF results introduces model-selection bias.",
        ],
    }
    output_json = candidate_dir / "matched_comparison_vs_densenet.json"
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    rows = []
    for name, entry in result["models"].items():
        metrics = entry["metrics"]
        rows.append(
            f"| {name} | {metrics['macro_auroc_defined_labels']:.4f} | "
            f"{metrics['macro_auprc_positive_labels']:.4f} | "
            f"{metrics['micro_precision_at_0_5']:.4f} | "
            f"{metrics['micro_recall_at_0_5']:.4f} | "
            f"{metrics['micro_f1_at_0_5']:.4f} |"
        )
    delta = result["paired_delta_efficientnet_minus_densenet"]["deltas"]
    report = "\n".join([
        "# EfficientNet-B0 versus DenseNet-121 on the capped 500-patient cohort",
        "",
        "**Date:** 2026-10-31  ",
        "**Purpose:** a controlled, exploratory architecture screen for Project 1. This is not clinical evidence.",
        "",
        "## Matched setup",
        "",
        "Both models use the same 500 patient rows, exact same five 400/100 patient-level fold assignments, "
        "ImageNet initialization, 224px input, preprocessing, optimizer, loss weights, augmentation, seeds, "
        "fixed two epochs per fold, and 0.5 threshold. The existing DenseNet predictions are reused; the "
        "EfficientNet model was trained only on the same 500-patient cohort. No official test rows were used.",
        "",
        "## Pooled out-of-fold results",
        "",
        "| Model | Macro AUROC | Macro AP | Micro precision @0.5 | Micro recall @0.5 | Micro F1 @0.5 |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "Paired patient-row bootstrap deltas (EfficientNet minus DenseNet): "
        + "; ".join(
            f"{key} {value['point_difference']:+.4f} "
            f"(95% CI {value['lower_2_5_percent']:+.4f} to {value['upper_97_5_percent']:+.4f})"
            for key, value in delta.items()
        ) + ".",
        "",
        "## Decision",
        "",
        "EfficientNet-B0 shows a small numerical increase in pooled AUROC, AP, and F1 relative to the "
        "reference, but thresholded detection remains poor and the two-epoch fit is not converged. The "
        "500-patient sample is too small and sparse to select a final architecture from this screen. "
        "Do not promote EfficientNet or claim a validated accuracy gain. Keep DenseNet as the existing "
        "demo checkpoint and treat EfficientNet as an unpromoted research candidate.",
        "",
        "## Limits and next step",
        "",
        "This result cannot establish good detection, clinical utility, or the 0.85 AUROC aspiration. "
        "The model-selection process has reused development data, several labels have few positives, "
        "and Hernia has zero positives. The next useful step is not another hybrid head: improve the "
        "evaluation sample and training protocol within the agreed 500-patient cap, including a fair "
        "convergence/early-stopping protocol, before any model promotion. Keep the frozen test sample "
        "untouched.",
        "",
        "Machine-readable scores and fold histories are in `data/NexviaProjectData/nih/oof_500_efficientnet_b0_20261031/`.",
    ]) + "\n"
    report_path = PROJECT_ROOT / "reports" / "500_image_efficientnet_comparison_2026-10-31.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"json": str(output_json), "report": str(report_path),
                      "metrics": {name: values["metrics"] for name, values in result["models"].items()},
                      "paired_delta": result["paired_delta_efficientnet_minus_densenet"]}, indent=2))


if __name__ == "__main__":
    main()
