"""Evaluate a fixed equal-score ensemble from matched 500-patient OOF outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_audit import NIH_LABELS  # noqa: E402
from scripts.run_500_image_oof_architecture_comparison import summarize  # noqa: E402


def stack(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    return np.column_stack([
        frame[f"score_{prefix}_{label}"].to_numpy(dtype=np.float64)
        for label in NIH_LABELS
    ])


def main() -> None:
    nih = PROJECT_ROOT / "data" / "NexviaProjectData" / "nih"
    run = nih / "oof_500_backbone_10epoch_20261031"
    predictions = pd.read_csv(run / "oof_predictions.csv")
    manifest = pd.read_csv(run / "cohort_manifest.csv")
    if len(predictions) != 500 or predictions["Image Index"].nunique() != 500:
        raise ValueError("Expected 500 unique OOF image rows")
    if predictions["Image Index"].tolist() != manifest["Image Index"].tolist():
        raise ValueError("OOF prediction order does not exactly match the cohort manifest")
    if manifest["Patient ID"].nunique() != 500:
        raise ValueError("Expected exactly one image per unique patient")

    y = np.column_stack([
        predictions[f"target_{label}"].to_numpy(dtype=np.float64)
        for label in NIH_LABELS
    ])
    dense = stack(predictions, "densenet121")
    efficient = stack(predictions, "efficientnet_b0")
    ensemble = (dense + efficient) / 2.0
    result = {
        "purpose": "post-hoc, fixed equal-weight score ensemble screen from matched OOF predictions",
        "images": len(predictions),
        "unique_patients": int(manifest["Patient ID"].nunique()),
        "folds": 5,
        "threshold": 0.5,
        "official_test_rows_used": 0,
        "ensemble_rule": "per-label probability = 0.5 * DenseNet-121 OOF score + 0.5 * EfficientNet-B0 OOF score; no learned weights or threshold tuning",
        "models": {
            "DenseNet-121": {"metrics": summarize(y, dense)},
            "EfficientNet-B0": {"metrics": summarize(y, efficient)},
            "Equal-average score ensemble": {"metrics": summarize(y, ensemble)},
        },
        "point_delta_ensemble_minus_densenet": {},
        "point_delta_ensemble_minus_efficientnet": {},
        "limitations": [
            "This is a post-hoc screen on the same 500-patient OOF cohort already used for architecture comparisons.",
            "Each component prediction is out-of-fold, but inspecting multiple model combinations adds model-selection bias.",
            "The result cannot establish external generalization, clinical utility, or diagnostic accuracy.",
            "No official-test rows were used and the live demo checkpoint was not changed.",
        ],
    }
    for model_name, reference in (("DenseNet-121", dense), ("EfficientNet-B0", efficient)):
        comparison = summarize(y, ensemble)
        baseline = summarize(y, reference)
        deltas = {
            key: float(comparison[key] - baseline[key])
            for key in ("macro_auroc_defined_labels", "macro_auprc_positive_labels",
                        "micro_precision_at_0_5", "micro_recall_at_0_5", "micro_f1_at_0_5")
        }
        field = ("point_delta_ensemble_minus_densenet" if model_name == "DenseNet-121"
                 else "point_delta_ensemble_minus_efficientnet")
        result[field] = {"reference": model_name, "deltas": deltas}
    json_path = PROJECT_ROOT / "reports" / "500_image_equal_average_ensemble_2026-10-31.json"
    md_path = PROJECT_ROOT / "reports" / "500_image_equal_average_ensemble_2026-10-31.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    names = list(result["models"])
    table = ["| Model | Macro AUROC | Macro AP | Precision @0.5 | Recall @0.5 | Micro F1 @0.5 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in names:
        m = result["models"][name]["metrics"]
        table.append(
            f"| {name} | {m['macro_auroc_defined_labels']:.4f} | "
            f"{m['macro_auprc_positive_labels']:.4f} | "
            f"{m['micro_precision_at_0_5']:.4f} | {m['micro_recall_at_0_5']:.4f} | "
            f"{m['micro_f1_at_0_5']:.4f} |"
        )
    report = [
        "# Fixed equal-average ensemble screen on matched 500-patient OOF predictions",
        "",
        "**Date:** 2026-10-31  ",
        "**Scope:** existing 500-image, 500-patient, five-fold OOF outputs; no retraining, image reads, or official-test rows.",
        "",
        "## Method",
        "",
        result["ensemble_rule"],
        "The ensemble rule was fixed without fitted stacking weights. Metrics use the same evaluator and fixed 0.5 cutoff as the component comparison.",
        "",
        "## Results",
        "",
        *table,
        "",
        "Paired patient-row bootstrap differences (ensemble minus each model):",
        "",
    ]
    for key, heading in (("point_delta_ensemble_minus_densenet", "DenseNet-121"),
                         ("point_delta_ensemble_minus_efficientnet", "EfficientNet-B0")):
        report.append(f"- Versus {heading}:")
        for metric, delta in result[key]["deltas"].items():
            report.append(f"  - {metric}: {delta:+.4f} (point difference; no interval calculated)")
    report += [
        "",
        "## Limits",
        "",
        *[f"- {item}" for item in result["limitations"]],
        "",
        "This is a cheap test of whether the two backbones' existing OOF score streams combine usefully. It does not train a new hybrid feature architecture or erase the selection bias from repeatedly inspecting this cohort. The production checkpoint remains unchanged.",
        "",
        f"Machine-readable output: `{json_path.relative_to(PROJECT_ROOT).as_posix()}`.",
    ]
    md_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(md_path),
        "metrics": {name: result["models"][name]["metrics"] for name in names},
        "delta_vs_densenet": result["point_delta_ensemble_minus_densenet"]["deltas"],
        "delta_vs_efficientnet": result["point_delta_ensemble_minus_efficientnet"]["deltas"],
    }, indent=2))


if __name__ == "__main__":
    main()
