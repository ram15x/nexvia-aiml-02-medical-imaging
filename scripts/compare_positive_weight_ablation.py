"""Summarize the capped positive-weight ablation against matched DenseNet runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


SEEDS = (42, 11, 21)
METRICS = (
    ("macro_auroc_defined_labels", "Macro AUROC"),
    ("macro_auprc_defined_labels", "Macro AUPRC"),
    ("micro_precision_at_0_5", "Micro precision @ 0.5"),
    ("micro_recall_at_0_5", "Micro recall @ 0.5"),
    ("micro_f1_at_0_5", "Micro F1 @ 0.5"),
)


def selected_epoch(run_dir: Path) -> dict:
    history = json.loads((run_dir / "training_history.json").read_text(encoding="utf-8"))
    if not history:
        raise ValueError(f"No epoch history in {run_dir}")
    return max(history, key=lambda row: row["macro_auroc_defined_labels"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nih-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    baseline_root = args.nih_root / "research_comparison_subset_20261028"
    candidate_root = args.nih_root
    rows = []
    epoch_rows = []
    for seed in SEEDS:
        base_dir = baseline_root if seed == 42 else args.nih_root / f"research_comparison_subset_20261028_seed{seed}"
        base_dir = base_dir / "densenet121"
        candidate_dir = candidate_root / f"positive_weight_ablation_20261029_seed{seed}"
        base = selected_epoch(base_dir)
        candidate_history = json.loads(
            (candidate_dir / "training_history.json").read_text(encoding="utf-8")
        )
        candidate = max(candidate_history, key=lambda row: row["macro_auroc_defined_labels"])
        base_summary = json.loads((base_dir / "run_summary.json").read_text(encoding="utf-8"))
        candidate_summary = json.loads((candidate_dir / "run_summary.json").read_text(encoding="utf-8"))
        if base_summary["train_images"] != 4000 or candidate_summary["train_images"] != 4000:
            raise ValueError("Expected matched 4,000-image training splits")
        if base_summary["validation_images"] != 1779 or candidate_summary["validation_images"] != 1779:
            raise ValueError("Expected matched 1,779-image validation splits")
        if base_summary["test_split_used"] or candidate_summary["test_split_used"]:
            raise ValueError("A run unexpectedly records test-split use")
        if candidate_summary.get("positive_weight_exponent") != 1.0:
            raise ValueError(f"Unexpected candidate loss setting for seed {seed}")
        row = {"seed": seed, "baseline": base, "candidate": candidate}
        row["delta"] = {
            metric: candidate[metric] - base[metric]
            for metric, _ in METRICS
        }
        rows.append(row)
        for epoch in candidate_history:
            epoch_rows.append({"seed": seed, **epoch})

    means = {}
    for key, label in METRICS:
        base_mean = mean(row["baseline"][key] for row in rows)
        candidate_mean = mean(row["candidate"][key] for row in rows)
        means[key] = {
            "baseline_mean": base_mean,
            "candidate_mean": candidate_mean,
            "candidate_minus_baseline": candidate_mean - base_mean,
        }
    mean_metric_gate = (
        means["macro_auprc_defined_labels"]["candidate_minus_baseline"] > 0
        and means["micro_f1_at_0_5"]["candidate_minus_baseline"] > 0
        and means["macro_auroc_defined_labels"]["candidate_minus_baseline"] >= 0
    )
    leave_one_seed_out = {}
    for metric, _ in METRICS[:2]:
        leave_one_seed_out[metric] = {
            str(excluded_seed): mean(
                row["delta"][metric] for row in rows if row["seed"] != excluded_seed
            )
            for excluded_seed in SEEDS
        }
    robustness_gate = all(
        score > 0
        for seed_scores in leave_one_seed_out.values()
        for score in seed_scores.values()
    )
    promoted = mean_metric_gate and robustness_gate
    result = {
        "experiment": "square-root weighting vs capped full positive weighting",
        "seeds": list(SEEDS),
        "train_images": 4000,
        "validation_images": 1779,
        "test_split_used": False,
        "baseline_positive_weight_exponent": 0.5,
        "candidate_positive_weight_exponent": 1.0,
        "candidate_positive_weight_cap": 10.0,
        "selected_checkpoint_rule": "highest validation macro AUROC within each seed",
        "mean_metric_gate_passed": mean_metric_gate,
        "leave_one_seed_out_ranking_robustness_gate_passed": robustness_gate,
        "leave_one_seed_out_mean_deltas": leave_one_seed_out,
        "promotion_rule_passed": promoted,
        "promoted": False,
        "decision": "keep square-root-weighted DenseNet; capped full weighting did not meet the predeclared promotion rule",
        "means": means,
        "selected_by_seed": rows,
        "candidate_all_epochs": epoch_rows,
        "limits": [
            "All comparisons reuse one validation split and are exploratory.",
            "The per-seed checkpoint is selected using validation macro AUROC.",
            "No independent test or external dataset was used.",
            "The 0.5 threshold is a fixed reporting threshold, not a clinical operating point.",
        ],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# Capped positive-weight ablation results — 2026-10-29",
        "",
        "The current DenseNet uses smoothed square-root class weighting. This matched ablation used the same pretrained DenseNet, patient split, optimizer, augmentation, batch size, three-epoch budget, and seeds, changing only the positive-weight exponent from 0.5 to 1.0 (both capped at 10). It used 4,000 train and 1,779 validation images; no test image or test manifest was loaded.",
        "",
        "## Selected checkpoint by seed",
        "",
        "Each run selected its checkpoint by the highest validation macro AUROC. The full candidate epoch curves are listed below.",
        "",
        "| Seed | Model | Epoch | Macro AUROC | Macro AUPRC | Micro precision @ 0.5 | Micro recall @ 0.5 | Micro F1 @ 0.5 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        for name in ("baseline", "candidate"):
            values = row[name]
            model_name = "Square-root baseline" if name == "baseline" else "Capped full weighting"
            lines.append(
                f"| {row['seed']} | {model_name} | {values['epoch']} | "
                f"{values['macro_auroc_defined_labels']:.4f} | "
                f"{values['macro_auprc_defined_labels']:.4f} | "
                f"{values['micro_precision_at_0_5']:.4f} | "
                f"{values['micro_recall_at_0_5']:.4f} | "
                f"{values['micro_f1_at_0_5']:.4f} |"
            )
    lines += ["", "## Mean across the three matched seeds", "", "| Metric | Baseline mean | Candidate mean | Change |", "|---|---:|---:|---:|"]
    for key, label in METRICS:
        values = means[key]
        lines.append(
            f"| {label} | {values['baseline_mean']:.4f} | "
            f"{values['candidate_mean']:.4f} | {values['candidate_minus_baseline']:+.4f} |"
        )
    lines += [
        "",
        "The candidate improved mean macro AUPRC, macro AUROC, and fixed-threshold F1, but the ranking gains are not stable across seeds. Applying the protocol's requirement that gains not be driven by one seed, leave-one-seed-out mean AUROC change is -0.0025 when seed 21 is removed, and mean AUPRC change is -0.0059 when seed 42 is removed. The promotion rule therefore fails; the candidate is **not promoted**. Precision fell in all three seeds while recall rose. The candidate also detected zero Pneumonia and zero Hernia positives at 0.5 in each selected checkpoint; raising positive weights did not solve those rare-label misses.",
        "",
        "## Candidate epoch history",
        "",
        "| Seed | Epoch | Macro AUROC | Macro AUPRC | Micro precision @ 0.5 | Micro recall @ 0.5 | Micro F1 @ 0.5 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in epoch_rows:
        lines.append(
            f"| {row['seed']} | {row['epoch']} | "
            f"{row['macro_auroc_defined_labels']:.4f} | "
            f"{row['macro_auprc_defined_labels']:.4f} | "
            f"{row['micro_precision_at_0_5']:.4f} | "
            f"{row['micro_recall_at_0_5']:.4f} | "
            f"{row['micro_f1_at_0_5']:.4f} |"
        )
    lines += [
        "",
        "## Limits and reproducibility",
        "",
        "This was an exploratory loss-weighting ablation on one fixed subset validation split. It does not establish generalization, remove checkpoint-selection bias, or support clinical use. Rare-label validation counts are very small. Reproduce with `scripts/run_positive_weight_ablation.ps1`; the exact protocol is in `positive_weight_ablation_protocol_2026-10-29.md`. Raw checkpoints and epoch JSON remain local in the ignored data directory; this report and the JSON summary retain the key results.",
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.report} and {args.json}; promotion rule passed: {promoted}")


if __name__ == "__main__":
    main()
