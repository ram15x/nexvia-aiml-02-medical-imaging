"""Recreate the three-seed, validation-only DenseNet head comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
NIH = ROOT / "data" / "NexviaProjectData" / "nih"
REPORTS = ROOT / "reports"
SEEDS = (42, 11, 21)
METRICS = {
    "macro_auroc_defined_labels": "Macro AUROC",
    "macro_auprc_defined_labels": "Macro AUPRC",
    "micro_f1_at_0_5": "Micro F1 @ 0.5",
}


def read_history(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def selected_record(history: list[dict], epochs: int = 3) -> dict:
    eligible = [row for row in history if row["epoch"] <= epochs]
    if len(eligible) != epochs:
        raise ValueError(f"Expected {epochs} epochs, got {len(eligible)}")
    return max(eligible, key=lambda row: row["macro_auroc_defined_labels"])


def run_summary(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    records: dict[int, dict[str, dict]] = {}
    for seed in SEEDS:
        baseline_dir = NIH / f"research_comparison_subset_20261028_seed{seed}" / "densenet121"
        if seed == 42:
            baseline_dir = NIH / "research_comparison_subset_20261028_epochs10_seed42" / "densenet121"
        candidate_dir = NIH / f"evidence_concentration_gate_20261029_seed{seed}"

        candidate_summary = run_summary(candidate_dir / "run_summary.json")
        if candidate_summary.get("test_split_used") is not False:
            raise ValueError(f"Candidate seed {seed} reports test access")
        if candidate_summary.get("train_images") != 4000 or candidate_summary.get("validation_images") != 1779:
            raise ValueError(f"Unexpected split sizes for candidate seed {seed}")
        if candidate_summary.get("epochs") != 3:
            raise ValueError(f"Unexpected candidate epoch count for seed {seed}")

        baseline_summary = run_summary(baseline_dir / "run_summary.json")
        if baseline_summary.get("test_split_used") is not False:
            raise ValueError(f"Baseline seed {seed} reports test access")
        baseline_rows = read_history(baseline_dir / "training_history.json")
        candidate_rows = read_history(candidate_dir / "training_history.json")
        records[seed] = {
            "DenseNet-121": selected_record(baseline_rows),
            "DenseNet + evidence-concentration gate": selected_record(candidate_rows),
        }

    result = {
        "selection": "highest validation macro AUROC among epochs 1-3; threshold fixed at 0.5 for F1",
        "split": {"train_images": 4000, "validation_images": 1779, "test_split_used": False},
        "seeds": {},
        "mean": {},
        "mean_delta_candidate_minus_baseline": {},
    }
    models = tuple(next(iter(records.values())).keys())
    for seed, model_rows in records.items():
        result["seeds"][str(seed)] = {
            model: {
                "selected_epoch": row["epoch"],
                **{key: row[key] for key in METRICS},
            }
            for model, row in model_rows.items()
        }
    for metric in METRICS:
        result["mean"][metric] = {
            model: sum(records[s][model][metric] for s in SEEDS) / len(SEEDS)
            for model in models
        }
        result["mean_delta_candidate_minus_baseline"][metric] = (
            result["mean"][metric][models[1]] - result["mean"][metric][models[0]]
        )

    REPORTS.mkdir(exist_ok=True)
    json_path = REPORTS / "evidence_concentration_gate_comparison_2026-10-29.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    seed_colors = ("#3478a8", "#d28b2f", "#538a54")
    mean_colors = ("#173f5f", "#8a4b08")
    x = (0, 1)
    for ax, (metric, title) in zip(axes, METRICS.items()):
        all_vals = [records[s][m][metric] for s in SEEDS for m in models]
        for seed, color in zip(SEEDS, seed_colors):
            vals = [records[seed][model][metric] for model in models]
            ax.plot(x, vals, marker="o", linewidth=1.4, alpha=0.85, color=color)
        means = [result["mean"][metric][model] for model in models]
        ax.scatter(x, means, marker="D", s=54, color=mean_colors, edgecolor="white", linewidth=0.8, zorder=5)
        ax.set_xticks(x, ["DenseNet", "Custom head"], rotation=18, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        low, high = min(all_vals), max(all_vals)
        margin = max(0.012, (high - low) * 0.35)
        ax.set_ylim(max(0, low - margin), min(1, high + margin))
    handles = [
        Line2D([0], [0], color=color, marker="o", label=f"Seed {seed}")
        for seed, color in zip(SEEDS, seed_colors)
    ] + [Line2D([0], [0], color="white", marker="D", markerfacecolor="#333333", label="Three-seed means")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.065), ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Matched 3-epoch validation screen (three seeds)", fontsize=13)
    fig.text(0.5, 0.015, "Diamonds: DenseNet mean (blue) and custom-head mean (orange). Test split unopened.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.16, 1, 0.92))
    png_path = REPORTS / "evidence_concentration_gate_comparison_2026-10-29.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {json_path}")
    print(f"Wrote {png_path}")
    for metric, title in METRICS.items():
        baseline = result["mean"][metric][models[0]]
        candidate = result["mean"][metric][models[1]]
        print(f"{title}: baseline={baseline:.6f}, candidate={candidate:.6f}, delta={candidate - baseline:+.6f}")
    for seed in SEEDS:
        print(seed, {model: {key: round(row[key], 6) for key in METRICS} for model, row in records[seed].items()})


if __name__ == "__main__":
    main()
