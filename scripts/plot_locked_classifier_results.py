"""Plot per-label results from the frozen, one-time NIH subset evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "reports" / "classifier_locked_subset_test_results_2026-10-29.json"
OUTPUT = ROOT / "reports" / "classifier_locked_subset_test_per_label_2026-10-29.png"


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    if result.get("split") != "official held-out test subset":
        raise ValueError("Expected a subset-scoped locked test result")
    if result.get("complete_test_split_evaluated") is not False:
        raise ValueError("Refusing to plot a result mislabeled as the complete official test")
    metrics = result["test_metrics"]
    n = int(result["test_images"])
    rows = []
    for label, values in metrics["per_label"].items():
        if values["auroc"] is None or values["auprc"] is None:
            continue
        prevalence = values["positive_count"] / n
        rows.append((label, int(values["positive_count"]), float(values["auroc"]), float(values["auprc"]), prevalence))
    rows.sort(key=lambda row: row[2])

    labels = [f"{label}  (n+={count})" for label, count, *_ in rows]
    auc = [row[2] for row in rows]
    ap = [row[3] for row in rows]
    prevalence = [row[4] for row in rows]
    y = list(range(len(rows)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 7.4), sharey=True)
    axes[0].barh(y, auc, color="#39799b")
    axes[0].axvline(0.5, color="#8b3f3f", linestyle="--", linewidth=1, label="AUROC 0.5 reference")
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("AUROC")
    axes[0].set_title("Ranking discrimination")
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)

    axes[1].barh(y, ap, color="#cf8b38", label="Average precision")
    axes[1].scatter(prevalence, y, color="#303030", marker="|", s=100, label="Positive prevalence baseline", zorder=4)
    axes[1].set_xlim(0, max(0.5, max(ap) * 1.12))
    axes[1].set_xlabel("Average precision (AUPRC)")
    axes[1].set_title("Positive-class retrieval")
    axes[1].legend(loc="lower right", frameon=False, fontsize=8)

    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(
        f"DenseNet-121 on held-out subset: {n} images, {result['test_patients']} patients",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.015,
        "Per-label positive counts shown at left. Fixed 0.5 threshold was used only for confusion metrics; no test tuning.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
