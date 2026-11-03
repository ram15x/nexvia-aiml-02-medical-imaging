"""Plot per-label validation metrics for the current demo checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    rows = sorted(report["per_label"], key=lambda item: item["auroc"])
    labels = [row["label"].replace("_", " ") for row in rows]
    y = np.arange(len(rows))
    auc = [row["auroc"] for row in rows]
    ap = [row["average_precision"] for row in rows]
    prevalence = [row["prevalence"] for row in rows]

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    fig, (auc_ax, ap_ax) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    fig.suptitle(
        "Current DenseNet on the selected validation split\n"
        "1,779 images · 548 patients · validation-only model-selection evidence",
        fontsize=14,
        fontweight="bold",
    )
    auc_ax.barh(y, auc, color="#4472C4", alpha=0.88)
    auc_ax.axvline(0.5, color="#333333", linestyle="--", linewidth=1, label="Random ranking")
    auc_ax.set_xlim(0.4, 1.0)
    auc_ax.set_xlabel("AUROC")
    auc_ax.set_title("Ranking discrimination")
    auc_ax.set_yticks(y, labels)
    auc_ax.grid(axis="x", alpha=0.22)
    auc_ax.legend(loc="lower right", frameon=False)

    ap_ax.barh(y, ap, color="#70AD47", alpha=0.88, label="Average precision")
    ap_ax.scatter(prevalence, y, marker="|", s=140, linewidths=2.0, color="#222222",
                  label="Positive prevalence reference", zorder=3)
    ap_ax.set_xlim(0, max(0.35, max(ap) * 1.12))
    ap_ax.set_xlabel("Average precision / prevalence")
    ap_ax.set_title("Rare-label ranking")
    ap_ax.grid(axis="x", alpha=0.22)
    ap_ax.legend(loc="lower right", frameon=False)

    for ax, values in ((auc_ax, auc), (ap_ax, ap)):
        for index, value in enumerate(values):
            ax.text(value + 0.006, index, f"{value:.2f}", va="center", fontsize=8, color="#333333")
    fig.text(
        0.5,
        0.015,
        "Per-label validation positives range from 5 to 217. The checkpoint was selected on this same split; "
        "these values are optimistic and are not independent test or clinical evidence.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
