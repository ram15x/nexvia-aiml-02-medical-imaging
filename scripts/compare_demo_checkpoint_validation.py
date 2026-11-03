"""Compare the current demo and larger-data DenseNet on validation data only."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_audit import NIH_LABELS
from train_classifier import NIHChestXrayDataset, evaluate, make_model, seed_everything


DATA = ROOT / "data" / "NexviaProjectData" / "nih"
MANIFEST = DATA / "trainval_available_20261028.csv"
IMAGE_ROOT = DATA / "images"
CHECKPOINTS = {
    "Previous demo checkpoint (3,466 train images)": ROOT / "artifacts" / "best_densenet121_shard001_v1.pt",
    "Larger-data candidate (10,761 train images)": DATA / "classifier_available_trainval_3epochs_20261028" / "best_densenet121.pt",
}
CURRENT_NAME, CANDIDATE_NAME = list(CHECKPOINTS)
OUTPUT_JSON = ROOT / "reports" / "validation_previous_vs_larger_candidate_2026-10-29.json"
OUTPUT_PNG = ROOT / "reports" / "validation_previous_vs_larger_candidate_2026-10-29.png"
METRIC_KEYS = (
    ("macro_auroc_defined_labels", "Macro AUROC"),
    ("macro_auprc_defined_labels", "Macro AUPRC"),
    ("micro_f1_at_0_5", "Micro F1 @ 0.5"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    seed_everything(42)
    torch.set_num_threads(min(8, torch.get_num_threads()))

    manifest_splits = set(pd.read_csv(MANIFEST, usecols=["split"])["split"].astype(str))
    if manifest_splits != {"train", "validation"}:
        raise ValueError("The comparison manifest must contain training and validation rows only")
    dataset = NIHChestXrayDataset(
        MANIFEST,
        IMAGE_ROOT,
        split="validation",
        limit=None,
        seed=42,
    )
    if len(dataset) != 1779 or dataset.rows["Patient ID"].nunique() != 548:
        raise ValueError("Validation manifest does not match the frozen 1,779-image / 548-patient split")
    if "test" in set(dataset.rows["split"].astype(str)):
        raise ValueError("A test row appeared in the validation dataset")

    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    rows = {}
    device = torch.device("cpu")
    current_path = CHECKPOINTS[CURRENT_NAME]
    current_checkpoint = torch.load(current_path, map_location="cpu", weights_only=False)
    if current_checkpoint.get("architecture") != "densenet121" or current_checkpoint.get("labels") != list(NIH_LABELS):
        raise ValueError(f"Unexpected architecture or label ordering in {current_path}")
    current_model = make_model(pretrained=False, architecture="densenet121")
    current_model.load_state_dict(current_checkpoint["state_dict"], strict=True)
    current_model.to(device)
    current_metrics, _, _ = evaluate(current_model, loader, device)
    rows[CURRENT_NAME] = {
        "checkpoint": str(current_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(current_path),
        "checkpoint_best_epoch": current_checkpoint.get("best_epoch"),
        "recomputed_validation_metrics": current_metrics,
        "metric_source": "recomputed on the common 1,779-row validation set",
    }
    del current_model, current_checkpoint

    candidate_path = CHECKPOINTS[CANDIDATE_NAME]
    candidate_checkpoint = torch.load(candidate_path, map_location="cpu", weights_only=False)
    if candidate_checkpoint.get("architecture") != "densenet121" or candidate_checkpoint.get("labels") != list(NIH_LABELS):
        raise ValueError(f"Unexpected architecture or label ordering in {candidate_path}")
    candidate_metrics = candidate_checkpoint.get("validation_metrics")
    if not candidate_metrics or candidate_metrics.get("images") != len(dataset):
        raise ValueError("Saved larger-data candidate metrics do not match the common validation size")
    rows[CANDIDATE_NAME] = {
        "checkpoint": str(candidate_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(candidate_path),
        "checkpoint_best_epoch": candidate_checkpoint.get("best_epoch"),
        "recomputed_validation_metrics": candidate_metrics,
        "metric_source": "saved from candidate training run on this exact validation manifest",
    }
    del candidate_checkpoint

    names = list(rows)
    deltas = {
        key: rows[names[1]]["recomputed_validation_metrics"][key]
        - rows[names[0]]["recomputed_validation_metrics"][key]
        for key, _ in METRIC_KEYS
    }
    result = {
        "dataset": "NIH ChestX-ray14",
        "split": "validation only; no held-out test manifest or test images accessed",
        "validation_images": len(dataset),
        "validation_patients": int(dataset.rows["Patient ID"].nunique()),
        "threshold_policy": "fixed 0.5; no validation threshold fitting",
        "selection_caveat": "Both checkpoints were selected using validation data, and the larger-data checkpoint used this exact validation split; the current checkpoint is rescored here against it. This comparison is exploratory and optimistic, not an independent test.",
        "models": rows,
        "candidate_minus_current_validation_delta": deltas,
    }
    OUTPUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.1))
    colors = ("#436b8c", "#c27b32")
    labels = ("Previous demo", "More train data")
    for ax, (key, title) in zip(axes, METRIC_KEYS):
        values = [rows[name]["recomputed_validation_metrics"][key] for name in names]
        bars = ax.bar(range(2), values, color=colors, width=0.62)
        ax.set_xticks(range(2), labels, rotation=18, ha="right")
        ax.set_ylim(0, max(values) * 1.22)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Common 1,779-image validation split (not a test result)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"JSON: {OUTPUT_JSON}")
    print(f"Plot: {OUTPUT_PNG}")
    for key, title in METRIC_KEYS:
        current = rows[names[0]]["recomputed_validation_metrics"][key]
        candidate = rows[names[1]]["recomputed_validation_metrics"][key]
        print(f"{title}: current={current:.6f}, larger-data={candidate:.6f}, delta={candidate-current:+.6f}")


if __name__ == "__main__":
    main()
