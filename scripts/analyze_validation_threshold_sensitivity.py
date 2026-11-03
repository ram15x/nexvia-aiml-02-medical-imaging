"""Measure how the current classifier's 0.5 cutoff affects validation F1.

This is a validation-only diagnostic. It does not train/select a model or read
the NIH test split. Thresholds are optimized on the same validation cohort used
for checkpoint selection, so optimized scores are explicitly optimistic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.modeling import ARCHITECTURE_NAMES
from data_audit import NIH_LABELS
from train_classifier import NIHChestXrayDataset, evaluate, make_model, seed_everything


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_f1_threshold(targets: np.ndarray, scores: np.ndarray) -> dict:
    """Select an empirical F1-maximizing threshold; ties choose nearest 0.5."""
    y = np.asarray(targets, dtype=np.int32).reshape(-1)
    p = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.shape != p.shape or len(y) == 0 or len(np.unique(y)) < 2:
        raise ValueError("threshold selection requires aligned scores with both target classes")
    precision, recall, thresholds = precision_recall_curve(y, p)
    if len(thresholds) == 0:
        raise ValueError("threshold selection produced no candidate thresholds")
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denom,
                   out=np.zeros_like(denom), where=denom > 0)
    best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=1e-12, atol=1e-12))
    chosen = int(best[np.argmin(np.abs(thresholds[best] - 0.5))])
    return {
        "threshold": float(thresholds[chosen]),
        "precision": float(precision[chosen]),
        "recall": float(recall[chosen]),
        "f1": float(f1[chosen]),
    }


def summarize_predictions(targets: np.ndarray, scores: np.ndarray,
                          thresholds: np.ndarray) -> dict:
    y = np.asarray(targets, dtype=np.uint8)
    p = np.asarray(scores, dtype=np.float64)
    cutoffs = np.asarray(thresholds, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != cutoffs.shape[0]:
        raise ValueError("targets, scores, and per-label thresholds have incompatible shapes")
    decisions = p >= cutoffs[None, :]
    return {
        "micro_precision": float(precision_score(y, decisions, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y, decisions, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y, decisions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y, decisions, average="macro", zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--minimum-positive-support", type=int, default=20,
                        help="Only labels at or above this validation support receive exploratory per-label thresholds.")
    args = parser.parse_args()
    if args.batch_size < 1 or args.minimum_positive_support < 1:
        parser.error("--batch-size and --minimum-positive-support must be positive")

    seed_everything(20261031)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = checkpoint.get("architecture")
    if architecture not in ARCHITECTURE_NAMES:
        raise ValueError("Checkpoint architecture is unsupported")
    if checkpoint.get("labels") != list(NIH_LABELS):
        raise ValueError("Checkpoint label order does not match NIH_LABELS")
    image_size = int(checkpoint.get("image_size", 224))

    validation = NIHChestXrayDataset(args.manifest, args.image_root, "validation", None,
                                     seed=20261031, image_size=image_size)
    if validation.missing_images:
        raise FileNotFoundError(f"Validation manifest has {validation.missing_images} missing images")
    if set(validation.rows["split"].astype(str)) != {"validation"}:
        raise ValueError("Validation loader unexpectedly contains non-validation rows")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    model = make_model(pretrained=False, architecture=architecture)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    fixed_metrics, targets, scores = evaluate(model, loader, device)

    per_label = {}
    thresholds = np.full(len(NIH_LABELS), 0.5, dtype=np.float64)
    for index, label in enumerate(NIH_LABELS):
        y = targets[:, index].astype(np.uint8)
        p = scores[:, index]
        support = int(y.sum())
        fixed = {
            "threshold": 0.5,
            "precision": float(precision_score(y, p >= 0.5, zero_division=0)),
            "recall": float(recall_score(y, p >= 0.5, zero_division=0)),
            "f1": float(f1_score(y, p >= 0.5, zero_division=0)),
        }
        row = {"positive_support": support, "fixed_0_5": fixed}
        if support >= args.minimum_positive_support:
            selected = select_f1_threshold(y, p)
            thresholds[index] = selected["threshold"]
            row["exploratory_f1_threshold"] = selected
            row["threshold_eligible"] = True
        else:
            row["exploratory_f1_threshold"] = None
            row["threshold_eligible"] = False
        per_label[label] = row

    global_selection = select_f1_threshold(targets, scores)
    global_thresholds = np.full(len(NIH_LABELS), global_selection["threshold"], dtype=np.float64)
    result = {
        "dataset": "NIH ChestX-ray14",
        "split": "development validation only",
        "images": int(len(targets)),
        "patients": int(validation.rows["Patient ID"].nunique()),
        "test_split_used": False,
        "image_size": image_size,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "validation_was_used_for_checkpoint_selection": True,
        "thresholds_are_deployment_recommendations": False,
        "threshold_policy": "empirical F1 maximum on this same validation cohort; ties choose closest to 0.5; per-label thresholds require at least the configured positive support",
        "minimum_positive_support": args.minimum_positive_support,
        "fixed_0_5_existing_metrics": {
            "micro_precision": fixed_metrics["micro_precision_at_0_5"],
            "micro_recall": fixed_metrics["micro_recall_at_0_5"],
            "micro_f1": fixed_metrics["micro_f1_at_0_5"],
            "macro_auroc": fixed_metrics["macro_auroc_defined_labels"],
            "macro_ap": fixed_metrics["macro_auprc_defined_labels"],
        },
        "single_global_f1_threshold": {
            **global_selection,
            "micro_metrics": summarize_predictions(targets, scores, global_thresholds),
        },
        "per_label_f1_thresholds": {
            "micro_metrics": summarize_predictions(targets, scores, thresholds),
            "labels": per_label,
        },
        "warning": "Both threshold optimization and prior checkpoint selection use this validation set. Optimized F1 is resubstitution/optimistic, is not a model-performance gain, and must not be presented as independent test performance.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
