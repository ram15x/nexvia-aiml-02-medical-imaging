"""Evaluate a frozen NIH DenseNet checkpoint once on the official held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from api.modeling import ARCHITECTURE_NAMES
from data_audit import NIH_LABELS
from train_classifier import NIHChestXrayDataset, evaluate, make_model, seed_everything


OFFICIAL_NIH_TEST_IMAGES = 25_596


def classify_test_scope(manifest_images: int, missing_images: int) -> tuple[str, bool]:
    """Label a manifest-defined subset accurately; local completeness isn't official completeness."""
    complete = manifest_images == OFFICIAL_NIH_TEST_IMAGES and missing_images == 0
    scope = "official held-out test" if complete else "official held-out test subset"
    return scope, complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-available-subset",
        action="store_true",
        help="Explicitly evaluate only local test images and label the output as a subset.",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("--batch-size must be positive and --workers cannot be negative")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("architecture")
    if architecture not in ARCHITECTURE_NAMES:
        raise ValueError("Checkpoint architecture is unsupported")
    if checkpoint.get("labels") != list(NIH_LABELS):
        raise ValueError("Checkpoint label order does not match NIH_LABELS")

    # This loader requires every official test image to exist. It never permits a subset.
    test = NIHChestXrayDataset(
        args.manifest,
        args.image_root,
        split="test",
        limit=None,
        seed=args.seed,
        allow_available_subset=args.allow_available_subset,
    )
    manifest_test_images = test.available_images + test.missing_images
    split_scope, complete_test = classify_test_scope(manifest_test_images, test.missing_images)
    loader = DataLoader(
        test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = make_model(pretrained=False, architecture=architecture)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    metrics, _, _ = evaluate(model, loader, device)

    result = {
        "dataset": "NIH ChestX-ray14",
        "split": split_scope,
        "test_images": len(test),
        "test_manifest_images": manifest_test_images,
        "official_test_images": OFFICIAL_NIH_TEST_IMAGES,
        "test_patients": int(test.rows["Patient ID"].nunique()),
        "test_images_missing_from_local_dataset": test.missing_images,
        "complete_test_split_evaluated": complete_test,
        "checkpoint": str(args.checkpoint),
        "architecture": architecture,
        "model_name": checkpoint.get("model_name", ARCHITECTURE_NAMES[architecture]),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "checkpoint_validation_metrics": checkpoint.get("validation_metrics"),
        "test_metrics": metrics,
        "threshold_policy": "fixed 0.5; no test-set threshold tuning",
        "test_split_used_for_training_or_selection": False,
        "clinical_use": "not suitable for diagnosis or care decisions",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
