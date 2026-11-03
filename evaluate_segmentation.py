"""Evaluate one frozen spleen checkpoint on its reserved case split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference

from train_segmentation import build_transform, make_model, read_cases, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True,
                        help="Frozen case_split.json created during training")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--foreground-threshold", type=float, default=None,
                        help="Override the checkpoint's validation-selected threshold (default: checkpoint value, or 0.5).")
    parser.add_argument("--flip-tta", action="store_true",
                        help="Average direct inference with axis-0 flipped inference.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("--workers cannot be negative")
    if args.foreground_threshold is not None and not 0.0 < args.foreground_threshold < 1.0:
        parser.error("--foreground-threshold must be between 0 and 1")

    seed_everything(args.seed)
    cases = read_cases(args.dataset_json, args.dataset_root)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    case_ids = [row["case_id"] for row in cases]
    expected = set(case_ids)
    split_ids = [item for key in ("train", "validation", "test") for item in split.get(key, [])]
    if set(split_ids) != expected or len(split_ids) != len(expected):
        raise ValueError("Frozen case split does not match the labeled cases in dataset.json")
    test_ids = split.get("test", [])
    if not test_ids:
        raise ValueError("Frozen case split contains no held-out test cases")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("architecture")
    if architecture not in {"monai_unet_3d", "monai_segresnet_3d"}:
        raise ValueError("Checkpoint architecture is not a supported MONAI spleen model")
    patch_size = tuple(checkpoint.get("patch_size", ()))
    if len(patch_size) != 3 or any(int(size) <= 0 for size in patch_size):
        raise ValueError("Checkpoint does not contain a valid three-dimensional patch size")
    foreground_threshold = (args.foreground_threshold if args.foreground_threshold is not None
                            else float(checkpoint.get("foreground_probability_threshold", 0.5)))

    indexed = {row["case_id"]: row for row in cases}
    test_cases = [indexed[case_id] for case_id in test_ids]
    dataset = Dataset(test_cases, transform=build_transform(False, patch_size))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(architecture).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    rows = []
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device)
            target = batch["label"].to(device)
            logits = sliding_window_inference(
                image, patch_size, sw_batch_size=1, predictor=model, overlap=0.25
            )
            probabilities = torch.softmax(logits, dim=1)
            if args.flip_tta:
                flipped_logits = sliding_window_inference(
                    torch.flip(image, dims=(2,)), patch_size,
                    sw_batch_size=1, predictor=model, overlap=0.25
                )
                flipped_probabilities = torch.flip(
                    torch.softmax(flipped_logits, dim=1), dims=(2,)
                )
                probabilities = 0.5 * probabilities + 0.5 * flipped_probabilities
            prediction = probabilities[:, 1] >= foreground_threshold
            truth = target[:, 0] == 1
            case_batch = batch.get("case_id")
            for index in range(image.shape[0]):
                predicted = prediction[index]
                actual = truth[index]
                intersection = int(torch.logical_and(predicted, actual).sum().item())
                predicted_voxels = int(predicted.sum().item())
                reference_voxels = int(actual.sum().item())
                denominator = predicted_voxels + reference_voxels
                dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
                case_id = str(case_batch[index]) if isinstance(case_batch, (list, tuple)) else test_ids[len(rows)]
                rows.append({
                    "case_id": case_id,
                    "dice": float(dice),
                    "predicted_voxels": predicted_voxels,
                    "reference_voxels": reference_voxels,
                })

    result = {
        "dataset": "MSD Task09 Spleen",
        "split": "frozen internal held-out cases from the labeled training cases",
        "test_cases": len(rows),
        "checkpoint": str(args.checkpoint),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "checkpoint_validation_case_mean_dice": checkpoint.get("validation_case_mean_dice"),
        "foreground_probability_threshold": foreground_threshold,
        "inference_augmentation": "fixed 50/50 direct and axis-0 flip probability average" if args.flip_tta else "none",
        "test_case_mean_dice": float(np.mean([row["dice"] for row in rows])),
        "metric": "per-case binary spleen Dice",
        "empty_mask_policy": "both prediction and reference empty gives Dice 1.0",
        "test_split_used_for_training_or_selection": False,
        "official_unlabeled_test_cases_evaluated": False,
        "per_case": rows,
        "clinical_use": "not suitable for diagnosis or care decisions",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
