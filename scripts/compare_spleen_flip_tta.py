"""Compare baseline MONAI 3D U-Net with a fixed axis-0 flip-TTA average on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_segmentation import build_transform, make_model, read_cases, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=(0.50, 0.65, 0.75, 0.80, 0.85))
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 0 or not args.thresholds or any(not 0 < value < 1 for value in args.thresholds):
        parser.error("workers must be nonnegative and thresholds must be between 0 and 1")

    seed_everything(42)
    cases = read_cases(args.dataset_json, args.dataset_root)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    ids = [row["case_id"] for row in cases]
    split_ids = [item for key in ("train", "validation", "test") for item in split.get(key, [])]
    if set(split_ids) != set(ids) or len(split_ids) != len(ids):
        raise ValueError("Frozen split does not match the labeled cases")
    validation_ids = split.get("validation", [])
    if not validation_ids:
        raise ValueError("Frozen split contains no validation cases")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("architecture")
    if architecture not in {"monai_unet_3d", "monai_segresnet_3d"}:
        raise ValueError("Unsupported segmentation architecture in checkpoint")
    patch_size = tuple(int(value) for value in checkpoint["patch_size"])
    indexed = {row["case_id"]: row for row in cases}
    dataset = Dataset([indexed[item] for item in validation_ids],
                      transform=build_transform(False, patch_size))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(architecture).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    cases_out = []

    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device)
            direct = sliding_window_inference(
                image, patch_size, sw_batch_size=1, predictor=model, overlap=0.25
            )
            flipped_image = torch.flip(image, dims=(2,))
            flipped_logits = sliding_window_inference(
                flipped_image, patch_size, sw_batch_size=1, predictor=model, overlap=0.25
            )
            unflipped_probability = torch.flip(torch.softmax(flipped_logits, dim=1), dims=(2,))[:, 1]
            direct_probability = torch.softmax(direct, dim=1)[:, 1]
            tta_probability = 0.5 * direct_probability + 0.5 * unflipped_probability
            cases_out.append({
                "case_id": str(batch["case_id"][0]),
                "direct": direct_probability[0].cpu().numpy().astype(np.float32),
                "flip_tta": tta_probability[0].cpu().numpy().astype(np.float32),
                "truth": (batch["label"][0, 0].cpu().numpy() == 1),
            })

    thresholds = sorted(set(float(value) for value in args.thresholds))
    results = {}
    for key in ("direct", "flip_tta"):
        threshold_results = []
        for threshold in thresholds:
            per_case = []
            for row in cases_out:
                prediction = row[key] >= threshold
                truth = row["truth"]
                predicted_count, truth_count = int(prediction.sum()), int(truth.sum())
                denominator = predicted_count + truth_count
                dice = 1.0 if denominator == 0 else 2.0 * int(np.logical_and(prediction, truth).sum()) / denominator
                per_case.append({"case_id": row["case_id"], "dice": float(dice),
                                 "predicted_voxels": predicted_count, "reference_voxels": truth_count})
            threshold_results.append({"threshold": threshold,
                                      "validation_case_mean_dice": float(np.mean([row["dice"] for row in per_case])),
                                      "per_case": per_case})
        results[key] = threshold_results

    output = {
        "dataset": "MSD Task09 Spleen",
        "split": "frozen six-case validation set only",
        "checkpoint": str(args.checkpoint),
        "architecture": architecture,
        "test_time_augmentation": "fixed 50/50 probability average between direct inference and axis-0 flipped inference, flipped probability restored before averaging",
        "thresholds": thresholds,
        "results": results,
        "held_out_cases_accessed": False,
        "selection_warning": "Threshold results use the small validation set and are not independent performance estimates.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    del model
    print(json.dumps({key: [(row["threshold"], round(row["validation_case_mean_dice"], 4))
                           for row in value] for key, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
