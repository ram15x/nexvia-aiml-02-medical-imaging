"""Select a coarse foreground probability threshold on MSD validation cases only."""

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
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=(0.35, 0.5, 0.65, 0.8))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.workers < 0 or not args.thresholds or any(not 0 < t < 1 for t in args.thresholds):
        parser.error("workers must be nonnegative and thresholds must be between 0 and 1")

    seed_everything(args.seed)
    cases = read_cases(args.dataset_json, args.dataset_root)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    ids = [row["case_id"] for row in cases]
    split_ids = [item for key in ("train", "validation", "test") for item in split.get(key, [])]
    if set(split_ids) != set(ids) or len(split_ids) != len(ids):
        raise ValueError("Frozen case split does not match the labeled cases")
    validation_ids = split.get("validation", [])
    if not validation_ids:
        raise ValueError("Frozen split contains no validation cases")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("architecture")
    if architecture not in {"monai_unet_3d", "monai_segresnet_3d"}:
        raise ValueError("Checkpoint architecture is not a supported MONAI spleen model")
    patch_size = tuple(int(x) for x in checkpoint.get("patch_size", ()))
    if len(patch_size) != 3 or any(x <= 0 for x in patch_size):
        raise ValueError("Checkpoint does not contain a valid patch size")

    indexed = {row["case_id"]: row for row in cases}
    validation_cases = [indexed[case_id] for case_id in validation_ids]
    dataset = Dataset(validation_cases, transform=build_transform(False, patch_size))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(architecture).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    rows = []
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device)
            label = batch["label"][0, 0].cpu().numpy() == 1
            logits = sliding_window_inference(
                image, patch_size, sw_batch_size=1, predictor=model, overlap=0.25
            )
            foreground_probability = torch.softmax(logits, dim=1)[0, 1].cpu().numpy().astype(np.float32)
            rows.append({
                "case_id": str(batch["case_id"][0]),
                "probability": foreground_probability,
                "truth": label,
            })

    results = []
    for threshold in sorted(set(float(t) for t in args.thresholds)):
        per_case = []
        for row in rows:
            predicted = row["probability"] >= threshold
            truth = row["truth"]
            intersection = int(np.logical_and(predicted, truth).sum())
            predicted_count = int(predicted.sum())
            truth_count = int(truth.sum())
            denominator = predicted_count + truth_count
            dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
            per_case.append({"case_id": row["case_id"], "dice": float(dice),
                             "predicted_voxels": predicted_count, "reference_voxels": truth_count})
        results.append({"threshold": threshold,
                        "validation_case_mean_dice": float(np.mean([x["dice"] for x in per_case])),
                        "per_case": per_case})
    selected = max(results, key=lambda x: (x["validation_case_mean_dice"], -abs(x["threshold"] - 0.5)))
    output = {
        "dataset": "MSD Task09 Spleen",
        "split": "frozen validation cases only",
        "checkpoint": str(args.checkpoint),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "checkpoint_validation_case_mean_dice_at_0_5": checkpoint.get("validation_case_mean_dice"),
        "threshold_selection": "coarse grid selected by validation case-mean Dice; ties prefer 0.5",
        "selected_threshold": selected["threshold"],
        "selected_validation_case_mean_dice": selected["validation_case_mean_dice"],
        "results": results,
        "held_out_cases_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
