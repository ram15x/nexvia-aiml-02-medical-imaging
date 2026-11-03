"""Validation-only error analysis for the frozen MSD Spleen model.

This script deliberately loads only validation cases. It never opens the
reserved internal held-out cases or the official challenge test set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from train_segmentation import build_transform, make_model, read_cases, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or args.workers < 0:
        parser.error("threshold must be in (0,1), workers must be nonnegative")

    seed_everything(args.seed)
    cases = read_cases(args.dataset_json, args.dataset_root)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    case_ids = [row["case_id"] for row in cases]
    split_ids = [case_id for key in ("train", "validation", "test")
                 for case_id in split.get(key, [])]
    if set(split_ids) != set(case_ids) or len(split_ids) != len(case_ids):
        raise ValueError("Frozen case split does not match the labeled cases")
    validation_ids = split.get("validation", [])
    if not validation_ids:
        raise ValueError("Frozen split contains no validation cases")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("architecture", "monai_unet_3d")
    if architecture not in {"monai_unet_3d", "monai_segresnet_3d"}:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
    patch_size = tuple(int(x) for x in checkpoint["patch_size"])
    indexed = {row["case_id"]: row for row in cases}
    dataset = Dataset([indexed[key] for key in validation_ids],
                      transform=build_transform(False, patch_size))
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(architecture).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    rows: list[dict] = []
    with torch.inference_mode():
        for batch in loader:
            logits = sliding_window_inference(
                batch["image"].to(device), patch_size, sw_batch_size=1,
                predictor=model, overlap=0.25)
            probabilities = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
            prediction = probabilities >= args.threshold
            truth = batch["label"][0, 0].cpu().numpy() == 1
            intersection = int(np.logical_and(prediction, truth).sum())
            pred_voxels = int(prediction.sum())
            truth_voxels = int(truth.sum())
            dice = 2 * intersection / max(pred_voxels + truth_voxels, 1)
            precision = intersection / max(pred_voxels, 1)
            recall = intersection / max(truth_voxels, 1)
            connectivity = ndimage.generate_binary_structure(3, 3)
            component_labels, component_count = ndimage.label(prediction, structure=connectivity)
            component_sizes = np.bincount(component_labels.ravel())[1:]
            largest = int(component_sizes.max()) if component_sizes.size else 0
            largest_component = component_labels == (int(np.argmax(component_sizes)) + 1) if component_sizes.size else np.zeros_like(prediction)
            lcc_intersection = int(np.logical_and(largest_component, truth).sum())
            lcc_voxels = int(largest_component.sum())
            _, truth_component_count = ndimage.label(truth, structure=connectivity)
            rows.append({
                "case_id": str(batch["case_id"][0]),
                "dice": float(dice),
                "precision": float(precision),
                "recall": float(recall),
                "predicted_voxels": pred_voxels,
                "reference_voxels": truth_voxels,
                "predicted_to_reference_volume_ratio": float(pred_voxels / max(truth_voxels, 1)),
                "predicted_connected_components_26": int(component_count),
                "largest_component_fraction": float(largest / max(pred_voxels, 1)),
                "largest_component_dice": float(2 * lcc_intersection / max(lcc_voxels + truth_voxels, 1)),
                "reference_connected_components_26": int(truth_component_count),
                "probability_p50": float(np.quantile(probabilities, 0.50)),
                "probability_p95": float(np.quantile(probabilities, 0.95)),
                "probability_p99": float(np.quantile(probabilities, 0.99)),
            })

    report = {
        "dataset": "MSD Task09 Spleen",
        "split": "frozen validation cases only",
        "architecture": architecture,
        "threshold": args.threshold,
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "case_count": len(rows),
        "mean_case_dice": float(np.mean([row["dice"] for row in rows])),
        "mean_case_precision": float(np.mean([row["precision"] for row in rows])),
        "mean_case_recall": float(np.mean([row["recall"] for row in rows])),
        "held_out_cases_accessed": False,
        "official_unlabeled_test_cases_accessed": False,
        "per_case": rows,
        "mean_largest_component_dice": float(np.mean([row["largest_component_dice"] for row in rows])),
        "interpretation_note": "Connected-component counts use 26-connectivity on the resampled validation grid. Largest-component Dice is a validation-only postprocessing ablation, not a clinically meaningful quality score or a promoted model change.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
