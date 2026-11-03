"""Validation-only equal-weight probability blend of frozen Spleen checkpoints."""

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


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("architecture")
    if architecture not in {"monai_unet_3d", "monai_segresnet_3d"}:
        raise ValueError(f"Unsupported architecture in {path}")
    model = make_model(architecture).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return checkpoint, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=(0.50, 0.65, 0.75, 0.80))
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 0 or not args.thresholds or any(not 0 < value < 1 for value in args.thresholds):
        parser.error("workers must be nonnegative and thresholds must be between 0 and 1")

    seed_everything(42)
    cases = read_cases(args.dataset_json, args.dataset_root)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    all_ids = [row["case_id"] for row in cases]
    split_ids = [item for key in ("train", "validation", "test") for item in split.get(key, [])]
    if set(split_ids) != set(all_ids) or len(split_ids) != len(all_ids):
        raise ValueError("Frozen split does not match the labeled data")
    validation_ids = split.get("validation", [])
    if not validation_ids:
        raise ValueError("Frozen split contains no validation cases")

    indexed = {row["case_id"]: row for row in cases}
    dataset = Dataset([indexed[item] for item in validation_ids],
                      transform=build_transform(False, (64, 64, 64)))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_meta, baseline = load_checkpoint(args.baseline_checkpoint, device)
    candidate_meta, candidate = load_checkpoint(args.candidate_checkpoint, device)
    if tuple(baseline_meta.get("patch_size", ())) != tuple(candidate_meta.get("patch_size", ())):
        raise ValueError("Checkpoint patch sizes differ; this comparison runner requires matched inference windows")
    patch_size = tuple(int(value) for value in baseline_meta["patch_size"])

    cases_out = []
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device)
            baseline_logits = sliding_window_inference(
                image, patch_size, sw_batch_size=1, predictor=baseline, overlap=0.25
            )
            candidate_logits = sliding_window_inference(
                image, patch_size, sw_batch_size=1, predictor=candidate, overlap=0.25
            )
            p_base = torch.softmax(baseline_logits, dim=1)[:, 1]
            p_candidate = torch.softmax(candidate_logits, dim=1)[:, 1]
            blend = (0.5 * p_base + 0.5 * p_candidate)[0].cpu().numpy().astype(np.float32)
            baseline_probability = p_base[0].cpu().numpy().astype(np.float32)
            truth = batch["label"][0, 0].cpu().numpy() == 1
            cases_out.append({
                "case_id": str(batch["case_id"][0]),
                "blend": blend,
                "baseline": baseline_probability,
                "truth": truth,
            })

    def evaluate(key: str, threshold: float) -> dict:
        per_case = []
        for row in cases_out:
            predicted = row[key] >= threshold
            truth = row["truth"]
            predicted_count = int(predicted.sum())
            truth_count = int(truth.sum())
            denominator = predicted_count + truth_count
            dice = 1.0 if denominator == 0 else 2.0 * int(np.logical_and(predicted, truth).sum()) / denominator
            per_case.append({"case_id": row["case_id"], "dice": float(dice),
                             "predicted_voxels": predicted_count, "reference_voxels": truth_count})
        return {"threshold": threshold,
                "validation_case_mean_dice": float(np.mean([item["dice"] for item in per_case])),
                "per_case": per_case}

    thresholds = sorted(set(float(value) for value in args.thresholds))
    results = {
        "baseline": [evaluate("baseline", value) for value in thresholds],
        "equal_weight_probability_blend": [evaluate("blend", value) for value in thresholds],
    }
    output = {
        "dataset": "MSD Task09 Spleen",
        "split": "frozen six-case validation set only",
        "blend": "fixed 0.5 * U-Net foreground probability + 0.5 * SegResNet foreground probability",
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "baseline_architecture": baseline_meta["architecture"],
        "candidate_checkpoint": str(args.candidate_checkpoint),
        "candidate_architecture": candidate_meta["architecture"],
        "thresholds": thresholds,
        "results": results,
        "held_out_cases_accessed": False,
        "selection_warning": "The validation set is small; these exploratory results are not independent performance estimates.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    del baseline, candidate
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
