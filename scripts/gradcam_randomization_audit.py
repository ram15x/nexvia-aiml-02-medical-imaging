"""Sanity-check DenseNet Grad-CAM sensitivity to model parameters.

This does not measure lesion localization or clinical correctness. It compares
Grad-CAM maps from a trained checkpoint with maps after progressively
randomizing classifier and encoder weights on a fixed validation case list.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.modeling import gradcam_target, image_to_tensor
from data_audit import NIH_LABELS
from train_classifier import make_model


def gradcam(model: torch.nn.Module, image: torch.Tensor, label: str,
            architecture: str, device: torch.device) -> tuple[np.ndarray, float]:
    activations: list[torch.Tensor] = []
    target_layer, channels_last = gradcam_target(model, architecture)
    handle = target_layer.register_forward_hook(
        lambda _module, _inputs, output: activations.append(output)
    )
    try:
        x = image.unsqueeze(0).to(device).requires_grad_(True)
        logits = model(x)
        idx = NIH_LABELS.index(label)
        probability = float(torch.sigmoid(logits[0, idx]).detach().cpu())
        activation = activations[-1]
        gradient = torch.autograd.grad(logits[0, idx], activation)[0]
        if channels_last:
            activation = activation.permute(0, 3, 1, 2)
            gradient = gradient.permute(0, 3, 1, 2)
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * activation).sum(dim=1))[0]
        heatmap = heatmap.detach().float().cpu().numpy()
        heatmap -= heatmap.min()
        maximum = float(heatmap.max())
        if maximum > 0:
            heatmap /= maximum
        return heatmap, probability
    finally:
        handle.remove()


def randomize_module(module: torch.nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.BatchNorm2d,
                              torch.nn.Linear)):
            layer.reset_parameters()


def randomized_model(checkpoint: dict, architecture: str, stage: str,
                     seed: int, device: torch.device) -> torch.nn.Module:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = make_model(pretrained=False, architecture=architecture)
    model.load_state_dict(checkpoint["state_dict"])
    if stage == "classifier_only":
        randomize_module(model.classifier)
    elif stage == "final_block_and_classifier":
        randomize_module(model.features.denseblock4)
        randomize_module(model.classifier)
    elif stage == "all_parameters":
        randomize_module(model)
    model.to(device).eval()
    return model


def correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if float(np.ptp(a)) == 0.0 or float(np.ptp(b)) == 0.0:
        return None
    result = spearmanr(a.ravel(), b.ravel()).statistic
    return float(result) if np.isfinite(result) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = checkpoint.get("architecture")
    if architecture != "densenet121" or checkpoint.get("labels") != list(NIH_LABELS):
        raise ValueError("Expected the current DenseNet-121 checkpoint and label order")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = int(checkpoint.get("image_size", 224))
    with args.cases.open(newline="", encoding="utf-8-sig") as f:
        cases = list(csv.DictReader(f))

    stages = ("trained", "classifier_only", "final_block_and_classifier", "all_parameters")
    models: dict[str, torch.nn.Module] = {}
    for i, stage in enumerate(stages):
        models[stage] = randomized_model(
            checkpoint, architecture, stage, 20261031 + i, device
        )

    rows = []
    for case in cases:
        path = args.image_root / case["Image Index"]
        with Image.open(path) as im:
            image = image_to_tensor(im.convert("RGB"), size)
        maps: dict[str, np.ndarray] = {}
        trained_probability = None
        for stage, model in models.items():
            with torch.enable_grad():
                heatmap, probability = gradcam(model, image, case["label"], architecture, device)
            maps[stage] = heatmap
            if stage == "trained":
                trained_probability = probability
        rows.append({
            "image": case["Image Index"],
            "label": case["label"],
            "case_type": case.get("case_type", ""),
            "trained_score": trained_probability,
            **{f"spearman_trained_vs_{stage}": correlation(maps["trained"], maps[stage])
               for stage in stages[1:]},
        })
        del maps
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "scope": "existing development-validation Grad-CAM cases only; no held-out test rows",
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_architecture": architecture,
        "device": str(device),
        "n_cases": len(rows),
        "interpretation_limit": "Parameter sensitivity sanity check only; does not establish localization or clinical correctness.",
        "cases": rows,
        "median_spearman_by_randomization_stage": {
            key: float(np.median(values)) if values else None
            for key in stages[1:]
            for values in [[r[f"spearman_trained_vs_{key}"] for r in rows
                            if r[f"spearman_trained_vs_{key}"] is not None]]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["median_spearman_by_randomization_stage"], indent=2))
    print(f"cases={len(rows)} device={device} output={args.output}")


if __name__ == "__main__":
    main()
