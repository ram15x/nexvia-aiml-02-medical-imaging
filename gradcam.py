"""Create a DenseNet Grad-CAM overlay for one NIH chest X-ray and label.

Grad-CAM is a coarse feature attribution visualization, not an explanation
that establishes causality or clinical correctness.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data_audit import NIH_LABELS
from api.modeling import gradcam_target, image_to_tensor
from train_classifier import make_model


def prepare(image_path: Path, size: int) -> tuple[torch.Tensor, Image.Image]:
    with Image.open(image_path) as source:
        original = source.convert("RGB").copy()
    tensor = image_to_tensor(original, size).unsqueeze(0)
    return tensor, original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--label", choices=NIH_LABELS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = checkpoint.get("architecture")
    if architecture not in {"densenet121", "efficientnet_b0", "swin_t", "densenet121_swin_t"} or checkpoint.get("labels") != list(NIH_LABELS):
        raise ValueError("Checkpoint architecture or label ordering does not match this tool")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(pretrained=False, architecture=architecture)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    target_layer, channels_last = gradcam_target(model, architecture)
    def capture_activation(_module, _inputs, output):
        activations.append(output)

    handle = target_layer.register_forward_hook(capture_activation)
    tensor, original = prepare(args.image, int(checkpoint.get("image_size", 224)))
    tensor = tensor.to(device)
    tensor = tensor.requires_grad_(True)
    logits = model(tensor)
    target_index = NIH_LABELS.index(args.label)
    probability = float(torch.sigmoid(logits[0, target_index]).detach().cpu())
    activation = activations[-1]
    gradient = torch.autograd.grad(logits[0, target_index], activation)[0]
    if channels_last:
        activation = activation.permute(0, 3, 1, 2)
        gradient = gradient.permute(0, 3, 1, 2)
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    heatmap = F.relu((weights * activation).sum(dim=1, keepdim=True))
    heatmap = F.interpolate(heatmap, size=(int(checkpoint.get("image_size", 224)),) * 2,
                            mode="bilinear", align_corners=False)[0, 0]
    heatmap = heatmap.detach().cpu().numpy()
    handle.remove()
    if float(heatmap.max()) > 0:
        heatmap /= float(heatmap.max())
    heat = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize(
        original.size, Image.Resampling.BILINEAR)
    heat_array = np.asarray(heat, dtype=np.float32) / 255.0
    base = np.asarray(original, dtype=np.float32)
    red_overlay = np.zeros_like(base)
    red_overlay[..., 0] = 255.0
    alpha = (heat_array * 0.45)[..., None]
    overlay = np.uint8(np.clip(base * (1 - alpha) + red_overlay * alpha, 0, 255))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(args.output)
    print(f"label={args.label} score={probability:.4f} output={args.output}")
    print(f"architecture={architecture}")
    print("Interpretation: Grad-CAM highlights feature-map attribution; it is not a clinical explanation.")


if __name__ == "__main__":
    main()
