"""KernelSHAP attribution over a coarse grid for one classifier output.

Each grid cell is either kept from the original image or replaced with neutral
grey. The Shapley value of a cell is its average marginal contribution to the
sigmoid score for the selected finding. Values are sampled, so the function
fixes the random seed for repeatable results and reports the additivity check
(baseline + sum of cell values should equal the prediction).

This is a model-behaviour explanation for the coursework demo. It is not a
clinical explanation and does not identify anatomy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from .modeling import image_to_tensor

GRID = 8
INPUT_SIZE = 224
NSAMPLES = 300
SEED = 0
GREY = 127.0


def _grid_name(row: int, col: int, grid: int = GRID) -> str:
    vertical = ("upper", "middle", "lower")[min(2, row * 3 // grid)]
    horizontal = ("left", "centre", "right")[min(2, col * 3 // grid)]
    return f"{vertical} {horizontal} (row {row + 1}, column {col + 1} of {grid})"


def _neutral_mask(z_row: np.ndarray, cell: int) -> np.ndarray:
    grid_values = z_row.reshape(GRID, GRID)
    return np.kron(grid_values, np.ones((cell, cell)))[..., None]


def shap_attribution(model: torch.nn.Module, image: Image.Image, label_index: int,
                     nsamples: int = NSAMPLES, seed: int = SEED) -> dict[str, Any]:
    """Return per-cell SHAP values for one finding, plus the additivity check."""
    import shap

    base = np.asarray(image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR),
                      dtype=np.float32)
    cell = INPUT_SIZE // GRID

    def masked_images(z: np.ndarray) -> np.ndarray:
        out = []
        for row in z:
            mask = _neutral_mask(row, cell)
            out.append(base * mask + GREY * (1.0 - mask))
        return np.stack(out).astype(np.uint8)

    def score(z: np.ndarray) -> np.ndarray:
        z = np.atleast_2d(z)
        chunks = []
        for start in range(0, len(z), 32):
            batch = torch.stack([image_to_tensor(Image.fromarray(a)) for a in masked_images(z[start:start + 32])])
            with torch.inference_mode():
                chunks.append(torch.sigmoid(model(batch))[:, label_index].cpu().numpy())
        return np.concatenate(chunks)

    state = np.random.get_state()
    try:
        np.random.seed(seed)
        explainer = shap.KernelExplainer(score, np.zeros((1, GRID * GRID)))
        values = np.asarray(explainer.shap_values(np.ones((1, GRID * GRID)),
                                                  nsamples=nsamples, silent=True)).reshape(-1)
        baseline = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    finally:
        np.random.set_state(state)

    prediction = float(score(np.ones((1, GRID * GRID)))[0])
    additivity_error = abs(baseline + float(values.sum()) - prediction)
    grid_values = values.reshape(GRID, GRID)
    order = np.argsort(-np.abs(values))[:5]
    top_cells = [{
        "row": int(i // GRID) + 1,
        "column": int(i % GRID) + 1,
        "region": _grid_name(int(i // GRID), int(i % GRID)),
        "value": round(float(values[i]), 6),
        "direction": "raises the score" if values[i] > 0 else "lowers the score",
    } for i in order if abs(values[i]) > 1e-6]
    return {
        "method": "KernelSHAP on an 8x8 grid with neutral-grey replacement",
        "grid": GRID,
        "nsamples": nsamples,
        "seed": seed,
        "baseline": round(baseline, 6),
        "prediction": round(prediction, 6),
        "additivity_error": round(additivity_error, 6),
        "values": [[round(float(v), 6) for v in row] for row in grid_values],
        "top_cells": top_cells,
        "image": base.astype(np.uint8),
    }


def overlay_png(image_array: np.ndarray, values: list[list[float]]) -> bytes:
    """Blend signed cell values onto the image: red raises the score, blue lowers it."""
    import io

    grid = np.asarray(values, dtype=np.float32)
    scale = float(np.max(np.abs(grid))) or 1.0
    normalised = np.clip(grid / scale, -1.0, 1.0)
    heat = np.kron(normalised, np.ones((INPUT_SIZE // GRID, INPUT_SIZE // GRID)))
    base = image_array.astype(np.float32)
    grey = np.stack([base.mean(axis=-1)] * 3, axis=-1)
    positive = np.zeros_like(base)
    positive[..., 0] = 255.0
    negative = np.zeros_like(base)
    negative[..., 2] = 255.0
    alpha = (np.abs(heat) * 0.55)[..., None]
    colour = np.where(heat[..., None] >= 0, positive, negative)
    blended = grey * (1 - alpha) + colour * alpha
    buffer = io.BytesIO()
    Image.fromarray(np.uint8(np.clip(blended, 0, 255))).save(buffer, format="PNG")
    return buffer.getvalue()
