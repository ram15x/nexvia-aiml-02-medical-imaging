"""Render a validation-only CT/ground-truth/prediction overlay montage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def orient(path: Path):
    return nib.as_closest_canonical(nib.load(str(path)))


def render(image_path: Path, truth_path: Path, prediction_path: Path,
           output_path: Path, threshold: float = 0.80) -> dict:
    image_nii = orient(image_path)
    truth_nii = orient(truth_path)
    prediction_nii = orient(prediction_path)
    if image_nii.shape != truth_nii.shape or image_nii.shape != prediction_nii.shape:
        raise ValueError("Canonical CT, reference, and prediction shapes do not match")
    if not np.allclose(image_nii.affine, truth_nii.affine, rtol=0, atol=1e-4):
        raise ValueError("Ground-truth affine does not match source CT")
    if not np.allclose(image_nii.affine, prediction_nii.affine, rtol=0, atol=1e-4):
        raise ValueError("Prediction affine does not match source CT")

    ct = np.asanyarray(image_nii.dataobj, dtype=np.float32)
    truth = np.asanyarray(truth_nii.dataobj) > 0
    prediction = np.asanyarray(prediction_nii.dataobj) > 0
    z_positions = np.flatnonzero(truth.any(axis=(0, 1)))
    if not len(z_positions):
        raise ValueError("Reference mask is empty; cannot choose spleen slices")
    quantiles = np.linspace(0.08, 0.92, 6)
    slice_indices = np.unique(np.rint(np.quantile(z_positions, quantiles)).astype(int))

    tile_size = 420
    header_height = 92
    footer_height = 74
    canvas = Image.new("RGB", (3 * tile_size, header_height + 2 * tile_size + footer_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((24, 17), "MSD spleen CT validation example", fill="#152333", font=font)
    draw.text((24, 43), "Green = reference · red = predicted mask · yellow = overlap", fill="#34495e", font=font)
    case_id = image_path.name.removesuffix(".nii.gz").removesuffix(".nii")
    draw.text((24, 65), f"Validation case {case_id} · threshold {threshold:.2f} · educational visualization", fill="#58697a", font=font)

    for index, z in enumerate(slice_indices[:6]):
        base = np.clip((ct[:, :, z] + 57.0) / (164.0 + 57.0), 0.0, 1.0)
        gray = (base * 255).astype(np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)
        gt_slice = truth[:, :, z]
        pred_slice = prediction[:, :, z]
        gt_only = gt_slice & ~pred_slice
        pred_only = pred_slice & ~gt_slice
        overlap = gt_slice & pred_slice
        rgb[gt_only] = 0.38 * rgb[gt_only] + 0.62 * np.asarray([20, 220, 90])
        rgb[pred_only] = 0.38 * rgb[pred_only] + 0.62 * np.asarray([245, 55, 55])
        rgb[overlap] = 0.25 * rgb[overlap] + 0.75 * np.asarray([255, 220, 0])
        tile = Image.fromarray(np.rot90(np.clip(rgb, 0, 255).astype(np.uint8)))
        tile = tile.resize((tile_size - 20, tile_size - 48), Image.Resampling.BILINEAR)
        x = (index % 3) * tile_size + 10
        y = header_height + (index // 3) * tile_size + 34
        canvas.paste(tile, (x, y))
        draw.text((x, y - 21), f"Axial slice {z}", fill="#152333", font=font)

    dice_source_grid = float(2 * np.logical_and(truth, prediction).sum()
                             / max(int(truth.sum()) + int(prediction.sum()), 1))
    draw.text((24, header_height + 2 * tile_size + 17),
              f"Source-grid Dice for this display: {dice_source_grid:.3f} · not comparable to the resampled-grid validation score",
              fill="#34495e", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return {
        "case_id": case_id,
        "split": "validation",
        "threshold": threshold,
        "slice_indices": slice_indices[:6].tolist(),
        "canonical_shape": list(image_nii.shape),
        "source_grid_dice_for_visual_review_only": dice_source_grid,
        "ground_truth_voxels_source_grid": int(truth.sum()),
        "predicted_voxels_source_grid": int(prediction.sum()),
        "predicted_to_reference_volume_ratio_source_grid": float(prediction.sum() / max(truth.sum(), 1)),
        "geometry_matches": True,
        "output_image": str(output_path),
        "interpretation_limit": "One validation example for visual review; the checkpoint and threshold used the validation split. Not an independent result or clinical evidence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.80,
                        help="Foreground probability threshold used to create the prediction.")
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    result = render(args.image, args.truth, args.prediction, args.output, args.threshold)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
