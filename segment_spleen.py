"""Run the coursework Spleen 3D U-Net on one NIfTI CT and save a mask.

Educational prototype only. This script is not validated for diagnosis or care.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train_segmentation import make_model


def build_inference_transform() -> Any:
    from monai.transforms import (
        Compose,
        CropForegroundd,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        Orientationd,
        ScaleIntensityRanged,
        Spacingd,
    )

    return Compose([
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        Spacingd(keys="image", pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
        ScaleIntensityRanged(keys="image", a_min=-57, a_max=164,
                             b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys="image", source_key="image"),
        EnsureTyped(keys="image", dtype=torch.float32, track_meta=True),
    ])


def keep_largest_component(mask: "np.ndarray") -> tuple["np.ndarray", int]:
    """Keep only the biggest 3D connected blob of a binary mask.

    The spleen is one organ, so small disconnected blobs are false positives. On the frozen validation
    and test cases this raised Dice for every case (see results/spleen/postprocess_study.json).
    Returns the cleaned mask and how many blobs were found.
    """
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    if count <= 1:
        return mask, count
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    return labels == (1 + int(np.argmax(sizes))), count


def run_inference(input_path: Path, checkpoint_path: Path, output_dir: Path,
                  threshold: float, device_name: str,
                  use_flip_tta: bool = False,
                  largest_component_only: bool = True) -> dict[str, Any]:
    from monai.data import MetaTensor
    from monai.inferers import sliding_window_inference
    from monai.transforms import Invertd, SaveImaged
    import nibabel as nib

    input_path = input_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    output_dir = output_dir.resolve()
    if input_path.suffix.lower() not in {".nii", ".gz"} or not input_path.name.lower().endswith((".nii", ".nii.gz")):
        raise ValueError("Input must be a .nii or .nii.gz NIfTI CT volume.")
    if not input_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("Input NIfTI or trained checkpoint was not found.")
    if not 0.0 < threshold < 1.0:
        raise ValueError("Threshold must be between 0 and 1.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("architecture")
    if architecture not in {"monai_unet_3d", "monai_segresnet_3d"}:
        raise ValueError("Checkpoint architecture is not a supported MONAI spleen model.")
    patch_size = tuple(int(value) for value in checkpoint.get("patch_size", (64, 64, 64)))
    model = make_model(architecture).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    original = nib.load(str(input_path))
    transform = build_inference_transform()
    data = transform({"image": str(input_path)})
    image = data["image"]
    batch = image.unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = sliding_window_inference(
            batch, patch_size, sw_batch_size=1, predictor=model, overlap=0.25
        )
        probability = torch.softmax(logits, dim=1)[:, 1]
        if use_flip_tta:
            flipped_batch = torch.flip(batch, dims=(2,))
            flipped_logits = sliding_window_inference(
                flipped_batch, patch_size, sw_batch_size=1, predictor=model, overlap=0.25
            )
            flipped_probability = torch.flip(torch.softmax(flipped_logits, dim=1), dims=(2,))[:, 1]
            probability = 0.5 * probability + 0.5 * flipped_probability
        probability = probability[0].cpu()
        binary = (probability >= threshold).numpy()
        blobs_found = 0
        if largest_component_only:
            binary, blobs_found = keep_largest_component(binary)
        prediction = torch.from_numpy(binary.astype(np.uint8)).unsqueeze(0)

    prediction_meta = MetaTensor(
        prediction,
        affine=image.affine,
        meta=image.meta,
        applied_operations=list(image.applied_operations),
    )
    data["pred"] = prediction_meta
    inverse = Invertd(
        keys="pred", transform=transform, orig_keys="image",
        nearest_interp=True, to_tensor=True,
    )
    restored = inverse(data)
    SaveImaged(
        keys="pred", output_dir=str(output_dir), output_postfix="spleen-mask",
        output_ext=".nii.gz", output_dtype=np.uint8, resample=False,
        separate_folder=False, print_log=False,
    )(restored)

    outputs = list(output_dir.glob("*.nii.gz"))
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one saved NIfTI mask, found {len(outputs)}.")
    mask_path = outputs[0]
    saved = nib.load(str(mask_path))
    mask = np.asanyarray(saved.dataobj)
    if mask.shape != original.shape:
        raise RuntimeError(f"Output shape {mask.shape} does not match input shape {original.shape}.")
    if not np.allclose(saved.affine, original.affine, rtol=0, atol=1e-4):
        raise RuntimeError("Output affine does not match the source NIfTI geometry.")
    values = np.unique(mask)
    if not set(values.tolist()).issubset({0, 1}):
        raise RuntimeError(f"Output mask is not binary: values {values.tolist()}.")

    voxel_ml = abs(float(np.linalg.det(original.affine[:3, :3]))) / 1000.0
    predicted_voxels = int(np.count_nonzero(mask))
    result = {
        "input": str(input_path),
        "mask": str(mask_path),
        "architecture": architecture,
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "validation_case_mean_dice": checkpoint.get("validation_case_mean_dice"),
        "threshold": threshold,
        "inference_augmentation": "fixed 50/50 direct and axis-0 flip probability average" if use_flip_tta else "none",
        "post_processing": "largest 3D connected component only" if largest_component_only else "none",
        "blobs_before_cleanup": blobs_found if largest_component_only else None,
        "input_shape": list(original.shape),
        "predicted_spleen_voxels": predicted_voxels,
        "estimated_predicted_volume_ml": round(predicted_voxels * voxel_ml, 2),
        "output_geometry_matches_input": True,
        "disclaimer": "Educational research output only; not a diagnosis or clinical segmentation.",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="One .nii or .nii.gz CT volume")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_spleen_unet.pt"))
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New or empty folder for the output binary NIfTI mask")
    parser.add_argument("--threshold", type=float, default=0.80,
                        help="Foreground probability cutoff selected on the six validation cases")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--flip-tta", action="store_true",
                        help="Average direct inference with axis-0 flipped inference (2x model compute).")
    args = parser.parse_args()
    try:
        from monai.transforms import Invertd, SaveImaged  # noqa: F401
        import nibabel  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install requirements-segmentation.txt in the selected Python environment.") from exc
    print(json.dumps(run_inference(args.input, args.checkpoint, args.output_dir,
                                   args.threshold, args.device, args.flip_tta), indent=2))


if __name__ == "__main__":
    main()
