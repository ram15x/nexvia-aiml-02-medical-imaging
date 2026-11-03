"""Train a MONAI 3D U-Net pilot on MSD Task09 Spleen.

Splits cases (never patches) into train/validation/test before preprocessing.
The test list is saved for later final evaluation but is never read here.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_cases(dataset_json: Path, dataset_root: Path) -> list[dict[str, str]]:
    source = json.loads(dataset_json.read_text(encoding="utf-8"))
    records = source.get("training")
    if not isinstance(records, list) or len(records) < 4:
        raise ValueError("Expected at least four labeled cases in dataset.json")
    root = dataset_root.resolve()
    cases = []
    seen = set()
    for row in records:
        paths = []
        for key in ("image", "label"):
            relative = row.get(key)
            if not relative:
                raise ValueError(f"Missing {key} in a training record")
            resolved = (root / relative).resolve()
            if root not in resolved.parents or not resolved.is_file():
                raise FileNotFoundError(f"Unsafe or missing dataset path: {relative}")
            paths.append(resolved)
        case_id = paths[0].name
        if case_id in seen:
            raise ValueError(f"Duplicate case id: {case_id}")
        seen.add(case_id)
        cases.append({"image": str(paths[0]), "label": str(paths[1]), "case_id": case_id})
    return cases


def frozen_split(cases: list[dict[str, str]], seed: int, path: Path) -> dict[str, list[str]]:
    if path.exists():
        split = json.loads(path.read_text(encoding="utf-8"))
        expected = {case["case_id"] for case in cases}
        actual = set().union(*(set(split.get(key, [])) for key in ("train", "validation", "test")))
        if actual != expected or sum(len(split.get(key, [])) for key in ("train", "validation", "test")) != len(expected):
            raise ValueError("Existing case_split.json does not match these cases; archive it before creating a new split")
        return split

    shuffled = sorted(cases, key=lambda row: row["case_id"])
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, round(n * 0.15))
    n_validation = max(1, round(n * 0.15))
    n_train = n - n_test - n_validation
    if n_train < 1:
        raise ValueError("Not enough cases for train, validation, and test")
    split = {
        "seed": seed,
        "policy": "case-level shuffled split: approximately 70% train, 15% validation, 15% test",
        "train": [row["case_id"] for row in shuffled[:n_train]],
        "validation": [row["case_id"] for row in shuffled[n_train:n_train + n_validation]],
        "test": [row["case_id"] for row in shuffled[n_train + n_validation:]],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return split


def build_transform(training: bool, patch_size: tuple[int, int, int],
                    augmentation_mode: str = "baseline") -> Any:
    from monai.transforms import (
        Compose, CropForegroundd, EnsureChannelFirstd, EnsureTyped, LoadImaged,
        Orientationd, RandAdjustContrastd, RandCropByPosNegLabeld, RandFlipd,
        RandRotate90d, RandScaleIntensityd, RandShiftIntensityd,
        ScaleIntensityRanged, Spacingd, SpatialPadd,
    )
    if augmentation_mode not in {"baseline", "spatial"}:
        raise ValueError("augmentation_mode must be 'baseline' or 'spatial'")
    operations = [
        LoadImaged(keys=("image", "label")),
        EnsureChannelFirstd(keys=("image", "label")),
        Orientationd(keys=("image", "label"), axcodes="RAS"),
        Spacingd(keys=("image", "label"), pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=("image", "label"), source_key="image"),
    ]
    if training:
        operations.extend([
            SpatialPadd(keys=("image", "label"), spatial_size=patch_size, method="symmetric"),
            RandCropByPosNegLabeld(keys=("image", "label"), label_key="label",
                                   spatial_size=patch_size, pos=1, neg=1, num_samples=2,
                                   image_key="image", image_threshold=0),
        ])
        if augmentation_mode == "baseline":
            operations.append(RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=0))
        else:
            operations.extend([
                RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=0),
                RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=1),
                RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=2),
                RandRotate90d(keys=("image", "label"), prob=0.5, max_k=3, spatial_axes=(0, 1)),
                RandScaleIntensityd(keys="image", factors=0.10, prob=0.5),
                RandShiftIntensityd(keys="image", offsets=0.10, prob=0.5),
                RandAdjustContrastd(keys="image", gamma=(0.8, 1.2), prob=0.3),
            ])
    operations.extend([
        EnsureTyped(keys="image", dtype=torch.float32, track_meta=True),
        EnsureTyped(keys="label", dtype=torch.long, track_meta=True),
    ])
    return Compose(operations)


SEGMENTATION_ARCHITECTURES = ("monai_unet_3d", "monai_segresnet_3d")


def write_json_atomic(path: Path, value: Any) -> None:
    """Replace a JSON artifact atomically so interrupted runs keep the last complete record."""
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def make_model(architecture: str = "monai_unet_3d") -> Any:
    if architecture == "monai_unet_3d":
        from monai.networks.nets import UNet
        return UNet(spatial_dims=3, in_channels=1, out_channels=2,
                    channels=(8, 16, 32, 64), strides=(2, 2, 2), num_res_units=2)
    if architecture == "monai_segresnet_3d":
        from monai.networks.nets import SegResNet
        return SegResNet(
            spatial_dims=3, init_filters=8, in_channels=1, out_channels=2,
            blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),
            upsample_mode="deconv",
        )
    raise ValueError(f"Unsupported segmentation architecture: {architecture}")


@torch.inference_mode()
def validate(model: torch.nn.Module, loader: Any, device: torch.device,
             patch_size: tuple[int, int, int]) -> tuple[float, list[dict]]:
    from monai.inferers import sliding_window_inference
    model.eval()
    rows = []
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["label"].to(device)
        logits = sliding_window_inference(image, patch_size, sw_batch_size=1, predictor=model, overlap=0.25)
        prediction = torch.argmax(logits, dim=1) == 1
        truth = target[:, 0] == 1
        for index in range(image.shape[0]):
            pred = prediction[index]
            actual = truth[index]
            intersection = int(torch.logical_and(pred, actual).sum().item())
            pred_count = int(pred.sum().item())
            truth_count = int(actual.sum().item())
            denominator = pred_count + truth_count
            dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
            case_id = f"validation-case-{len(rows) + 1}"
            case_ids = batch.get("case_id")
            if isinstance(case_ids, (list, tuple)) and index < len(case_ids):
                case_id = str(case_ids[index])
            rows.append({"case_id": case_id, "dice": float(dice),
                         "predicted_voxels": pred_count, "reference_voxels": truth_count})
    return (float(np.mean([row["dice"] for row in rows])) if rows else float("nan"), rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--limit-train-cases", type=int, default=None,
                        help="Engineering pilot cap; not a full-data result.")
    parser.add_argument("--foreground-only-dice", action="store_true",
                        help="Exclude background from the Dice component of DiceCE; cross-entropy is unchanged.")
    parser.add_argument("--augmentation-mode", choices=("baseline", "spatial"), default="baseline",
                        help="Use the historical single-axis flip, or a stronger 3D/intensity augmentation set.")
    parser.add_argument("--architecture", choices=SEGMENTATION_ARCHITECTURES,
                        default="monai_unet_3d", help="3D network used for this run.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or any(d <= 0 or d % 8 for d in args.patch_size):
        parser.error("epochs/batch size must be positive and each patch dimension must be a positive multiple of 8")

    seed_everything(args.seed)
    try:
        from monai.data import DataLoader, Dataset, list_data_collate
        from monai.losses import DiceCELoss
    except ImportError as exc:
        raise RuntimeError("Install the MONAI and NiBabel dependencies for this runtime") from exc
    cases = read_cases(args.dataset_json, args.dataset_root)
    split = frozen_split(cases, args.seed, args.output / "case_split.json")
    indexed = {row["case_id"]: row for row in cases}
    train_cases = [indexed[case_id] for case_id in split["train"]]
    validation_cases = [indexed[case_id] for case_id in split["validation"]]
    if args.limit_train_cases is not None:
        train_cases = train_cases[:args.limit_train_cases]

    patch_size = tuple(args.patch_size)
    train_ds = Dataset(train_cases, transform=build_transform(True, patch_size, args.augmentation_mode))
    validation_ds = Dataset(validation_cases, transform=build_transform(False, patch_size))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=list_data_collate,
                              pin_memory=torch.cuda.is_available())
    validation_loader = DataLoader(validation_ds, batch_size=1, shuffle=False,
                                   num_workers=args.workers, pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(args.architecture).to(device)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True,
                               include_background=not args.foreground_only_dice)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0
    history = []
    checkpoint_filename = {
        "monai_unet_3d": "best_spleen_unet.pt",
        "monai_segresnet_3d": "best_spleen_segresnet.pt",
    }[args.architecture]

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, batches = 0.0, 0
        for batch in train_loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(image)
                loss = loss_function(logits, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            batches += 1

        mean_dice, per_case = validate(model, validation_loader, device, patch_size)
        record = {"epoch": epoch, "train_loss": total_loss / max(batches, 1),
                  "validation_case_mean_dice": mean_dice, "validation_cases": per_case}
        history.append(record)
        print(json.dumps(record), flush=True)
        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save({
                "architecture": args.architecture,
                "state_dict": model.state_dict(),
                "patch_size": patch_size,
                "spacing_mm": [1.5, 1.5, 2.0],
                "best_epoch": epoch,
                "validation_case_mean_dice": mean_dice,
                "warning": "Educational research artifact only; not for clinical use.",
            }, args.output / checkpoint_filename)
        write_json_atomic(args.output / "training_history.json", history)

    summary = {"device": str(device), "train_cases": len(train_cases),
               "validation_cases": len(validation_cases), "test_cases": len(split["test"]),
               "test_split_used": False, "limit_train_cases": args.limit_train_cases,
               "epochs_requested": args.epochs, "batch_size": args.batch_size,
               "learning_rate": args.lr, "patch_size": list(patch_size),
               "foreground_only_dice": args.foreground_only_dice,
               "augmentation_mode": args.augmentation_mode,
               "architecture": args.architecture,
               "checkpoint": checkpoint_filename,
               "case_split_file": "case_split.json",
               "metric": "per-case binary spleen Dice; both-empty prediction/reference is defined as 1.0",
               "preprocessing": "RAS orientation, 1.5x1.5x2.0 mm spacing, CT clip [-57,164] scaled to [0,1]",
               "warning": "Small-sample educational prototype; not for clinical use."}
    write_json_atomic(args.output / "run_summary.json", summary)


if __name__ == "__main__":
    main()
