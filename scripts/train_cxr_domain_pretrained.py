"""Fine-tune a CheXpert-pretrained DenseNet on the fixed NIH subset.

This development experiment reads only the train and validation rows from a
patient-disjoint manifest. It never opens the held-out test manifest/images.
The XRV checkpoint is CheXpert-only; the combined XRV checkpoint is excluded
because it contains NIH training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from data_audit import NIH_LABELS


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def label_vector(raw: str) -> np.ndarray:
    labels = set(str(raw).split("|"))
    labels.discard("No Finding")
    unknown = labels - set(NIH_LABELS)
    if unknown:
        raise ValueError(f"Unexpected NIH labels: {sorted(unknown)}")
    return np.asarray([float(label in labels) for label in NIH_LABELS], dtype=np.float32)


class NIHXRVSubset(Dataset):
    def __init__(self, manifest: Path, image_root: Path, split: str) -> None:
        frame = pd.read_csv(manifest)
        required = {"Image Index", "Finding Labels", "Patient ID", "split"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        frame = frame.loc[frame["split"] == split].copy().reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"Manifest has no {split!r} rows")
        self.image_root = image_root.resolve()
        self.paths: list[Path] = []
        for filename in frame["Image Index"].astype(str):
            path = (self.image_root / filename).resolve()
            if path.parent != self.image_root:
                raise ValueError(f"Unsafe filename in manifest: {filename}")
            if not path.is_file():
                raise FileNotFoundError(path)
            self.paths.append(path)
        self.rows = frame
        self.targets = np.stack([label_vector(value) for value in frame["Finding Labels"]])
        self.training = split == "train"
        self.center_crop = xrv.datasets.XRayCenterCrop()
        self.resize = xrv.datasets.XRayResizer(224)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.paths[index]) as source:
            pixels = np.asarray(source.convert("L"), dtype=np.uint8)[None, :, :]
        pixels = self.center_crop(pixels)
        pixels = self.resize(pixels)
        if self.training and random.random() < 0.5:
            pixels = np.flip(pixels, axis=2).copy()
        normalized = xrv.utils.normalize(pixels, maxval=255.0)
        image = torch.from_numpy(np.ascontiguousarray(normalized, dtype=np.float32))
        return image, torch.from_numpy(self.targets[index])


def make_model(weights_path: Path) -> torch.nn.Module:
    model = xrv.models.DenseNet(weights="densenet121-res224-chex", cache_dir=str(weights_path.parent))
    if Path(model.weights_filename_local).resolve() != weights_path.resolve():
        raise ValueError("The loaded XRV weight path differs from the requested checkpoint")
    model.op_threshs = None  # train raw logits; XRV operating-point remapping is inference-only
    model.classifier = nn.Linear(1024, len(NIH_LABELS))  # NIH labels differ; train a fresh head
    return model


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    ys, ps = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            ys.append(labels.numpy())
            ps.append(torch.sigmoid(logits).cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    aucs, aps = [], []
    per_label = {}
    for index, label in enumerate(NIH_LABELS):
        positives = int(y[:, index].sum())
        auc = float(roc_auc_score(y[:, index], p[:, index])) if len(np.unique(y[:, index])) == 2 else None
        ap = float(average_precision_score(y[:, index], p[:, index])) if positives else None
        if auc is not None:
            aucs.append(auc)
        if ap is not None:
            aps.append(ap)
        pred = p[:, index] >= 0.5
        actual = y[:, index].astype(bool)
        per_label[label] = {
            "positives": positives,
            "auroc": auc,
            "auprc": ap,
            "true_positive_at_0_5": int(np.logical_and(actual, pred).sum()),
            "false_positive_at_0_5": int(np.logical_and(~actual, pred).sum()),
        }
    return {
        "images": int(len(y)),
        "macro_auroc": float(np.mean(aucs)) if aucs else None,
        "macro_auprc": float(np.mean(aps)) if aps else None,
        "micro_f1_at_0_5": float(2 * np.logical_and(y.astype(bool), p >= 0.5).sum() /
                                  max(int(y.astype(bool).sum() + (p >= 0.5).sum()), 1)),
        "per_label": per_label,
    }


def checkpoint_selection_score(metrics: dict, selection_metric: str) -> float | None:
    """Return the declared validation metric used to keep a checkpoint."""
    metric_key = {
        "macro_auprc": "macro_auprc",
        "macro_auroc": "macro_auroc",
    }.get(selection_metric)
    if metric_key is None:
        raise ValueError(f"Unsupported checkpoint selection metric: {selection_metric}")
    value = metrics.get(metric_key)
    return float(value) if value is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-metric", choices=("macro_auprc", "macro_auroc"),
                        default="macro_auprc",
                        help="Validation metric for best-checkpoint selection; macro AP is primary.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch size must be positive; workers cannot be negative")
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    train = NIHXRVSubset(args.manifest, args.image_root, "train")
    validation = NIHXRVSubset(args.manifest, args.image_root, "validation")
    train_patients = set(train.rows["Patient ID"].astype(str))
    validation_patients = set(validation.rows["Patient ID"].astype(str))
    if train_patients & validation_patients:
        raise ValueError("Train and validation contain overlapping patient IDs")
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=torch.cuda.is_available())
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.workers, pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(args.weights).to(device)
    positives = train.targets.sum(axis=0)
    ratio = (len(train) - positives + 1.0) / (positives + 1.0)
    pos_weight = np.sqrt(ratio).clip(1.0, 10.0).astype(np.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score = -float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        metrics = evaluate(model, validation_loader, device)
        row = {"epoch": epoch, "train_bce": float(np.mean(losses)), **metrics}
        history.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "per_label"}), flush=True)
        selection_score = checkpoint_selection_score(metrics, args.selection_metric)
        if selection_score is not None and selection_score > best_score:
            best_score = selection_score
            torch.save({
                "architecture": "torchxrayvision_densenet121",
                "initialization": "densenet121-res224-chex (CheXpert-only)",
                "state_dict": model.state_dict(),
                "labels": NIH_LABELS,
                "input_normalization": "XRV grayscale, center crop, resize 224, scale 8-bit pixels to [-1024,1024]",
                "threshold": 0.5,
                "best_epoch": epoch,
                "validation_metrics": metrics,
                "checkpoint_selection_metric": args.selection_metric,
                "warning": "Educational subset experiment; not for diagnosis or clinical use.",
            }, args.output / "best_chexpert_initialized.pt")
    weights_sha = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {
        "architecture": "TorchXRayVision DenseNet-121",
        "initialization": "CheXpert-only checkpoint; NIH-containing combined checkpoint excluded",
        "xrv_version": xrv.__version__,
        "weight_sha256": weights_sha,
        "manifest_sha256": manifest_sha,
        "train_images": len(train),
        "train_patients": len(train_patients),
        "validation_images": len(validation),
        "validation_patients": len(validation_patients),
        "patient_overlap": 0,
        "heldout_or_official_test_accessed": False,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "device": str(device),
        "validation_checkpoint_selection": f"highest {args.selection_metric}",
        "selection_metric": args.selection_metric,
        "history_file": "history.json",
    }
    (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
