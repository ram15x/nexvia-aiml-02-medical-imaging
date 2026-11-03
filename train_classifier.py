"""Train a small NIH ChestX-ray14 multi-label DenseNet baseline.

This is a coursework experiment, not a clinical model. It trains on the
patient-disjoint `train` rows in a manifest and selects checkpoints on the
`validation` rows. It never reads the held-out `test` split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import InterpolationMode, RandomAffine
from torchvision.transforms import functional as TF

from api.modeling import ARCHITECTURE_NAMES, IMAGE_SIZE, build_classifier, image_to_tensor
from data_audit import NIH_LABELS


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def labels_to_vector(raw: str) -> np.ndarray:
    labels = set(str(raw).split("|"))
    labels.discard("No Finding")
    unknown = labels - set(NIH_LABELS)
    if unknown:
        raise ValueError(f"Unexpected NIH labels: {sorted(unknown)}")
    return np.asarray([float(label in labels) for label in NIH_LABELS], dtype=np.float32)


def positive_class_weights(targets: np.ndarray, exponent: float = 0.5,
                           cap: float = 10.0) -> np.ndarray:
    """Return smoothed, capped BCE positive weights from training targets only."""
    values = np.asarray(targets, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(NIH_LABELS) or not len(values):
        raise ValueError("targets must be a non-empty 2D matrix with one column per NIH label")
    if exponent not in {0.5, 1.0} or cap < 1.0:
        raise ValueError("exponent must be 0.5 or 1.0 and cap must be at least 1")
    positives = values.sum(axis=0)
    negatives = len(values) - positives
    ratio = (negatives + 1.0) / (positives + 1.0)
    weights = np.sqrt(ratio) if exponent == 0.5 else ratio
    return weights.clip(1.0, cap).astype(np.float32)


class MaskedNegativeBCE(nn.Module):
    """Keep positive and No Finding supervision; mask unmentioned labels in abnormal rows.

    This is an explicit weak-label hypothesis for NIH report-mined targets, not a
    claim that every unmentioned finding is actually present. The denominator is
    kept equal to ordinary BCE (batch * labels), so the ablation changes only
    which negative terms contribute gradients. Positive-class weights are supplied
    from the same full training targets as the BCE control.
    """

    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight.detach().clone())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(dtype=logits.dtype)
        per_label = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none",
        )
        no_finding = targets.sum(dim=1, keepdim=True).eq(0)
        observed = targets.bool() | no_finding
        return (per_label * observed.to(per_label.dtype)).sum() / targets.numel()


class NIHChestXrayDataset(Dataset):
    def __init__(self, manifest: Path, image_root: Path, split: str,
                 limit: int | None, seed: int, image_size: int = 224,
                 allow_available_subset: bool = False,
                 additional_image_roots: tuple[Path, ...] = (),
                 augmentation_policy: str = "flip_only"):
        if augmentation_policy not in {"flip_only", "light_affine"}:
            raise ValueError(f"Unknown augmentation policy: {augmentation_policy}")
        frame = pd.read_csv(manifest)
        required = {"Image Index", "Finding Labels", "Patient ID", "split"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        frame = frame.loc[frame["split"] == split].copy()
        if frame.empty:
            raise ValueError(f"Manifest has no rows for split={split!r}")
        self.image_roots = [image_root.resolve(), *(root.resolve() for root in additional_image_roots)]
        def safe_path(name: str) -> Path:
            if Path(name).name != name:
                raise ValueError(f"Unsafe image filename in manifest: {name}")
            for root in self.image_roots:
                path = (root / name).resolve()
                if path.parent != root:
                    raise ValueError(f"Unsafe image filename in manifest: {name}")
                if path.is_file():
                    return path
            return self.image_roots[0] / name

        available = frame["Image Index"].astype(str).map(lambda name: safe_path(name).is_file())
        self.missing_images = int((~available).sum())
        self.available_images = int(available.sum())
        if self.missing_images and not allow_available_subset:
            first_missing = str(frame.loc[~available, "Image Index"].iloc[0])
            raise FileNotFoundError(
                f"{self.missing_images} images are missing from split={split}; first: {first_missing}. "
                "Use --allow-available-subset only for a documented partial-data pilot."
            )
        if allow_available_subset:
            frame = frame.loc[available].copy()
        if frame.empty:
            raise ValueError(f"No available images for split={split!r}")
        if limit is not None and len(frame) > limit:
            frame = frame.sample(n=limit, random_state=seed).sort_values("Image Index")
        self.rows = frame.reset_index(drop=True)
        self.image_size = image_size
        self.augment = split == "train"
        self.augmentation_policy = augmentation_policy
        self.paths: list[Path] = []
        for name in self.rows["Image Index"].astype(str):
            path = safe_path(name)
            if not path.is_file():
                raise FileNotFoundError(path)
            self.paths.append(path)
        self.targets = np.stack([labels_to_vector(value)
                                 for value in self.rows["Finding Labels"]])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.paths[index]) as source:
            image = source.convert("RGB")
            # Laterality is not a target in this 14-finding task. The flip is a
            # conservative training-only augmentation; validation remains fixed.
            if self.augment and random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if self.augment and self.augmentation_policy == "light_affine":
                angle, translation, scale, shear = RandomAffine.get_params(
                    degrees=(-5, 5),
                    translate=(0.02, 0.02),
                    scale_ranges=(0.97, 1.03),
                    shears=None,
                    img_size=[image.height, image.width],
                )
                image = TF.affine(
                    image, angle=angle, translate=translation, scale=scale,
                    shear=shear, interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                )
        tensor = image_to_tensor(image, self.image_size)
        return tensor, torch.from_numpy(self.targets[index])


def make_model(pretrained: bool, architecture: str = "densenet121",
               label_adjacency: np.ndarray | None = None) -> nn.Module:
    try:
        return build_classifier(architecture, len(NIH_LABELS), pretrained=pretrained,
                                label_adjacency=label_adjacency)
    except ImportError as exc:
        raise RuntimeError(
            "Install a torchvision build compatible with the active PyTorch runtime"
        ) from exc


def training_label_graph(targets: np.ndarray) -> np.ndarray:
    """Build a positive-PMI label graph from training targets only."""
    positives = (np.asarray(targets) > 0.5).astype(np.float64)
    if positives.ndim != 2 or positives.shape[1] != len(NIH_LABELS):
        raise ValueError("targets must be a 2D array with one column per NIH label")
    sample_count = len(positives)
    if sample_count == 0:
        raise ValueError("cannot build a label graph from an empty training split")
    counts = positives.T @ positives
    prevalence = (positives.sum(axis=0) + 1.0) / (sample_count + 2.0)
    joint = (counts + 1.0) / (sample_count + 2.0)
    denominator = np.outer(prevalence, prevalence)
    pmi = np.log(joint / np.maximum(denominator, 1e-12))
    np.fill_diagonal(pmi, 0.0)
    return np.maximum(pmi, 0.0).astype(np.float32)


def safe_auc(targets: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(targets)) < 2:
        return None
    return float(roc_auc_score(targets, scores))


def checkpoint_selection_score(metrics: dict, selection_metric: str) -> float | None:
    """Return the declared validation metric used to choose a checkpoint."""
    metric_keys = {
        "macro_auprc": "macro_auprc_defined_labels",
        "macro_auroc": "macro_auroc_defined_labels",
    }
    if selection_metric not in metric_keys:
        raise ValueError(f"Unsupported checkpoint selection metric: {selection_metric}")
    value = metrics.get(metric_keys[selection_metric])
    return float(value) if value is not None else None


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    all_targets, all_scores = [], []
    for images, targets in loader:
        scores = torch.sigmoid(model(images.to(device))).cpu().numpy()
        all_targets.append(targets.numpy())
        all_scores.append(scores)
    y = np.concatenate(all_targets)
    p = np.concatenate(all_scores)
    per_label = {}
    aucs, aps = [], []
    binary = p >= 0.5
    for column, label in enumerate(NIH_LABELS):
        auc = safe_auc(y[:, column], p[:, column])
        ap = float(average_precision_score(y[:, column], p[:, column])) if y[:, column].sum() else None
        actual = y[:, column].astype(bool)
        predicted = binary[:, column]
        per_label[label] = {
            "positive_count": int(actual.sum()),
            "auroc": auc,
            "auprc": ap,
            "confusion_at_0_5": {
                "true_positive": int(np.logical_and(actual, predicted).sum()),
                "false_positive": int(np.logical_and(~actual, predicted).sum()),
                "false_negative": int(np.logical_and(actual, ~predicted).sum()),
                "true_negative": int(np.logical_and(~actual, ~predicted).sum()),
            },
        }
        if auc is not None:
            aucs.append(auc)
        if ap is not None:
            aps.append(ap)
    threshold = 0.5
    metrics = {
        "images": int(len(y)),
        "threshold": threshold,
        "macro_auroc_defined_labels": float(np.mean(aucs)) if aucs else None,
        "macro_auprc_defined_labels": float(np.mean(aps)) if aps else None,
        "micro_precision_at_0_5": float(precision_score(y, binary, average="micro", zero_division=0)),
        "micro_recall_at_0_5": float(recall_score(y, binary, average="micro", zero_division=0)),
        "micro_f1_at_0_5": float(f1_score(y, binary, average="micro", zero_division=0)),
        "per_label": per_label,
    }
    return metrics, y, p


def save_latest_checkpoint(path: Path, *, model: nn.Module, optimizer: torch.optim.Optimizer,
                           scaler: torch.amp.GradScaler, epoch: int, next_batch: int,
                           epoch_order: list[int] | None, running_loss: float, seen: int,
                           history: list[dict], best_score: float, manifest_sha256: str,
                           run_config: dict) -> None:
    """Keep enough state to resume after Colab or a local session disconnects."""
    state = {
        "checkpoint_version": 1,
        "epoch": epoch,
        "next_batch": next_batch,
        "epoch_order": epoch_order,
        "running_loss": running_loss,
        "seen": seen,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "history": history,
        "best_score": best_score,
        "manifest_sha256": manifest_sha256,
        "run_config": run_config,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


class AsymmetricLoss(nn.Module):
    """ASL for multi-label targets, normalized to a per-label mean.

    Hyperparameters follow the official multi-label recipe (gamma_neg=4,
    gamma_pos=0, clip=0.05). The focal weight is detached, as in the official
    implementation, to avoid differentiating through the modulation factor.
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0,
                 clip: float = 0.05, eps: float = 1e-8) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        prob_pos = torch.sigmoid(logits)
        prob_neg = 1.0 - prob_pos
        if self.clip > 0:
            prob_neg = (prob_neg + self.clip).clamp(max=1.0)
        log_likelihood = (
            targets * torch.log(prob_pos.clamp(min=self.eps))
            + (1.0 - targets) * torch.log(prob_neg.clamp(min=self.eps))
        )
        with torch.no_grad():
            pt = prob_pos * targets + prob_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            focal_weight = torch.pow(1.0 - pt, gamma)
        return -(log_likelihood * focal_weight).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--additional-image-root", type=Path, action="append", default=[],
                        help="Extra image directory; may be repeated to train from multiple shards without copying them.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", choices=tuple(ARCHITECTURE_NAMES),
                        default="densenet121",
                        help="Classifier architecture; fusion keeps both pretrained encoders frozen.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--image-size", type=int, choices=(224, 320), default=IMAGE_SIZE,
                        help="Input side length; 320 is for the controlled resolution screen.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--selection-metric", choices=("macro_auprc", "macro_auroc"),
                        default="macro_auprc",
                        help="Validation metric for best-checkpoint selection; macro AP is the project primary metric.")
    parser.add_argument("--limit-train", type=int, default=None,
                        help="Deterministic engineering pilot cap; not a full-data result.")
    parser.add_argument("--limit-validation", type=int, default=None)
    parser.add_argument("--allow-available-subset", action="store_true",
                        help="Ignore missing files only for a clearly documented partial-data pilot.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positive-weight-exponent", type=float, choices=(0.5, 1.0), default=0.5,
                        help="Exponent on smoothed negative/positive ratio; 0.5 is the established square-root baseline, 1.0 is stronger positive weighting (still capped at 10).")
    parser.add_argument("--augmentation-policy", choices=("flip_only", "light_affine"),
                        default="flip_only",
                        help="Use horizontal flips only, or add small train-only affine changes.")
    parser.add_argument("--loss", choices=("weighted_bce", "masked_bce", "asymmetric"), default="weighted_bce",
                        help="Loss function. masked_bce keeps positive labels and No Finding negatives, masking unmentioned labels on abnormal rows; it tests a weak-label assumption.")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Train from random initialization instead of ImageNet DenseNet121 weights.")
    parser.add_argument("--checkpoint-every-steps", type=int, default=1000,
                        help="Write an atomic resumable checkpoint every N optimizer steps.")
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="Resume from a last_checkpoint.pt created with the same manifest and run settings.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.checkpoint_every_steps < 1:
        parser.error("--epochs, --batch-size, and --checkpoint-every-steps must be positive")

    seed_everything(args.seed)
    train = NIHChestXrayDataset(args.manifest, args.image_root, "train", args.limit_train,
                                args.seed, image_size=args.image_size,
                                allow_available_subset=args.allow_available_subset,
                                additional_image_roots=tuple(args.additional_image_root),
                                augmentation_policy=args.augmentation_policy)
    validation = NIHChestXrayDataset(args.manifest, args.image_root, "validation", args.limit_validation,
                                     args.seed, image_size=args.image_size,
                                     allow_available_subset=args.allow_available_subset,
                                     additional_image_roots=tuple(args.additional_image_root))
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.workers, pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_architectures = {"densenet121_label_graph", "densenet121_label_graph_residual"}
    label_adjacency = (training_label_graph(train.targets)
                       if args.architecture in graph_architectures else None)
    graph_sha256 = (hashlib.sha256(label_adjacency.tobytes()).hexdigest()
                    if label_adjacency is not None else None)
    model = make_model(
        pretrained=not args.no_pretrained and args.resume_from is None,
        architecture=args.architecture,
        label_adjacency=label_adjacency,
    ).to(device)

    # The BCE reference uses square-root positive weights; ASL is a separate
    # multi-label loss strategy and does not additionally apply those weights.
    if args.loss in {"weighted_bce", "masked_bce"}:
        pos_weight = positive_class_weights(train.targets, args.positive_weight_exponent, 10.0)
        pos_weight_tensor = torch.tensor(pos_weight, device=device)
        if args.loss == "masked_bce":
            criterion = MaskedNegativeBCE(pos_weight_tensor)
            loss_description = "MaskedBCE(positive labels + No Finding negatives; unmentioned labels masked)"
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            loss_description = "BCEWithLogitsLoss"
    else:
        criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
        loss_description = "AsymmetricLoss(gamma_neg=4,gamma_pos=0,clip=0.05,mean)"
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    args.output.mkdir(parents=True, exist_ok=True)
    best_score = -float("inf")
    history = []
    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    run_config = {
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "selection_metric": args.selection_metric,
        "loss": args.loss,
        "negative_label_policy": (
            "positive_targets_and_no_finding_negatives_only"
            if args.loss == "masked_bce" else "all_zero_targets_treated_as_negative"
        ),
        "positive_weight_exponent": args.positive_weight_exponent if args.loss in {"weighted_bce", "masked_bce"} else None,
        "positive_weight_cap": 10.0 if args.loss in {"weighted_bce", "masked_bce"} else None,
        "asl_gamma_neg": 4.0 if args.loss == "asymmetric" else None,
        "asl_gamma_pos": 0.0 if args.loss == "asymmetric" else None,
        "asl_clip": 0.05 if args.loss == "asymmetric" else None,
        "seed": args.seed, "train_limit": args.limit_train,
        "validation_limit": args.limit_validation,
        "allow_available_subset": args.allow_available_subset,
        "image_roots": [str(args.image_root.resolve()),
                        *(str(root.resolve()) for root in args.additional_image_root)],
        "pretrained": not args.no_pretrained, "image_size": args.image_size,
        "architecture": args.architecture,
        "augmentation_policy": args.augmentation_policy,
        "label_graph_sha256": graph_sha256,
    }
    start_epoch, resume_batch, resume_order = 1, 0, None
    running_loss, seen = 0.0, 0
    if args.resume_from is not None:
        state = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if state.get("checkpoint_version") != 1:
            raise ValueError("Unsupported resume checkpoint version")
        if state.get("manifest_sha256") != manifest_sha256:
            raise ValueError("The manifest differs from the one used by the resume checkpoint")
        previous_config = state.get("run_config")
        if not isinstance(previous_config, dict):
            raise ValueError("Resume checkpoint is missing its training settings")
        previous_config.setdefault("architecture", "densenet121")
        previous_config.setdefault("positive_weight_exponent", 0.5)
        previous_config.setdefault("positive_weight_cap", 10.0)
        previous_config.setdefault("augmentation_policy", "flip_only")
        previous_config.setdefault("selection_metric", "macro_auroc")
        fixed_keys = (set(previous_config) | set(run_config)) - {"epochs"}
        changed = [key for key in sorted(fixed_keys)
                   if previous_config.get(key) != run_config.get(key)]
        if changed:
            raise ValueError(f"Training settings differ from the resume checkpoint: {changed}")
        if args.epochs < int(previous_config.get("epochs", 0)):
            raise ValueError("The resumed epoch target cannot be lower than the saved target")
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        scaler.load_state_dict(state["scaler_state"])
        history = state["history"]
        best_score = state["best_score"]
        start_epoch = int(state["epoch"])
        resume_batch = int(state["next_batch"])
        resume_order = state["epoch_order"]
        running_loss, seen = float(state["running_loss"]), int(state["seen"])
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if device.type == "cuda" and state["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        print(json.dumps({"event": "resumed", "epoch": start_epoch,
                          "next_batch": resume_batch, "history_epochs": len(history)}), flush=True)

    training_start = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        if epoch == start_epoch and resume_order is not None:
            order = resume_order
            first_batch = resume_batch
        else:
            generator = torch.Generator().manual_seed(args.seed + epoch)
            order = torch.randperm(len(train), generator=generator).tolist()
            first_batch = 0
            running_loss, seen = 0.0, 0
        remaining = order[first_batch * args.batch_size:]
        epoch_train = Subset(train, remaining)
        train_loader = DataLoader(epoch_train, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.workers, pin_memory=torch.cuda.is_available())
        for local_batch, (images, targets) in enumerate(train_loader):
            batch_index = first_batch + local_batch
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * len(images)
            seen += len(images)
            if (batch_index + 1) % args.checkpoint_every_steps == 0:
                save_latest_checkpoint(
                    args.output / "last_checkpoint.pt", model=model, optimizer=optimizer,
                    scaler=scaler, epoch=epoch, next_batch=batch_index + 1,
                    epoch_order=order, running_loss=running_loss, seen=seen,
                    history=history, best_score=best_score,
                    manifest_sha256=manifest_sha256, run_config=run_config,
                )
                print(json.dumps({"event": "checkpoint", "epoch": epoch,
                                  "step": batch_index + 1, "images_seen": seen}), flush=True)

        metrics, _, _ = evaluate(model, validation_loader, device)
        train_loss = running_loss / max(seen, 1)
        record = {"epoch": epoch, "train_loss": train_loss, "loss_name": args.loss,
                  "epoch_elapsed_seconds": time.perf_counter() - epoch_start, **metrics}
        if args.loss in {"weighted_bce", "masked_bce"}:
            record["train_bce"] = train_loss
        history.append(record)
        print(json.dumps(record), flush=True)
        selection_score = checkpoint_selection_score(metrics, args.selection_metric)
        if selection_score is not None and selection_score > best_score:
            best_score = selection_score
            torch.save({
                "architecture": args.architecture,
                "model_name": ARCHITECTURE_NAMES[args.architecture],
                "labels": list(NIH_LABELS),
                "state_dict": model.state_dict(),
                "label_adjacency": (torch.as_tensor(label_adjacency)
                                    if label_adjacency is not None else None),
                "label_graph_source": ("training_split_only_positive_pmi_v1"
                                       if label_adjacency is not None else None),
                "validation_metrics": metrics,
                "checkpoint_selection_metric": args.selection_metric,
                "image_size": args.image_size,
                "normalization": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "version": "imagenet_rgb_bilinear_square_v1",
                },
                "seed": args.seed,
                "loss": loss_description,
                "negative_label_policy": run_config["negative_label_policy"],
                "positive_weight_exponent": run_config["positive_weight_exponent"],
                "positive_weight_cap": run_config["positive_weight_cap"],
                "asl_gamma_neg": run_config["asl_gamma_neg"],
                "asl_gamma_pos": run_config["asl_gamma_pos"],
                "asl_clip": run_config["asl_clip"],
                "training_images": len(train),
                "validation_images": len(validation),
                "best_epoch": epoch,
                "warning": "Educational research artifact only; not for clinical use.",
            }, args.output / f"best_{args.architecture}.pt")

        save_latest_checkpoint(
            args.output / "last_checkpoint.pt", model=model, optimizer=optimizer,
            scaler=scaler, epoch=epoch + 1, next_batch=0, epoch_order=None,
            running_loss=0.0, seen=0, history=history, best_score=best_score,
            manifest_sha256=manifest_sha256, run_config=run_config,
        )

    (args.output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    run = {
        "device": str(device), "epochs": args.epochs,
        "train_images": len(train), "validation_images": len(validation),
        "train_images_available_before_limit": train.available_images,
        "validation_images_available_before_limit": validation.available_images,
        "train_images_missing_from_manifest": train.missing_images,
        "validation_images_missing_from_manifest": validation.missing_images,
        "allow_available_subset": args.allow_available_subset,
        "train_limit_applied": args.limit_train is not None,
        "validation_limit_applied": args.limit_validation is not None,
        "checkpoint_selection": f"highest validation {args.selection_metric} across labels with both classes present",
        "selection_metric": args.selection_metric,
        "loss": loss_description,
        "negative_label_policy": run_config["negative_label_policy"],
        "positive_weight_exponent": run_config["positive_weight_exponent"],
        "positive_weight_cap": run_config["positive_weight_cap"],
        "asl_gamma_neg": run_config["asl_gamma_neg"],
        "asl_gamma_pos": run_config["asl_gamma_pos"],
        "asl_clip": run_config["asl_clip"],
        "architecture": args.architecture,
        "model_name": ARCHITECTURE_NAMES[args.architecture],
        "augmentation_policy": args.augmentation_policy,
        "image_size": args.image_size,
        "elapsed_seconds": time.perf_counter() - training_start,
        "peak_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "peak_cuda_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
        ),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "test_split_used": False,
        "label_source": "NIH ChestX-ray14 report-mined labels; uncertain or unmentioned findings are not resolved",
    }
    (args.output / "run_summary.json").write_text(json.dumps(run, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
