"""Screen a staged ImageNet DenseNet fine-tuning workflow on the fixed 500 patients.

Each fold starts from ImageNet weights (never the NIH-trained app checkpoint),
warms the classifier head for two epochs, then unfreezes DenseNet's final block
and norm for eight epochs. Predictions are from the final epoch only.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_audit import NIH_LABELS  # noqa: E402
from scripts.run_500_image_oof_architecture_comparison import summarize  # noqa: E402
from train_classifier import (  # noqa: E402
    NIHChestXrayDataset,
    evaluate,
    make_model,
    positive_class_weights,
    seed_everything,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_trainable(model: torch.nn.Module, phase: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    if phase == "partial":
        for module in (model.features.denseblock4, model.features.norm5):
            for parameter in module.parameters():
                parameter.requires_grad_(True)


def configure_train_mode(model: torch.nn.Module, phase: str) -> None:
    model.train()
    # Keep all frozen feature blocks, including their batch-normalization
    # statistics, fixed. Only the selected final block adapts in phase two.
    model.features.eval()
    if phase == "partial":
        model.features.denseblock4.train()
        model.features.norm5.train()
    model.classifier.train()


def train_phase(model: torch.nn.Module, loader: DataLoader, targets: np.ndarray,
                device: torch.device, phase: str, epochs: int, seed: int) -> list[dict]:
    set_trainable(model, phase)
    if phase == "head":
        groups = [{"params": model.classifier.parameters(), "lr": 1e-3}]
    else:
        groups = [
            {"params": model.features.denseblock4.parameters(), "lr": 1e-5},
            {"params": model.features.norm5.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-3},
        ]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_class_weights(targets, 0.5, 10.0), device=device)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    for epoch in range(1, epochs + 1):
        configure_train_mode(model, phase)
        running_loss = 0.0
        seen = 0
        for step, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * len(images)
            seen += len(images)
        record = {
            "phase": phase,
            "phase_epoch": epoch,
            "train_bce": running_loss / max(seen, 1),
            "training_rows": seen,
            "seed": seed,
        }
        history.append(record)
        print(json.dumps({"event": "epoch_complete", **record}), flush=True)
    return history


def main() -> None:
    nih = PROJECT_ROOT / "data" / "NexviaProjectData" / "nih"
    reference = nih / "oof_500_backbone_10epoch_20261031"
    manifest_path = reference / "cohort_manifest.csv"
    image_root = nih / "images_224_fresh_15k_20261029"
    output = nih / "oof_500_staged_densenet_20261031"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(manifest_path)
    if len(cohort) != 500 or cohort["Patient ID"].nunique() != 500:
        raise ValueError("Expected the frozen 500-image / 500-patient cohort")
    if "test" in set(cohort["split"].astype(str)):
        raise ValueError("Official-test rows are forbidden in this experiment")
    reference_frame = pd.read_csv(reference / "oof_predictions.csv")
    if reference_frame["Image Index"].tolist() != cohort["Image Index"].tolist():
        raise ValueError("Reference OOF rows do not exactly match the frozen cohort order")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = np.column_stack([
        np.asarray([float(label in str(raw).split("|")) for raw in cohort["Finding Labels"]],
                   dtype=np.float32)
        for label in NIH_LABELS
    ])
    oof = np.full_like(y, np.nan)
    histories = {}
    fold_hashes = {}
    started = time.perf_counter()
    for fold in range(1, 6):
        fold_manifest = reference / "fold_manifests" / f"fold_{fold:02d}.csv"
        fold_hashes[str(fold)] = digest(fold_manifest)
        seed = 20261031 + fold
        seed_everything(seed)
        train = NIHChestXrayDataset(fold_manifest, image_root, "train", None, seed, 224)
        validation = NIHChestXrayDataset(fold_manifest, image_root, "validation", None, seed, 224)
        if len(train) != 400 or len(validation) != 100:
            raise ValueError(f"Fold {fold} is not 400 train / 100 held out")
        train_loader = DataLoader(train, batch_size=4, shuffle=True, num_workers=0,
                                  pin_memory=device.type == "cuda")
        validation_loader = DataLoader(validation, batch_size=8, shuffle=False, num_workers=0,
                                       pin_memory=device.type == "cuda")
        model = make_model(pretrained=True, architecture="densenet121").to(device)
        history = []
        history += train_phase(model, train_loader, train.targets, device, "head", 2, seed)
        history += train_phase(model, train_loader, train.targets, device, "partial", 8, seed)
        model.eval()
        metrics, fold_y, fold_scores = evaluate(model, validation_loader, device)
        positions = cohort["Image Index"].astype(str).isin(
            validation.rows["Image Index"].astype(str)
        ).to_numpy()
        if positions.sum() != 100 or np.isfinite(oof[positions]).any():
            raise ValueError(f"Invalid or duplicate OOF assignment in fold {fold}")
        oof[positions] = fold_scores
        histories[str(fold)] = {
            "seed": seed,
            "training_history": history,
            "fixed_final_epoch_metrics": metrics,
            "train_images": len(train),
            "validation_images": len(validation),
        }
        print(json.dumps({"event": "fold_complete", "fold": fold,
                          "metrics": metrics}), flush=True)
        del model, train_loader, validation_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not np.isfinite(oof).all():
        raise RuntimeError("Missing or non-finite OOF predictions")
    prediction_frame = pd.DataFrame({"Image Index": cohort["Image Index"].astype(str)})
    for i, label in enumerate(NIH_LABELS):
        prediction_frame[f"target_{label}"] = y[:, i].astype(int)
        prediction_frame[f"score_staged_densenet_{label}"] = oof[:, i]
    prediction_frame.to_csv(output / "oof_predictions.csv", index=False)
    cohort.to_csv(output / "cohort_manifest.csv", index=False)
    (output / "fold_history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")

    base_scores = np.column_stack([
        reference_frame[f"score_densenet121_{label}"].to_numpy(dtype=np.float32)
        for label in NIH_LABELS
    ])
    candidate_metrics = summarize(y, oof)
    baseline_metrics = summarize(y, base_scores)
    metric_keys = ("macro_auroc_defined_labels", "macro_auprc_positive_labels",
                   "micro_precision_at_0_5", "micro_recall_at_0_5", "micro_f1_at_0_5")
    summary = {
        "purpose": "fixed staged-fine-tuning screen on the existing capped cohort",
        "cohort_images": 500,
        "unique_patients": int(cohort["Patient ID"].nunique()),
        "folds": 5,
        "train_validation_per_fold": [400, 100],
        "image_size": 224,
        "batch_size": 4,
        "initialization": "Torchvision ImageNet DenseNet-121 weights; not the NIH-trained demo checkpoint",
        "schedule": [
            {"phase": "head only", "epochs": 2, "head_lr": 0.001},
            {"phase": "unfreeze denseblock4 + norm5 and head", "epochs": 8,
             "backbone_lr": 0.00001, "head_lr": 0.001},
        ],
        "prediction_epoch": 10,
        "threshold": 0.5,
        "official_test_rows_used": 0,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "cohort_manifest_sha256": digest(manifest_path),
        "fold_manifest_sha256": fold_hashes,
        "baseline_full_finetune_metrics": baseline_metrics,
        "staged_finetune_metrics": candidate_metrics,
        "point_delta_staged_minus_full": {
            key: float(candidate_metrics[key] - baseline_metrics[key]) for key in metric_keys
        },
        "fold_history": histories,
        "limitations": [
            "The 500 patients were inspected in prior architecture screens; this remains exploratory and model-selection biased.",
            "Several findings have very few positives in the cohort, and Hernia has none.",
            "Point estimates only; no bootstrap interval was calculated in this bounded screen.",
            "A 0.5 threshold is a fixed comparison point, not a validated operating threshold.",
            "No official-test rows were used; the production checkpoint was not changed.",
        ],
    }
    (output / "staged_finetune_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Staged DenseNet fine-tuning screen on the capped 500-patient cohort",
        "",
        "**Date:** 2026-10-31  ",
        "**Status:** exploratory development result; no held-out test use or checkpoint promotion.",
        "",
        "## Protocol",
        "",
        "Each of the same five patient-level folds starts from ImageNet DenseNet-121 weights. The NIH-trained app checkpoint is deliberately not used because the 500 patients overlap its training and checkpoint-selection data.",
        "",
        "1. Freeze the encoder and train only the classifier head for 2 epochs (head LR 1e-3).",
        "2. Unfreeze DenseNet denseblock4, norm5, and the classifier for 8 epochs (backbone LR 1e-5; head LR 1e-3). Earlier feature blocks and their batch-normalization statistics stay frozen.",
        "3. Use the same 400/100 fold manifests, 224 input, batch 4, square-root positive weights, train-only horizontal flip, and fixed 0.5 threshold. Score epoch 10; no checkpoint or threshold is selected on the held-out fold.",
        "",
        "## Out-of-fold results",
        "",
        "| Workflow | Macro AUROC | Macro AP | Precision @0.5 | Recall @0.5 | Micro F1 @0.5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in (("Full fine-tune baseline", baseline_metrics),
                          ("Staged fine-tune", candidate_metrics)):
        report.append(
            f"| {name} | {metrics['macro_auroc_defined_labels']:.4f} | "
            f"{metrics['macro_auprc_positive_labels']:.4f} | "
            f"{metrics['micro_precision_at_0_5']:.4f} | "
            f"{metrics['micro_recall_at_0_5']:.4f} | {metrics['micro_f1_at_0_5']:.4f} |"
        )
    report += ["", "Point deltas (staged minus full fine-tune; no interval calculated):", ""]
    report += [f"- {key}: {value:+.4f}" for key, value in summary["point_delta_staged_minus_full"].items()]
    report += [
        "",
        "## Limits and decision",
        "",
        *[f"- {item}" for item in summary["limitations"]],
        "",
        "This is a controlled workflow comparison, not proof of clinical usefulness or an independent estimate. Do not promote based on this result alone.",
        "",
        f"Reproduction files: `{output.relative_to(PROJECT_ROOT).as_posix()}`.",
    ]
    report_path = PROJECT_ROOT / "reports" / "500_image_staged_densenet_finetune_2026-10-31.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "metrics": {
        "baseline": baseline_metrics,
        "staged": candidate_metrics,
        "delta": summary["point_delta_staged_minus_full"],
    }, "device": str(device), "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
