"""Compare fixed, equal-weight soft-voting ensembles on NIH validation only.

This script never constructs a test dataset. It averages sigmoid probabilities
from the already-trained three-seed DenseNet and label-graph residual runs.
No ensemble weights, calibration parameters, or thresholds are fit here.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

# Make the project root importable when this file is run directly from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_audit import NIH_LABELS
from train_classifier import NIHChestXrayDataset, make_model, seed_everything, safe_auc


SEED_RUNS = {
    42: "research_comparison_subset_20261028",
    11: "research_comparison_subset_20261028_seed11",
    21: "research_comparison_subset_20261028_seed21",
}
MODELS = {
    "densenet121": "densenet121",
    "label_graph_residual": "densenet121_label_graph_residual",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_scores(targets: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    binary = scores >= 0.5
    per_label: dict[str, Any] = {}
    aucs: list[float] = []
    aps: list[float] = []
    for column, label in enumerate(NIH_LABELS):
        y = targets[:, column].astype(np.uint8)
        p = scores[:, column]
        auc = safe_auc(y, p)
        ap = float(average_precision_score(y, p)) if y.sum() else None
        predicted = binary[:, column]
        tp = int(np.logical_and(y == 1, predicted).sum())
        fp = int(np.logical_and(y == 0, predicted).sum())
        fn = int(np.logical_and(y == 1, ~predicted).sum())
        per_label[label] = {
            "positive_count": int(y.sum()),
            "auroc": auc,
            "auprc": ap,
            "precision_at_0_5": tp / (tp + fp) if tp + fp else 0.0,
            "recall_at_0_5": tp / (tp + fn) if tp + fn else 0.0,
            "f1_at_0_5": (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }
        if auc is not None:
            aucs.append(auc)
        if ap is not None:
            aps.append(ap)
    return {
        "images": int(targets.shape[0]),
        "threshold": 0.5,
        "macro_auroc_defined_labels": float(np.mean(aucs)) if aucs else None,
        "macro_auprc_defined_labels": float(np.mean(aps)) if aps else None,
        "micro_precision_at_0_5": float(precision_score(targets, binary, average="micro", zero_division=0)),
        "micro_recall_at_0_5": float(recall_score(targets, binary, average="micro", zero_division=0)),
        "micro_f1_at_0_5": float(f1_score(targets, binary, average="micro", zero_division=0)),
        "multilabel_brier": float(np.mean((scores - targets) ** 2)),
        "per_label": per_label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    root = args.project_root.resolve()
    data_root = (args.data_root or root / "data" / "NexviaProjectData" / "nih").resolve()
    # Use only the frozen bounded-subset manifest, not the full-corpus manifest.
    # The full manifest cannot be satisfied by this local 7,271-image subset.
    manifest = data_root / "metadata" / "colab_t4_subset" / "trainval_manifest.csv"
    image_root = data_root / "images"
    output_dir = (args.output_dir or data_root / "validation_ensembles_20261029").resolve()
    if not manifest.is_file() or not image_root.is_dir():
        raise FileNotFoundError("The local NIH manifest or image subset is unavailable")

    validation = NIHChestXrayDataset(
        manifest, image_root, split="validation", limit=None, seed=42,
        allow_available_subset=False,
    )
    frame = pd.read_csv(manifest, usecols=["Patient ID", "split"])
    train_patients = set(frame.loc[frame["split"] == "train", "Patient ID"].astype(str))
    validation_patients = set(validation.rows["Patient ID"].astype(str))
    if train_patients.intersection(validation_patients):
        raise ValueError("Patient overlap detected between train and validation")
    if len(validation) != 1779:
        raise ValueError(f"Expected the full 1,779-image validation split; found {len(validation)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(42)
    loader = DataLoader(
        validation, batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    targets: np.ndarray | None = None
    predictions: dict[str, np.ndarray] = {}
    checkpoint_info: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    specs = [
        (seed, model_key, model_arch)
        for seed, run_dir in SEED_RUNS.items()
        for model_key, model_arch in MODELS.items()
    ]

    for index, (seed, model_key, architecture) in enumerate(specs, start=1):
        run_dir = SEED_RUNS[seed]
        relative_checkpoint = Path(run_dir) / architecture / f"best_{architecture}.pt"
        checkpoint_path = data_root / relative_checkpoint
        summary_path = checkpoint_path.parent / "run_summary.json"
        if not checkpoint_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"Missing trained checkpoint or run summary: {relative_checkpoint}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("test_split_used") is not False:
            raise ValueError(f"Run summary does not certify a train/validation-only run: {relative_checkpoint}")
        if summary.get("train_images") != 4000 or summary.get("validation_images") != 1779:
            raise ValueError(f"Run split does not match the frozen subset protocol: {relative_checkpoint}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("architecture") != architecture:
            raise ValueError(f"Architecture mismatch in {relative_checkpoint}")
        if checkpoint.get("seed") != seed:
            raise ValueError(f"Seed mismatch in {relative_checkpoint}")
        if checkpoint.get("labels") != list(NIH_LABELS):
            raise ValueError(f"Label order mismatch in {relative_checkpoint}")
        if checkpoint.get("training_images") != 4000 or checkpoint.get("validation_images") != 1779:
            raise ValueError(f"Checkpoint split size mismatch in {relative_checkpoint}")

        print(f"[{index}/{len(specs)}] Evaluating {model_key}, seed {seed}, epoch {checkpoint.get('best_epoch')} on validation only", flush=True)
        model = make_model(
            pretrained=False,
            architecture=architecture,
            label_adjacency=checkpoint.get("label_adjacency"),
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device)
        model.eval()

        batch_targets: list[np.ndarray] = []
        batch_scores: list[np.ndarray] = []
        with torch.inference_mode():
            for images, labels in loader:
                logits = model(images.to(device, non_blocking=device.type == "cuda"))
                batch_scores.append(torch.sigmoid(logits).cpu().numpy())
                batch_targets.append(labels.numpy())
        y = np.concatenate(batch_targets)
        scores = np.concatenate(batch_scores)
        if targets is None:
            targets = y
        elif not np.array_equal(targets, y):
            raise ValueError("Validation labels/order differed between model passes")

        key = f"{model_key}_seed{seed}"
        predictions[key] = scores
        checkpoint_info[key] = {
            "architecture": architecture,
            "seed": seed,
            "best_epoch": checkpoint["best_epoch"],
            "path": relative_checkpoint.as_posix(),
            "sha256": sha256_file(checkpoint_path),
            "validation_metrics": summarize_scores(y, scores),
        }
        del model, checkpoint, batch_targets, batch_scores, scores, y
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert targets is not None
    groups = {
        "DenseNet three-seed soft vote": [predictions[f"densenet121_seed{s}"] for s in SEED_RUNS],
        "Label-graph residual three-seed soft vote": [predictions[f"label_graph_residual_seed{s}"] for s in SEED_RUNS],
        "Six-model cross-architecture soft vote": list(predictions.values()),
    }
    ensemble_results = {}
    ensemble_predictions = {}
    for name, members in groups.items():
        averaged = np.mean(np.stack(members, axis=0), axis=0, dtype=np.float64).astype(np.float32)
        ensemble_predictions[name] = averaged
        ensemble_results[name] = {
            "members": len(members),
            "rule": "uniform mean of sigmoid probabilities; no fitted weights or calibration",
            "metrics": summarize_scores(targets, averaged),
        }

    manifest_hash = sha256_file(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "validation_probabilities.npz", targets=targets, **{
        f"model_{key}": value for key, value in predictions.items()
    }, **{
        f"ensemble_{index}": value for index, value in enumerate(ensemble_predictions.values())
    })
    result = {
        "dataset": "NIH ChestX-ray14 retained patient-disjoint subset",
        "split": "validation only",
        "images": len(validation),
        "patients": int(validation.rows["Patient ID"].nunique()),
        "patient_overlap_train_validation": False,
        "manifest_sha256": manifest_hash,
        "device": str(device),
        "batch_size": args.batch_size,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "threshold": 0.5,
        "test_split_loaded_or_evaluated": False,
        "checkpoint_selection_note": "Each member was previously selected by validation macro AUROC; these ensemble comparisons are exploratory validation results and are not an independent estimate.",
        "checkpoints": checkpoint_info,
        "ensembles": ensemble_results,
    }
    result_path = output_dir / "ensemble_metrics.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(result_path), "device": str(device), "ensembles": ensemble_results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
