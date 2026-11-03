"""Exploratory comparison of frozen NIH candidates on the historically viewed 2,250-patient split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_audit import NIH_LABELS  # noqa: E402
from train_classifier import NIHChestXrayDataset, make_model, seed_everything  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(y: np.ndarray, p: np.ndarray) -> dict:
    aucs, aps = [], []
    per_label = {}
    predicted = p >= 0.5
    for col, label in enumerate(NIH_LABELS):
        target, scores = y[:, col], p[:, col]
        auc = float(roc_auc_score(target, scores)) if np.unique(target).size == 2 else None
        ap = float(average_precision_score(target, scores)) if target.sum() else None
        if auc is not None:
            aucs.append(auc)
        if ap is not None:
            aps.append(ap)
        per_label[label] = {
            "positive": int(target.sum()), "negative": int(len(target) - target.sum()),
            "auroc": auc, "average_precision": ap,
            "precision_at_0_5": float(precision_score(target, predicted[:, col], zero_division=0)),
            "recall_at_0_5": float(recall_score(target, predicted[:, col], zero_division=0)),
            "f1_at_0_5": float(f1_score(target, predicted[:, col], zero_division=0)),
        }
    flat_y, flat_p = y.ravel(), p.ravel()
    return {
        "macro_auroc": float(np.mean(aucs)) if aucs else None,
        "macro_auroc_label_count": len(aucs),
        "macro_average_precision": float(np.mean(aps)) if aps else None,
        "macro_average_precision_label_count": len(aps),
        "micro_auroc": float(roc_auc_score(flat_y, flat_p)) if np.unique(flat_y).size == 2 else None,
        "micro_average_precision": float(average_precision_score(flat_y, flat_p)) if flat_y.sum() else None,
        "micro_precision_at_0_5": float(precision_score(flat_y, flat_p >= 0.5, zero_division=0)),
        "micro_recall_at_0_5": float(recall_score(flat_y, flat_p >= 0.5, zero_division=0)),
        "micro_f1_at_0_5": float(f1_score(flat_y, flat_p >= 0.5, zero_division=0)),
        "labels": per_label,
    }


def aggregate_bootstrap_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | None]:
    aucs, aps = [], []
    for col in range(y.shape[1]):
        target, scores = y[:, col], p[:, col]
        if np.unique(target).size == 2:
            aucs.append(float(roc_auc_score(target, scores)))
        if target.sum():
            aps.append(float(average_precision_score(target, scores)))
    return {
        "macro_auroc": float(np.mean(aucs)) if aucs else None,
        "macro_average_precision": float(np.mean(aps)) if aps else None,
        "micro_f1_at_0_5": float(f1_score(y.ravel(), p.ravel() >= 0.5, zero_division=0)),
    }


def score_checkpoint(checkpoint_path: Path, dataset: NIHChestXrayDataset,
                     batch_size: int, device: torch.device) -> tuple[np.ndarray, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("labels") != list(NIH_LABELS):
        raise ValueError(f"Checkpoint label order mismatch: {checkpoint_path}")
    architecture = checkpoint.get("architecture")
    if architecture not in {"densenet121", "efficientnet_b0"}:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
    adjacency = checkpoint.get("label_adjacency")
    model = make_model(pretrained=False, architecture=architecture,
                       label_adjacency=adjacency)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    all_scores = []
    with torch.inference_mode():
        for images, _ in loader:
            logits = model(images.to(device, non_blocking=device.type == "cuda"))
            all_scores.append(torch.sigmoid(logits).cpu().numpy())
    scores = np.concatenate(all_scores, axis=0)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "architecture": architecture,
        "training_images": checkpoint.get("training_images"),
        "image_size": int(checkpoint.get("image_size", 224)),
    }
    del model, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--primary-checkpoint", type=Path, required=True)
    parser.add_argument("--secondary-checkpoint", type=Path, required=True)
    parser.add_argument("--primary-train-manifest", type=Path, required=True)
    parser.add_argument("--secondary-train-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-name", default="primary_8k")
    parser.add_argument("--secondary-name", default="secondary_10k")
    parser.add_argument("--ensemble-name", default="equal_average_8k_10k")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap_samples < 100:
        parser.error("batch size must be positive and use at least 100 bootstrap samples")
    for path in (args.holdout_manifest, args.image_root, args.primary_checkpoint,
                 args.secondary_checkpoint, args.primary_train_manifest,
                 args.secondary_train_manifest):
        if not path.exists():
            raise FileNotFoundError(path)
    # Validate that the report location is writable before spending time on inference.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    holdout = pd.read_csv(args.holdout_manifest)
    if len(holdout) != 2250 or holdout["Patient ID"].nunique() != 2250:
        raise ValueError("Expected the complete 2,250-image / 2,250-patient internal holdout")
    holdout_ids = set(holdout["Patient ID"].astype(str))
    overlap = {}
    for name, path in (("primary", args.primary_train_manifest),
                       ("secondary", args.secondary_train_manifest)):
        train_rows = pd.read_csv(path)
        if "Patient ID" not in train_rows or "split" not in train_rows:
            raise ValueError(f"Invalid training manifest: {path}")
        ids = set(train_rows.loc[train_rows["split"] == "train", "Patient ID"].astype(str))
        overlap[name] = len(holdout_ids & ids)
        if overlap[name]:
            raise ValueError(f"Patient leakage: {overlap[name]} holdout patients overlap {name} train split")

    dataset = NIHChestXrayDataset(args.holdout_manifest, args.image_root, "test", None, args.seed)
    if dataset.missing_images or len(dataset) != 2250:
        raise FileNotFoundError(f"Expected all 2,250 holdout images; missing={dataset.missing_images}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    y = dataset.targets.astype(np.int32)
    scores_8k, metadata_8k = score_checkpoint(args.primary_checkpoint, dataset, args.batch_size, device)
    scores_10k, metadata_10k = score_checkpoint(args.secondary_checkpoint, dataset, args.batch_size, device)
    if scores_8k.shape != y.shape or scores_10k.shape != y.shape:
        raise ValueError("Checkpoint output shape does not match holdout labels")
    score_sets = {
        args.primary_name: scores_8k,
        args.secondary_name: scores_10k,
        args.ensemble_name: (scores_8k + scores_10k) / 2.0,
    }
    summaries = {name: summarize(y, p) for name, p in score_sets.items()}

    unique_patients = dataset.rows["Patient ID"].astype(str).to_numpy()
    groups = {pid: np.flatnonzero(unique_patients == pid) for pid in np.unique(unique_patients)}
    rng = np.random.default_rng(args.seed)
    comparisons = {
        f"{args.secondary_name}_minus_{args.primary_name}":
            (args.secondary_name, args.primary_name),
        f"{args.ensemble_name}_minus_{args.primary_name}":
            (args.ensemble_name, args.primary_name),
    }
    paired = {}
    for comparison, (candidate, baseline) in comparisons.items():
        deltas = {key: [] for key in ("macro_auroc", "macro_average_precision", "micro_f1_at_0_5")}
        for _ in range(args.bootstrap_samples):
            selected = rng.choice(list(groups), size=len(groups), replace=True)
            indices = np.concatenate([groups[pid] for pid in selected])
            candidate_metrics = aggregate_bootstrap_metrics(y[indices], score_sets[candidate][indices])
            baseline_metrics = aggregate_bootstrap_metrics(y[indices], score_sets[baseline][indices])
            for key in deltas:
                a, b = candidate_metrics[key], baseline_metrics[key]
                if a is not None and b is not None and np.isfinite(a - b):
                    deltas[key].append(float(a - b))
        paired[comparison] = {}
        for key, values in deltas.items():
            a, b = summaries[candidate][key], summaries[baseline][key]
            paired[comparison][key] = {
                "observed_delta": a - b if a is not None and b is not None else None,
                "paired_patient_bootstrap_95_ci": (
                    [float(v) for v in np.quantile(values, [0.025, 0.975])] if values else None
                ),
                "valid_resamples": len(values),
            }

    rows = dataset.rows.reset_index(drop=True)
    output = pd.DataFrame({"Image Index": rows["Image Index"],
                           "Patient ID": rows["Patient ID"].astype(str)})
    for i, label in enumerate(NIH_LABELS):
        output[f"target_{label}"] = y[:, i]
        for name, scores in score_sets.items():
            output[f"score_{name}_{label}"] = scores[:, i]
    output.to_csv(args.output_dir / "predictions.csv", index=False)
    per_label_rows = []
    for candidate_name, summary in summaries.items():
        for label, label_metrics in summary["labels"].items():
            per_label_rows.append({"candidate": candidate_name, "label": label, **label_metrics})
    pd.DataFrame(per_label_rows).to_csv(args.output_dir / "per_label_metrics.csv", index=False)
    result = {
        "dataset": "NIH ChestX-ray14-derived internal patient holdout",
        "scope": "exploratory only; this 2,250-patient holdout was previously evaluated once for an older checkpoint",
        "rows": len(dataset), "unique_patients": int(len(holdout_ids)),
        "patient_overlap_with_training_rows": overlap,
        "threshold": 0.5, "threshold_tuned": False,
        "official_NIH_test_used": False,
        "candidates": {args.primary_name: metadata_8k, args.secondary_name: metadata_10k},
        "results": summaries, "paired_comparisons": paired,
        "positive_support_by_label": {label: int(y[:, i].sum()) for i, label in enumerate(NIH_LABELS)},
        "holdout_manifest_sha256": file_sha256(args.holdout_manifest),
        "bootstrap_samples": args.bootstrap_samples, "bootstrap_seed": args.seed,
        "device": str(device),
        "caveat": "Patient-disjoint from candidate training, but already viewed for an older model; exploratory comparison only, not a pristine final test or clinical validation.",
        "clinical_use": "not suitable for diagnosis or care decisions",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"device": device.type, "rows": len(dataset), "overlap": overlap,
                      "results": {name: {key: value for key, value in summary.items()
                                         if key != "labels"}
                                  for name, summary in summaries.items()},
                      "paired_comparisons": paired}, indent=2))


if __name__ == "__main__":
    main()
