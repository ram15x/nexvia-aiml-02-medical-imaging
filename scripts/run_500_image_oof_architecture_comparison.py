"""Patient-level 5-fold OOF architecture screen on a fixed 500-image cohort.

Each architecture trains on 400 patients and predicts the other 100. Every
patient receives exactly one out-of-fold prediction. A fixed epoch count avoids
choosing an epoch on the fold used for scoring. This remains a small-data
screen, not an independent clinical or official-test evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_audit import NIH_LABELS  # noqa: E402
from api.modeling import ARCHITECTURE_NAMES  # noqa: E402
from train_classifier import (  # noqa: E402
    NIHChestXrayDataset,
    evaluate,
    make_model,
    seed_everything,
)

ARCHITECTURES = (
    "densenet121",
    "densenet121_evidence_concentration_gate",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(targets: np.ndarray, scores: np.ndarray) -> dict:
    per_label = {}
    aucs, aps = [], []
    binary = scores >= 0.5
    for col, label in enumerate(NIH_LABELS):
        y, p = targets[:, col], scores[:, col]
        positives = int(y.sum())
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
        ap = float(average_precision_score(y, p)) if positives else None
        tp = int(np.logical_and(y == 1, binary[:, col]).sum())
        fp = int(np.logical_and(y == 0, binary[:, col]).sum())
        fn = positives - tp
        per_label[label] = {
            "positive_count": positives,
            "auroc": auc,
            "auprc": ap,
            "precision_at_0_5": tp / (tp + fp) if tp + fp else 0.0,
            "recall_at_0_5": tp / positives if positives else None,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        }
        if auc is not None:
            aucs.append(auc)
        if ap is not None:
            aps.append(ap)
    return {
        "images": len(targets),
        "macro_auroc_defined_labels": float(np.mean(aucs)) if aucs else None,
        "macro_auprc_positive_labels": float(np.mean(aps)) if aps else None,
        "micro_precision_at_0_5": float(precision_score(
            targets, binary, average="micro", zero_division=0)),
        "micro_recall_at_0_5": float(recall_score(
            targets, binary, average="micro", zero_division=0)),
        "micro_f1_at_0_5": float(f1_score(
            targets, binary, average="micro", zero_division=0)),
        "per_label": per_label,
    }


def bootstrap_intervals(targets: np.ndarray, scores: np.ndarray,
                        *, seed: int, replicates: int = 1000) -> dict:
    rng = np.random.default_rng(seed)
    values = {"macro_auroc_defined_labels": [],
              "macro_auprc_positive_labels": [], "micro_f1_at_0_5": []}
    n = len(targets)
    for _ in range(replicates):
        idx = rng.integers(0, n, size=n)
        y, p = targets[idx], scores[idx]
        aucs, aps = [], []
        for col in range(y.shape[1]):
            positives = int(y[:, col].sum())
            if 0 < positives < len(y):
                aucs.append(float(roc_auc_score(y[:, col], p[:, col])))
            if positives:
                aps.append(float(average_precision_score(y[:, col], p[:, col])))
        if aucs:
            values["macro_auroc_defined_labels"].append(float(np.mean(aucs)))
        if aps:
            values["macro_auprc_positive_labels"].append(float(np.mean(aps)))
        values["micro_f1_at_0_5"].append(float(f1_score(
            y, p >= 0.5, average="micro", zero_division=0)))
    return {
        "method": "patient-row bootstrap; percentile 95% intervals",
        "replicates": replicates,
        "seed": seed,
        "intervals": {
            key: {"lower_2_5_percent": float(np.percentile(v, 2.5)),
                  "upper_97_5_percent": float(np.percentile(v, 97.5))}
            for key, v in values.items() if v
        },
        "caveat": "Intervals quantify sampling variation only; architecture screening on the same OOF outputs still creates model-selection bias.",
    }


def paired_delta_intervals(targets: np.ndarray, baseline: np.ndarray,
                           candidate: np.ndarray, *, seed: int,
                           replicates: int = 1000) -> dict:
    """Paired patient-row bootstrap for candidate-minus-baseline metric deltas."""
    rng = np.random.default_rng(seed)
    names = ("macro_auroc_defined_labels", "macro_auprc_positive_labels",
             "micro_f1_at_0_5")
    samples = {name: [] for name in names}
    for _ in range(replicates):
        idx = rng.integers(0, len(targets), size=len(targets))
        baseline_metrics = summarize(targets[idx], baseline[idx])
        candidate_metrics = summarize(targets[idx], candidate[idx])
        for name in names:
            left, right = baseline_metrics[name], candidate_metrics[name]
            if left is not None and right is not None:
                samples[name].append(float(right - left))
    return {
        "direction": "candidate minus baseline",
        "method": "paired patient-row bootstrap; percentile 95% intervals",
        "replicates": replicates,
        "seed": seed,
        "deltas": {
            name: {
                "point_difference": float(
                    summarize(targets, candidate)[name]
                    - summarize(targets, baseline)[name]
                ),
                "lower_2_5_percent": float(np.percentile(values, 2.5)),
                "upper_97_5_percent": float(np.percentile(values, 97.5)),
            }
            for name, values in samples.items() if values
        },
        "caveat": "The intervals quantify sampling variation only; comparing models on these OOF outputs still creates model-selection bias.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--architectures", nargs="+", choices=tuple(ARCHITECTURE_NAMES),
                        default=None,
                        help="Optional matched architecture subset; default reproduces the original DenseNet/gate screen.")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("Epoch count must be at least one")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    architectures = tuple(args.architectures or ARCHITECTURES)

    frame = pd.read_csv(args.pilot_manifest)
    required = {"Image Index", "Finding Labels", "Patient ID", "split"}
    if not required <= set(frame.columns):
        raise ValueError("Pilot manifest does not have the expected columns")
    if len(frame) != 500 or frame["Patient ID"].nunique() != 500:
        raise ValueError("The frozen cohort must contain exactly 500 unique patients")
    if set(frame["split"].unique()) - {"train", "validation"}:
        raise ValueError("Official test rows are forbidden")
    frame = frame.sort_values("Image Index").reset_index(drop=True)
    frame.to_csv(args.output_dir / "cohort_manifest.csv", index=False)
    image_positions = {name: i for i, name in enumerate(frame["Image Index"].astype(str))}
    y_all = np.zeros((len(frame), len(NIH_LABELS)), dtype=np.float32)
    scores = {arch: np.full_like(y_all, np.nan) for arch in architectures}
    seen = {arch: np.zeros(len(frame), dtype=bool) for arch in architectures}
    fold_summaries = {arch: [] for arch in architectures}
    folds_dir = args.output_dir / "fold_manifests"
    folds_dir.mkdir()
    kfold = KFold(n_splits=5, shuffle=True, random_state=args.seed)

    for fold_index, (train_idx, val_idx) in enumerate(kfold.split(frame), start=1):
        fold_seed = args.seed + fold_index
        fold_frame = frame.copy()
        fold_frame["split"] = "train"
        fold_frame.loc[val_idx, "split"] = "validation"
        train_patients = set(fold_frame.iloc[train_idx]["Patient ID"])
        val_patients = set(fold_frame.iloc[val_idx]["Patient ID"])
        if len(train_idx) != 400 or len(val_idx) != 100 or train_patients & val_patients:
            raise ValueError(f"Fold {fold_index} violates the patient-level 400/100 split")
        fold_manifest = folds_dir / f"fold_{fold_index:02d}.csv"
        fold_frame.to_csv(fold_manifest, index=False)
        print(json.dumps({"event": "fold_start", "fold": fold_index,
                          "train": len(train_idx), "validation": len(val_idx),
                          "seed": fold_seed}), flush=True)

        for architecture in architectures:
            run_dir = args.output_dir / f"{architecture}_fold_{fold_index:02d}"
            command = [
                sys.executable, str(PROJECT_ROOT / "train_classifier.py"),
                "--manifest", str(fold_manifest),
                "--image-root", str(args.image_root),
                "--output", str(run_dir),
                "--architecture", architecture,
                "--epochs", str(args.epochs),
                "--image-size", "224",
                "--batch-size", "4",
                "--workers", "0",
                "--lr", "0.0001",
                "--seed", str(fold_seed),
                "--positive-weight-exponent", "0.5",
                "--checkpoint-every-steps", "1000",
            ]
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)

            last_path = run_dir / "last_checkpoint.pt"
            final_state = torch.load(last_path, map_location="cpu", weights_only=False)
            if int(final_state["epoch"]) != args.epochs + 1:
                raise RuntimeError(f"Fold did not complete fixed {args.epochs}-epoch run: {run_dir}")
            checkpoint_hash = sha256(last_path)
            model = make_model(pretrained=False, architecture=architecture)
            model.load_state_dict(final_state["model_state"], strict=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            dataset = NIHChestXrayDataset(
                fold_manifest, args.image_root, split="validation", limit=None,
                seed=fold_seed, image_size=224, allow_available_subset=False,
            )
            if dataset.missing_images or len(dataset) != 100:
                raise FileNotFoundError(f"Fold {fold_index} validation image missing or incomplete")
            loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0,
                                pin_memory=device.type == "cuda")
            metrics, y_fold, p_fold = evaluate(model, loader, device)
            rows = dataset.rows["Image Index"].astype(str).tolist()
            positions = np.asarray([image_positions[name] for name in rows], dtype=int)
            if seen[architecture][positions].any():
                raise ValueError("An image received more than one OOF prediction")
            y_all[positions] = y_fold
            scores[architecture][positions] = p_fold
            seen[architecture][positions] = True
            fold_summaries[architecture].append({
                "fold": fold_index,
                "seed": fold_seed,
                "fixed_prediction_epoch": args.epochs,
                "last_checkpoint_sha256": checkpoint_hash,
                "validation_metrics": metrics,
            })
            # Keep metrics/history and OOF scores; discard large model/optimizer states.
            for checkpoint_path in run_dir.glob("*.pt"):
                checkpoint_path.unlink()
            print(json.dumps({"event": "fold_complete", "fold": fold_index,
                              "architecture": architecture,
                              "fixed_epoch": args.epochs,
                              "fixed_epoch_auroc": metrics["macro_auroc_defined_labels"],
                              "fixed_epoch_ap": metrics["macro_auprc_defined_labels"],
                              "fixed_epoch_f1": metrics["micro_f1_at_0_5"]}), flush=True)

    if not all(seen[architecture].all() for architecture in architectures):
        raise RuntimeError("Some cases did not receive exactly one OOF prediction")
    if not np.array_equal(y_all, np.stack([
        np.asarray([float(label in str(raw).split("|")) for label in NIH_LABELS], dtype=np.float32)
        for raw in frame["Finding Labels"]
    ])):
        raise RuntimeError("OOF target order does not match the cohort manifest")

    prediction_frame = pd.DataFrame({"Image Index": frame["Image Index"].astype(str)})
    for col, label in enumerate(NIH_LABELS):
        prediction_frame[f"target_{label}"] = y_all[:, col].astype(int)
    for architecture in architectures:
        for col, label in enumerate(NIH_LABELS):
            prediction_frame[f"score_{architecture}_{label}"] = scores[architecture][:, col]
    prediction_frame.to_csv(args.output_dir / "oof_predictions.csv", index=False)

    result = {
        "purpose": "500-image patient-level 5-fold out-of-fold architecture screen",
        "cohort_images": len(frame),
        "unique_patients": int(frame["Patient ID"].nunique()),
        "folds": 5,
        "train_images_per_fold": 400,
        "validation_images_per_fold": 100,
        "epochs_per_fold": args.epochs,
        "seed_base": args.seed,
        "input_size": 224,
        "batch_size": 4,
        "test_rows_used": 0,
        "pilot_manifest_sha256": sha256(args.pilot_manifest),
        "cohort_manifest_sha256": sha256(args.output_dir / "cohort_manifest.csv"),
        "training_recipe": f"ImageNet initialization; AdamW lr=1e-4 wd=1e-4; square-root pos weights capped at 10; train-only horizontal flip; fixed epoch {args.epochs} predictions; no fold checkpoint selection.",
        "models": {},
        "limitations": [
            "The 500 patients were already used in the one-fold pilot, so this is exploratory rather than a pristine benchmark.",
            "The 500-image cohort contains no Hernia positives and sparse positives for several findings.",
            "Selecting an architecture after comparing these OOF metrics introduces model-selection bias.",
            "This does not establish clinical performance or an 85% target.",
        ],
    }
    for architecture in architectures:
        result["models"][architecture] = {
            "oof_metrics": summarize(y_all, scores[architecture]),
            "patient_bootstrap_95_percent_intervals": bootstrap_intervals(
                y_all, scores[architecture], seed=args.seed
            ),
            "per_fold": fold_summaries[architecture],
        }
    if set(("densenet121", "densenet121_evidence_concentration_gate")) <= set(architectures):
        result["paired_delta_intervals_gate_minus_densenet"] = paired_delta_intervals(
            y_all, scores["densenet121"],
            scores["densenet121_evidence_concentration_gate"], seed=args.seed
        )
    if "densenet121" in architectures:
        for candidate in architectures:
            if candidate in {"densenet121", "densenet121_evidence_concentration_gate"}:
                continue
            result[f"paired_delta_intervals_{candidate}_minus_densenet"] = paired_delta_intervals(
                y_all, scores["densenet121"], scores[candidate], seed=args.seed
            )
    out_json = args.output_dir / "oof_comparison.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"event": "comparison_complete",
                      "summary": {arch: result["models"][arch]["oof_metrics"]
                                  for arch in architectures}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
