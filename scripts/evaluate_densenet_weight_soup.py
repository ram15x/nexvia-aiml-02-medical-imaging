"""Build a same-seed DenseNet weight soup, recalibrate BatchNorm on train data, and score a fixed internal holdout."""

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
from torch.optim.swa_utils import update_bn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_audit import NIH_LABELS  # noqa: E402
from train_classifier import NIHChestXrayDataset, make_model, seed_everything  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(y: np.ndarray, p: np.ndarray) -> dict:
    aucs, aps, per_label = [], [], {}
    binary = p >= 0.5
    for col, label in enumerate(NIH_LABELS):
        target, probability = y[:, col], p[:, col]
        auc = float(roc_auc_score(target, probability)) if np.unique(target).size == 2 else None
        ap = float(average_precision_score(target, probability)) if target.sum() else None
        if auc is not None:
            aucs.append(auc)
        if ap is not None:
            aps.append(ap)
        per_label[label] = {
            "positive": int(target.sum()), "negative": int(len(target) - target.sum()),
            "auroc": auc, "average_precision": ap,
            "precision_at_0_5": float(precision_score(target, binary[:, col], zero_division=0)),
            "recall_at_0_5": float(recall_score(target, binary[:, col], zero_division=0)),
            "f1_at_0_5": float(f1_score(target, binary[:, col], zero_division=0)),
        }
    flat_y, flat_p = y.ravel(), p.ravel()
    return {
        "macro_auroc": float(np.mean(aucs)), "macro_average_precision": float(np.mean(aps)),
        "micro_auroc": float(roc_auc_score(flat_y, flat_p)),
        "micro_average_precision": float(average_precision_score(flat_y, flat_p)),
        "micro_precision_at_0_5": float(precision_score(flat_y, flat_p >= 0.5, zero_division=0)),
        "micro_recall_at_0_5": float(recall_score(flat_y, flat_p >= 0.5, zero_division=0)),
        "micro_f1_at_0_5": float(f1_score(flat_y, flat_p >= 0.5, zero_division=0)),
        "labels": per_label,
    }


def aggregate_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    aucs, aps = [], []
    for col in range(y.shape[1]):
        target, score = y[:, col], p[:, col]
        if np.unique(target).size == 2:
            aucs.append(float(roc_auc_score(target, score)))
        if target.sum():
            aps.append(float(average_precision_score(target, score)))
    return {
        "macro_auroc": float(np.mean(aucs)),
        "macro_average_precision": float(np.mean(aps)),
        "micro_f1_at_0_5": float(f1_score(y.ravel(), p.ravel() >= 0.5, zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-8k", type=Path, required=True)
    parser.add_argument("--checkpoint-10k", type=Path, required=True)
    parser.add_argument("--train-manifest-8k", type=Path, required=True)
    parser.add_argument("--train-manifest-10k", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--comparison-predictions", type=Path, required=True,
                        help="Saved frozen 8k/10k/probability-average predictions on the identical holdout")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap_samples < 100:
        parser.error("batch size must be positive and use at least 100 bootstrap samples")
    inputs = (args.checkpoint_8k, args.checkpoint_10k, args.train_manifest_8k,
              args.train_manifest_10k, args.holdout_manifest, args.image_root,
              args.comparison_predictions)
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
    # Fail fast if the output folder cannot be created.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ck8 = torch.load(args.checkpoint_8k, map_location="cpu", weights_only=True)
    ck10 = torch.load(args.checkpoint_10k, map_location="cpu", weights_only=True)
    for name, ck in (("8k", ck8), ("10k", ck10)):
        if ck.get("architecture") != "densenet121" or ck.get("labels") != list(NIH_LABELS):
            raise ValueError(f"Unexpected model architecture or label order in {name} checkpoint")
    for key in ("image_size", "seed"):
        if ck8.get(key) != ck10.get(key):
            raise ValueError(f"Candidate checkpoints differ in {key}; refusing a soup")
    if ck8.get("seed") is None:
        raise ValueError("Cannot verify shared initialization seed")
    state8, state10 = ck8["state_dict"], ck10["state_dict"]
    if state8.keys() != state10.keys():
        raise ValueError("Checkpoint state-dictionary keys differ")
    soup_state = {}
    for key in state8:
        a, b = state8[key], state10[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"Checkpoint tensor mismatch at {key}")
        soup_state[key] = ((a + b) / 2).to(a.dtype) if torch.is_floating_point(a) else a.clone()

    train8 = pd.read_csv(args.train_manifest_8k)
    train10 = pd.read_csv(args.train_manifest_10k)
    holdout_frame = pd.read_csv(args.holdout_manifest)
    ids8 = set(train8.loc[train8["split"] == "train", "Patient ID"].astype(str))
    ids10 = set(train10.loc[train10["split"] == "train", "Patient ID"].astype(str))
    holdout_ids = set(holdout_frame["Patient ID"].astype(str))
    if len(ids8) != 8000 or len(ids10) != 10000 or len(holdout_ids) != len(holdout_frame):
        raise ValueError("Training or holdout manifest cardinality is not as expected")
    if not ids8.issubset(ids10):
        raise ValueError("Expected the 8k training patients to be nested inside the 10k cohort")
    if holdout_ids & ids8 or holdout_ids & ids10:
        raise ValueError("Holdout patient overlaps candidate training patients")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(pretrained=False, architecture="densenet121",
                       label_adjacency=ck8.get("label_adjacency"))
    model.load_state_dict(soup_state, strict=True)
    model.to(device)

    # Re-estimate BatchNorm running statistics on the 8k common training cohort only.
    calibration_ds = NIHChestXrayDataset(args.train_manifest_8k, args.image_root,
                                         "train", None, args.seed, image_size=224)
    if calibration_ds.missing_images or len(calibration_ds) != 8000:
        raise FileNotFoundError(f"Expected all 8,000 calibration images; missing={calibration_ds.missing_images}")
    calibration_ds.augment = False
    calibration_loader = DataLoader(calibration_ds, batch_size=args.batch_size,
                                    shuffle=False, num_workers=0,
                                    pin_memory=device.type == "cuda")
    update_bn(calibration_loader, model, device=device)
    model.eval()

    holdout_ds = NIHChestXrayDataset(args.holdout_manifest, args.image_root,
                                     "test", None, args.seed, image_size=224)
    if holdout_ds.missing_images or len(holdout_ds) != len(holdout_frame):
        raise FileNotFoundError(f"Holdout has missing images: {holdout_ds.missing_images}")
    holdout_loader = DataLoader(holdout_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=0,
                                pin_memory=device.type == "cuda")
    all_scores = []
    with torch.inference_mode():
        for images, _ in holdout_loader:
            all_scores.append(torch.sigmoid(model(images.to(device, non_blocking=device.type == "cuda"))).cpu().numpy())
    y, soup_scores = holdout_ds.targets.astype(np.int32), np.concatenate(all_scores, axis=0)

    prior = pd.read_csv(args.comparison_predictions)
    aligned = prior.set_index("Image Index").reindex(holdout_ds.rows["Image Index"].astype(str))
    if aligned.isna().any().any() or len(prior) != len(holdout_ds):
        raise ValueError("Prior candidate predictions do not cover the exact holdout rows")
    for col, label in enumerate(NIH_LABELS):
        if not np.array_equal(aligned[f"target_{label}"].to_numpy(dtype=np.int32), y[:, col]):
            raise ValueError(f"Saved prediction labels disagree with holdout manifest for {label}")
    p8 = aligned[[f"score_primary_8k_{label}" for label in NIH_LABELS]].to_numpy(dtype=float)
    p10 = aligned[[f"score_secondary_10k_{label}" for label in NIH_LABELS]].to_numpy(dtype=float)
    p_avg = aligned[[f"score_equal_average_8k_10k_{label}" for label in NIH_LABELS]].to_numpy(dtype=float)
    score_sets = {"weight_soup_8k10k": soup_scores, "primary_8k": p8,
                  "secondary_10k": p10, "probability_average_8k10k": p_avg}
    summaries = {name: summarize(y, scores) for name, scores in score_sets.items()}

    patient_ids = holdout_ds.rows["Patient ID"].astype(str).to_numpy()
    unique = np.unique(patient_ids)
    groups = {pid: np.flatnonzero(patient_ids == pid) for pid in unique}
    rng = np.random.default_rng(args.seed)
    comparisons = {"weight_soup_minus_8k": ("weight_soup_8k10k", "primary_8k"),
                   "weight_soup_minus_probability_average": ("weight_soup_8k10k", "probability_average_8k10k")}
    paired = {}
    for name, (candidate, baseline) in comparisons.items():
        deltas = {key: [] for key in ("macro_auroc", "macro_average_precision", "micro_f1_at_0_5")}
        for _ in range(args.bootstrap_samples):
            selected = rng.choice(unique, size=len(unique), replace=True)
            rows = np.concatenate([groups[pid] for pid in selected])
            a = aggregate_metrics(y[rows], score_sets[candidate][rows])
            b = aggregate_metrics(y[rows], score_sets[baseline][rows])
            for key in deltas:
                deltas[key].append(float(a[key] - b[key]))
        paired[name] = {}
        for key, vals in deltas.items():
            a, b = summaries[candidate][key], summaries[baseline][key]
            paired[name][key] = {
                "observed_delta": float(a - b),
                "paired_patient_bootstrap_95_ci": [float(x) for x in np.quantile(vals, [0.025, 0.975])],
                "valid_resamples": len(vals),
            }

    soup_path = args.output_dir / "densenet121_weight_soup_8k10k_bn_recalibrated.pt"
    soup_checkpoint = {
        "architecture": "densenet121", "model_name": "DenseNet-121 weight soup (8k+10k)",
        "labels": list(NIH_LABELS), "state_dict": model.cpu().state_dict(),
        "image_size": int(ck8.get("image_size", 224)), "seed": ck8.get("seed"),
        "training_images": 10000, "weight_soup_alpha_10k": 0.5,
        "batchnorm_recalibration_images": 8000,
        "batchnorm_recalibration_manifest_sha256": sha256(args.train_manifest_8k),
        "source_checkpoint_sha256": {"8k": sha256(args.checkpoint_8k), "10k": sha256(args.checkpoint_10k)},
        "selection_evaluation": "exploratory internal holdout; not for default promotion",
    }
    torch.save(soup_checkpoint, soup_path)
    pred = pd.DataFrame({"Image Index": holdout_ds.rows["Image Index"].astype(str),
                         "Patient ID": patient_ids})
    for col, label in enumerate(NIH_LABELS):
        pred[f"target_{label}"] = y[:, col]
        pred[f"score_weight_soup_{label}"] = soup_scores[:, col]
    pred.to_csv(args.output_dir / "weight_soup_predictions.csv", index=False)
    per_label_rows = []
    for name, summary in summaries.items():
        for label, values in summary["labels"].items():
            per_label_rows.append({"candidate": name, "label": label, **values})
    pd.DataFrame(per_label_rows).to_csv(args.output_dir / "weight_soup_per_label_metrics.csv", index=False)
    result = {
        "dataset": "NIH ChestX-ray14-derived internal patient holdout",
        "scope": "exploratory only; this 2,250-patient holdout was previously scored for an older checkpoint",
        "rows": len(holdout_ds), "unique_patients": len(unique),
        "threshold": 0.5, "threshold_tuned": False, "official_NIH_test_used": False,
        "candidates": {"weight_soup": {"checkpoint": str(soup_path), "checkpoint_sha256": sha256(soup_path)},
                       "source_8k": {"checkpoint": str(args.checkpoint_8k), "checkpoint_sha256": sha256(args.checkpoint_8k)},
                       "source_10k": {"checkpoint": str(args.checkpoint_10k), "checkpoint_sha256": sha256(args.checkpoint_10k)}},
        "results": summaries, "paired_comparisons": paired,
        "batchnorm_calibration_images": len(calibration_ds),
        "batchnorm_calibration_train_patient_overlap_with_holdout": len(set(calibration_ds.rows["Patient ID"].astype(str)) & holdout_ids),
        "device": str(device), "bootstrap_samples": args.bootstrap_samples, "bootstrap_seed": args.seed,
        "caveat": "This fixed cohort has prior use in older-model evaluation; intervals do not correct for development selection. Do not promote based on this check.",
    }
    (args.output_dir / "weight_soup_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "rows": len(holdout_ds), "results": {
        name: {key: value for key, value in summary.items() if key != "labels"}
        for name, summary in summaries.items()}, "paired_comparisons": paired,
        "checkpoint_sha256": sha256(soup_path)}, indent=2))


if __name__ == "__main__":
    main()
