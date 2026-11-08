"""Compare standard NIH metrics with a conservative No-Finding-negative sensitivity analysis."""

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
from train_classifier import NIHChestXrayDataset, make_model  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score(checkpoint_path: Path, dataset: NIHChestXrayDataset,
          batch_size: int, device: torch.device) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("labels") != list(NIH_LABELS):
        raise ValueError(f"Label order mismatch in {checkpoint_path}")
    architecture = checkpoint.get("architecture")
    model = make_model(pretrained=False, architecture=architecture,
                       label_adjacency=checkpoint.get("label_adjacency"))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    batches = []
    with torch.inference_mode():
        for images, _ in loader:
            logits = model(images.to(device, non_blocking=device.type == "cuda"))
            batches.append(torch.sigmoid(logits).cpu().numpy())
    del model, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(batches, axis=0)


def summarize(y: np.ndarray, p: np.ndarray, *, clean_negatives_only: bool) -> dict:
    no_finding = y.sum(axis=1) == 0
    macro_auc, macro_ap = [], []
    pooled_y, pooled_p = [], []
    per_label = {}
    for column, label in enumerate(NIH_LABELS):
        selected = ((no_finding | (y[:, column] == 1))
                    if clean_negatives_only else np.ones(len(y), dtype=bool))
        target, score_values = y[selected, column], p[selected, column]
        auc = (float(roc_auc_score(target, score_values))
               if np.unique(target).size == 2 else None)
        ap = (float(average_precision_score(target, score_values))
              if target.sum() else None)
        if auc is not None:
            macro_auc.append(auc)
        if ap is not None:
            macro_ap.append(ap)
        predicted = score_values >= 0.5
        per_label[label] = {
            "n_evaluated": int(len(target)),
            "positive": int(target.sum()),
            "negative": int(len(target) - target.sum()),
            "auroc": auc,
            "average_precision": ap,
            "precision_at_0_5": float(precision_score(target, predicted, zero_division=0)),
            "recall_at_0_5": float(recall_score(target, predicted, zero_division=0)),
            "f1_at_0_5": float(f1_score(target, predicted, zero_division=0)),
        }
        pooled_y.append(target)
        pooled_p.append(score_values)
    flat_y, flat_p = np.concatenate(pooled_y), np.concatenate(pooled_p)
    return {
        "images": int(len(y)),
        "evaluation_policy": (
            "per label: positive report-mined cases vs No Finding cases; other abnormal cases excluded"
            if clean_negatives_only else "all report-mined zero labels treated as negative"
        ),
        "macro_auroc": float(np.mean(macro_auc)) if macro_auc else None,
        "macro_auroc_label_count": len(macro_auc),
        "macro_average_precision": float(np.mean(macro_ap)) if macro_ap else None,
        "macro_average_precision_label_count": len(macro_ap),
        "micro_precision_at_0_5": float(precision_score(flat_y, flat_p >= 0.5, zero_division=0)),
        "micro_recall_at_0_5": float(recall_score(flat_y, flat_p >= 0.5, zero_division=0)),
        "micro_f1_at_0_5": float(f1_score(flat_y, flat_p >= 0.5, zero_division=0)),
        "per_label": per_label,
    }


def paired_bootstrap(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray,
                     *, clean_negatives_only: bool, iterations: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    deltas = {"macro_auroc": [], "macro_average_precision": [], "micro_f1_at_0_5": []}
    for _ in range(iterations):
        indices = rng.integers(0, len(y), size=len(y))
        control = summarize(y[indices], baseline[indices], clean_negatives_only=clean_negatives_only)
        trial = summarize(y[indices], candidate[indices], clean_negatives_only=clean_negatives_only)
        for metric in deltas:
            before, after = control[metric], trial[metric]
            if before is not None and after is not None:
                deltas[metric].append(after - before)
    result = {}
    for metric, values in deltas.items():
        result[metric] = {
            "mean_delta": float(np.mean(values)) if values else None,
            "ci_95": ([float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
                      if values else None),
            "valid_bootstraps": len(values),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation",
                        help="Use validation for development screening, or a patient-disjoint local test cohort for exploratory follow-up.")
    parser.add_argument("--cohort-description", default=None)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()
    for path in (args.manifest, args.image_root, args.baseline, args.candidate):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.batch_size < 1 or args.bootstrap < 100:
        parser.error("batch size must be positive and bootstrap must be at least 100")

    dataset = NIHChestXrayDataset(args.manifest, args.image_root, args.split, None,
                                  args.seed, image_size=224)
    if dataset.missing_images:
        raise FileNotFoundError(f"Missing validation images: {dataset.missing_images}")
    frame = dataset.rows
    y = dataset.targets
    train_frame = pd.read_csv(args.manifest)
    train_ids = set(train_frame.loc[train_frame["split"] == "train", "Patient ID"].astype(str))
    evaluation_ids = set(frame["Patient ID"].astype(str))
    overlap = train_ids & evaluation_ids
    if overlap:
        raise ValueError(f"Training/evaluation patient overlap: {len(overlap)}")
    no_finding = np.asarray(["No Finding" in str(value).split("|")
                             for value in frame["Finding Labels"]], dtype=bool)
    if not np.array_equal(no_finding, y.sum(axis=1) == 0):
        raise ValueError("Manifest No Finding labels disagree with the encoded targets")
    patient_ids = frame["Patient ID"].astype(str)
    if patient_ids.duplicated().any():
        raise ValueError("Validation manifest must contain unique patients for patient bootstrap")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_scores = score(args.baseline, dataset, args.batch_size, device)
    candidate_scores = score(args.candidate, dataset, args.batch_size, device)
    output = {
        "cohort": args.cohort_description or (
            "development validation; already used for checkpoint selection"
            if args.split == "validation" else "local NIH patient-disjoint test cohort; exploratory only"
        ),
        "patients": int(len(frame)),
        "no_finding_negatives": int(no_finding.sum()),
        "positive_rows": int((~no_finding).sum()),
        "official_test_used": False,
        "bootstrap": {"unit": "patient", "iterations": args.bootstrap, "seed": args.seed},
        "baseline": {
            "path": str(args.baseline), "sha256": sha256(args.baseline),
            "standard_weak_label_metrics": summarize(y, baseline_scores, clean_negatives_only=False),
            "clean_negative_sensitivity_metrics": summarize(y, baseline_scores, clean_negatives_only=True),
        },
        "masked_loss_candidate": {
            "path": str(args.candidate), "sha256": sha256(args.candidate),
            "standard_weak_label_metrics": summarize(y, candidate_scores, clean_negatives_only=False),
            "clean_negative_sensitivity_metrics": summarize(y, candidate_scores, clean_negatives_only=True),
        },
        "paired_bootstrap_differences_candidate_minus_baseline": {
            "standard_weak_label_metrics": paired_bootstrap(
                y, baseline_scores, candidate_scores, clean_negatives_only=False,
                iterations=args.bootstrap, seed=args.seed,
            ),
            "clean_negative_sensitivity_metrics": paired_bootstrap(
                y, baseline_scores, candidate_scores, clean_negatives_only=True,
                iterations=args.bootstrap, seed=args.seed + 1,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "device": str(device),
        "baseline_standard": output["baseline"]["standard_weak_label_metrics"],
        "candidate_standard": output["masked_loss_candidate"]["standard_weak_label_metrics"],
        "baseline_clean_negative": output["baseline"]["clean_negative_sensitivity_metrics"],
        "candidate_clean_negative": output["masked_loss_candidate"]["clean_negative_sensitivity_metrics"],
        "paired_bootstrap": output["paired_bootstrap_differences_candidate_minus_baseline"],
        "output": str(args.output),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
