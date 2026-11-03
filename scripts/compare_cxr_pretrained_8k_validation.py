"""Re-score the matched 8k CXR-pretrained candidate and DenseNet references.

Uses one previously used 1,000-patient development-validation panel. It never
reads test rows and writes aggregate metrics only (no patient-level predictions).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_audit import NIH_LABELS  # noqa: E402
from scripts.train_cxr_domain_pretrained import NIHXRVSubset, make_model as make_xrv_model  # noqa: E402
from train_classifier import NIHChestXrayDataset, evaluate, make_model as make_imagenet_model  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    aps, aucs = [], []
    for index in range(y.shape[1]):
        column = y[:, index]
        if column.sum() > 0:
            aps.append(float(average_precision_score(column, scores[:, index])))
        if np.unique(column).size == 2:
            aucs.append(float(roc_auc_score(column, scores[:, index])))
    binary = scores >= 0.5
    return {
        "images": int(len(y)),
        "macro_auprc": float(np.mean(aps)) if aps else None,
        "macro_auprc_labels": len(aps),
        "macro_auroc": float(np.mean(aucs)) if aucs else None,
        "macro_auroc_labels": len(aucs),
        "micro_precision_at_0_5": float(
            np.logical_and(y.astype(bool), binary).sum() / max(int(binary.sum()), 1)
        ),
        "micro_recall_at_0_5": float(
            np.logical_and(y.astype(bool), binary).sum() / max(int(y.sum()), 1)
        ),
        "micro_f1_at_0_5": float(f1_score(y, binary, average="micro", zero_division=0)),
        "positive_counts_by_label": {
            label: int(y[:, index].sum()) for index, label in enumerate(NIH_LABELS)
        },
    }


def bootstrap_delta(y: np.ndarray, reference: np.ndarray, candidate: np.ndarray,
                   replicates: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    ap_delta, auc_delta, f1_delta = [], [], []
    for _ in range(replicates):
        indices = rng.integers(0, len(y), size=len(y))
        truth = y[indices]
        paired = []
        for scores in (reference[indices], candidate[indices]):
            aps, aucs = [], []
            for column in range(truth.shape[1]):
                labels = truth[:, column]
                if labels.sum() > 0 and np.unique(labels).size == 2:
                    aps.append(float(average_precision_score(labels, scores[:, column])))
                if np.unique(labels).size == 2:
                    aucs.append(float(roc_auc_score(labels, scores[:, column])))
            predicted = scores >= 0.5
            paired.append((
                float(np.mean(aps)) if aps else np.nan,
                float(np.mean(aucs)) if aucs else np.nan,
                float(f1_score(truth, predicted, average="micro", zero_division=0)),
            ))
        if np.isfinite(paired[0][0]) and np.isfinite(paired[1][0]):
            ap_delta.append(paired[1][0] - paired[0][0])
        if np.isfinite(paired[0][1]) and np.isfinite(paired[1][1]):
            auc_delta.append(paired[1][1] - paired[0][1])
        f1_delta.append(paired[1][2] - paired[0][2])

    def interval(values: list[float]) -> dict:
        values = np.asarray(values, dtype=np.float64)
        return {
            "valid_replicates": int(len(values)),
            "median": float(np.median(values)),
            "percentile_95_ci": [float(np.quantile(values, .025)),
                                  float(np.quantile(values, .975))],
        }

    return {
        "method": "paired resampling of unique validation patients; macro AP and AUROC per replicate use labels with both classes present",
        "requested_replicates": replicates,
        "seed": seed,
        "macro_ap_candidate_minus_reference": interval(ap_delta),
        "macro_auroc_candidate_minus_reference": interval(auc_delta),
        "micro_f1_candidate_minus_reference": interval(f1_delta),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--active-checkpoint", type=Path, required=True)
    parser.add_argument("--ap-reference-checkpoint", type=Path, required=True)
    parser.add_argument("--xrv-candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--xrv-pretrained-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=9918)
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        parser.error("Use at least 100 bootstrap replicates")

    standard_data = NIHChestXrayDataset(args.manifest, args.image_root, "validation", None,
                                       args.seed, image_size=224)
    xrv_data = NIHXRVSubset(args.manifest, args.image_root, "validation")
    if standard_data.missing_images or len(standard_data.rows) != 1000:
        raise RuntimeError("Expected exactly 1,000 available validation images")
    if standard_data.rows["Patient ID"].nunique() != 1000:
        raise RuntimeError("Validation rows are not 1,000 unique patients")
    if not standard_data.rows["Image Index"].astype(str).tolist() == \
            xrv_data.rows["Image Index"].astype(str).tolist():
        raise RuntimeError("Image order differs between the two preprocessing paths")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    standard_loader = DataLoader(standard_data, batch_size=8, shuffle=False, num_workers=0,
                                 pin_memory=device.type == "cuda")
    xrv_loader = DataLoader(xrv_data, batch_size=4, shuffle=False, num_workers=0,
                             pin_memory=device.type == "cuda")

    active = torch.load(args.active_checkpoint, map_location="cpu", weights_only=False)
    active_model = make_imagenet_model(False, architecture=active["architecture"],
                                       label_adjacency=active.get("label_adjacency")).to(device)
    active_model.load_state_dict(active["state_dict"])
    _, y, active_scores = evaluate(active_model, standard_loader, device)
    active_metrics = metrics(y, active_scores)
    del active_model

    ap_reference = torch.load(args.ap_reference_checkpoint, map_location="cpu", weights_only=False)
    ap_model = make_imagenet_model(False, architecture=ap_reference["architecture"],
                                   label_adjacency=ap_reference.get("label_adjacency")).to(device)
    ap_model.load_state_dict(ap_reference["state_dict"])
    _, ap_y, ap_scores = evaluate(ap_model, standard_loader, device)
    ap_metrics = metrics(ap_y, ap_scores)
    del ap_model

    candidate = torch.load(args.xrv_candidate_checkpoint, map_location="cpu", weights_only=False)
    xrv_model = make_xrv_model(args.xrv_pretrained_weights).to(device)
    xrv_model.load_state_dict(candidate["state_dict"])
    _, xrv_y, xrv_scores = evaluate(xrv_model, xrv_loader, device)
    xrv_metrics = metrics(xrv_y, xrv_scores)
    if not (np.array_equal(y, ap_y) and np.array_equal(y, xrv_y)):
        raise RuntimeError("Targets or patient ordering differ between model evaluations")

    reference_metrics = ap_metrics
    result = {
        "analysis": "same-split model comparison; previously used development validation only",
        "manifest_sha256": sha256(args.manifest),
        "images": 1000,
        "unique_patients": int(standard_data.rows["Patient ID"].nunique()),
        "missing_images": 0,
        "official_test_used": False,
        "candidate_best_epoch": candidate.get("best_epoch"),
        "candidate_selection_metric": candidate.get("checkpoint_selection_metric"),
        "candidate_seed": candidate.get("seed", 20261031),
        "candidate_training_images": candidate.get("training_images", 8000),
        "candidate_checkpoint_sha256": sha256(args.xrv_candidate_checkpoint),
        "active_checkpoint_sha256": sha256(args.active_checkpoint),
        "ap_reference_checkpoint_sha256": sha256(args.ap_reference_checkpoint),
        "metrics": {
            "active_imagenet_densenet": active_metrics,
            "macro_ap_selected_imagenet_densenet": ap_metrics,
            "chexpert_pretrained_xrv_densenet": xrv_metrics,
        },
        "candidate_minus_macro_ap_selected_imagenet": {
            key: (xrv_metrics[key] - reference_metrics[key]
                  if xrv_metrics[key] is not None and reference_metrics[key] is not None else None)
            for key in ("macro_auprc", "macro_auroc", "micro_f1_at_0_5",
                        "micro_precision_at_0_5", "micro_recall_at_0_5")
        },
        "paired_patient_bootstrap_candidate_vs_macro_ap_selected_imagenet": bootstrap_delta(
            y, ap_scores, xrv_scores, args.bootstrap_replicates, args.seed
        ),
        "caveat": "The validation panel has prior development use and selected checkpoints. This comparison is exploratory and cannot establish independent generalization or clinical use.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "saved": str(args.output),
        "candidate_best_epoch": result["candidate_best_epoch"],
        "metrics": {name: {key: value for key, value in item.items()
                            if key in {"macro_auprc", "macro_auroc", "micro_f1_at_0_5"}}
                    for name, item in result["metrics"].items()},
        "candidate_minus_macro_ap_selected_imagenet": result[
            "candidate_minus_macro_ap_selected_imagenet"],
        "bootstrap": result["paired_patient_bootstrap_candidate_vs_macro_ap_selected_imagenet"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
