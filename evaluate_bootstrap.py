"""Per-label metrics with patient-level bootstrap intervals for a frozen NIH checkpoint.

Writes per-image predictions (for audit) and a JSON summary. Resampling is by patient,
so images from the same patient stay together.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from api.modeling import ARCHITECTURE_NAMES
from data_audit import NIH_LABELS
from train_classifier import NIHChestXrayDataset, evaluate, make_model, seed_everything


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if y.min() == y.max():
        return None
    return float(roc_auc_score(y, p))


def _ap(y: np.ndarray, p: np.ndarray) -> float | None:
    if y.sum() == 0:
        return None
    return float(average_precision_score(y, p))


def patient_bootstrap(y: np.ndarray, p: np.ndarray, patients: np.ndarray,
                      n_boot: int, seed: int) -> dict:
    """Percentile 95% intervals from patient-level resampling of per-label AUROC and AUPRC."""
    unique, inverse = np.unique(patients, return_inverse=True)
    groups = [np.where(inverse == k)[0] for k in range(len(unique))]
    rng = np.random.default_rng(seed)
    auc_draws = {label: [] for label in NIH_LABELS}
    ap_draws = {label: [] for label in NIH_LABELS}
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[k] for k in pick])
        for column, label in enumerate(NIH_LABELS):
            auc = _auc(y[idx, column], p[idx, column])
            ap = _ap(y[idx, column], p[idx, column])
            if auc is not None:
                auc_draws[label].append(auc)
            if ap is not None:
                ap_draws[label].append(ap)

    def interval(values: list[float]) -> list[float] | None:
        if len(values) < n_boot // 2:
            return None
        lo, hi = np.percentile(values, [2.5, 97.5])
        return [round(float(lo), 4), round(float(hi), 4)]

    return {label: {"auroc_95ci": interval(auc_draws[label]),
                    "auprc_95ci": interval(ap_draws[label])} for label in NIH_LABELS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-available-subset", action="store_true")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("architecture")
    if architecture not in ARCHITECTURE_NAMES:
        raise ValueError("Checkpoint architecture is unsupported")
    if checkpoint.get("labels") != list(NIH_LABELS):
        raise ValueError("Checkpoint label order does not match NIH_LABELS")

    test = NIHChestXrayDataset(args.manifest, args.image_root, split="test", limit=None,
                               seed=args.seed, allow_available_subset=args.allow_available_subset)
    loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = make_model(pretrained=False, architecture=architecture)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    metrics, y, p = evaluate(model, loader, device)

    patients = test.rows["Patient ID"].astype(str).to_numpy()
    images = test.rows["Image Index"].astype(str).to_numpy()
    assert len(patients) == len(y) == len(p)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preds = pd.DataFrame(p, columns=[f"score_{label}" for label in NIH_LABELS])
    preds.insert(0, "patient_id", patients)
    preds.insert(0, "image_index", images)
    preds.to_csv(args.output_dir / "per_image_predictions.csv", index=False)

    intervals = patient_bootstrap(y, p, patients, args.n_boot, args.seed)
    per_label = {}
    for column, label in enumerate(NIH_LABELS):
        base = metrics["per_label"][label] if "per_label" in metrics else {}
        per_label[label] = {**base, **intervals[label]}
    summary = {
        "images": int(len(y)),
        "patients": int(len(np.unique(patients))),
        "bootstrap_resamples": args.n_boot,
        "bootstrap_unit": "patient",
        "macro": {k: v for k, v in metrics.items() if k != "per_label"},
        "per_label": per_label,
    }
    (args.output_dir / "per_label_intervals.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("images", "patients", "macro")}, indent=2))


if __name__ == "__main__":
    main()
