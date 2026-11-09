"""Subgroup performance of the frozen classifier on the NIH sample.

Groups: patient sex and age band. Metrics are computed within each group from the
per-image predictions written by evaluate_bootstrap.py. A label is scored only when
the group has at least MIN_POSITIVES positives and MIN_NEGATIVES negatives for it,
so unstable estimates are reported as not evaluated rather than as numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
    "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
MIN_POSITIVES = 5
MIN_NEGATIVES = 5
THRESHOLD = 0.5
BOOTSTRAP_RESAMPLES = 500
SEED = 42


def age_band(age: float) -> str:
    if age < 40:
        return "<40"
    if age < 60:
        return "40-59"
    return "60+"


def _truth(frame: pd.DataFrame) -> np.ndarray:
    return np.stack([frame["Finding Labels"].str.contains(label, regex=False)
                     .astype(int).to_numpy() for label in LABELS], axis=1)


def _macro_auroc(truth: np.ndarray, p: np.ndarray) -> float | None:
    values = [roc_auc_score(truth[:, j], p[:, j]) for j in range(truth.shape[1])
              if truth[:, j].sum() >= MIN_POSITIVES and (len(truth) - truth[:, j].sum()) >= MIN_NEGATIVES]
    return float(np.mean(values)) if values else None


def patient_bootstrap_ci(frame: pd.DataFrame, scores: pd.DataFrame) -> tuple[float | None, float | None]:
    """Percentile 95% interval for macro AUROC, resampling patients with replacement."""
    truth = _truth(frame)
    p = scores.to_numpy()
    patients = frame["Patient ID"].to_numpy()
    groups = {pid: np.flatnonzero(patients == pid) for pid in np.unique(patients)}
    keys = list(groups)
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        chosen = rng.choice(len(keys), size=len(keys), replace=True)
        idx = np.concatenate([groups[keys[i]] for i in chosen])
        value = _macro_auroc(truth[idx], p[idx])
        if value is not None:
            draws.append(value)
    if not draws:
        return None, None
    low, high = np.percentile(draws, [2.5, 97.5])
    return round(float(low), 4), round(float(high), 4)


def group_metrics(frame: pd.DataFrame, scores: pd.DataFrame, group_type: str, group: str) -> dict:
    truth = _truth(frame)
    p = scores.to_numpy()
    aurocs, auprcs, evaluated = [], [], 0
    tp = fp = fn = 0
    for j in range(len(LABELS)):
        y = truth[:, j]
        pos, neg = int(y.sum()), int(len(y) - y.sum())
        pred = p[:, j] >= THRESHOLD
        tp += int((pred & (y == 1)).sum())
        fp += int((pred & (y == 0)).sum())
        fn += int((~pred & (y == 1)).sum())
        if pos >= MIN_POSITIVES and neg >= MIN_NEGATIVES:
            evaluated += 1
            aurocs.append(roc_auc_score(y, p[:, j]))
            auprcs.append(average_precision_score(y, p[:, j]))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    return {
        "group_type": group_type,
        "group": group,
        "images": int(len(frame)),
        "patients": int(frame["Patient ID"].nunique()),
        "labels_evaluated": evaluated,
        "macro_auroc": round(float(np.mean(aurocs)), 4) if aurocs else None,
        "macro_auprc": round(float(np.mean(auprcs)), 4) if auprcs else None,
        "micro_f1_at_0_5": round(float(f1), 4) if f1 == f1 else None,
        "macro_auroc_95ci_patient_bootstrap": patient_bootstrap_ci(frame, scores),
    }


def run(manifest: Path, predictions: Path, output: Path) -> pd.DataFrame:
    meta = pd.read_csv(manifest)
    preds = pd.read_csv(predictions)
    if not (meta["Image Index"].to_numpy() == preds["image_index"].to_numpy()).all():
        raise ValueError("Manifest and predictions are not in the same image order")
    score_cols = [f"score_{label}" for label in LABELS]
    rows = [group_metrics(meta, preds[score_cols], "all", "all")]
    for sex, idx in meta.groupby("Patient Gender").groups.items():
        rows.append(group_metrics(meta.loc[idx], preds.loc[idx, score_cols], "sex", str(sex)))
    bands = meta["Patient Age"].map(age_band)
    for band in ["<40", "40-59", "60+"]:
        idx = bands[bands == band].index
        rows.append(group_metrics(meta.loc[idx], preds.loc[idx, score_cols], "age_band", band))
    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.manifest, args.predictions, args.output).to_string(index=False))


if __name__ == "__main__":
    main()
