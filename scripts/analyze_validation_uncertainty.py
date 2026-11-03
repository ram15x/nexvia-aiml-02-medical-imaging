"""Patient-clustered uncertainty analysis for the current NIH validation scores.

This script loads validation images only. It does not train, tune thresholds,
read test images, or select a checkpoint. Its bootstrap intervals describe
sampling variability on the already-used validation set; they are not an
independent generalization estimate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from api.modeling import build_classifier, image_to_tensor


class ValidationImages(Dataset):
    def __init__(self, manifest: Path, metadata: Path, image_root: Path, labels: list[str]):
        with manifest.open(newline="", encoding="utf-8-sig") as stream:
            rows = [row for row in csv.DictReader(stream)
                    if row.get("split") == "validation"]
        if not rows:
            raise ValueError("Manifest has no validation rows")
        self.rows = rows
        self.root = image_root.resolve()
        self.labels = labels
        self.targets = np.asarray([
            [float(label in set(row["Finding Labels"].split("|")) - {"No Finding"})
             for label in labels]
            for row in rows
        ], dtype=np.float32)
        self.patient_codes, self.patient_ids = self._patient_codes()
        wanted = {row["Image Index"] for row in rows}
        metadata_by_image = {}
        with metadata.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                name = row.get("Image Index")
                if name not in wanted:
                    continue
                try:
                    age = int(float(row.get("Patient Age", "")))
                except (TypeError, ValueError):
                    age = -1
                age_group = (
                    "0-39" if 0 <= age < 40 else
                    "40-59" if 40 <= age < 60 else
                    "60-100" if 60 <= age <= 100 else
                    "Unknown or out of range"
                )
                sex = row.get("Patient Sex", "").strip().upper()
                sex_group = sex if sex in {"F", "M"} else "Unknown"
                view = row.get("View Position", "").strip().upper()
                view_group = view if view in {"AP", "PA"} else "Other"
                metadata_by_image[name] = {
                    "age_group": age_group,
                    "sex": sex_group,
                    "view": view_group,
                }
        missing = wanted - metadata_by_image.keys()
        if missing:
            raise ValueError(f"Metadata missing for {len(missing)} validation images")
        self.groups = {
            field: np.asarray([metadata_by_image[row["Image Index"]][field] for row in rows])
            for field in ("age_group", "sex", "view")
        }

    def _patient_codes(self) -> tuple[np.ndarray, list[str]]:
        patient_ids = [row["Patient ID"] for row in self.rows]
        unique_ids = sorted(set(patient_ids))
        mapping = {patient: index for index, patient in enumerate(unique_ids)}
        return np.asarray([mapping[patient] for patient in patient_ids]), unique_ids

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        name = self.rows[index]["Image Index"]
        path = (self.root / name).resolve()
        if path.parent != self.root or not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            return image_to_tensor(image.convert("RGB"))


def rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def auc_ap(target: np.ndarray, score: np.ndarray) -> tuple[float | None, float | None]:
    positive = target.astype(bool)
    positives = int(positive.sum())
    negatives = len(target) - positives
    auc = None
    if positives and negatives:
        ranks = rankdata_average(score)
        auc = float((ranks[positive].sum() - positives * (positives + 1) / 2)
                    / (positives * negatives))
    ap = None
    if positives:
        order = np.argsort(-score, kind="mergesort")
        sorted_score = score[order]
        sorted_positive = positive[order].astype(np.int64)
        cumulative_tp = np.cumsum(sorted_positive)
        group_ends = np.r_[sorted_score[:-1] != sorted_score[1:], True]
        tp_at_end = cumulative_tp[group_ends]
        total_at_end = np.flatnonzero(group_ends) + 1
        new_tp = np.diff(np.r_[0, tp_at_end])
        ap = float(np.sum(new_tp / positives * (tp_at_end / total_at_end)))
    return auc, ap


def summarize(y: np.ndarray, p: np.ndarray) -> dict:
    aucs, aps = [], []
    prevalence = []
    for column in range(y.shape[1]):
        auc, ap = auc_ap(y[:, column], p[:, column])
        if auc is not None:
            aucs.append(auc)
        if ap is not None:
            aps.append(ap)
        prevalence.append(float(y[:, column].mean()))
    predicted = p >= 0.5
    tp = int(np.logical_and(predicted, y > 0.5).sum())
    fp = int(np.logical_and(predicted, y <= 0.5).sum())
    fn = int(np.logical_and(~predicted, y > 0.5).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {
        "images": int(len(y)),
        "patients": None,
        "macro_auroc": float(np.mean(aucs)) if aucs else None,
        "macro_auroc_defined_labels": len(aucs),
        "macro_auprc_average_precision": float(np.mean(aps)) if aps else None,
        "macro_auprc_defined_labels": len(aps),
        "macro_positive_prevalence_baseline": float(np.mean(prevalence)),
        "micro_precision_at_0_5": tp / max(tp + fp, 1),
        "micro_recall_at_0_5": tp / max(tp + fn, 1),
        "micro_f1_at_0_5": f1,
        "micro_brier": float(np.mean((p - y) ** 2)),
        "per_label_positive_counts": y.sum(axis=0).astype(int).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.bootstrap_replicates < 100 or args.batch_size < 1:
        parser.error("Use at least 100 bootstrap replicates and a positive batch size")

    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    labels = checkpoint["labels"]
    data = ValidationImages(args.manifest, args.metadata, args.image_root, labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    model = build_classifier(checkpoint["architecture"], len(labels), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    chunks = []
    with torch.inference_mode():
        for batch in loader:
            chunks.append(torch.sigmoid(model(batch.to(device))).cpu().numpy())
    scores = np.concatenate(chunks)
    targets = data.targets

    point = summarize(targets, scores)
    point["patients"] = len(data.patient_ids)
    point["labels"] = labels
    point["threshold_policy"] = "fixed 0.5; no threshold tuning"

    subgroup_metrics = {}
    for field, values in data.groups.items():
        subgroup_metrics[field] = {}
        for category in sorted(set(values)):
            mask = values == category
            if int(mask.sum()) < 25:
                continue
            summary = summarize(targets[mask], scores[mask])
            summary["patients"] = len(set(data.patient_codes[mask].tolist()))
            subgroup_metrics[field][category] = summary

    # Resample patients as clusters, preserving all images from each sampled patient.
    groups = [np.flatnonzero(data.patient_codes == patient)
              for patient in range(len(data.patient_ids))]
    rng = np.random.default_rng(args.seed)
    boot = {key: [] for key in (
        "macro_auroc", "macro_auprc_average_precision", "micro_f1_at_0_5", "micro_brier"
    )}
    for _ in range(args.bootstrap_replicates):
        sampled = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in sampled])
        summary = summarize(targets[indices], scores[indices])
        for key in boot:
            value = summary[key]
            if value is not None and np.isfinite(value):
                boot[key].append(value)
    intervals = {
        key: {
            "lower_2_5_percentile": float(np.percentile(values, 2.5)),
            "upper_97_5_percentile": float(np.percentile(values, 97.5)),
            "valid_replicates": len(values),
            "requested_replicates": args.bootstrap_replicates,
        }
        for key, values in boot.items() if values
    }
    output = {
        "analysis": "patient-cluster bootstrap on the already-used NIH validation split",
        "split": "validation only; test images not loaded",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "manifest": str(args.manifest),
        "device": str(device),
        "point_estimates": point,
        "validation_subgroups": subgroup_metrics,
        "subgroup_interpretation_limit": (
            "Descriptive only; group sizes and label prevalence differ, the same validation "
            "split selected the checkpoint, and observed gaps do not establish bias or causation."
        ),
        "patient_cluster_bootstrap_95_percentile_intervals": intervals,
        "interpretation_limit": (
            "Descriptive uncertainty on this validation sample only. The checkpoint was selected "
            "using this validation split, so intervals do not correct model-selection bias and "
            "are not an independent generalization estimate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
