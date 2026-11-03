"""Evaluate a frozen pilot checkpoint on validation cases excluded from selection.

This is a descriptive validation analysis, not a held-out test. It excludes the
pilot's 100 checkpoint-selection cases and never reads official test rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.modeling import ARCHITECTURE_NAMES
from data_audit import NIH_LABELS
from train_classifier import NIHChestXrayDataset, evaluate, make_model, seed_everything


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patient_bootstrap_intervals(
    targets: np.ndarray, scores: np.ndarray, *, replicates: int, seed: int
) -> dict[str, object]:
    """Percentile intervals from resampling unique patient rows with replacement."""
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "macro_auroc_defined_labels": [],
        "macro_auprc_positive_labels": [],
        "micro_f1_at_0_5": [],
    }
    n = len(targets)
    for _ in range(replicates):
        index = rng.integers(0, n, size=n)
        y = targets[index]
        p = scores[index]
        aucs, aps = [], []
        for column in range(y.shape[1]):
            positives = y[:, column].sum()
            if 0 < positives < len(y):
                aucs.append(float(roc_auc_score(y[:, column], p[:, column])))
            if positives > 0:
                aps.append(float(average_precision_score(y[:, column], p[:, column])))
        if aucs:
            values["macro_auroc_defined_labels"].append(float(np.mean(aucs)))
        if aps:
            values["macro_auprc_positive_labels"].append(float(np.mean(aps)))
        values["micro_f1_at_0_5"].append(
            float(f1_score(y, p >= 0.5, average="micro", zero_division=0))
        )
    return {
        "method": "patient-row bootstrap with replacement; percentile 95% interval",
        "replicates": replicates,
        "seed": seed,
        "warning": "Intervals describe sampling variation only; they do not correct validation reuse or checkpoint/model-selection bias.",
        "intervals": {
            name: {
                "lower_2_5_percent": float(np.percentile(samples, 2.5)),
                "upper_97_5_percent": float(np.percentile(samples, 97.5)),
            }
            for name, samples in values.items() if samples
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap_replicates < 100:
        parser.error("--batch-size must be positive and --bootstrap-replicates at least 100")

    full_columns, full_rows = read_rows(args.full_manifest)
    pilot_columns, pilot_rows = read_rows(args.pilot_manifest)
    if full_columns != pilot_columns:
        raise ValueError("Full and pilot manifest schemas differ")
    if any(row["split"] == "test" for row in full_rows):
        raise ValueError("Input manifest contains test rows; refusing evaluation")
    selected_ids = {row["Image Index"] for row in pilot_rows
                    if row["split"] == "validation"}
    selected_patients = {row["Patient ID"] for row in pilot_rows
                         if row["split"] == "validation"}
    if len(selected_ids) != 100 or len(selected_patients) != 100:
        raise ValueError("Expected the frozen 100-case pilot validation split")

    validation_rows = [row for row in full_rows if row["split"] == "validation"]
    eval_rows = [row for row in validation_rows
                 if row["Image Index"] not in selected_ids]
    if len(validation_rows) != 2250 or len(eval_rows) != 2150:
        raise ValueError("Expected 2,250 full validation rows minus 100 pilot rows")
    if selected_patients & {row["Patient ID"] for row in eval_rows}:
        raise ValueError("Pilot selection and evaluation remainder share patients")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    eval_manifest = args.output.with_name("validation_remainder_manifest.csv")
    with eval_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=full_columns)
        writer.writeheader()
        writer.writerows(eval_rows)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = checkpoint.get("architecture")
    if architecture not in ARCHITECTURE_NAMES:
        raise ValueError("Checkpoint architecture is unsupported")
    if architecture != "densenet121":
        raise ValueError("This frozen evaluation is defined for DenseNet-121 only")
    if checkpoint.get("labels") != list(NIH_LABELS):
        raise ValueError("Checkpoint label order does not match NIH_LABELS")

    dataset = NIHChestXrayDataset(
        eval_manifest, args.image_root, split="validation", limit=None,
        seed=args.seed, image_size=int(checkpoint.get("image_size", 224)),
        allow_available_subset=False,
    )
    if dataset.missing_images:
        raise FileNotFoundError(
            f"Validation remainder has {dataset.missing_images} missing images"
        )
    if dataset.rows["Patient ID"].nunique() != len(dataset):
        raise ValueError("Patient bootstrap requires one validation image per patient")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    model = make_model(pretrained=False, architecture=architecture)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    metrics, targets, scores = evaluate(model, loader, device)
    intervals = patient_bootstrap_intervals(
        targets, scores, replicates=args.bootstrap_replicates, seed=args.seed
    )

    result = {
        "dataset": "NIH ChestX-ray14",
        "split": "development validation remainder; not an independent test",
        "images": len(dataset),
        "patients": int(dataset.rows["Patient ID"].nunique()),
        "pilot_validation_cases_excluded": len(selected_ids),
        "official_test_rows_used": 0,
        "missing_images": dataset.missing_images,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch_selected_on_pilot_100": checkpoint.get("best_epoch"),
        "checkpoint_architecture": architecture,
        "full_manifest_sha256": sha256(args.full_manifest),
        "pilot_manifest_sha256": sha256(args.pilot_manifest),
        "evaluation_manifest_sha256": sha256(eval_manifest),
        "threshold_policy": "fixed 0.5; no threshold fitting",
        "metrics": metrics,
        "patient_bootstrap_95_percent_intervals": intervals,
        "caveat": (
            "The model checkpoint was selected using the excluded 100-case pilot validation set. "
            "This remainder gives a broader descriptive validation estimate, but the source "
            "validation cohort has been used in earlier project experiments and is not a blind test."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
