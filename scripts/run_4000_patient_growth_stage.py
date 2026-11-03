"""One incremental 2,500-vs-4,000-patient DenseNet learning-curve stage.

The candidate adds 1,500 training patients to the frozen 2,500-patient cohort.
Both are scored on the same 1,000-patient development validation panel. No
official-test rows or images are loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_audit import NIH_LABELS  # noqa: E402
from scripts.run_1000_image_learning_curve import (  # noqa: E402
    draw_ap_chart,
    label_support,
    metrics,
    paired_bootstrap,
    sha256,
)
from train_classifier import NIHChestXrayDataset, evaluate, make_model  # noqa: E402


def make_manifests(output: Path, project_root: Path, image_root: Path,
                   selection_seed: int) -> tuple[Path, Path, dict]:
    nih = project_root / "data" / "NexviaProjectData" / "nih"
    metadata = nih / "metadata"
    split_dir = metadata / "fresh_15k_patient_split_20261029"
    source_train_path = split_dir / "train_manifest.csv"
    source_val_path = split_dir / "validation_manifest.csv"
    pilot_path = metadata / "pilot_500_patient_split_20261030" / "pilot_manifest.csv"
    prior_stage = nih / "learning_curve_2500_train_1000_val_20261031"
    prior_800_path = prior_stage / "manifest_2500_train_1000_validation.csv"
    prior_1000_path = prior_stage / "manifest_2500_train_1000_validation.csv"

    source_train = pd.read_csv(source_train_path)
    source_val = pd.read_csv(source_val_path)
    pilot = pd.read_csv(pilot_path)
    prior_800 = pd.read_csv(prior_800_path)
    prior_validation = pd.read_csv(prior_1000_path)
    if list(source_train.columns) != list(source_val.columns):
        raise ValueError("Train and validation source manifest schemas differ")
    if not {"Image Index", "Finding Labels", "Patient ID", "split"} <= set(source_train.columns):
        raise ValueError("Source manifest schema is incomplete")
    if len(source_train) != 10500 or len(source_val) != 2250:
        raise ValueError("Unexpected 15k split sizes; stop before training")
    if source_train["Patient ID"].nunique() != len(source_train):
        raise ValueError("Source train manifest is not patient-disjoint")
    if source_val["Patient ID"].nunique() != len(source_val):
        raise ValueError("Source validation manifest is not patient-disjoint")

    pilot_train_ids = set(pilot.loc[pilot["split"] == "train", "Image Index"].astype(str))
    pilot_val_ids = set(pilot.loc[pilot["split"] == "validation", "Image Index"].astype(str))
    prior_train = prior_800.loc[prior_800["split"] == "train"].copy()
    prior_train["Image Index"] = prior_train["Image Index"].astype(str)
    prior_val_ids = set(prior_validation.loc[
        prior_validation["split"] == "validation", "Image Index"].astype(str))
    if len(prior_train) != 2500 or prior_train["Patient ID"].nunique() != 2500:
        raise ValueError("Existing baseline is not the expected 2,500 unique patients")
    if not set(prior_train["Image Index"]) <= set(source_train["Image Index"].astype(str)):
        raise ValueError("Prior 2,500-patient training cohort does not match this source split")

    candidate_pool = source_train.loc[
        ~source_train["Image Index"].astype(str).isin(pilot_train_ids | set(prior_train["Image Index"]))
    ]
    additions = candidate_pool.sample(n=1500, random_state=selection_seed).copy()
    prior_train["split"] = "train"
    additions["split"] = "train"
    train_4000 = pd.concat([prior_train, additions], ignore_index=True)
    train_4000 = train_4000.drop_duplicates("Image Index")
    if len(train_4000) != 4000 or train_4000["Patient ID"].nunique() != 4000:
        raise ValueError("Candidate must contain 4,000 unique training patients")

    validation = prior_validation.loc[prior_validation["split"] == "validation"].copy()
    if len(validation) != 1000 or validation["Patient ID"].nunique() != 1000:
        raise ValueError("The frozen validation panel is not 1,000 unique patients")
    validation["split"] = "validation"
    train_ids = set(train_4000["Patient ID"])
    val_ids = set(validation["Patient ID"])
    if train_ids & val_ids:
        raise ValueError("Train/validation patient leakage")
    if set(validation["Image Index"].astype(str)) & pilot_val_ids:
        raise ValueError("New validation panel overlaps the earlier 500-patient pilot")

    columns = list(source_train.columns)
    baseline = pd.concat([prior_train[columns], validation[columns]], ignore_index=True)
    candidate = pd.concat([train_4000[columns], validation[columns]], ignore_index=True)
    baseline_path = output / "manifest_2500_train_1000_validation.csv"
    candidate_path = output / "manifest_4000_train_1000_validation.csv"
    baseline.to_csv(baseline_path, index=False)
    candidate.to_csv(candidate_path, index=False)

    missing = [name for name in candidate["Image Index"].astype(str)
               if not (image_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} selected image files are missing; first {missing[0]}")
    if set(candidate["split"].unique()) != {"train", "validation"}:
        raise ValueError("Unexpected split found; official-test rows must not be included")

    audit = {
        "purpose": "incremental learning-curve comparison: nested 2,500 vs 4,000 train patients",
        "training_patients_baseline": 2500,
        "training_patients_candidate": 4000,
        "additional_training_patients": 1500,
        "shared_validation_patients": 1000,
        "unique_patients_in_larger_cohort": 5000,
        "baseline_train_is_nested_in_candidate": True,
        "both_scores_use_same_validation_panel": True,
        "validation_panel_frozen_from_prior_stage": True,
        "earlier_500_patient_pilot_validation_excluded": True,
        "official_test_rows_used": 0,
        "new_downloads": 0,
        "patient_overlap": False,
        "image_root": str(image_root.resolve()),
        "all_selected_images_present": True,
        "selection_seed": selection_seed,
        "training_seed": 20261031,
        "train_label_positive_counts_4000": label_support(train_4000),
        "validation_label_positive_counts_1000": label_support(validation),
        "source_train_manifest_sha256": sha256(source_train_path),
        "source_validation_manifest_sha256": sha256(source_val_path),
        "prior_2500_manifest_sha256": sha256(prior_800_path),
        "baseline_manifest_sha256": sha256(baseline_path),
        "candidate_manifest_sha256": sha256(candidate_path),
        "caveat": "Source validation pool has prior development use (including a 900-image stage); this is not a fresh independent test.",
    }
    (output / "selection_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return baseline_path, candidate_path, audit


def score_checkpoint(checkpoint: Path, manifest: Path, image_root: Path,
                     seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = make_model(pretrained=False, architecture="densenet121")
    model.load_state_dict(state["model_state"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = NIHChestXrayDataset(manifest, image_root, "validation", None, seed,
                                  image_size=224, allow_available_subset=False)
    if len(dataset) != 1000 or dataset.missing_images:
        raise ValueError("Validation panel is incomplete")
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    _, y, scores = evaluate(model, loader, device)
    names = dataset.rows["Image Index"].astype(str).tolist()
    return y, scores, names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=20261101)
    parser.add_argument("--training-seed", type=int, default=20261031)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_manifest, candidate_manifest, audit = make_manifests(
        args.output_dir, args.project_root, args.image_root, args.selection_seed)
    print(json.dumps({"event": "cohort_ready", "train_2500": 2500, "train_4000": 4000,
                      "validation": 1000, "support": audit["validation_label_positive_counts_1000"]}, indent=2), flush=True)

    baseline_checkpoint = (args.project_root / "data" / "NexviaProjectData" / "nih" /
        "learning_curve_2500_train_1000_val_20261031" / "densenet121_train2500" / "last_checkpoint.pt")
    baseline_state = torch.load(baseline_checkpoint, map_location="cpu", weights_only=False)
    if int(baseline_state["epoch"]) != 11 or int(baseline_state["run_config"]["seed"]) != args.training_seed:
        raise ValueError("The existing 2,500-patient reference is not the matched fixed-10-epoch seed")

    run_dir = args.output_dir / "densenet121_train4000"
    command = [
        sys.executable, str(args.project_root / "train_classifier.py"),
        "--manifest", str(candidate_manifest), "--image-root", str(args.image_root),
        "--output", str(run_dir), "--architecture", "densenet121",
        "--epochs", str(args.epochs), "--image-size", "224", "--batch-size", "4",
        "--workers", "0", "--lr", "0.0001", "--seed", str(args.training_seed),
        "--positive-weight-exponent", "0.5", "--checkpoint-every-steps", "1000",
    ]
    start = time.perf_counter()
    subprocess.run(command, cwd=args.project_root, check=True)
    elapsed = time.perf_counter() - start
    candidate_checkpoint = run_dir / "last_checkpoint.pt"
    candidate_state = torch.load(candidate_checkpoint, map_location="cpu", weights_only=False)
    if int(candidate_state["epoch"]) != args.epochs + 1:
        raise RuntimeError("The candidate did not complete the fixed final epoch")

    y_base, p_base, names_base = score_checkpoint(
        baseline_checkpoint, baseline_manifest, args.image_root, args.training_seed)
    y_candidate, p_candidate, names_candidate = score_checkpoint(
        candidate_checkpoint, candidate_manifest, args.image_root, args.training_seed)
    if names_base != names_candidate or not np.array_equal(y_base, y_candidate):
        raise ValueError("Baseline and candidate were not scored on identical validation cases")
    baseline_metrics = metrics(y_base, p_base)
    candidate_metrics = metrics(y_candidate, p_candidate)
    delta = paired_bootstrap(y_base, p_base, p_candidate, seed=args.selection_seed)
    # The shared helper retains the old 400-vs-800 key name. Rename it here so
    # this stage's report states the actual candidate-minus-baseline direction.
    delta["deltas_candidate_minus_baseline"] = delta.pop("deltas_800_minus_400")
    summary = {
        "experiment": "incremental data-size comparison: nested 2,500 vs 4,000 training patients",
        "total_unique_patients": 5000,
        "official_test_used": False,
        "same_architecture_settings_seed_and_validation": True,
        "training": {
            "architecture": "ImageNet DenseNet-121",
            "image_size": 224,
            "epochs": args.epochs,
            "batch_size": 4,
            "learning_rate": 0.0001,
            "positive_weight_exponent": 0.5,
            "training_seed": args.training_seed,
            "candidate_elapsed_seconds": elapsed,
            "candidate_run_summary": json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8")),
        },
        "baseline_2500": {
            "checkpoint_sha256": sha256(baseline_checkpoint),
            "metrics": baseline_metrics,
        },
        "candidate_4000": {
            "checkpoint_sha256": sha256(candidate_checkpoint),
            "metrics": candidate_metrics,
        },
        "paired_bootstrap_4000_minus_2500": delta,
        "interpretation": "Single-seed development result only. The source validation pool has prior reuse; do not promote unless a repeat confirms the direction.",
    }
    (args.output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    predictions = pd.DataFrame({"Image Index": names_base})
    for col, label in enumerate(NIH_LABELS):
        predictions[f"target_{label}"] = y_base[:, col].astype(int)
        predictions[f"score_train2500_{label}"] = p_base[:, col]
        predictions[f"score_train4000_{label}"] = p_candidate[:, col]
    predictions.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    labels = [label for label in NIH_LABELS if candidate_metrics["per_label"][label]["average_precision"] is not None]
    draw_ap_chart(
        args.output_dir / "average_precision_by_finding.png", labels,
        [baseline_metrics["per_label"][label]["average_precision"] for label in labels],
        [candidate_metrics["per_label"][label]["average_precision"] for label in labels],
        [candidate_metrics["per_label"][label]["prevalence"] for label in labels],
    )
    print(json.dumps({"event": "complete", "baseline_2500": baseline_metrics,
                      "candidate_4000": candidate_metrics, "paired_delta": delta}, indent=2), flush=True)


if __name__ == "__main__":
    main()
