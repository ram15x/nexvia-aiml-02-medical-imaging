"""Matched 400-vs-800 training-size comparison on a fixed 200-patient holdout.

Uses only locally available NIH ChestX-ray14 images, keeps the official test
split out of the workflow, and scores the final fixed epoch rather than choosing
an epoch from the holdout. This is an exploratory development comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_audit import NIH_LABELS  # noqa: E402
from train_classifier import NIHChestXrayDataset, evaluate, make_model  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label_support(frame: pd.DataFrame) -> dict[str, int]:
    labels = frame["Finding Labels"].fillna("").astype(str)
    return {name: int(labels.map(lambda raw: name in raw.split("|")).sum())
            for name in NIH_LABELS}


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    predicted = p >= 0.5
    per_label: dict[str, dict] = {}
    aucs, aps = [], []
    for i, label in enumerate(NIH_LABELS):
        positives = int(y[:, i].sum())
        auc = float(roc_auc_score(y[:, i], p[:, i])) if 0 < positives < len(y) else None
        ap = float(average_precision_score(y[:, i], p[:, i])) if positives else None
        tp = int(np.logical_and(y[:, i] == 1, predicted[:, i]).sum())
        fp = int(np.logical_and(y[:, i] == 0, predicted[:, i]).sum())
        per_label[label] = {
            "positive_count": positives,
            "prevalence": positives / len(y),
            "auroc": auc,
            "average_precision": ap,
            "precision_at_0_5": tp / (tp + fp) if tp + fp else 0.0,
            "recall_at_0_5": tp / positives if positives else None,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": positives - tp,
        }
        if auc is not None:
            aucs.append(auc)
        if ap is not None:
            aps.append(ap)
    return {
        "patients": int(len(y)),
        "macro_auroc_defined_labels": float(np.mean(aucs)) if aucs else None,
        "macro_average_precision_positive_labels": float(np.mean(aps)) if aps else None,
        "micro_precision_at_0_5": float(precision_score(y, predicted, average="micro", zero_division=0)),
        "micro_recall_at_0_5": float(recall_score(y, predicted, average="micro", zero_division=0)),
        "micro_f1_at_0_5": float(f1_score(y, predicted, average="micro", zero_division=0)),
        "per_label": per_label,
    }


def paired_bootstrap(y: np.ndarray, base: np.ndarray, larger: np.ndarray,
                     *, seed: int, replicates: int = 1000) -> dict:
    rng = np.random.default_rng(seed)
    fields = ("macro_auroc_defined_labels", "macro_average_precision_positive_labels",
              "micro_f1_at_0_5")
    samples = {field: [] for field in fields}
    for _ in range(replicates):
        idx = rng.integers(0, len(y), size=len(y))
        left, right = metrics(y[idx], base[idx]), metrics(y[idx], larger[idx])
        for field in fields:
            if left[field] is not None and right[field] is not None:
                samples[field].append(right[field] - left[field])
    return {
        "replicates": replicates,
        "method": "paired patient-row bootstrap percentile intervals; exploratory only",
        "deltas_800_minus_400": {
            field: {
                "point_difference": metrics(y, larger)[field] - metrics(y, base)[field],
                "lower_2_5_percent": float(np.percentile(vals, 2.5)),
                "upper_97_5_percent": float(np.percentile(vals, 97.5)),
            }
            for field, vals in samples.items() if vals
        },
        "warning": "Small holdout and prior development reuse limit inference; bootstrap intervals do not remove model-selection bias.",
    }


def draw_ap_chart(path: Path, labels: list[str], ap400: list[float],
                  ap800: list[float], prevalence: list[float]) -> None:
    width, height = 1500, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        small = ImageFont.truetype("arial.ttf", 15)
        title_font = ImageFont.truetype("arial.ttf", 27)
    except OSError:
        font = small = title_font = ImageFont.load_default()
    left, right, top, bottom = 80, 35, 100, 180
    chart_w, chart_h = width - left - right, height - top - bottom
    max_value = max(0.1, max(ap400 + ap800 + prevalence) * 1.12)
    draw.text((left, 32), "Same 200-patient holdout: average precision and prevalence", fill="#20252b", font=title_font)
    draw.text((left, 68), "AP baseline depends on each finding's prevalence; rare labels have wide uncertainty.", fill="#515b66", font=small)
    for tick in range(6):
        value = max_value * tick / 5
        y = top + chart_h - int(chart_h * value / max_value)
        draw.line((left, y, width - right, y), fill="#e4e8ec", width=1)
        draw.text((18, y - 8), f"{value:.2f}", fill="#515b66", font=small)
    draw.line((left, top, left, top + chart_h), fill="#68737d", width=2)
    draw.line((left, top + chart_h, width - right, top + chart_h), fill="#68737d", width=2)
    slot = chart_w / max(len(labels), 1)
    bar_w = min(22, slot * 0.28)
    for i, label in enumerate(labels):
        center = left + slot * (i + 0.5)
        for value, color, offset in ((ap400[i], "#6b8fb3", -bar_w * 0.58),
                                     (ap800[i], "#d17a52", bar_w * 0.58)):
            x1 = center + offset - bar_w / 2
            x2 = center + offset + bar_w / 2
            y1 = top + chart_h - int(chart_h * value / max_value)
            draw.rectangle((x1, y1, x2, top + chart_h), fill=color)
        pv_y = top + chart_h - int(chart_h * prevalence[i] / max_value)
        draw.line((center - 16, pv_y, center + 16, pv_y), fill="#222222", width=4)
        draw.text((center, top + chart_h + 12), label.replace("_", "\n"), fill="#313940", font=small, anchor="ma", align="center")
    legend_y = height - 35
    legend = [("#6b8fb3", "400 training patients"), ("#d17a52", "800 training patients"), ("#222222", "Holdout prevalence")]
    x = left
    for color, label in legend:
        if color == "#222222":
            draw.line((x, legend_y + 9, x + 22, legend_y + 9), fill=color, width=4)
        else:
            draw.rectangle((x, legend_y + 2, x + 18, legend_y + 17), fill=color)
        draw.text((x + 28, legend_y), label, fill="#313940", font=font)
        x += 290
    image.save(path)


def make_manifests(args: argparse.Namespace, output: Path) -> tuple[Path, Path]:
    base = args.project_root / "data" / "NexviaProjectData" / "nih" / "metadata"
    source = base / "fresh_15k_patient_split_20261029"
    train_path = source / "train_manifest.csv"
    validation_path = source / "validation_manifest.csv"
    pilot_path = base / "pilot_500_patient_split_20261030" / "pilot_manifest.csv"
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    pilot = pd.read_csv(pilot_path)
    expected = set(train.columns)
    if set(validation.columns) != expected or set(pilot.columns) != expected:
        raise ValueError("Source and pilot manifest schemas differ")
    for frame in (train, validation, pilot):
        required = {"Image Index", "Finding Labels", "Patient ID", "split"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Manifest missing required columns: {required - set(frame.columns)}")
    if set(pilot["split"].unique()) - {"train", "validation"}:
        raise ValueError("Pilot manifest contains a forbidden split")
    old_train = set(pilot.loc[pilot["split"] == "train", "Image Index"].astype(str))
    old_validation = set(pilot.loc[pilot["split"] == "validation", "Image Index"].astype(str))
    train = train.loc[~train["Image Index"].astype(str).isin(old_train)]
    validation = validation.loc[~validation["Image Index"].astype(str).isin(old_validation)]
    train = train.sample(n=800, random_state=args.seed).copy()
    validation = validation.sample(n=200, random_state=args.seed).copy()
    train["split"] = "train"
    validation["split"] = "validation"
    combined = pd.concat([train, validation], ignore_index=True)
    if len(combined) != 1000 or combined["Patient ID"].nunique() != 1000:
        raise ValueError("Expected exactly 1,000 distinct patients")
    if set(train["Patient ID"]) & set(validation["Patient ID"]):
        raise ValueError("Patient leakage across train and validation")
    if len(set(combined["split"])) != 2:
        raise ValueError("Unexpected data split")

    baseline_train = train.sample(n=400, random_state=args.seed + 1).copy()
    baseline = pd.concat([baseline_train, validation], ignore_index=True)
    baseline_path = output / "manifest_400_train_200_validation.csv"
    larger_path = output / "manifest_800_train_200_validation.csv"
    baseline.to_csv(baseline_path, index=False)
    combined.to_csv(larger_path, index=False)

    root = args.image_root
    missing = [str(root / name) for name in combined["Image Index"].astype(str)
               if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} selected images are missing; first: {missing[0]}")
    audit = {
        "purpose": "matched learning-curve comparison; 400 vs 800 ImageNet DenseNet-121 training patients",
        "unique_patients_in_cohort": 1000,
        "training_patients_candidate": 800,
        "training_patients_baseline": 400,
        "shared_validation_patients": 200,
        "baseline_training_is_subset_of_candidate": True,
        "official_test_rows_used": 0,
        "new_downloads": 0,
        "all_selected_images_present": True,
        "patient_overlap": False,
        "excluded_previous_500_train_images": len(old_train),
        "excluded_previous_500_validation_images": len(old_validation),
        "sampling_seed": args.seed,
        "train_label_positive_counts_800": label_support(train),
        "validation_label_positive_counts_200": label_support(validation),
        "source_train_manifest_sha256": sha256(train_path),
        "source_validation_manifest_sha256": sha256(validation_path),
        "pilot_manifest_sha256": sha256(pilot_path),
        "baseline_manifest_sha256": sha256(baseline_path),
        "candidate_manifest_sha256": sha256(larger_path),
        "validation_note": "Validation is a random 200-patient development subset, not the official test. Some rare labels have very low support.",
    }
    (output / "selection_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return baseline_path, larger_path


def train_and_score(args: argparse.Namespace, manifest: Path, run_dir: Path,
                    expected_train: int, *, resume: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    command = [
        sys.executable, str(args.project_root / "train_classifier.py"),
        "--manifest", str(manifest), "--image-root", str(args.image_root),
        "--output", str(run_dir), "--architecture", "densenet121",
        "--epochs", str(args.epochs), "--image-size", "224", "--batch-size", "4",
        "--workers", "0", "--lr", "0.0001", "--seed", str(args.seed),
        "--positive-weight-exponent", "0.5", "--checkpoint-every-steps", "1000",
    ]
    last = run_dir / "last_checkpoint.pt"
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        if not last.is_file():
            raise FileNotFoundError(f"Cannot resume; checkpoint is missing: {last}")
        command.extend(["--resume-from", str(last)])
    start = time.perf_counter()
    subprocess.run(command, cwd=args.project_root, check=True)
    elapsed = time.perf_counter() - start
    state = torch.load(last, map_location="cpu", weights_only=False)
    if int(state["epoch"]) != args.epochs + 1:
        raise RuntimeError(f"Fixed {args.epochs}-epoch run did not complete: {run_dir}")
    model = make_model(pretrained=False, architecture="densenet121")
    model.load_state_dict(state["model_state"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = NIHChestXrayDataset(manifest, args.image_root, "validation", None,
                                  args.seed, image_size=224, allow_available_subset=False)
    if len(dataset) != 200 or dataset.missing_images:
        raise ValueError("Validation set is incomplete")
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    _, y, p = evaluate(model, loader, device)
    details = {
        "architecture": "ImageNet initialized DenseNet-121",
        "train_images": expected_train,
        "validation_images": len(dataset),
        "epochs": args.epochs,
        "scored_checkpoint": "last_checkpoint.pt; fixed final epoch, not best-validation checkpoint",
        "validation_used_for_gradient_updates": False,
        "elapsed_seconds_including_subprocess": elapsed,
        "checkpoint_sha256": sha256(last),
        "training_run_summary": json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8")),
    }
    return y, p, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--resume", action="store_true",
                        help="Continue both existing matched runs from their latest checkpoints")
    parser.add_argument("--reuse-manifests-from", type=Path, default=None,
                        help="Reuse a previously frozen pair of manifests, without resampling patients")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    output = args.output_dir
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.resume or args.reuse_manifests_from is not None:
        source = args.reuse_manifests_from if args.reuse_manifests_from is not None else output
        if args.resume and args.reuse_manifests_from is not None:
            raise ValueError("Choose either --resume or --reuse-manifests-from, not both")
        baseline_source = source / "manifest_400_train_200_validation.csv"
        larger_source = source / "manifest_800_train_200_validation.csv"
        if not baseline_source.is_file() or not larger_source.is_file():
            raise FileNotFoundError("Frozen manifests are missing from the requested source directory")
        baseline_manifest = output / baseline_source.name
        larger_manifest = output / larger_source.name
        if source.resolve() != output.resolve():
            shutil.copy2(baseline_source, baseline_manifest)
            shutil.copy2(larger_source, larger_manifest)
            audit_source = source / "selection_audit.json"
            if audit_source.is_file():
                shutil.copy2(audit_source, output / audit_source.name)
    else:
        baseline_manifest, larger_manifest = make_manifests(args, output)
    print(json.dumps({"event": "selection_complete", "patients": 1000,
                      "train_400": str(baseline_manifest), "train_800": str(larger_manifest)}, indent=2), flush=True)

    y400, p400, detail400 = train_and_score(
        args, baseline_manifest, output / "densenet121_train400", 400, resume=args.resume)
    y800, p800, detail800 = train_and_score(
        args, larger_manifest, output / "densenet121_train800", 800, resume=args.resume)
    if not np.array_equal(y400, y800):
        raise ValueError("The two runs did not produce identical validation targets/order")
    summary400, summary800 = metrics(y400, p400), metrics(y800, p800)
    paired = paired_bootstrap(y400, p400, p800, seed=args.seed)
    summary = {
        "experiment": "matched 400-vs-800 training-size comparison",
        "total_unique_patients": 1000,
        "training_seed": args.seed,
        "official_test_used": False,
        "metrics_at_threshold": 0.5,
        "baseline_train400": {"run": detail400, "metrics": summary400},
        "candidate_train800": {"run": detail800, "metrics": summary800},
        "paired_bootstrap": paired,
        "interpretation": "Single development holdout comparison; small sample and low rare-label support make results exploratory. Not clinical evidence.",
    }
    (output / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    predictions = pd.DataFrame({"Image Index": pd.read_csv(baseline_manifest).query("split == 'validation'")["Image Index"].astype(str)})
    for i, label in enumerate(NIH_LABELS):
        predictions[f"target_{label}"] = y400[:, i].astype(int)
        predictions[f"score_train400_{label}"] = p400[:, i]
        predictions[f"score_train800_{label}"] = p800[:, i]
    predictions.to_csv(output / "validation_predictions.csv", index=False)

    labels = [name for name in NIH_LABELS if summary400["per_label"][name]["average_precision"] is not None]
    ap400 = [summary400["per_label"][name]["average_precision"] for name in labels]
    ap800 = [summary800["per_label"][name]["average_precision"] for name in labels]
    prevalence = [summary400["per_label"][name]["prevalence"] for name in labels]
    draw_ap_chart(output / "average_precision_by_finding.png", labels, ap400, ap800, prevalence)
    print(json.dumps({"event": "experiment_complete", "metrics_400": summary400,
                      "metrics_800": summary800, "paired_bootstrap": paired}, indent=2), flush=True)


if __name__ == "__main__":
    main()
