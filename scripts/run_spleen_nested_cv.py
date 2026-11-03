"""Run five-fold case-level nested cross-validation for the MSD Spleen pilot.

Each outer fold stays untouched during training and checkpoint selection. A
separate inner validation split is used by the existing training script. Outer
fold evaluation uses a fixed 0.5 foreground threshold; thresholds are not tuned
on the outer cases. Outputs are kept outside the source tree and can resume by
skipping folds with complete summaries.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_segmentation import read_cases  # noqa: E402


def make_folds(case_ids: list[str], n_folds: int, seed: int) -> list[list[str]]:
    shuffled = sorted(case_ids)
    random.Random(seed).shuffle(shuffled)
    return [shuffled[index::n_folds] for index in range(n_folds)]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--augmentation-mode", choices=("baseline", "spatial"), default="baseline")
    args = parser.parse_args()
    if args.epochs < 1 or args.folds < 2 or args.workers < 0:
        parser.error("epochs and folds must be positive (folds >= 2); workers cannot be negative")

    cases = read_cases(args.dataset_json, args.dataset_root)
    case_ids = [row["case_id"] for row in cases]
    if len(case_ids) != 41:
        raise ValueError(f"Expected all 41 labeled MSD Spleen training cases, found {len(case_ids)}")
    outer_folds = make_folds(case_ids, args.folds, args.seed)
    expected = set(case_ids)
    flattened = [case_id for fold in outer_folds for case_id in fold]
    if set(flattened) != expected or len(flattened) != len(expected):
        raise RuntimeError("Outer folds are not a disjoint cover of all labeled cases")

    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "protocol.json", {
        "dataset": "MSD Task09 Spleen",
        "labeled_cases": 41,
        "outer_folds": args.folds,
        "outer_split_seed": args.seed,
        "epochs": args.epochs,
        "model": "MONAI 3D U-Net (same as baseline)",
        "augmentation_mode": args.augmentation_mode,
        "checkpoint_selection": "inner validation case-mean Dice at foreground threshold 0.5",
        "outer_evaluation_threshold": 0.5,
        "outer_cases_used_for_training_or_checkpoint_selection": False,
        "official_unlabeled_test_cases_used": False,
        "warning": "Small-cohort internal cross-validation; not clinical evidence or an external test.",
        "folds": outer_folds,
    })
    summarize(args.output, args.folds, args.epochs)

    for index, outer_test in enumerate(outer_folds, start=1):
        fold_dir = args.output / f"fold_{index:02d}"
        split_path = fold_dir / "case_split.json"
        train_summary = fold_dir / "run_summary.json"
        evaluation_path = fold_dir / "outer_evaluation.json"
        if train_summary.is_file() and evaluation_path.is_file():
            print(f"Fold {index}/{args.folds}: complete; reusing recorded outputs", flush=True)
            continue

        outer_test_set = set(outer_test)
        outer_train = [case_id for case_id in case_ids if case_id not in outer_test_set]
        inner = outer_train.copy()
        random.Random(args.seed + index).shuffle(inner)
        n_validation = max(1, round(len(inner) * 0.2))
        validation = inner[:n_validation]
        training = inner[n_validation:]
        split = {
            "seed": args.seed + index,
            "policy": "nested case-level CV: outer fold held out; inner 20% of outer training cases used for validation",
            "train": training,
            "validation": validation,
            "test": outer_test,
        }
        fold_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(split_path, split)

        train_cmd = [
            sys.executable, "-u", str(ROOT / "train_segmentation.py"),
            "--dataset-json", str(args.dataset_json.resolve()),
            "--dataset-root", str(args.dataset_root.resolve()),
            "--output", str(fold_dir.resolve()),
            "--epochs", str(args.epochs), "--batch-size", "1", "--workers", str(args.workers),
            "--lr", "0.001", "--patch-size", "64", "64", "64",
            "--seed", str(args.seed + index), "--augmentation-mode", args.augmentation_mode,
        ]
        log_path = fold_dir / "training.log"
        print(f"Fold {index}/{args.folds}: training {len(training)} cases, validating on {len(validation)}; outer {len(outer_test)} cases remain untouched", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(train_cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

        eval_cmd = [
            sys.executable, "-u", str(ROOT / "evaluate_segmentation.py"),
            "--dataset-json", str(args.dataset_json.resolve()),
            "--dataset-root", str(args.dataset_root.resolve()),
            "--split", str(split_path.resolve()),
            "--checkpoint", str((fold_dir / "best_spleen_unet.pt").resolve()),
            "--output", str(evaluation_path.resolve()),
            "--foreground-threshold", "0.5", "--workers", str(args.workers),
            "--seed", str(args.seed + index),
        ]
        with (fold_dir / "evaluation.log").open("w", encoding="utf-8") as log:
            subprocess.run(eval_cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads(evaluation_path.read_text(encoding="utf-8"))
        print(f"Fold {index}/{args.folds}: outer mean Dice {result['test_case_mean_dice']:.4f}", flush=True)
        summarize(args.output, args.folds, args.epochs)

    summarize(args.output, args.folds, args.epochs)


def summarize(output: Path, n_folds: int, epochs: int) -> None:
    fold_results = []
    per_case = []
    for index in range(1, n_folds + 1):
        path = output / f"fold_{index:02d}" / "outer_evaluation.json"
        if not path.is_file():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        fold_results.append({"fold": index, "case_count": result["test_cases"],
                             "mean_case_dice": result["test_case_mean_dice"]})
        per_case.extend(result["per_case"])
    scores = [float(row["dice"]) for row in per_case]
    summary = {
        "status": "complete" if len(fold_results) == n_folds else "in_progress",
        "dataset": "MSD Task09 Spleen",
        "outer_folds_completed": len(fold_results),
        "outer_folds_requested": n_folds,
        "epochs_per_fold": epochs,
        "case_count": len(scores),
        "out_of_fold_mean_case_dice": float(statistics.fmean(scores)) if scores else None,
        "out_of_fold_median_case_dice": float(statistics.median(scores)) if scores else None,
        "out_of_fold_case_dice_sd": float(statistics.stdev(scores)) if len(scores) > 1 else None,
        "folds": fold_results,
        "per_case": per_case,
        "interpretation": "Internal development estimate only; all 41 labeled cases contribute one outer prediction. The official unlabeled challenge set remains unused.",
    }
    atomic_json(output / "cross_validation_summary.json", summary)
    print(json.dumps({key: summary[key] for key in (
        "status", "outer_folds_completed", "case_count", "out_of_fold_mean_case_dice",
        "out_of_fold_median_case_dice", "out_of_fold_case_dice_sd")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
