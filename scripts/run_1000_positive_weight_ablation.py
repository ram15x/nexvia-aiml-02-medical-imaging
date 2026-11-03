"""Test stronger positive weighting on the frozen 800/200 learning-curve split."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_audit import NIH_LABELS  # noqa: E402
from scripts.run_1000_image_learning_curve import metrics, paired_bootstrap, draw_ap_chart  # noqa: E402
from train_classifier import NIHChestXrayDataset, evaluate, make_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="Original 1,000-patient experiment folder (seed-20261031)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference_predictions = pd.read_csv(args.reference_dir / "validation_predictions.csv")
    manifest = args.reference_dir / "manifest_800_train_200_validation.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    run_dir = args.output_dir / "densenet121_train800_positive_weight_1_0"
    command = [
        sys.executable, str(PROJECT_ROOT / "train_classifier.py"),
        "--manifest", str(manifest), "--image-root", str(args.image_root),
        "--output", str(run_dir), "--architecture", "densenet121",
        "--epochs", str(args.epochs), "--image-size", "224", "--batch-size", "4",
        "--workers", "0", "--lr", "0.0001", "--seed", str(args.seed),
        "--positive-weight-exponent", "1.0", "--checkpoint-every-steps", "1000",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    checkpoint = run_dir / "last_checkpoint.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(state["epoch"]) != args.epochs + 1:
        raise RuntimeError("The fixed final epoch was not reached")
    model = make_model(pretrained=False, architecture="densenet121")
    model.load_state_dict(state["model_state"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = NIHChestXrayDataset(manifest, args.image_root, "validation", None,
                                  args.seed, image_size=224, allow_available_subset=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    _, y, candidate_scores = evaluate(model, loader, device)

    names = dataset.rows["Image Index"].astype(str).tolist()
    if names != reference_predictions["Image Index"].astype(str).tolist():
        raise ValueError("Reference and candidate validation rows are not identically ordered")
    y_ref = reference_predictions[[f"target_{label}" for label in NIH_LABELS]].to_numpy(dtype=np.float32)
    baseline_scores = reference_predictions[[f"score_train800_{label}" for label in NIH_LABELS]].to_numpy(dtype=np.float32)
    if not np.array_equal(y, y_ref):
        raise ValueError("Reference and candidate targets do not match")

    baseline_metrics = metrics(y, baseline_scores)
    candidate_metrics = metrics(y, candidate_scores)
    paired = paired_bootstrap(y, baseline_scores, candidate_scores, seed=args.seed)
    summary = {
        "experiment": "positive-class weighting ablation on same 800 train / 200 validation patients",
        "single_change": "positive-weight exponent 0.5 -> 1.0, both capped at 10",
        "training_seed": args.seed,
        "epochs": args.epochs,
        "official_test_used": False,
        "baseline": {"weight_exponent": 0.5, "metrics": baseline_metrics},
        "candidate": {
            "weight_exponent": 1.0,
            "metrics": candidate_metrics,
            "checkpoint": str(checkpoint),
            "training_run_summary": json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8")),
        },
        "paired_bootstrap": paired,
        "caveat": "One seed and a 200-patient development holdout with very low rare-label support; do not promote from this result alone.",
    }
    (args.output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = pd.DataFrame({"Image Index": names})
    for i, label in enumerate(NIH_LABELS):
        rows[f"target_{label}"] = y[:, i].astype(int)
        rows[f"score_sqrt_weight_{label}"] = baseline_scores[:, i]
        rows[f"score_full_weight_{label}"] = candidate_scores[:, i]
    rows.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    labels = [label for label in NIH_LABELS if candidate_metrics["per_label"][label]["average_precision"] is not None]
    ap_base = [baseline_metrics["per_label"][label]["average_precision"] for label in labels]
    ap_cand = [candidate_metrics["per_label"][label]["average_precision"] for label in labels]
    prevalence = [candidate_metrics["per_label"][label]["prevalence"] for label in labels]
    draw_ap_chart(args.output_dir / "average_precision_by_finding.png", labels, ap_base, ap_cand, prevalence)
    print(json.dumps({"baseline": baseline_metrics, "candidate": candidate_metrics,
                      "paired_bootstrap": paired}, indent=2), flush=True)


if __name__ == "__main__":
    main()
