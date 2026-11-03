"""Matched weighted-BCE vs Asymmetric Loss ablation on the 2,500-patient cohort."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_audit import NIH_LABELS  # noqa: E402
from scripts.run_1000_image_learning_curve import (  # noqa: E402
    draw_ap_chart, metrics, paired_bootstrap, sha256,
)
from train_classifier import NIHChestXrayDataset, evaluate, make_model  # noqa: E402


def score(checkpoint: Path, manifest: Path, image_root: Path, seed: int):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = make_model(pretrained=False, architecture="densenet121")
    model.load_state_dict(state["model_state"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = NIHChestXrayDataset(
        manifest, image_root, "validation", None, seed,
        image_size=224, allow_available_subset=False,
    )
    if len(dataset) != 1000 or dataset.missing_images:
        raise ValueError("The shared validation panel is incomplete")
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    _, y, probabilities = evaluate(model, loader, device)
    return y, probabilities, dataset.rows["Image Index"].astype(str).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stage = ROOT / "data" / "NexviaProjectData" / "nih" / "learning_curve_2500_train_1000_val_20261031"
    # This manifest is the 2,500-train candidate, despite its legacy filename.
    manifest = stage / "manifest_2500_train_1000_validation.csv"
    frame = pd.read_csv(manifest)
    train = frame.loc[frame["split"] == "train"]
    validation = frame.loc[frame["split"] == "validation"]
    if len(train) != 2500 or train["Patient ID"].nunique() != 2500:
        raise ValueError("Expected 2,500 unique training patients")
    if len(validation) != 1000 or validation["Patient ID"].nunique() != 1000:
        raise ValueError("Expected the frozen 1,000-case validation panel")
    if set(train["Patient ID"]) & set(validation["Patient ID"]):
        raise ValueError("Train/validation patient overlap")

    baseline_checkpoint = stage / "densenet121_train2500" / "last_checkpoint.pt"
    baseline_state = torch.load(baseline_checkpoint, map_location="cpu", weights_only=False)
    baseline_cfg = baseline_state.get("run_config", {})
    if int(baseline_state["epoch"]) != 11 or int(baseline_cfg.get("seed", -1)) != args.seed:
        raise ValueError("BCE reference is not the expected final 10-epoch seed")

    run_dir = args.output_dir / "densenet121_asymmetric_2500"
    command = [
        sys.executable, str(ROOT / "train_classifier.py"),
        "--manifest", str(manifest), "--image-root", str(args.image_root),
        "--output", str(run_dir), "--architecture", "densenet121",
        "--loss", "asymmetric", "--epochs", str(args.epochs),
        "--image-size", "224", "--batch-size", "4", "--workers", "0",
        "--lr", "0.0001", "--seed", str(args.seed),
        "--checkpoint-every-steps", "1000",
    ]
    start = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True)
    elapsed = time.perf_counter() - start
    candidate_checkpoint = run_dir / "last_checkpoint.pt"
    candidate_state = torch.load(candidate_checkpoint, map_location="cpu", weights_only=False)
    if int(candidate_state["epoch"]) != args.epochs + 1:
        raise RuntimeError("ASL candidate did not finish the fixed final epoch")
    if candidate_state["run_config"].get("loss") != "asymmetric":
        raise RuntimeError("Candidate checkpoint does not record Asymmetric Loss")

    y_base, p_base, names_base = score(baseline_checkpoint, manifest, args.image_root, args.seed)
    y_asl, p_asl, names_asl = score(candidate_checkpoint, manifest, args.image_root, args.seed)
    if names_base != names_asl or not np.array_equal(y_base, y_asl):
        raise ValueError("BCE and ASL were not scored on identical validation cases")
    base_metrics, asl_metrics = metrics(y_base, p_base), metrics(y_asl, p_asl)
    delta = paired_bootstrap(y_base, p_base, p_asl, seed=args.seed)
    delta["deltas_candidate_minus_baseline"] = delta.pop("deltas_800_minus_400")

    summary = {
        "experiment": "matched BCE vs Asymmetric Loss on the same 2,500-patient cohort",
        "official_test_used": False,
        "training_patients": 2500,
        "validation_patients": 1000,
        "same_architecture_seed_manifest_preprocessing_epochs_and_optimizer": True,
        "only_intended_change": "loss strategy: weighted BCE vs Asymmetric Loss",
        "training": {
            "architecture": "ImageNet DenseNet-121",
            "image_size": 224,
            "epochs": args.epochs,
            "batch_size": 4,
            "learning_rate": 0.0001,
            "seed": args.seed,
            "candidate_loss": "ASL gamma_neg=4, gamma_pos=0, clip=0.05; mean over batch-label elements; no BCE pos_weight",
            "candidate_elapsed_seconds": elapsed,
            "candidate_run_summary": json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8")),
        },
        "bce_reference": {"checkpoint_sha256": sha256(baseline_checkpoint), "metrics": base_metrics},
        "asl_candidate": {"checkpoint_sha256": sha256(candidate_checkpoint), "metrics": asl_metrics},
        "paired_bootstrap_asl_minus_bce": delta,
        "validation_caveat": "The 1,000-case panel has prior development reuse and only 2 Hernia, 3 Edema, and 5 Pneumonia positives. This is a screening ablation, not final independent validation.",
    }
    (args.output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pred = pd.DataFrame({"Image Index": names_base})
    for i, label in enumerate(NIH_LABELS):
        pred[f"target_{label}"] = y_base[:, i].astype(int)
        pred[f"score_bce_{label}"] = p_base[:, i]
        pred[f"score_asl_{label}"] = p_asl[:, i]
    pred.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    visible = [label for label in NIH_LABELS if asl_metrics["per_label"][label]["average_precision"] is not None]
    draw_ap_chart(
        args.output_dir / "average_precision_by_finding.png", visible,
        [base_metrics["per_label"][label]["average_precision"] for label in visible],
        [asl_metrics["per_label"][label]["average_precision"] for label in visible],
        [asl_metrics["per_label"][label]["prevalence"] for label in visible],
    )
    print(json.dumps({"event": "complete", "bce": base_metrics,
                      "asl": asl_metrics, "paired_delta": delta}, indent=2), flush=True)


if __name__ == "__main__":
    main()
