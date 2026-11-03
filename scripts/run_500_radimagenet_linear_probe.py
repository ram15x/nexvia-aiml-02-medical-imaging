"""Evaluate a frozen RadImageNet DenseNet-121 plus regularized linear heads.

Only the fixed 500-patient development cohort is used. The external RadImageNet
encoder is frozen; each label head and feature scaler are fit inside the
existing patient-level training folds. No official-test rows are read.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api.modeling import build_classifier  # noqa: E402
from data_audit import NIH_LABELS  # noqa: E402
from scripts.run_500_image_oof_architecture_comparison import summarize  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CXRFeatureDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, image_root: Path):
        self.rows = rows.reset_index(drop=True)
        self.paths = [image_root / str(name) for name in self.rows["Image Index"]]
        missing = [str(path) for path in self.paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} cohort images; first: {missing[0]}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as src:
            image = np.asarray(src.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR),
                               dtype=np.float32) / 255.0
        # RadImageNet's official PyTorch example scales input intensities to [-1, 1].
        image = (image - 0.5) / 0.5
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


def main() -> None:
    nih = PROJECT_ROOT / "data" / "NexviaProjectData" / "nih"
    reference = nih / "oof_500_backbone_10epoch_20261031"
    manifest_path = reference / "cohort_manifest.csv"
    image_root = nih / "images_224_fresh_15k_20261029"
    weights_path = nih / "torch-cache" / "radimagenet_densenet121.pt"
    output = nih / "oof_500_radimagenet_linear_probe_20261031"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(manifest_path)
    if len(cohort) != 500 or cohort["Patient ID"].nunique() != 500:
        raise ValueError("Expected exactly 500 images from 500 unique patients")
    if "test" in set(cohort["split"].astype(str)):
        raise ValueError("Official-test rows are forbidden")
    reference_predictions = pd.read_csv(reference / "oof_predictions.csv")
    if reference_predictions["Image Index"].tolist() != cohort["Image Index"].tolist():
        raise ValueError("Reference predictions and cohort rows do not align")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_classifier("densenet121", len(NIH_LABELS), pretrained=False)
    rad_state = torch.load(weights_path, map_location="cpu", weights_only=True)
    feature_state = {
        key[len("backbone.0."):]: value
        for key, value in rad_state.items()
        if key.startswith("backbone.0.")
    }
    model.features.load_state_dict(feature_state, strict=True)
    for parameter in model.features.parameters():
        parameter.requires_grad_(False)
    model.features.eval().to(device)

    started = time.perf_counter()
    dataset = CXRFeatureDataset(cohort, image_root)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    embeddings = []
    with torch.inference_mode():
        for images in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            features = F.relu(model.features(images), inplace=False)
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
            embeddings.append(features.cpu().numpy())
    X = np.concatenate(embeddings).astype(np.float32)
    if X.shape != (500, 1024) or not np.isfinite(X).all():
        raise ValueError(f"Unexpected or non-finite embedding array: {X.shape}")
    np.save(output / "radimagenet_features_500x1024.npy", X)

    label_matrix = np.column_stack([
        np.asarray([float(label in str(raw).split("|")) for raw in cohort["Finding Labels"]],
                   dtype=np.int32)
        for label in NIH_LABELS
    ])
    scores = np.zeros_like(label_matrix, dtype=np.float32)
    fold_hashes = {}
    fold_records = []
    for fold in range(1, 6):
        fold_path = reference / "fold_manifests" / f"fold_{fold:02d}.csv"
        fold_hashes[str(fold)] = sha256(fold_path)
        split = pd.read_csv(fold_path)
        train_names = set(split.loc[split["split"] == "train", "Image Index"].astype(str))
        val_names = set(split.loc[split["split"] == "validation", "Image Index"].astype(str))
        if len(train_names) != 400 or len(val_names) != 100 or train_names & val_names:
            raise ValueError(f"Fold {fold} violates the 400/100 patient split")
        train_idx = np.flatnonzero(cohort["Image Index"].astype(str).isin(train_names).to_numpy())
        val_idx = np.flatnonzero(cohort["Image Index"].astype(str).isin(val_names).to_numpy())
        if len(train_idx) != 400 or len(val_idx) != 100:
            raise ValueError(f"Fold {fold} rows do not align to the frozen cohort")

        scaler = StandardScaler().fit(X[train_idx])
        x_train = scaler.transform(X[train_idx])
        x_val = scaler.transform(X[val_idx])
        fold_predictions = np.zeros((len(val_idx), len(NIH_LABELS)), dtype=np.float32)
        class_counts = {}
        for col, label in enumerate(NIH_LABELS):
            y_train = label_matrix[train_idx, col]
            positives = int(y_train.sum())
            class_counts[label] = positives
            if positives == 0 or positives == len(y_train):
                fold_predictions[:, col] = 0.0
                continue
            head = LogisticRegression(
                C=0.1, class_weight="balanced", solver="liblinear",
                max_iter=1000, random_state=20261031 + fold,
            )
            head.fit(x_train, y_train)
            fold_predictions[:, col] = head.predict_proba(x_val)[:, 1]
        scores[val_idx] = fold_predictions
        fold_records.append({
            "fold": fold,
            "train_images": len(train_idx),
            "heldout_images": len(val_idx),
            "train_positive_counts": class_counts,
        })

    if not np.isfinite(scores).all():
        raise RuntimeError("The OOF scores are not all finite")
    result = {
        "purpose": "frozen external medical-feature encoder plus regularized per-label linear probes",
        "images": 500,
        "unique_patients": 500,
        "official_test_rows_used": 0,
        "encoder": "Official RadImageNet DenseNet-121 feature extractor; all weights frozen; classifier head discarded",
        "encoder_sha256": sha256(weights_path),
        "encoder_input": "Grayscale NIH PNGs replicated to RGB, resized 224x224, scaled to [-1,1] per the official RadImageNet PyTorch example",
        "features": {"shape": list(X.shape), "pooling": "ReLU after DenseNet features, global average pool"},
        "probe": "Fold-local StandardScaler followed by per-label L2 LogisticRegression(C=0.1, class_weight='balanced', liblinear); no threshold fitting",
        "folds": 5,
        "fold_manifest_sha256": fold_hashes,
        "fold_records": fold_records,
        "threshold": 0.5,
        "metrics": summarize(label_matrix, scores),
        "baseline_full_finetune_densenet_metrics": summarize(
            label_matrix,
            np.column_stack([
                reference_predictions[f"score_densenet121_{label}"].to_numpy(dtype=np.float32)
                for label in NIH_LABELS
            ]),
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "The 500-patient cohort has been repeatedly inspected; this is a screening result, not an independent benchmark.",
            "Several findings have very low positive support and Hernia has no positives.",
            "Balanced logistic heads produce uncalibrated scores; threshold 0.5 is only a fixed comparison point.",
            "No confidence interval was computed; do not promote from point estimates alone.",
            "The encoder was pretrained on CT/MRI/ultrasound, not chest radiographs; transfer may not help this task.",
        ],
    }
    pred_frame = pd.DataFrame({"Image Index": cohort["Image Index"].astype(str)})
    for col, label in enumerate(NIH_LABELS):
        pred_frame[f"target_{label}"] = label_matrix[:, col]
        pred_frame[f"score_radimagenet_linear_{label}"] = scores[:, col]
    pred_frame.to_csv(output / "oof_predictions.csv", index=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    baseline = result["baseline_full_finetune_densenet_metrics"]
    candidate = result["metrics"]
    report = [
        "# RadImageNet frozen-feature linear probe on the 500-patient cohort",
        "",
        "**Date:** 2026-10-31  ",
        "**Scope:** exactly the existing 500 images / 500 patients, same five held-out folds; no official-test rows.",
        "",
        "## Protocol",
        "",
        "The official RadImageNet DenseNet-121 feature weights were loaded with PyTorch's safe weights-only loader. The encoder is frozen; its classifier is discarded. Features are extracted once, then each fold fits a training-only scaler and 14 regularized one-vs-rest logistic heads. The fixed fold-held-out predictions are pooled for the reported metrics.",
        "",
        "## Results",
        "",
        "| Model | Macro AUROC | Macro AP | Precision @0.5 | Recall @0.5 | Micro F1 @0.5 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| ImageNet DenseNet-121 full fine-tune | {baseline['macro_auroc_defined_labels']:.4f} | {baseline['macro_auprc_positive_labels']:.4f} | {baseline['micro_precision_at_0_5']:.4f} | {baseline['micro_recall_at_0_5']:.4f} | {baseline['micro_f1_at_0_5']:.4f} |",
        f"| RadImageNet frozen encoder + linear heads | {candidate['macro_auroc_defined_labels']:.4f} | {candidate['macro_auprc_positive_labels']:.4f} | {candidate['micro_precision_at_0_5']:.4f} | {candidate['micro_recall_at_0_5']:.4f} | {candidate['micro_f1_at_0_5']:.4f} |",
        "",
        "## Limits",
        "",
        *[f"- {item}" for item in result["limitations"]],
        "",
        "This probe tests a medically pretrained representation without training a larger architecture. A different AUROC/AP tradeoff is not by itself a promotion; compare the paired OOF predictions and require a predeclared improvement before changing the demo model.",
        "",
        f"Output directory: `{output.relative_to(PROJECT_ROOT).as_posix()}`. The compressed multi-model archive was removed after extracting only the 28.4 MB DenseNet weights.",
    ]
    report_path = PROJECT_ROOT / "reports" / "500_image_radimagenet_linear_probe_2026-10-31.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "radimagenet_metrics": candidate,
        "baseline_metrics": baseline,
        "elapsed_seconds": result["elapsed_seconds"],
        "weights_sha256": result["encoder_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
