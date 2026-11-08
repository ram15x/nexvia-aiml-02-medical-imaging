"""Evaluate frozen NIH-trained checkpoints on CheXpert's radiologist-labeled val set.

This is an external validation tool, not a training or test-set selection script.
Only the five findings shared by NIH ChestX-ray14 and CheXpert are evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from api.modeling import build_classifier, image_to_tensor


LABEL_MAP = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Pleural Effusion": "Effusion",
}


def patient_id(path: str) -> str:
    match = re.search(r"patient\d+", path)
    if not match:
        raise ValueError(f"Could not extract de-identified patient ID from path: {path}")
    return match.group(0)


def patient_bootstrap_ci(y: np.ndarray, p: np.ndarray, patients: np.ndarray,
                         *, seed: int = 20261031, samples: int = 1000) -> dict[str, list[float] | None]:
    """Percentile intervals from patient-cluster resampling, excluding one-class draws."""
    unique = np.unique(patients)
    groups = {pid: np.flatnonzero(patients == pid) for pid in unique}
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    aps: list[float] = []
    f1s: list[float] = []
    for _ in range(samples):
        selected = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([groups[pid] for pid in selected])
        target = y[indices]
        score = p[indices]
        if len(np.unique(target)) == 2:
            aucs.append(float(roc_auc_score(target, score)))
        if target.sum() > 0:
            aps.append(float(average_precision_score(target, score)))
        f1s.append(float(f1_score(target, score >= 0.5, zero_division=0)))

    def interval(values: list[float]) -> list[float] | None:
        return [float(x) for x in np.quantile(values, [0.025, 0.975])] if values else None

    return {"auroc": interval(aucs), "average_precision": interval(aps), "f1_at_0_5": interval(f1s)}


class CheXpertValidation(Dataset):
    def __init__(self, image_dir: Path, labels_csv: Path) -> None:
        self.image_dir = image_dir
        frame = pd.read_csv(labels_csv)
        required = {"Path", "Frontal/Lateral", *LABEL_MAP}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"CheXpert validation CSV is missing columns: {sorted(missing)}")
        frame = frame[frame["Frontal/Lateral"].astype(str).str.lower() == "frontal"].copy()
        frame["Path"] = frame["Path"].astype(str)
        if frame.empty:
            raise ValueError("No frontal images found in the CheXpert validation labels")
        if frame["Path"].duplicated().any():
            raise ValueError("Duplicate image paths in the CheXpert validation labels")
        self.rows = frame.reset_index(drop=True)
        self.image_size = 224

    def __len__(self) -> int:
        return len(self.rows)

    def _image_path(self, raw_path: str) -> Path:
        root = self.image_dir.resolve()
        parts = Path(raw_path.replace("\\", "/")).parts
        for marker in ("valid", "val"):
            if marker in parts:
                candidate = self.image_dir.joinpath(*parts[parts.index(marker) + 1 :])
                if candidate.is_file() and candidate.resolve().is_relative_to(root):
                    return candidate
        candidate = self.image_dir / Path(raw_path).name
        if candidate.is_file() and candidate.resolve().is_relative_to(root):
            return candidate
        raise FileNotFoundError(f"Image not found under {self.image_dir}: {raw_path}")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        row = self.rows.iloc[index]
        path = self._image_path(row["Path"])
        with Image.open(path) as image:
            tensor = image_to_tensor(image, self.image_size)
        return tensor, row["Path"]


def evaluate_one(checkpoint_path: Path, dataset: CheXpertValidation,
                 device: torch.device, batch_size: int) -> tuple[dict, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    labels = checkpoint.get("labels")
    if not isinstance(labels, list) or len(labels) != 14:
        raise ValueError(f"Unexpected label list in checkpoint: {checkpoint_path}")
    model = build_classifier(
        checkpoint["architecture"], len(labels), pretrained=False,
        label_adjacency=checkpoint.get("label_adjacency"),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    dataset.image_size = int(checkpoint.get("image_size", 224))
    model.to(device).eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for images, batch_paths in loader:
            probs = torch.sigmoid(model(images.to(device))).cpu().numpy()
            outputs.append(probs)
    predictions = np.concatenate(outputs)
    scores: dict[str, dict] = {}
    patient_ids = dataset.rows["Path"].astype(str).map(patient_id).to_numpy()
    macro_ap_bootstrap: list[float] = []
    rng = np.random.default_rng(20261031)
    unique_patients = np.unique(patient_ids)
    patient_rows = {pid: np.flatnonzero(patient_ids == pid) for pid in unique_patients}
    macro_ap_by_label: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for chex_label, nih_label in LABEL_MAP.items():
        target = pd.to_numeric(dataset.rows[chex_label], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(target) & np.isin(target, (0.0, 1.0))
        col = labels.index(nih_label)
        y = target[valid].astype(np.int32)
        p = predictions[valid, col]
        ids = patient_ids[valid]
        ci = patient_bootstrap_ci(y, p, ids)
        scores[chex_label] = {
            "n": int(valid.sum()),
            "positive": int(y.sum()),
            "negative": int(len(y) - y.sum()),
            "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "average_precision": float(average_precision_score(y, p)) if y.sum() else None,
            "f1_at_0_5": float(f1_score(y, p >= 0.5, zero_division=0)),
            "patient_bootstrap_95_ci": ci,
        }
        macro_ap_by_label[chex_label] = (target, predictions[:, col], valid)

    for _ in range(1000):
        selected = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        indices = np.concatenate([patient_rows[pid] for pid in selected])
        aps: list[float] = []
        for target, probability, valid in macro_ap_by_label.values():
            keep = valid[indices]
            yb = target[indices][keep].astype(np.int32)
            pb = probability[indices][keep]
            if yb.sum() > 0:
                aps.append(float(average_precision_score(yb, pb)))
        if aps:
            macro_ap_bootstrap.append(float(np.mean(aps)))
    point_aps = [x["average_precision"] for x in scores.values()
                 if x["average_precision"] is not None]
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "architecture": checkpoint["architecture"],
        "training_images": checkpoint.get("training_images"),
        "selected_epoch": checkpoint.get("best_epoch"),
        "checkpoint_labels": labels,
        "image_count_frontal": len(dataset),
        "patient_count": len(unique_patients),
        "macro_average_precision_patient_bootstrap_95_ci": (
            [float(x) for x in np.quantile(macro_ap_bootstrap, [0.025, 0.975])]
            if macro_ap_bootstrap else None
        ),
        "macro_average_precision": float(np.mean(point_aps)) if point_aps else None,
        "labels": scores,
    }, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True,
                        help="Directory containing CheXpert/val frontal images")
    parser.add_argument("--labels-csv", type=Path, required=True,
                        help="CheXpert/val_labels.csv from the CheXlocalize release")
    parser.add_argument("--model", action="append", required=True, metavar="NAME=CHECKPOINT",
                        help="Repeat for each frozen candidate, e.g. 8k=path\\best_densenet121.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not args.image_dir.is_dir() or not args.labels_csv.is_file():
        raise FileNotFoundError("CheXpert validation image directory or labels CSV was not found")

    models = []
    for item in args.model:
        if "=" not in item:
            parser.error("Each --model must use NAME=CHECKPOINT")
        name, raw_path = item.split("=", 1)
        path = Path(raw_path)
        if not name or not path.is_file():
            raise FileNotFoundError(f"Invalid model name/checkpoint: {item}")
        models.append((name, path))

    dataset = CheXpertValidation(args.image_dir, args.labels_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    prediction_rows = []
    for name, checkpoint in models:
        result, predictions = evaluate_one(checkpoint, dataset, device, args.batch_size)
        result["candidate_name"] = name
        model_labels = result.pop("checkpoint_labels")
        results.append(result)
        for row_i, path in enumerate(dataset.rows["Path"].astype(str)):
            row = {
                "candidate": name,
                "path": path,
                "patient_id": patient_id(path),
            }
            for chex_label, nih_label in LABEL_MAP.items():
                target = pd.to_numeric(dataset.rows.iloc[row_i][chex_label], errors="coerce")
                row[f"target_{chex_label}"] = None if pd.isna(target) else float(target)
                row[f"score_{chex_label}"] = float(predictions[row_i, model_labels.index(nih_label)])
            prediction_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "chexpert_radiologist_validation_metrics.json").write_text(
        json.dumps({
            "dataset": "CheXlocalize CheXpert validation subset",
            "scope": "external validation; frontal images only; five shared findings",
            "device": str(device),
            "threshold": 0.5,
            "threshold_tuned_on_this_set": False,
            "official_NIH_test_used": False,
            "clinical_use": "not suitable for diagnosis or care decisions",
            "results": results,
        }, indent=2), encoding="utf-8")
    pd.DataFrame(prediction_rows).to_csv(
        args.output_dir / "chexpert_radiologist_validation_predictions.csv", index=False)
    print(json.dumps({"output_dir": str(args.output_dir), "results": results}, indent=2))


if __name__ == "__main__":
    main()
