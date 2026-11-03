"""Validate local NIH ChestX-ray14 and MSD Task09 metadata before training.

This module writes metadata manifests only. It does not copy image data or make
clinical predictions. Keep source datasets outside version control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

NIH_LABELS = (
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
)


def _read_names(path: Path) -> set[str]:
    names = {line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
             if line.strip()}
    if not names:
        raise ValueError(f"Split file is empty: {path}")
    if any(Path(name).name != name or name in {".", ".."} for name in names):
        raise ValueError(f"Split file must contain bare filenames only: {path}")
    return names


def _patient_id(filename: str) -> str:
    # NIH names follow 00000001_000.png; split at the first underscore.
    if "_" not in filename:
        raise ValueError(f"Unexpected NIH filename (patient ID missing): {filename}")
    return filename.split("_", 1)[0]


def patient_split(patient_id: str, validation_fraction: float = 0.15) -> str:
    """Stable patient-level train/validation split, independent of row order."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    bucket = int(hashlib.sha256(patient_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "validation" if bucket < validation_fraction else "train"


def build_nih_manifest(metadata_csv: Path, train_val_list: Path, test_list: Path,
                       image_root: Path, output_dir: Path,
                       validation_fraction: float = 0.15,
                       check_files: bool = False) -> dict:
    """Build a patient-disjoint train/validation/test manifest from NIH metadata."""
    metadata = pd.read_csv(metadata_csv)
    required = {"Image Index", "Finding Labels", "Patient ID"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    if metadata.empty or metadata["Image Index"].isna().any():
        raise ValueError("Metadata is empty or contains missing image names")
    if metadata["Image Index"].duplicated().any():
        raise ValueError("Image Index must be unique in NIH metadata")

    trainval, test = _read_names(train_val_list), _read_names(test_list)
    overlap_names = trainval & test
    if overlap_names:
        raise ValueError(f"Image appears in both official splits: {sorted(overlap_names)[:3]}")
    indexed = set(metadata["Image Index"].astype(str))
    unknown = (trainval | test) - indexed
    if unknown:
        raise ValueError(f"Split files refer to images absent from metadata: {sorted(unknown)[:3]}")
    missing_from_splits = indexed - trainval - test
    if missing_from_splits:
        raise ValueError(f"Metadata images absent from supplied splits: {len(missing_from_splits)}")

    rows = metadata[["Image Index", "Finding Labels", "Patient ID"]].copy()
    rows["Image Index"] = rows["Image Index"].astype(str)
    rows["Patient ID"] = rows["Patient ID"].astype(str)
    rows["split"] = rows["Image Index"].map(
        lambda name: "test" if name in test else patient_split(_patient_id(name), validation_fraction))
    overlap_patients = set(rows.loc[rows["split"] == "test", "Patient ID"]) & set(
        rows.loc[rows["split"].isin(["train", "validation"]), "Patient ID"])
    if overlap_patients:
        raise ValueError(f"Patient leakage across official test split: {len(overlap_patients)} patients")

    if check_files:
        absent = [name for name in rows["Image Index"] if not (image_root / name).is_file()]
        if absent:
            raise FileNotFoundError(f"{len(absent)} image files missing; first: {absent[0]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_dir / "nih_manifest.csv", index=False)
    summary = {
        "dataset": "NIH ChestX-ray14",
        "images": int(len(rows)),
        "patients": int(rows["Patient ID"].nunique()),
        "split_images": {str(k): int(v) for k, v in rows["split"].value_counts().items()},
        "positive_label_image_counts": _label_counts(rows["Finding Labels"]),
        "label_policy": "No Finding is an all-zero vector; other findings are multi-label positives; uncertain labels are not inferred.",
        "validation_fraction_of_official_trainval_patients": validation_fraction,
        "check_files": check_files,
    }
    (output_dir / "nih_manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _label_counts(label_rows: Iterable[str]) -> dict[str, int]:
    counts = {label: 0 for label in NIH_LABELS}
    for value in label_rows:
        labels = set(str(value).split("|"))
        labels.discard("No Finding")
        unknown = labels - set(NIH_LABELS)
        if unknown:
            raise ValueError(f"Unknown NIH labels found: {sorted(unknown)}")
        for label in labels:
            counts[label] += 1
    return counts


def build_msd_manifest(dataset_json: Path, dataset_root: Path,
                       output_dir: Path, check_files: bool = False) -> dict:
    """Validate Task09 train image/label pairing and emit a metadata manifest."""
    source = json.loads(dataset_json.read_text(encoding="utf-8"))
    training = source.get("training")
    if not isinstance(training, list) or not training:
        raise ValueError("MSD dataset.json must contain a nonempty training list")
    rows = []
    seen: set[str] = set()
    for record in training:
        image, label = record.get("image"), record.get("label")
        if not image or not label:
            raise ValueError("Every MSD training record must have image and label")
        case_id = Path(image).name
        if case_id in seen:
            raise ValueError(f"Duplicate MSD case: {case_id}")
        seen.add(case_id)
        for relative in (image, label):
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError(f"Unsafe relative path in dataset.json: {relative}")
            if check_files and not (dataset_root / relative).is_file():
                raise FileNotFoundError(f"Missing MSD file: {dataset_root / relative}")
        rows.append({"case_id": case_id, "image": image, "label": label})
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "msd_manifest.csv", index=False)
    summary = {"dataset": source.get("name", "MSD Task09 Spleen"),
               "training_cases": len(rows), "check_files": check_files,
               "note": "The MSD training list does not define a validation/test policy. Create and freeze a case-level split before tuning; do not split patches across folds."}
    (output_dir / "msd_manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="dataset", required=True)
    nih = commands.add_parser("nih")
    nih.add_argument("--metadata", type=Path, required=True)
    nih.add_argument("--train-val-list", type=Path, required=True)
    nih.add_argument("--test-list", type=Path, required=True)
    nih.add_argument("--image-root", type=Path, required=True)
    nih.add_argument("--output", type=Path, required=True)
    nih.add_argument("--validation-fraction", type=float, default=0.15)
    nih.add_argument("--check-files", action="store_true")
    msd = commands.add_parser("msd")
    msd.add_argument("--dataset-json", type=Path, required=True)
    msd.add_argument("--dataset-root", type=Path, required=True)
    msd.add_argument("--output", type=Path, required=True)
    msd.add_argument("--check-files", action="store_true")
    args = parser.parse_args()
    if args.dataset == "nih":
        result = build_nih_manifest(args.metadata, args.train_val_list, args.test_list,
                                    args.image_root, args.output,
                                    args.validation_fraction, args.check_files)
    else:
        result = build_msd_manifest(args.dataset_json, args.dataset_root, args.output, args.check_files)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
