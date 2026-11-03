"""Create an 800-train/100-validation cohort nested around the 500-image pilot.

This is a bounded learning-curve stage, not a test-set manifest. The pilot's
400 training and 100 validation rows are retained exactly; 400 new training
patients are sampled from the same official train pool with a fixed seed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(label for label in row["Finding Labels"].split("|")
                      if label and label != "No Finding")
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--additional-train-count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    if args.additional_train_count < 1 or args.additional_train_count > 400:
        raise ValueError("This frozen stage may add at most 400 training images")

    nih = args.project_root / "data" / "NexviaProjectData" / "nih"
    source_dir = nih / "metadata" / "fresh_15k_patient_split_20261029"
    source_train = source_dir / "train_manifest.csv"
    source_validation = source_dir / "validation_manifest.csv"
    pilot = nih / "metadata" / "pilot_500_patient_split_20261030" / "pilot_manifest.csv"
    columns, train_rows = read_csv(source_train)
    validation_columns, validation_rows = read_csv(source_validation)
    pilot_columns, pilot_rows = read_csv(pilot)
    if columns != validation_columns or columns != pilot_columns:
        raise ValueError("Source/pilot manifest schemas differ")

    train_ids = {row["Image Index"] for row in train_rows}
    validation_ids = {row["Image Index"] for row in validation_rows}
    pilot_train = [row for row in pilot_rows if row["split"] == "train"]
    pilot_validation = [row for row in pilot_rows if row["split"] == "validation"]
    if len(pilot_train) != 400 or len(pilot_validation) != 100:
        raise ValueError("Expected the frozen 400-train/100-validation pilot")
    if not {row["Image Index"] for row in pilot_train} <= train_ids:
        raise ValueError("Pilot training rows are not a subset of the source train pool")
    if not {row["Image Index"] for row in pilot_validation} <= validation_ids:
        raise ValueError("Pilot validation rows are not a subset of the source validation pool")

    used = {row["Image Index"] for row in pilot_train}
    eligible = [row for row in train_rows if row["Image Index"] not in used]
    additions = random.Random(args.seed).sample(eligible, args.additional_train_count)
    for row in additions:
        row = dict(row)
        row["split"] = "train"
        pilot_train.append(row)

    combined = sorted(pilot_train + pilot_validation,
                      key=lambda row: (row["split"], row["Image Index"]))
    patients_by_split = {
        split: {row["Patient ID"] for row in combined if row["split"] == split}
        for split in ("train", "validation")
    }
    if len(patients_by_split["train"]) != len(pilot_train):
        raise ValueError("Training patients are not unique")
    if len(patients_by_split["validation"]) != len(pilot_validation):
        raise ValueError("Validation patients are not unique")
    if patients_by_split["train"] & patients_by_split["validation"]:
        raise ValueError("Train/validation patients overlap")
    if {row["split"] for row in combined} != {"train", "validation"}:
        raise ValueError("Unexpected split in stage manifest")
    if len(combined) > 900:
        raise ValueError("Stage exceeds the frozen 900-image budget")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "nested_train800_val100_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(combined)
    audit = {
        "purpose": "controlled learning-curve stage after freezing the DenseNet-121 reference architecture",
        "total_images": len(combined),
        "train_images": len(pilot_train),
        "validation_images": len(pilot_validation),
        "train_unique_patients": len(patients_by_split["train"]),
        "validation_unique_patients": len(patients_by_split["validation"]),
        "patient_overlap": False,
        "test_rows": 0,
        "existing_500_pilot_rows_retained": 500,
        "additional_train_rows": len(additions),
        "additional_train_seed": args.seed,
        "validation_set_unchanged_from_500_pilot": True,
        "train_positive_image_counts": label_counts(pilot_train),
        "validation_positive_image_counts": label_counts(pilot_validation),
        "source_train_sha256": sha256(source_train),
        "source_validation_sha256": sha256(source_validation),
        "pilot_manifest_sha256": sha256(pilot),
        "stage_manifest_sha256": sha256(manifest),
    }
    (args.output_dir / "selection_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
