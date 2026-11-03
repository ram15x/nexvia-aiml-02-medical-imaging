"""Make a 400-train/100-validation engineering pilot from the frozen NIH split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=400)
    parser.add_argument("--validation-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()
    if args.train_count + args.validation_count > 500:
        raise ValueError("Engineering pilot is capped at 500 total images")

    source_dir = args.project_root / "data" / "NexviaProjectData" / "nih" / "metadata" / "fresh_15k_patient_split_20261029"
    train_path = source_dir / "train_manifest.csv"
    validation_path = source_dir / "validation_manifest.csv"
    columns, train_rows = load_csv(train_path)
    validation_columns, validation_rows = load_csv(validation_path)
    if columns != validation_columns:
        raise ValueError("Source train and validation manifest schemas differ")
    if len({row["Patient ID"] for row in train_rows}) != len(train_rows):
        raise ValueError("Source training manifest has repeated patients")
    if len({row["Patient ID"] for row in validation_rows}) != len(validation_rows):
        raise ValueError("Source validation manifest has repeated patients")
    if {row["Patient ID"] for row in train_rows} & {row["Patient ID"] for row in validation_rows}:
        raise ValueError("Source train and validation patients overlap")

    rng = random.Random(args.seed)
    selected_train = rng.sample(train_rows, args.train_count)
    selected_validation = rng.sample(validation_rows, args.validation_count)
    for row in selected_train:
        row["split"] = "train"
    for row in selected_validation:
        row["split"] = "validation"
    combined = sorted(selected_train + selected_validation, key=lambda row: (row["split"], row["Image Index"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "pilot_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(combined)

    labels = sorted({
        label
        for row in combined
        for label in row["Finding Labels"].split("|")
        if label and label != "No Finding"
    })
    counts = {
        split: {
            label: sum(label in row["Finding Labels"].split("|") for row in combined if row["split"] == split)
            for label in labels
        }
        for split in ("train", "validation")
    }
    audit = {
        "purpose": "engineering pipeline smoke check only; not a model-performance estimate",
        "total_images": len(combined),
        "maximum_allowed_images": 500,
        "train_images": len(selected_train),
        "validation_images": len(selected_validation),
        "unique_patients": len({row["Patient ID"] for row in combined}),
        "patient_overlap_train_validation": bool(
            {row["Patient ID"] for row in selected_train}
            & {row["Patient ID"] for row in selected_validation}
        ),
        "reserved_holdout_used": False,
        "selection": "uniform seeded sample from the existing patient-disjoint train and validation manifests",
        "seed": args.seed,
        "label_image_counts": counts,
        "source_train_manifest_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "source_validation_manifest_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
        "pilot_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    if audit["patient_overlap_train_validation"]:
        raise RuntimeError("Pilot split has patient overlap")
    (args.output_dir / "selection_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
