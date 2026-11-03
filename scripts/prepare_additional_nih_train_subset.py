"""Create a patient-disjoint, bounded NIH training extension for Project 1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def read_patient_ids(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "Patient ID" not in (reader.fieldnames or []):
            raise ValueError(f"Missing Patient ID column: {path}")
        return {row["Patient ID"] for row in reader if row.get("Patient ID")}


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    nih = args.project_root / "data" / "NexviaProjectData" / "nih"
    metadata = nih / "metadata"
    current = metadata / "fresh_15k_patient_split_20261029"
    out = args.output_dir
    official_trainval = set(
        (metadata / "train_val_list.txt").read_text(encoding="utf-8-sig").splitlines()
    )
    exclusion_paths = [
        nih / "shard001_trainval_manifest_20261028.csv",
        nih / "trainval_available_20261028.csv",
        metadata / "colab_t4_subset" / "trainval_manifest.csv",
        metadata / "colab_t4_subset" / "test_manifest.csv",
        current / "train_manifest.csv",
        current / "validation_manifest.csv",
        current / "holdout_manifest.csv",
    ]
    excluded: set[str] = set()
    for path in exclusion_paths:
        if path.exists():
            excluded.update(read_patient_ids(path))

    source = metadata / "Data_Entry_2017_v2020.csv"
    columns, data = read_rows(source)
    per_patient: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in data:
        image = row.get("Image Index", "")
        pid = row.get("Patient ID", "")
        if image in official_trainval and pid and pid not in excluded:
            per_patient[pid].append(row)

    patient_ids = sorted(per_patient)
    if args.count > len(patient_ids):
        raise ValueError(f"Requested {args.count} patients, only {len(patient_ids)} are eligible")
    rng = random.Random(args.seed)
    selected_patients = set(rng.sample(patient_ids, args.count))
    selected = []
    for pid in sorted(selected_patients):
        row = dict(rng.choice(per_patient[pid]))
        row["split"] = "train"
        selected.append(row)
    selected.sort(key=lambda row: row["Image Index"])

    selected_ids = {row["Patient ID"] for row in selected}
    if len(selected_ids) != args.count or selected_ids & excluded:
        raise RuntimeError("Patient-level selection or exclusion validation failed")
    if len({row["Image Index"] for row in selected}) != len(selected):
        raise RuntimeError("Selected image names are not unique")

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "additional_train_manifest.csv", columns + ["split"], selected)
    base_train = read_rows(current / "train_manifest.csv")[1]
    validation = read_rows(current / "validation_manifest.csv")[1]
    merged = base_train + selected + validation
    write_csv(out / "train_15p5k_validation_2p25k_manifest.csv", columns + ["split"], merged)

    label_names = sorted({
        label
        for row in selected
        for label in row.get("Finding Labels", "").split("|")
        if label and label != "No Finding"
    })
    counts = {
        label: sum(label in row.get("Finding Labels", "").split("|") for row in selected)
        for label in label_names
    }
    digest = hashlib.sha256((out / "train_15p5k_validation_2p25k_manifest.csv").read_bytes()).hexdigest()
    audit = {
        "source": "NIH ChestX-ray14 official train_val_list and Data_Entry_2017_v2020.csv",
        "seed": args.seed,
        "additional_training_images": len(selected),
        "additional_training_unique_patients": len(selected_ids),
        "eligible_unused_patients": len(patient_ids),
        "excluded_existing_project_patients": len(excluded),
        "excluded_patients_not_sampled": True,
        "additional_training_label_image_counts": counts,
        "combined_training_images": len(base_train) + len(selected),
        "validation_images": len(validation),
        "train_validation_patient_overlap": bool(
            {row["Patient ID"] for row in base_train + selected}
            & {row["Patient ID"] for row in validation}
        ),
        "holdout_patients_added_to_training": bool(
            selected_ids & read_patient_ids(current / "holdout_manifest.csv")
        ),
        "merged_manifest_sha256": digest,
        "selection": "uniform seeded sample of eligible unique patients, then one seeded random image per selected patient",
        "images_downloaded_or_loaded": False,
    }
    if audit["train_validation_patient_overlap"] or audit["holdout_patients_added_to_training"]:
        raise RuntimeError("Split overlap detected")
    (out / "selection_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
