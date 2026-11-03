"""Freeze a small, paired train/validation manifest for the resolution screen.

The script only reads the supplied train/validation manifest. It rejects a test
split, samples training rows once with a recorded seed, and retains the complete
validation split so both image-size arms use identical examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_audit import NIH_LABELS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--train-images", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    frame = pd.read_csv(args.source_manifest)
    required = {"Image Index", "Finding Labels", "Patient ID", "split"}
    if required - set(frame.columns):
        raise ValueError(f"Manifest missing columns: {sorted(required - set(frame.columns))}")
    if set(frame["split"].astype(str)) != {"train", "validation"}:
        raise ValueError("Source must contain train and validation only; test rows are forbidden")
    if frame["Image Index"].duplicated().any():
        raise ValueError("Image Index values must be unique")

    training = frame.loc[frame["split"] == "train"].copy()
    validation = frame.loc[frame["split"] == "validation"].copy()
    if len(training) < args.train_images or validation.empty:
        raise ValueError("Requested training sample is unavailable or validation is empty")
    overlap = set(training["Patient ID"].astype(str)) & set(validation["Patient ID"].astype(str))
    if overlap:
        raise ValueError(f"Source has patient leakage: {len(overlap)} overlapping patients")

    selected_train = training.sample(n=args.train_images, random_state=args.seed)
    selected = pd.concat([selected_train, validation], ignore_index=True)
    selected = selected.sort_values(["split", "Image Index"]).reset_index(drop=True)
    if set(selected["split"].astype(str)) != {"train", "validation"}:
        raise AssertionError("Selected manifest unexpectedly contains a non-development split")

    def label_counts(rows: pd.DataFrame) -> dict[str, int]:
        counts = {label: 0 for label in NIH_LABELS}
        for raw in rows["Finding Labels"].astype(str):
            for label in set(raw.split("|")) - {"No Finding"}:
                if label not in counts:
                    raise ValueError(f"Unknown label in manifest: {label}")
                counts[label] += 1
        return counts

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_manifest, index=False)
    audit = {
        "source_manifest": str(args.source_manifest),
        "source_manifest_sha256": sha256(args.source_manifest),
        "selected_manifest": str(args.output_manifest),
        "selected_manifest_sha256": sha256(args.output_manifest),
        "selection_seed": args.seed,
        "selection_method": "uniform image-row sample from the existing train split; patient IDs remain wholly separated from validation",
        "source_test_rows_present": False,
        "selected_rows": int(len(selected)),
        "train_images": int(len(selected_train)),
        "train_patients": int(selected_train["Patient ID"].nunique()),
        "validation_images": int(len(validation)),
        "validation_patients": int(validation["Patient ID"].nunique()),
        "patient_overlap": 0,
        "train_positive_image_counts": label_counts(selected_train),
        "validation_positive_image_counts": label_counts(validation),
    }
    args.output_audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
