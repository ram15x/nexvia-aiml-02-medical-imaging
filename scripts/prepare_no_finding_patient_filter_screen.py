"""Build matched manifests to screen ambiguous NIH "No Finding" negatives.

The patient-level positivity lookup is restricted to the official train/validation
image list. No official-test rows or labels are consulted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has_finding(value: object) -> bool:
    labels = {part.strip() for part in str(value).split("|")}
    return bool(labels - {"", "No Finding", "nan"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--official-trainval-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=20261103)
    args = parser.parse_args()

    source = pd.read_csv(args.manifest, dtype={"Image Index": str})
    required = {"Image Index", "Finding Labels", "Patient ID", "split"}
    if not required <= set(source.columns):
        raise ValueError(f"Manifest is missing columns: {sorted(required - set(source.columns))}")
    if set(source["split"].astype(str)) != {"train", "validation"}:
        raise ValueError("Expected a train/validation-only manifest; test rows are forbidden")
    if source["Image Index"].duplicated().any() or source["Patient ID"].duplicated().any():
        raise ValueError("Input must contain one image per unique patient")
    train_ids = set(source.loc[source.split == "train", "Patient ID"].astype(str))
    val_ids = set(source.loc[source.split == "validation", "Patient ID"].astype(str))
    if train_ids & val_ids:
        raise ValueError("Patient overlap exists between train and validation")

    official_trainval = {
        line.strip()
        for line in args.official_trainval_list.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }
    if not set(source["Image Index"].astype(str)) <= official_trainval:
        raise ValueError("Input contains an image outside the official train/validation list")

    metadata = pd.read_csv(args.metadata, dtype={"Image Index": str})
    metadata = metadata.loc[metadata["Image Index"].astype(str).isin(official_trainval),
                            ["Patient ID", "Finding Labels"]].copy()
    metadata["_has_finding"] = metadata["Finding Labels"].map(has_finding)
    patient_positive = metadata.groupby("Patient ID")["_has_finding"].any()

    source["_is_no_finding"] = source["Finding Labels"].astype(str).eq("No Finding")
    source["_patient_has_other_finding"] = (
        source["Patient ID"].map(patient_positive).fillna(False).astype(bool)
    )
    ambiguous = source["_is_no_finding"] & source["_patient_has_other_finding"]
    filtered = source.loc[~ambiguous].copy()

    matched_rows: list[pd.DataFrame] = []
    split_audit: dict[str, dict[str, int]] = {}
    for split in ("train", "validation"):
        group = source.loc[source["split"] == split]
        positive = group.loc[~group["_is_no_finding"]]
        all_negative = group.loc[group["_is_no_finding"]]
        clean_negative = all_negative.loc[~all_negative["_patient_has_other_finding"]]
        if len(clean_negative) > len(all_negative):
            raise AssertionError("Filtered negative count exceeds the original count")
        sampled_negative = all_negative.sample(
            n=len(clean_negative), random_state=args.selection_seed
        )
        baseline_rows = pd.concat([positive, sampled_negative], ignore_index=True)
        candidate_rows = pd.concat([positive, clean_negative], ignore_index=True)
        if len(baseline_rows) != len(candidate_rows):
            raise AssertionError("Matched train/validation cohorts differ in size")
        if set(baseline_rows.loc[~baseline_rows["_is_no_finding"], "Image Index"]) != set(
            candidate_rows.loc[~candidate_rows["_is_no_finding"], "Image Index"]
        ):
            raise AssertionError("Positive examples differ between matched cohorts")
        baseline_rows["split"] = split
        candidate_rows["split"] = split
        # Both models must be scored on the exact same clean validation cases.
        # Only the training negatives differ between the two manifests.
        matched_rows.append(baseline_rows if split == "train" else candidate_rows)
        split_audit[split] = {
            "input_images": int(len(group)),
            "input_no_finding": int(len(all_negative)),
            "ambiguous_no_finding_removed_from_candidate": int(
                (group["_is_no_finding"] & group["_patient_has_other_finding"]).sum()
            ),
            "clean_no_finding": int(len(clean_negative)),
            "positive_images_kept_in_both": int(len(positive)),
            "matched_images_per_cohort": int(len(candidate_rows)),
        }

    baseline = pd.concat(matched_rows, ignore_index=True)
    candidate = filtered.loc[filtered["split"].isin(["train", "validation"])].copy()
    baseline = baseline.sort_values(["split", "Image Index"]).drop(
        columns=["_is_no_finding", "_patient_has_other_finding"]
    )
    candidate = candidate.sort_values(["split", "Image Index"]).drop(
        columns=["_is_no_finding", "_patient_has_other_finding"]
    )
    baseline_validation = set(baseline.loc[baseline["split"] == "validation", "Image Index"])
    candidate_validation = set(candidate.loc[candidate["split"] == "validation", "Image Index"])
    if baseline_validation != candidate_validation:
        raise AssertionError("Matched cohorts must share identical validation images")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    baseline_path = args.output_dir / "manifest_matched_unfiltered.csv"
    candidate_path = args.output_dir / "manifest_patient_filtered.csv"
    baseline.to_csv(baseline_path, index=False)
    candidate.to_csv(candidate_path, index=False)
    audit = {
        "rule": "For No Finding rows only, exclude the image when any other image for the same patient in the official train/validation list has a positive finding.",
        "purpose": "Matched-size, same-positive-example screen of patient-level No Finding filtering.",
        "selection_seed_for_unfiltered_negative_sample": args.selection_seed,
        "official_test_consulted": False,
        "input_manifest_sha256": sha256(args.manifest),
        "source_metadata_sha256": sha256(args.metadata),
        "official_trainval_list_sha256": sha256(args.official_trainval_list),
        "patient_overlap_train_validation": False,
        "same_positive_images_in_matched_cohorts": True,
        "validation_cases_identical_across_cohorts": True,
        "split_counts": split_audit,
        "baseline_manifest_sha256": sha256(baseline_path),
        "candidate_manifest_sha256": sha256(candidate_path),
        "caveat": "This is a label-policy hypothesis, not proof that every No Finding image from a patient with another positive is mislabeled. Compare on the same filtered validation rows and retain the original model as reference.",
    }
    (args.output_dir / "selection_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
