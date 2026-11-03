"""Create a seeded, patient-disjoint sample from locally available unused NIH test images.

This prepares a one-time evaluation manifest. It never copies image pixels and
does not inspect model outputs or use the sample for training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-manifest", type=Path, required=True)
    parser.add_argument("--prior-test-manifest", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")

    official = pd.read_csv(args.official_manifest)
    prior_test = pd.read_csv(args.prior_test_manifest)
    development = pd.read_csv(args.development_manifest)
    required = {"Image Index", "Finding Labels", "Patient ID", "split"}
    for path, frame in ((args.official_manifest, official),
                        (args.prior_test_manifest, prior_test),
                        (args.development_manifest, development)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    already_tested_patients = set(prior_test["Patient ID"].astype(str))
    development_patients = set(development["Patient ID"].astype(str))
    image_root = args.image_root.resolve()
    available = official.loc[official["split"].eq("test")].copy()
    available["Patient ID"] = available["Patient ID"].astype(str)
    available = available.loc[
        ~available["Patient ID"].isin(already_tested_patients | development_patients)
    ]
    available = available.loc[
        available["Image Index"].astype(str).map(
            lambda name: (image_root / Path(name).name).is_file()
            and Path(name).name == name
        )
    ]
    if len(available) < args.count:
        raise ValueError(
            f"Only {len(available)} eligible unused official test images are local; "
            f"cannot create the requested {args.count}-image sample."
        )

    sample = available.sample(n=args.count, random_state=args.seed).copy()
    sample["split"] = "test"
    sample = sample[["Image Index", "Finding Labels", "Patient ID", "split"]]
    if sample["Image Index"].duplicated().any():
        raise RuntimeError("Sample contains duplicate image filenames")
    if set(sample["Patient ID"]) & (already_tested_patients | development_patients):
        raise RuntimeError("Sample overlaps a previously used or development patient")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.sort_values("Image Index").to_csv(args.output, index=False)
    print(f"images={len(sample)}")
    print(f"patients={sample['Patient ID'].nunique()}")
    print(f"eligible_unused_images={len(available)}")
    print(f"seed={args.seed}")
    print(f"manifest={args.output.resolve()}")


if __name__ == "__main__":
    main()
