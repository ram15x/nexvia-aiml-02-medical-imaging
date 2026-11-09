"""Map the NIH sample manifest onto the column names the classifier evaluator expects.

Deterministic: the output depends only on the input manifest, so the DVC stage that
runs it is reproducible from the tracked sample manifest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED = {"image_index", "patient_id", "finding_labels", "split", "patient_age", "patient_sex"}


def build(source: Path, target: Path) -> int:
    frame = pd.read_csv(source)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Sample manifest is missing columns: {sorted(missing)}")
    if not frame["split"].eq("official_test").all():
        raise ValueError("Only official_test rows are supported by this sample manifest")
    out = pd.DataFrame({
        "Image Index": frame["image_index"].astype(str),
        "Patient ID": frame["patient_id"].astype(int),
        "Finding Labels": frame["finding_labels"].fillna("No Finding").astype(str),
        "Patient Age": frame["patient_age"],
        "Patient Gender": frame["patient_sex"],
        "View Position": frame["view_position"],
        "split": frame["split"].map({"official_test": "test"}),
    })
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    return len(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    print(f"wrote {build(args.source, args.target)} rows to {args.target}")


if __name__ == "__main__":
    main()
