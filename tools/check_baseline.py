"""Compare a fresh classifier evaluation against the frozen baseline.

Usage:
    python tools/check_baseline.py results/nih_sample/per_label_intervals.json

The baseline (results/nih_sample/baseline.json) records the served DenseNet-121 checkpoint's
scores on the 999-image NIH sample. A candidate passes only if its macro AUROC is within
TOLERANCE of the baseline or higher, and no single label's AUROC drops by more than
LABEL_TOLERANCE. Exit code 1 means a regression.
"""
import argparse
import json
import sys
from pathlib import Path

BASELINE = Path(__file__).resolve().parent.parent / "results" / "nih_sample" / "baseline.json"
TOLERANCE = 0.002
LABEL_TOLERANCE = 0.01


def compare(baseline: dict, candidate: dict) -> list[str]:
    problems = []
    base_macro = baseline["macro_auroc"]
    cand_macro = candidate["macro"]["macro_auroc_defined_labels"]
    if cand_macro < base_macro - TOLERANCE:
        problems.append(f"macro AUROC {cand_macro:.4f} is below baseline {base_macro:.4f}")
    for label, base_auroc in baseline["per_label_auroc"].items():
        cand_auroc = candidate["per_label"].get(label, {}).get("auroc")
        if cand_auroc is None:
            problems.append(f"{label}: AUROC missing from candidate")
        elif cand_auroc < base_auroc - LABEL_TOLERANCE:
            problems.append(f"{label}: AUROC {cand_auroc:.4f} is below baseline {base_auroc:.4f}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="per_label_intervals.json from a fresh evaluation")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    problems = compare(baseline, candidate)
    if problems:
        print("REGRESSION")
        for problem in problems:
            print(" -", problem)
        return 1
    print("OK: candidate matches or beats the frozen baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
