import copy
import json
from pathlib import Path

from tools.check_baseline import BASELINE, compare

ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "results" / "nih_sample" / "per_label_intervals.json"


def _load():
    return json.loads(BASELINE.read_text()), json.loads(METRICS.read_text())


def test_committed_metrics_match_frozen_baseline():
    baseline, candidate = _load()
    assert compare(baseline, candidate) == []


def test_macro_drop_is_reported_as_regression():
    baseline, candidate = _load()
    worse = copy.deepcopy(candidate)
    worse["macro"]["macro_auroc_defined_labels"] -= 0.05
    problems = compare(baseline, worse)
    assert any("macro AUROC" in problem for problem in problems)


def test_single_label_drop_is_reported_as_regression():
    baseline, candidate = _load()
    worse = copy.deepcopy(candidate)
    label = next(iter(baseline["per_label_auroc"]))
    worse["per_label"][label]["auroc"] -= 0.05
    problems = compare(baseline, worse)
    assert any(label in problem for problem in problems)
