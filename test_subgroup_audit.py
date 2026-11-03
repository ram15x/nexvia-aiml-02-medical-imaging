from __future__ import annotations

import unittest

import numpy as np

from scripts.audit_subgroup_performance import (
    bootstrap_group,
    eligible_labels,
    metrics_for,
)
from data_audit import NIH_LABELS


class SubgroupAuditTests(unittest.TestCase):
    def test_metrics_for_perfect_rank_and_threshold_predictions(self) -> None:
        y = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8)
        scores = np.array([[0.1, 0.1], [0.2, 0.9], [0.9, 0.2], [0.8, 0.8]])
        result = metrics_for(y, scores, ["a", "b"])
        self.assertAlmostEqual(result["macro_auroc"], 1.0)
        self.assertAlmostEqual(result["macro_ap"], 1.0)
        self.assertAlmostEqual(result["micro_sensitivity"], 1.0)
        self.assertAlmostEqual(result["micro_fpr"], 0.0)
        self.assertAlmostEqual(result["micro_f1"], 1.0)

    def test_support_filter_requires_minimum_positive_and_negative_examples(self) -> None:
        ids = np.array(["F"] * 30 + ["M"] * 30)
        y = np.zeros((60, len(NIH_LABELS)), dtype=np.int8)
        y[:10, 0] = 1
        y[30:40, 0] = 1
        y[:9, 1] = 1
        y[30:40, 1] = 1
        included = eligible_labels(y, ids, [("Female", "F"), ("Male", "M")])
        self.assertEqual(included, [0])

    def test_bootstrap_returns_fixed_seeded_metric_samples(self) -> None:
        y = np.array([[0], [0], [1], [1]] * 10, dtype=np.int8)
        scores = np.array([[0.1], [0.2], [0.8], [0.9]] * 10, dtype=np.float64)
        indices = np.arange(len(y))
        first = bootstrap_group(y, scores, indices, ["a"], 100, np.random.default_rng(7))
        second = bootstrap_group(y, scores, indices, ["a"], 100, np.random.default_rng(7))
        self.assertEqual(int(first["valid_replicates"][0]), 100)
        np.testing.assert_array_equal(first["macro_auroc"], second["macro_auroc"])
        np.testing.assert_array_equal(first["micro_f1"], second["micro_f1"])


if __name__ == "__main__":
    unittest.main()
