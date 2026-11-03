"""Run a descriptive demographic subgroup audit on saved 8k predictions.

This script reads prediction and metadata tables only. It does not load images,
train a model, change thresholds, or write patient-level outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_audit import NIH_LABELS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COHORT = ROOT / "data/NexviaProjectData/nih/candidate_holdout_additional_5k_20261031"
DEFAULT_PREDICTIONS = DEFAULT_COHORT / "predictions_additional_5k.csv"
DEFAULT_MANIFEST = DEFAULT_COHORT / "manifest_additional_5000_validation.csv"
DEFAULT_REFERENCE = DEFAULT_COHORT / "metrics_additional_5k.json"
DEFAULT_ACTIVE_CHECKPOINT = ROOT / "artifacts/best_densenet121.pt"
DEFAULT_CANDIDATE_CHECKPOINT = (
    ROOT / "data/NexviaProjectData/nih/learning_curve_8000_train_1000_val_20261031"
    / "densenet121_train8000/best_densenet121.pt"
)
THRESHOLD = 0.5
MIN_POSITIVES_PER_GROUP = 10
BOOTSTRAP_SEED = 20261101

GROUPS: dict[str, list[tuple[str, str]]] = {
    "sex": [("Female", "F"), ("Male", "M")],
    "age": [("0-39", "0-39"), ("40-59", "40-59"), ("60+", "60+")],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint_weights(active_path: Path, candidate_path: Path) -> dict[str, Any]:
    import torch

    active = torch.load(active_path, map_location="cpu", weights_only=True)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
    if active.get("architecture") != "densenet121" or candidate.get("architecture") != "densenet121":
        raise ValueError("Both checkpoints must be DenseNet-121.")
    if active.get("labels") != candidate.get("labels") or active.get("labels") != list(NIH_LABELS):
        raise ValueError("Checkpoint label order does not match the project label order.")
    active_state = active.get("state_dict", {})
    candidate_state = candidate.get("state_dict", {})
    if active_state.keys() != candidate_state.keys():
        raise ValueError("Active and prediction-reference checkpoint keys differ.")
    if not all(torch.equal(active_state[key], candidate_state[key]) for key in active_state):
        raise ValueError("Saved predictions do not match the active checkpoint weights.")
    return {
        "active_checkpoint_sha256": sha256_file(active_path),
        "prediction_reference_checkpoint_sha256": sha256_file(candidate_path),
        "state_dict_tensor_exact_match": True,
    }


def load_cohort(
    predictions_path: Path,
    manifest_path: Path,
    reference_path: Path,
    active_checkpoint_path: Path,
    candidate_checkpoint_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    predictions = pd.read_csv(predictions_path)
    manifest = pd.read_csv(manifest_path)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    required_manifest = {"Image Index", "Patient ID", "Patient Age", "Patient Sex", "Finding Labels"}
    missing = required_manifest - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    if predictions["Image Index"].duplicated().any() or manifest["Image Index"].duplicated().any():
        raise ValueError("Image Index must be unique in both input tables.")
    if predictions["Patient ID"].duplicated().any() or manifest["Patient ID"].duplicated().any():
        raise ValueError("This audit expects one image per patient.")

    joined = predictions.merge(
        manifest[list(required_manifest)],
        on="Image Index",
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_prediction", "_manifest"),
    )
    if len(joined) != 5000 or not (joined["_merge"] == "both").all():
        raise ValueError("Prediction rows and the 5,000-row manifest do not match exactly.")
    if not np.array_equal(
        joined["Patient ID_prediction"].to_numpy(),
        joined["Patient ID_manifest"].to_numpy(),
    ):
        raise ValueError("Prediction and manifest patient IDs are misaligned.")
    joined = joined.drop(columns="_merge").rename(
        columns={
            "Patient ID_prediction": "Patient ID",
            "Patient Age": "Patient Age",
            "Patient Sex": "Patient Sex",
        }
    )

    label_sets = joined["Finding Labels"].fillna("").astype(str).str.split("|")
    targets: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for label in NIH_LABELS:
        target_col = f"target_{label}"
        score_col = f"score_train8000_{label}"
        if target_col not in joined or score_col not in joined:
            raise ValueError(f"Missing saved 8k prediction columns for {label}.")
        expected = label_sets.map(lambda items: int(label in items)).to_numpy(dtype=np.int8)
        actual = pd.to_numeric(joined[target_col], errors="raise").to_numpy(dtype=np.int8)
        if not np.array_equal(actual, expected):
            raise ValueError(f"Saved target column does not match the metadata labels: {label}.")
        targets.append(actual)
        scores.append(pd.to_numeric(joined[score_col], errors="raise").to_numpy(dtype=np.float64))

    y = np.column_stack(targets)
    score_matrix = np.column_stack(scores)
    if not np.isfinite(score_matrix).all() or ((score_matrix < 0) | (score_matrix > 1)).any():
        raise ValueError("Prediction scores must be finite values in [0, 1].")

    age = pd.to_numeric(joined["Patient Age"], errors="raise")
    if age.isna().any() or (age < 0).any():
        raise ValueError("Age metadata contains missing or negative values.")
    joined["age_group"] = pd.cut(
        age, bins=[-1, 39, 59, 130], labels=["0-39", "40-59", "60+"]
    ).astype(str)
    if joined["age_group"].eq("nan").any():
        raise ValueError("Age metadata falls outside the declared 0-130 range.")
    if not joined["Patient Sex"].isin(["F", "M"]).all():
        raise ValueError("Sex metadata contains values outside F/M; define handling before analysis.")

    audit = reference.get("audit", {})
    expected_manifest_hash = audit.get("evaluation_manifest_sha256")
    actual_manifest_hash = sha256_file(manifest_path)
    if expected_manifest_hash and expected_manifest_hash != actual_manifest_hash:
        raise ValueError("Manifest hash does not match the recorded evaluation audit.")
    candidate_result = reference.get("results", {}).get("train8000", {})
    if not candidate_result:
        raise ValueError("Reference metrics do not contain the train8000 candidate.")
    weights = verify_checkpoint_weights(active_checkpoint_path, candidate_checkpoint_path)

    global_auc = float(roc_auc_score(y, score_matrix, average="macro"))
    global_ap = float(average_precision_score(y, score_matrix, average="macro"))
    global_f1 = float(f1_score(y.ravel(), (score_matrix >= THRESHOLD).ravel(), zero_division=0))
    expected_metrics = {
        "macro_auroc_defined_labels": global_auc,
        "macro_auprc_defined_labels": global_ap,
        "micro_f1_at_0_5": global_f1,
    }
    for key, actual in expected_metrics.items():
        if not np.isclose(actual, float(candidate_result[key]), atol=1e-6, rtol=0):
            raise ValueError(f"Global {key} does not reproduce the saved reference metric.")

    joined["Patient ID"] = joined["Patient ID"].astype(str)
    metadata = {
        "rows": int(len(joined)),
        "unique_patients": int(joined["Patient ID"].nunique()),
        "sex_counts": {str(k): int(v) for k, v in joined["Patient Sex"].value_counts().to_dict().items()},
        "age_group_counts": {
            str(k): int(v) for k, v in joined["age_group"].value_counts().sort_index().to_dict().items()
        },
        "manifest_sha256": actual_manifest_hash,
        "predictions_sha256": sha256_file(predictions_path),
        "global_metric_reproduction": expected_metrics,
        "checkpoint_identity": weights,
        "threshold": THRESHOLD,
    }
    return joined, y, score_matrix, metadata


def metrics_for(y: np.ndarray, scores: np.ndarray, labels: list[str]) -> dict[str, float]:
    decisions = scores >= THRESHOLD
    tp = np.logical_and(decisions, y == 1).sum(dtype=np.int64)
    fp = np.logical_and(decisions, y == 0).sum(dtype=np.int64)
    fn = np.logical_and(~decisions, y == 1).sum(dtype=np.int64)
    tn = np.logical_and(~decisions, y == 0).sum(dtype=np.int64)
    positives = tp + fn
    negatives = fp + tn
    return {
        "macro_auroc": float(roc_auc_score(y, scores, average="macro")),
        "macro_ap": float(average_precision_score(y, scores, average="macro")),
        "mean_label_prevalence": float(y.mean(axis=0).mean()),
        "micro_sensitivity": float(tp / positives) if positives else float("nan"),
        "micro_fpr": float(fp / negatives) if negatives else float("nan"),
        "micro_precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "micro_f1": float(f1_score(y.ravel(), decisions.ravel(), zero_division=0)),
        "positive_decisions": int(tp + fp),
        "n_labels": len(labels),
    }


def eligible_labels(y: np.ndarray, group_ids: np.ndarray, groups: list[tuple[str, str]]) -> list[int]:
    eligible: list[int] = []
    for label_idx, label in enumerate(NIH_LABELS):
        is_eligible = True
        for name, code in groups:
            if group_ids.dtype.kind in "iu":
                mask = group_ids == int(code)
            else:
                mask = group_ids == code
            n = int(mask.sum())
            positives = int(y[mask, label_idx].sum())
            negatives = n - positives
            if positives < MIN_POSITIVES_PER_GROUP or negatives < MIN_POSITIVES_PER_GROUP:
                is_eligible = False
        if is_eligible:
            eligible.append(label_idx)
    return eligible


def bootstrap_group(
    y: np.ndarray,
    scores: np.ndarray,
    indices: np.ndarray,
    labels: list[str],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    metric_names = ["macro_auroc", "macro_ap", "mean_label_prevalence", "micro_sensitivity", "micro_fpr", "micro_precision", "micro_f1"]
    out = {name: np.empty(n_bootstrap, dtype=np.float64) for name in metric_names}
    valid = 0
    attempts = 0
    while valid < n_bootstrap and attempts < n_bootstrap * 2:
        attempts += 1
        sampled = rng.choice(indices, size=len(indices), replace=True)
        ys, ss = y[sampled], scores[sampled]
        positives = ys.sum(axis=0)
        if np.any(positives == 0) or np.any(positives == len(sampled)):
            continue
        result = metrics_for(ys, ss, labels)
        for name in metric_names:
            out[name][valid] = result[name]
        valid += 1
    if valid < max(100, int(n_bootstrap * 0.9)):
        raise RuntimeError(f"Only {valid}/{n_bootstrap} bootstrap replicates were valid.")
    return {**{key: values[:valid] for key, values in out.items()}, "valid_replicates": np.asarray([valid])}


def interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def add_ci(row: dict[str, Any], samples: dict[str, np.ndarray]) -> dict[str, Any]:
    for metric in samples:
        if metric == "valid_replicates":
            continue
        low, high = interval(samples[metric])
        row[f"{metric}_ci_low"] = low
        row[f"{metric}_ci_high"] = high
    row["bootstrap_valid_replicates"] = int(samples["valid_replicates"][0])
    return row


def per_label_rows(frame: pd.DataFrame, y: np.ndarray, scores: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension, specs in GROUPS.items():
        column = "Patient Sex" if dimension == "sex" else "age_group"
        for group_name, group_code in specs:
            mask = frame[column].to_numpy() == group_code
            group_y, group_scores = y[mask], scores[mask]
            for label_idx, label in enumerate(NIH_LABELS):
                truth = group_y[:, label_idx]
                pred = group_scores[:, label_idx] >= THRESHOLD
                positives = int(truth.sum())
                negatives = int(len(truth) - positives)
                tp = int(np.logical_and(pred, truth == 1).sum())
                fp = int(np.logical_and(pred, truth == 0).sum())
                fn = positives - tp
                tn = negatives - fp
                rows.append({
                    "dimension": dimension,
                    "group": group_name,
                    "label": label,
                    "n_patients": int(mask.sum()),
                    "positive_support": positives,
                    "negative_support": negatives,
                    "prevalence": positives / int(mask.sum()),
                    "auroc": float(roc_auc_score(truth, group_scores[:, label_idx])) if positives and negatives else None,
                    "average_precision": float(average_precision_score(truth, group_scores[:, label_idx])) if positives else None,
                    "sensitivity_at_0_5": tp / positives if positives else None,
                    "fpr_at_0_5": fp / negatives if negatives else None,
                    "precision_at_0_5": tp / (tp + fp) if tp + fp else 0.0,
                    "included_in_dimension_macro": False,
                })
    return rows


def svg_chart(summary_rows: list[dict[str, Any]], output: Path) -> None:
    width, height = 1200, 780
    colors = {"Female": "#3366cc", "Male": "#dc3912", "0-39": "#3366cc", "40-59": "#109618", "60+": "#dc3912"}
    panels = [
        ("sex", "Sex groups · macro AUROC", "macro_auroc", 0.45, 1.0, 45, 55),
        ("sex", "Sex groups · macro average precision", "macro_ap", 0.0, 0.40, 640, 55),
        ("age", "Age groups · macro AUROC", "macro_auroc", 0.45, 1.0, 45, 420),
        ("age", "Age groups · macro average precision", "macro_ap", 0.0, 0.40, 640, 420),
    ]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#152238}.title{font-size:19px;font-weight:bold}.axis{font-size:12px}.note{font-size:13px;fill:#475569}</style>',
             '<text x="45" y="28" class="title">8k DenseNet subgroup performance · 5,000-patient NIH development cohort</text>']
    for dimension, title, metric, y_min, y_max, x0, y0 in panels:
        panel_w, panel_h = 510, 300
        left, top = x0 + 55, y0 + 38
        plot_w, plot_h = 405, 205
        parts.append(f'<text x="{x0}" y="{y0 + 18}" class="title">{html.escape(title)}</text>')
        for tick_i in range(6):
            val = y_min + (y_max - y_min) * tick_i / 5
            y = top + plot_h * (1 - tick_i / 5)
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#dbe3ee"/>')
            parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="axis">{val:.2f}</text>')
        rows = [row for row in summary_rows if row["dimension"] == dimension]
        for i, row in enumerate(rows):
            x = left + plot_w * (i + 0.5) / len(rows)
            value = row[metric]
            low, high = row[f"{metric}_ci_low"], row[f"{metric}_ci_high"]
            map_y = lambda v: top + plot_h * (1 - (v - y_min) / (y_max - y_min))
            y, y_low, y_high = map_y(value), map_y(low), map_y(high)
            color = colors[row["group"]]
            parts.append(f'<line x1="{x:.1f}" y1="{y_low:.1f}" x2="{x:.1f}" y2="{y_high:.1f}" stroke="{color}" stroke-width="2"/>')
            parts.append(f'<line x1="{x-7:.1f}" y1="{y_low:.1f}" x2="{x+7:.1f}" y2="{y_low:.1f}" stroke="{color}" stroke-width="2"/>')
            parts.append(f'<line x1="{x-7:.1f}" y1="{y_high:.1f}" x2="{x+7:.1f}" y2="{y_high:.1f}" stroke="{color}" stroke-width="2"/>')
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"/>')
            parts.append(f'<text x="{x:.1f}" y="{top+plot_h+22}" text-anchor="middle" class="axis">{html.escape(row["group"])} (n={row["n_patients"]})</text>')
        parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#334155"/>')
        parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#334155"/>')
    parts.append('<text x="45" y="765" class="note">Dots: subgroup estimate; whiskers: 95% patient-bootstrap interval. Common supported label sets differ by dimension. AP depends on subgroup prevalence.</text>')
    parts.append('</svg>')
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference-metrics", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--active-checkpoint", type=Path, default=DEFAULT_ACTIVE_CHECKPOINT)
    parser.add_argument("--prediction-checkpoint", type=Path, default=DEFAULT_CANDIDATE_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/subgroup_performance_2026-11-01")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates for this descriptive interval report.")

    frame, y, scores, metadata = load_cohort(
        args.predictions, args.manifest, args.reference_metrics,
        args.active_checkpoint, args.prediction_checkpoint,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []
    included_by_dimension: dict[str, list[str]] = {}
    group_bootstraps: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for dimension, specs in GROUPS.items():
        column = "Patient Sex" if dimension == "sex" else "age_group"
        eligible = eligible_labels(y, frame[column].to_numpy(), specs)
        if not eligible:
            raise ValueError(f"No labels meet the minimum support rule for {dimension}.")
        included = [NIH_LABELS[i] for i in eligible]
        included_by_dimension[dimension] = included
        group_bootstraps[dimension] = {}
        for group_name, group_code in specs:
            mask = frame[column].to_numpy() == group_code
            indices = np.flatnonzero(mask)
            group_y = y[mask][:, eligible]
            group_scores = scores[mask][:, eligible]
            point = metrics_for(group_y, group_scores, included)
            boot = bootstrap_group(y[:, eligible], scores[:, eligible], indices, included, args.bootstrap, rng)
            group_bootstraps[dimension][group_name] = boot
            row: dict[str, Any] = {
                "dimension": dimension,
                "group": group_name,
                "n_patients": int(mask.sum()),
                "positive_label_assignments": int(y[mask].sum()),
                "included_labels": "|".join(included),
                "minimum_positive_support_each_label": MIN_POSITIVES_PER_GROUP,
                **point,
            }
            summary_rows.append(add_ci(row, boot))

        for first_idx in range(len(specs)):
            for second_idx in range(first_idx + 1, len(specs)):
                first, second = specs[first_idx][0], specs[second_idx][0]
                first_point = next(row for row in summary_rows if row["dimension"] == dimension and row["group"] == first)
                second_point = next(row for row in summary_rows if row["dimension"] == dimension and row["group"] == second)
                row = {"dimension": dimension, "contrast": f"{first} minus {second}"}
                for metric in ["macro_auroc", "macro_ap", "micro_sensitivity", "micro_fpr", "micro_f1"]:
                    delta = float(first_point[metric] - second_point[metric])
                    boot_delta = (
                        group_bootstraps[dimension][first][metric]
                        - group_bootstraps[dimension][second][metric]
                    )
                    low, high = interval(boot_delta)
                    row[f"{metric}_difference"] = delta
                    row[f"{metric}_difference_ci_low"] = low
                    row[f"{metric}_difference_ci_high"] = high
                difference_rows.append(row)

    label_rows = per_label_rows(frame, y, scores)
    for row in label_rows:
        row["included_in_dimension_macro"] = row["label"] in included_by_dimension[row["dimension"]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "subgroup_summary.csv", index=False)
    pd.DataFrame(difference_rows).to_csv(args.output_dir / "subgroup_differences.csv", index=False)
    pd.DataFrame(label_rows).to_csv(args.output_dir / "subgroup_per_label.csv", index=False)
    svg_chart(summary_rows, args.output_dir / "subgroup_metrics.svg")
    payload = {
        "analysis": "descriptive post-selection demographic subgroup audit",
        "cohort": "5,000-patient NIH ChestX-ray14 train/validation development cohort",
        "model": "current deployed 8k DenseNet-121; candidate and active state_dict tensors match exactly",
        "score_column_family": "score_train8000_*",
        "threshold": THRESHOLD,
        "age_groups": ["0-39", "40-59", "60+"],
        "sex_groups": ["Female", "Male"],
        "bootstrap": {"replicates": args.bootstrap, "seed": BOOTSTRAP_SEED, "unit": "patient within subgroup"},
        "minimum_positive_support_per_label_per_group": MIN_POSITIVES_PER_GROUP,
        "included_labels_by_dimension": included_by_dimension,
        "excluded_labels_by_dimension": {
            dimension: [label for label in NIH_LABELS if label not in included]
            for dimension, included in included_by_dimension.items()
        },
        "input_validation": metadata,
        "limitations": [
            "This cohort was used to select/promote the 8k checkpoint; subgroup results are exploratory and post-selection.",
            "Labels are mined from reports and are not radiologist-adjudicated ground truth.",
            "Subgroup differences are descriptive, not proof of algorithmic bias, causation, or fairness.",
            "The evaluation is single-source, does not test external generalization, and has no race/ethnicity metadata.",
            "Macro average precision depends on subgroup label prevalence; the displayed prevalence and per-label supports must be considered.",
            "No group-specific thresholds were tuned or deployed; all threshold metrics use 0.5.",
            "The minimum-support filter limits unstable macro summaries but does not make the sample clinically sufficient.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "rows": metadata["rows"],
        "patients": metadata["unique_patients"],
        "included_labels_by_dimension": included_by_dimension,
        "group_summary": [
            {k: row[k] for k in ["dimension", "group", "n_patients", "macro_auroc", "macro_ap", "micro_sensitivity", "micro_fpr", "micro_f1"]}
            for row in summary_rows
        ],
        "differences": difference_rows,
        "checkpoint_identity": metadata["checkpoint_identity"],
    }, indent=2))


if __name__ == "__main__":
    main()
