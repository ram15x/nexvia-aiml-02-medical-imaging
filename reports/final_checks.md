# Project 1 final checks — November 3, 2026

Inference-only closeout. No training, data download, or checkpoint modification was performed. Start time: 14:32:43 IST; 30-minute stop time: 15:02:43 IST. Local timezone: India Standard Time.

## 1. Old versus new segmentation checkpoints

Both checkpoints were scored on the same frozen six validation cases and six internal holdout cases using argmax, sliding-window inference, and 0.25 overlap. No threshold tuning or flip TTA was used for this comparison. Preprocessing followed the same transform code; the checkpoint metadata selected different ROI sizes (old 64³, new 96³), so this is matched by cases and metric, not identical compute configuration.

The old file's SHA-256 was verified as `414266BA20B24D9B4987FAE348BC26E33D14CDD1CF12D1077B35623937C55750`, matching the hash supplied in the request. The new checkpoint SHA-256 is `46622DE72AC91C7E17D36DA19D1088E90E1AD836A0D6E00533B37B805B5CB7A4`. Command: `Get-FileHash <checkpoint-path> -Algorithm SHA256`.

| Split | Case | Old checkpoint Dice | New checkpoint Dice | New − old |
|---|---|---:|---:|---:|
| Validation | spleen_10 | 0.256255 | 0.903045 | +0.646791 |
| Validation | spleen_63 | 0.209226 | 0.849549 | +0.640323 |
| Validation | spleen_46 | 0.185134 | 0.811124 | +0.625990 |
| Validation | spleen_17 | 0.285927 | 0.715347 | +0.429421 |
| Validation | spleen_18 | 0.369031 | 0.846035 | +0.477005 |
| Validation | spleen_2 | 0.256523 | 0.870947 | +0.614423 |
| **Validation mean (6)** |  | **0.260349** | **0.832675** | **+0.572325** |
| Internal holdout | spleen_26 | 0.236326 | 0.716717 | +0.480391 |
| Internal holdout | spleen_27 | 0.100523 | 0.776963 | +0.676439 |
| Internal holdout | spleen_29 | 0.126380 | 0.753736 | +0.627355 |
| Internal holdout | spleen_12 | 0.495936 | 0.732219 | +0.236283 |
| Internal holdout | spleen_19 | 0.307301 | 0.905025 | +0.597723 |
| Internal holdout | spleen_9 | 0.341686 | 0.844988 | +0.503302 |
| **Internal holdout mean (6)** |  | **0.268025** | **0.788275** | **+0.520249** |

Source: `AUDIT_WORKSPACE\work\old_checkpoint_inference_20261103\final_inference_metrics.json` and `AUDIT_WORKSPACE\work\segmentation_retrain_20261103_1132\final_inference\final_inference_metrics.json`. The latter checkpoint was selected on the six validation cases, so its validation mean is optimistic. The internal holdout is also not globally untouched (details below). These scores are evidence of a large difference on these saved cases, not an independent generalization estimate.

## 2. Internal holdout history and leakage audit

The internal holdout contains **6 cases**: spleen_26, spleen_27, spleen_29, spleen_12, spleen_19, spleen_9. The old run and November 3 final run each exclude all six from their own 29-case training split (`data\NexviaProjectData\msd_segmentation_run\case_split.json` and `work\segmentation_retrain_20261103_1132\case_split.json`). They are labeled cases inside Task09's official `training` partition; “holdout” here means a project-internal split, not the dataset's official test data.

Across all 11 local `case_split.json` files under the project data and workspace run folders, five of the six cases were assigned to training in other nested-CV folds: spleen_29, spleen_12, spleen_19, spleen_9 in fold 01, and spleen_26 plus spleen_9 in fold 02. The union is five; spleen_27 is the only case not found in any audited training list. Sources: `data\NexviaProjectData\msd_segmentation_nested_cv_20261029\fold_01\case_split.json` and `fold_02\case_split.json`. Therefore this set must not be described as a globally untouched test set.

Saved prior score history found for these cases:

- Three full-six score configurations predate this audit: old model direct/thresholded evaluation (mean 0.327201, with the same per-case values saved in both `msd_segmentation_run\heldout_evaluation.json` and `msd_segresnet_seed42_20261030\final_holdout_baseline_direct.json`); old model horizontal-flip TTA (`final_holdout_flip_tta.json`, mean 0.341193); and new candidate argmax (`work\segmentation_retrain_20261103_1132\final_inference\final_inference_metrics.json`, mean 0.788275). The two direct-output files have identical per-case metrics and identify the same checkpoint; whether the second file is a rerun or a copied result is **NOT VERIFIED**.
- Nested-CV histories add 20 validation score records for spleen_26 (`fold_01\training_history.json`), 10 for spleen_19 and 10 for spleen_27 (`fold_02\training.log`), plus one outer-fold evaluation for spleen_27 (`fold_01\outer_evaluation.json`). These are repeated fold validation/test records, not independent new cohorts.

Thus, counting saved per-case score records (not independent studies), prior records per case are: spleen_26 23; spleen_27 14; spleen_29 3; spleen_12 3; spleen_19 13; spleen_9 3. This count includes the three full-six configurations and the listed nested-CV records, but counts the duplicate direct JSON only once. Search scope: all `case_split.json` files under `data\NexviaProjectData` and `work`, plus all JSON/log files under local `msd_*` runs containing those six IDs and Dice fields.

## 3. API checkpoint configuration and live HTTP smoke

The project packages `artifacts\best_spleen_unet_candidate_20261103.pt`, SHA-256 `46622DE72AC91C7E17D36DA19D1088E90E1AD836A0D6E00533B37B805B5CB7A4`. `.env` selects it through `NEXVIA_SEGMENTATION_CHECKPOINT=artifacts/best_spleen_unet_candidate_20261103.pt`; `docker-compose.yml` points inside the read-only artifacts mount to `/app/artifacts/best_spleen_unet_candidate_20261103.pt`. The protected `artifacts\best_spleen_unet.pt` still hashes to `414266BA20B24D9B4987FAE348BC26E33D14CDD1CF12D1077B35623937C55750`.

A live Uvicorn HTTP run returned `/health` 200, `/predict` 200, and `/segment/spleen` 200 for spleen_10, spleen_63, and spleen_17. The segmenter ran on CUDA. Full request times were 9.518, 3.715, and 5.492 seconds; helper times were 7.294, 2.990, and 4.067 seconds; response bodies were 369,888, 437,144, and 415,067 bytes. The earlier TestClient stall did not recur; its cause is **NOT VERIFIED**. Docker API/web HTTP checks later passed on Nov 3; the Docker image uses CPU-only PyTorch, but the runtime device was not logged.

The live `/predict` request took 3.012 seconds and returned 1,192 bytes with 14 scores. A blank PNG received HTTP 422 (107 bytes, 0.002 seconds) and the near-constant-intensity error; a real PNG received HTTP 200. The guard tests 8-bit grayscale variance ≤1.0. Browser upload and the fresh pinned Windows environment passed on Nov 3. Orthanc remains **NOT VERIFIED**. Detailed measurements are in `reports\api_regression.md`.
## 4. Classifier ensemble metrics on the 2,138 panel

The `0.8028` versus `0.7926` comparison uses the nine labels with at least 30 positive patients: Atelectasis (139), Cardiomegaly (48), Effusion (117), Infiltration (279), Mass (85), Nodule (112), Consolidation (38), Fibrosis (30), Pleural_Thickening (52). It is the direct, no-TTA comparison.

| Metric | Soup | Soup + 15.5k probability average |
|---|---:|---:|
| Macro AUROC, 9 labels (support ≥30) | 0.792600 | 0.802830 |
| Macro AP, 9 labels (support ≥30) | 0.225797 | 0.233117 |
| Macro AUROC, all 14 labels | 0.817696 | 0.831570 |
| Macro AP, all 14 labels | 0.172447 | 0.182422 |

Source: `work\classifier_ensemble_inference_20261103\classifier_ensemble_inference_report.json`, keys `metrics.soup_direct`, `metrics.ensemble_direct_mean_soup_15p5k`, `labels_ge30`, and `positive_support_by_label`. The result remains provisional because the 15.5k manifest link is **NOT VERIFIED** and the panel was reused.

## 5. Requested wording scan

Searched the current `README.md`, `reports\project_closeout.md`, `deliverables\README.md`, and v20 deck slide text plus speaker notes for the exact requested strings. No matching line was found for `validated`, `improved over`, `accuracy %`, `Orthanc integrated`, `end-to-end`, or `working demo`.

The only literal `clinical` hits are below. They are dataset naming or explicit limitations, not positive performance claims:

| File | Line/location | Context |
|---|---:|---|
| `README.md` | 27 | “not clinical-performance claims” (negative limitation) |
| `README.md` | 44 | “not a clinical explanation” (limitation) |
| `README.md` | 72 | “NIH Clinical Center” (proper name) |
| `README.md` | 141 | “explain clinical reasoning” (says Grad-CAM does not) |
| `README.md` | 145 | “Neither capability establishes clinical performance” (negative limitation) |
| `README.md` | 151 | “not clinical evidence” (negative limitation) |
| `reports\project_closeout.md` | 78 | “Neither supports a clinical-performance claim” (negative limitation) |
| Deck speaker notes | slide 9 | “no independent-test or clinical-performance claim is made” (negative limitation) |

No overclaim was found in these hits. The README also uses the nearby phrase “no accuracy estimate” at line 13 for the unscored API smoke; it does not state an accuracy percentage. It says historical Orthanc behavior was not reverified and explicitly does not claim current Orthanc integration (`README.md` lines 47–49, 137, 139, 151). Slide 4 was corrected to say both in-process API paths passed while leaving live HTTP/browser/Orthanc/fresh install unverified.

## 6. Number-to-source inventory for visible slide text

Slide numbers are page labels. All other visible numeric claims are mapped below; checkpoint IDs and model names are included where they contain digits.

| Slide | Visible numbers / identifiers | Source file(s) |
|---:|---|---|
| 1 | AIML-02, Project 1, 3 November 2026; page 01 | `README.md`; `reports\project_closeout.md` |
| 2 | Page 02; 14 findings; NIH ChestX-ray14; 3D MSD | `reports\classifier_results.md`; `data\NexviaProjectData\Task09_Spleen\dataset.json` |
| 3 | Page 03; ChestX-ray14; 8k/10k; DenseNet-121; 6c32; 5,000 patients; 13 labels; ≥30 positives; 2,138 patients; 9 labels | `reports\classifier_results.md`; `inference_audit_2026-11-03\audit_report_data.json` |
| 4 | Page 04; DenseNet-121; 14 scores; 3D; MONAI 3D U-Net | `train_classifier.py`; `train_segmentation.py`; `api\app.py`; in-process smoke files named in section 3 |
| 5 | Page 05; 50/50; 8k/10k; DenseNet-121; 6c32; checkpoint 5634; 3,222 patients; 3 epochs | `reports\classifier_results.md`, checkpoint/cohort and soup sections |
| 6 | Page 06; 5,000 patients; 13 labels; ≥30 positives; 6c32; AUROC 0.7755/0.7953 and delta +0.0200 [0.0120, 0.0284]; AP 0.1519/0.1594 and delta +0.0083 [0.0009, 0.0168]; 2,000 resamples; 95% | `inference_audit_2026-11-03\panel_5k_metrics.json`; `panel_5k_paired_patient_bootstrap_2000.json`; `panel_5k_common_clean_manifest.csv` |
| 7 | Page 07; 2,138 patients; 9 labels; ≥30 positives; 6c32; AUROC 0.7700/0.7926 and delta +0.0226 [0.0110, 0.0336]; AP 0.2093/0.2258 and delta +0.0166 [0.0004, 0.0328]; 2,000 resamples; 95% | `inference_audit_2026-11-03\panel_2250_metrics.json`; `panel_2250_paired_patient_bootstrap_2000.json`; `panel_2250_common_clean_manifest.csv` |
| 8 | Page 08; 6 validation cases; epoch 125; Dice 0.8327; 6 holdout cases; Dice 0.7883; 29 training volumes; ROI 96³; 134 epochs; historical Dice 0.3170 | `work\segmentation_retrain_20261103_1132\run_summary.json`; `final_inference\final_inference_metrics.json`; `data\NexviaProjectData\msd_segmentation_run\validation_error_diagnosis.json` |
| 9 | Page 09; 2,138 patients; 15.5k checkpoint; AUROC 0.8028 vs 0.7926; 9 labels; ≥30 positives; 2,000 resamples; delta +0.0102; 95% CI [+0.0042, +0.0168]; soup SHA prefix 082bc9 | `work\classifier_ensemble_inference_20261103\classifier_ensemble_inference_report.json`; `demo_verification.md` |
| 10 | Page 10; no model or metric numbers | Slide content is a student reflection prompt |

Deck inspected: v20 was the prior copy; corrected v21 is packaged separately. The v21 copy opens in python-pptx and all XML parts parse; visual PowerPoint rendering remains **NOT VERIFIED**. ZIP and XML checks pass; visual rendering remains **NOT VERIFIED** because PowerPoint COM returned `80070005 (E_ACCESSDENIED)`.


