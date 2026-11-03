# Project 1 closeout — November 3, 2026

## Scope and evidence standard

This closeout covers the chest X-ray classifier and spleen CT segmentation coursework prototype. Every result below points to a saved report or run artifact. “Development panel” and “validation” are used literally: neither is an independent test set. The NIH official test split was not read during the final retrain or current inference audit. The protected old segmentation, soup, and 15.5k checkpoints were not modified. Live HTTP, browser uploads, fresh Windows installation, and Docker API/web checks passed on Nov 3. Orthanc remains **NOT VERIFIED**; Docker segmentation is configured for CPU and its runtime device was not logged.

## Change log since Oct 31 commit

Compared the Oct 31 commit (`61ec318`) with the current worktree using `git show` for committed blobs and a read-only content comparison. `api/app.py` now passes through checkpoint `promotion_evaluation` metadata, rejects near-constant chest images, and uses spleen threshold 0.50 with flip TTA disabled. `docker-compose.yml` now points to the new segmentation candidate checkpoint. `.env` and `requirements-windows-tested.txt` had no baseline version in that commit, so a before/after content comparison for those files is unavailable. The ordinary `git diff` invocation failed because the project directory was not recognized as a Git work tree in that shell; the committed-blob comparison is the basis for these findings.

## Classifier results

### Weight soup versus the earlier 8k reference (6c32)

The 50/50 DenseNet-121 weight soup from the 8k and 10k checkpoints, after BatchNorm recalibration, was evaluated on Nov 3 against the earlier 8k reference, checkpoint 6c32; no classifier checkpoint changed that day. Metrics are macro AUROC / macro AP on labels with at least 30 positive patients; intervals are paired patient bootstrap 95% percentile intervals over 2,000 resamples.

| Reused panel | Labels | 6c32 AUROC / AP | Soup AUROC / AP | Soup minus 6c32 AUROC (95% CI) | AP delta (95% CI) |
|---|---:|---:|---:|---:|---:|
| 5,000 patients | 13 | 0.7755 / 0.1519 | 0.7953 / 0.1594 | +0.0200 [+0.0120, +0.0284] | +0.0083 [+0.0009, +0.0168] |
| 2,138 common-clean patients | 9 | 0.7700 / 0.2093 | 0.7926 / 0.2258 | +0.0226 [+0.0110, +0.0336] | +0.0166 [+0.0004, +0.0328] |

These values and panel definitions are in `classifier_results.md`, “Common-clean inference” results and its paired-bootstrap tables. The panels were reused during development, so they do not establish independent generalization. The 2,138 panel is conditional on the assumed 15.5k manifest relationship; that checkpoint-to-manifest link is **NOT VERIFIED**. The earlier checkpoint 5634 is historical context only: 3,222-patient training cohort, 3 epochs (`classifier_results.md`, checkpoint and cohort table).

### Provisional soup + 15.5k probability ensemble

An inference-only candidate averaged sigmoid outputs from soup SHA-256 `082bc9dc5ff0399a6eeb07ae4dd9146ab258c39829e2c19a5c3eccc2c1cd8cc8` and 15.5k SHA-256 `bf144a8a0844f6256b7cda7d0aab5aa355f75036bfd409ad2fd1a9b23cc40831`. On the 2,138-patient panel, macro AUROC for the nine labels with at least 30 positives was 0.8028 for the direct ensemble and 0.7926 for soup alone. The paired 2,000-resample patient bootstrap gave median delta +0.0102, 95% interval [+0.0042, +0.0168]. With horizontal-flip TTA, the ensemble scored 0.8046 versus 0.7967 for soup; median delta +0.0079, interval [+0.0027, +0.0134].

The ensemble is an undeployed inference-only candidate, not an improvement to the API classifier. Although its paired interval on this reused panel is above zero, the 15.5k manifest binding is **NOT VERIFIED**, and the panel is not independent. The candidate was not integrated into the API; no classifier checkpoint changed on Nov 3. Full aggregate metrics, support counts, hashes, reproduction differences, and caveats are summarized in the inference comparison audit. Row-level predictions and raw scoring artifacts remain in the private audit workspace and are intentionally excluded from this release.

The ensemble report records small direct-output differences from saved scores (maximum absolute difference 0.000218 for soup and 0.000127 for 15.5k); the reason was **NOT VERIFIED**. It accepted the fresh outputs under a 1e-3 tolerance. Do not describe these as bit-identical reproductions.

## Spleen CT segmentation

### Bounded final-day retrain

The new run used 29 labeled training volumes and 6 validation volumes; a separate six-case internal holdout was scored once again, but those cases had already appeared in an earlier `heldout_evaluation.json`, so this is a repeat holdout evaluation, not a fresh final test. Configuration: 96³ ROI, 4 crops per volume with positive:negative sampling 1:1, batch size 2, AMP, CacheDataset, AdamW at 1e-4, DiceCE loss, and validation every 5 epochs. Preprocessing stayed RAS, 1.5×1.5×2.0 mm spacing, HU clip [-57,164] scaled to [0,1], foreground crop, and axis-0 flip. Source: `AUDIT_WORKSPACE/work\segmentation_retrain_20261103_1132/run_summary.json` keys `train_cases`, `validation_cases`, `heldout_cases_not_used_for_training_or_selection`, `patch_size`, `samples_per_volume`, `positive_negative_crop_ratio`, `batch_size`, `amp`, `loss`, `optimizer`, `learning_rate`, `preprocessing`, `validation_interval_epochs`.

The run completed 134 epochs / 2,010 optimizer steps. Best validation epoch was 125 with mean argmax Dice 0.8327 across the same six cases used for checkpoint selection. Per-case argmax Dice at that selected checkpoint: spleen_10 0.9030; spleen_63 0.8495; spleen_46 0.8111; spleen_17 0.7153; spleen_18 0.8460; spleen_2 0.8709. The validation mean is optimistic because the run selected the best epoch on these validation cases; it is not an unbiased estimate. Per-case exploratory largest-component Dice: 0.9344, 0.8886, 0.8698, 0.7173, 0.8635, 0.9385 in the same order. Full precision and voxel counts: `AUDIT_WORKSPACE/work\segmentation_retrain_20261103_1132/final_inference/final_inference_metrics.json`, keys `validation.mean_argmax_dice`, `validation.per_case`, `validation.largest_component`.

Validation checkpoints every five epochs (training loss at that epoch; mean validation Dice):

| Epoch | Train loss | Validation Dice | Epoch | Train loss | Validation Dice |
|---:|---:|---:|---:|---:|---:|
| 5 | 1.291701 | 0.029758 | 70 | 0.424512 | 0.573538 |
| 10 | 1.159454 | 0.044033 | 75 | 0.403178 | 0.672708 |
| 15 | 1.074418 | 0.066635 | 80 | 0.383977 | 0.615973 |
| 20 | 0.978257 | 0.125960 | 85 | 0.352579 | 0.706552 |
| 25 | 0.891686 | 0.154942 | 90 | 0.354906 | 0.674764 |
| 30 | 0.827284 | 0.175092 | 95 | 0.344890 | 0.571659 |
| 35 | 0.778020 | 0.325762 | 100 | 0.315977 | 0.688968 |
| 40 | 0.702875 | 0.265164 | 105 | 0.324582 | 0.648376 |
| 45 | 0.648377 | 0.455616 | 110 | 0.293772 | 0.799400 |
| 50 | 0.598093 | 0.516710 | 115 | 0.269093 | 0.755441 |
| 55 | 0.505300 | 0.491980 | 120 | 0.280905 | 0.789654 |
| 60 | 0.462944 | 0.490448 | 125 | 0.284500 | 0.832675 |
| 65 | 0.439564 | 0.625889 | 130 | 0.256387 | 0.798365 |

These are the saved values from `AUDIT_WORKSPACE/work\segmentation_retrain_20261103_1132/training_history.json`. The validation score fluctuated rather than rising monotonically. Epoch 125 was the best among checkpoints evaluated through epoch 130; epoch 134 completed after the last scheduled validation.

Repeated internal-holdout argmax mean was 0.7883 over six cases, with exploratory largest-component scores in the same inference metrics file. All six cases were excluded from the Nov 3 final run’s training and selection, but they had prior project evaluations; five appeared as training cases in separate nested-CV folds. This is not a fresh project-wide independent test. Whether earlier scores influenced configuration choices is NOT RECORDED. The file explicitly records `independent_fresh_test: false`. No official unlabeled test images were read (`run_summary.json`, key `official_unlabeled_test_images_read`).

The user did not provide a submission cutoff; the run-start evidence records an 18:00 IST assumption and T_stop 16:30 IST. Training ended earlier, at the 2.5-hour cap, after completing and saving epoch 134; the run summary records `stop_reason: 2.5_hours`, effective deadline 14:01:32 IST and elapsed time 9,045.34 seconds, about 45 seconds beyond the nominal cap to finish and save the epoch already in progress. The earlier stop rule did not trigger: epoch 100 validation mean was above 0.35 in `AUDIT_WORKSPACE/work\segmentation_retrain_20261103_1132/training_history.json`. Training logs and history: `AUDIT_WORKSPACE/work\segmentation_retrain_20261103_1132/run.log` and `training_history.json`.

The new validation mean exceeds the user's adoption threshold of 0.417 (the earlier baseline 0.3170 plus 0.10). A byte-for-byte copy now exists at `artifacts/best_spleen_unet_candidate_20261103.pt`, SHA-256 `46622DE72AC91C7E17D36DA19D1088E90E1AD836A0D6E00533B37B805B5CB7A4`. The protected `artifacts/best_spleen_unet.pt` remains unchanged at SHA-256 `414266BA20B24D9B4987FAE348BC26E33D14CDD1CF12D1077B35623937C55750`. The `.env` uses the relative project path `artifacts/best_spleen_unet_candidate_20261103.pt`. Docker Compose points inside its artifact mount to `/app/artifacts/best_spleen_unet_candidate_20261103.pt`. Docker API/web checks passed over HTTP on Nov 3; its image uses CPU-only PyTorch, while runtime device was not logged. Do not treat this comparison as a clean benchmark: the validation set is small and selected the checkpoint, while the repeated internal holdout is not fresh.

The threshold comparison is also not protocol-matched: the old 0.3170 used a threshold chosen on the six validation cases, while the new 0.8327 uses argmax. The API now uses threshold 0.50 with flip TTA disabled, matching the requested argmax-equivalent setting more closely. This API setting change does not alter the stored validation metric. All six validation overlays remain in the dated run's `final_inference/` output.

The previous Dice 0.3170 at threshold 0.80, and largest-component mean 0.4200, are historical baseline figures from the old 20-epoch model; its threshold was selected on those same six validation cases. The separate 0.3272 is the old model's previously evaluated internal holdout result. They are not measurements of the new run and must not be mixed with its argmax metrics. See the earlier audit in `demo_verification.md` and its cited JSON sources.

### API and Docker checks

The API segmentation call now uses threshold 0.50 and no flip TTA; its previous configuration was threshold 0.80 plus axis-0 flip probability averaging. A one-command live Uvicorn check returned HTTP 200 from `/health`, HTTP 200 from `/predict`, and HTTP 200 from `/segment/spleen` for spleen_10, spleen_63, and spleen_17. The selected device was CUDA. Segmentation helper times were 7.294, 2.990, and 4.067 seconds; full HTTP request times were 9.518, 3.715, and 5.492 seconds. Response body sizes were 369,888, 437,144, and 415,067 bytes. The earlier TestClient stall did not recur with the live server; its specific cause is **NOT VERIFIED**. The server process was terminated after the requests.

The `/predict` response was 1,192 bytes and completed in 3.012 seconds. A blank PNG returned HTTP 422 with the new near-constant-image error (107-byte response, 0.002 seconds). The guard rejects grayscale variance ≤1.0 on the 8-bit intensity scale. A real PNG returned HTTP 200 with 14 scores. Docker Compose uses the new relative artifact under `/app/artifacts`. Docker API/web checks, browser uploads, and a fresh pinned Windows install passed on Nov 3. Orthanc remains **NOT VERIFIED**. Docker uses CPU-only PyTorch per its image configuration; runtime device was not logged. Exact live HTTP measurements are in the read-only audit output `f4_live_http_result.json` outside this project. The consolidated deployment, input-guard, and regression evidence is in `api_regression.md`.

## Limitations

- Both classifier development panels were reused; neither is an independent test set.
- The 15.5k checkpoint-to-training-manifest link is **NOT VERIFIED**, so the ensemble clean-panel claim remains provisional and conditional.
- NIH labels are report-mined and were not manually adjudicated in this work.
- Absolute classifier performance is modest; macro AP is prevalence-sensitive.
- The six-case segmentation validation set selected the checkpoint. The repeat internal holdout was excluded from the final run’s training and selection, but has prior project evaluation history (including nested-CV training use for five cases); it is not a project-wide independent test.
- The classifier ensemble remains an undeployed inference-only candidate, not an improvement to the API classifier. Live HTTP, browser uploads, and Docker API/web checks passed; Orthanc remains unverified.
- The new segmentation configuration values (96³ ROI, 4 samples per volume, balanced sampling, batch 2, AMP, AdamW 1e-4, DiceCE) and argmax threshold were fixed in the user-directed retrain instructions before that run. Epoch 125 was selected by validation Dice only. The six internal holdout cases had been scored in earlier project work; whether earlier holdout results influenced the choice of this configuration is **NOT VERIFIED**. No threshold or epoch was selected on those holdout cases in the recorded final run.
- This is an educational, non-diagnostic prototype.

## Commands and reproducibility

The final inference scripts and score outputs are retained in the dated audit workspace. To replay them, set `NEXVIA_AUDIT_WORKSPACE` to that workspace root. The local workspace path is intentionally not embedded in this report.

```powershell
& '.\data\NexviaProjectData\aiml02-venv310\Scripts\python.exe' "$env:NEXVIA_AUDIT_WORKSPACE\work\segmentation_retrain_20261103_1132\final_inference.py"
& '.\data\NexviaProjectData\aiml02-venv310\Scripts\python.exe' "$env:NEXVIA_AUDIT_WORKSPACE\work\segmentation_retrain_20261103_1132\api_segment_smoke.py"
& '.\data\NexviaProjectData\aiml02-venv310\Scripts\python.exe' "$env:NEXVIA_AUDIT_WORKSPACE\work\classifier_ensemble_inference_20261103.py"
Get-FileHash '.\artifacts\best_spleen_unet_candidate_20261103.pt' -Algorithm SHA256
Get-FileHash '.\artifacts\best_spleen_unet.pt' -Algorithm SHA256
```

The saved training summary and output JSONs are the authoritative run records. The exact shell wrapper that launched the long segmentation training process is **NOT VERIFIED** from a persisted command transcript; the command above invokes the saved training script directly but does not reconstruct the original wrapper or timing environment. API network listener failure is reported above rather than treated as a pass.

## Presentation verification

The final PPTX passed ZIP integrity and XML parsing checks for all 10 slides and the updated speaker notes. Visual rendering is **NOT VERIFIED**: PowerPoint COM startup failed with `Retrieving the COM class factory ... failed ... 80070005 (E_ACCESSDENIED)`. The error occurred when running `New-Object -ComObject PowerPoint.Application`; no slide-render output was produced.





