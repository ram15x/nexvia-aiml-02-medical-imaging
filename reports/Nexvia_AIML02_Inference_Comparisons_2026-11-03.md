# Nexvia AIML-02 inference-only comparison report

**Date:** 2026-11-03  
**Scope:** Requested inference-only scoring on manifest-defined common-clean panels. No training, downloads, checkpoint edits, official test list access, or official test image access occurred.

## Important scope limitation

The 2,250 panel had 112 patients in the former-demo training manifest. Those 112 were excluded before scoring. The 15.5k seed-42 checkpoint’s exact training-manifest link is **NOT VERIFIED**: the checkpoint metadata and run summary contain counts but no manifest path/hash. For a provisional clean-set calculation only, I used the named 15.5k cohort manifest from the seed-11 protocol; under that mapping it contributes zero overlap with the 2,250 panel. Therefore 15.5k results below are conditional on that manifest mapping and are not verified as clean for that checkpoint. The 5k panel was scored without the 15.5k model, as instructed.

## 1. Checkpoint identity and manifest linkage

| Requested candidate | File | SHA-256 | Manifest linkage |
|---|---|---|---|
| 5634 former demo | `data\NexviaProjectData\nih\classifier_available_trainval_3epochs_20261028\best_densenet121.pt` | `5634f21659d2cb8bf57bb90ff45797bd998beef15e1231e7f916f953f610db87` | Verified by reports/classifier_available_trainval_experiment_2026-10-28.md naming trainval_available_20261028.csv; run summary counts match. |
| 6c32 pre-soup | `artifacts\best_densenet121_pre_soup_20261102.pt` | `6c32434bafc2b09b5c7487f9d9442d60addfd91d3baa6b070549f76611a01899` | Verified by reports/checkpoint_selection_macro_ap_20261102.json exact 8k manifest path and SHA, paired to historical 6c32 checkpoint hash. |
| E385 8k | `data\NexviaProjectData\nih\learning_curve_8000_train_1000_val_20261031\densenet121_train8000\best_densenet121.pt` | `e3856243ad7e3295c79505a112105ba466122ba1cb030c54c1250a3687d12a10` | Verified by checkpoint_promotion_2026-10-31.md naming E385 as exact source for 6c32; state_dicts were tensor-equal; 6c32 manifest SHA matches E385 8k manifest bytes. |
| 10k / 4c79 soup source | `data\NexviaProjectData\nih\learning_curve_10000_train_1000_val_20261031\densenet121_train10000\best_densenet121.pt` | `4c7998c1dc751d0c8f887d28450c357c48b018c0b7c6606196f3f35fc607f772` | Manifest path is explicitly assigned by frozen_candidates README reproduction command; checkpoint hash and path recorded by 2250 metrics and soup metadata. Run summary confirms 10k/1k counts but does not itself include manifest hash. |
| soup | `reports\frozen_candidates_2250_internal_holdout_2026-11-01\densenet121_weight_soup_8k10k_bn_recalibrated.pt` | `082bc9dc5ff0399a6eeb07ae4dd9146ab258c39829e2c19a5c3eccc2c1cd8cc8` | Checkpoint metadata records source hashes 6c32 and 4c79; also records 8k BN calibration manifest SHA. |
| 15.5k | `data\NexviaProjectData\nih\imagenet_densenet121_full15500_seed42_5epochs_20261030\best_densenet121.pt` | `bf144a8a0844f6256b7cda7d0aab5aa355f75036bfd409ad2fd1a9b23cc40831` | NOT VERIFIED for this seed-42 checkpoint. Checkpoint metadata/run summary record 15,500 train/2,250 val but no manifest path/hash. The named manifest is from a seed-11 protocol; used provisionally for overlap filtering. |

The soup checkpoint metadata records source hashes **6c32 pre-soup** (`6c32434b…`) and **10k/4c79** (`4c7998c1…`). At soup creation, the 6c32 file was recorded at `artifacts/best_densenet121.pt`; it is now preserved at `artifacts/best_densenet121_pre_soup_20261102.pt`. The other source is `data/NexviaProjectData/nih/learning_curve_10000_train_1000_val_20261031/densenet121_train10000/best_densenet121.pt`. These are the two source files identified by `weight_soup_metrics.json` and the soup checkpoint metadata. 10k and 4c79 are the same checkpoint and were scored once. The 6c32 and E385 files have different file hashes but their state-dictionary tensors compare exactly equal; their outputs in the saved predictions are also identical.

**Manifest-link evidence:** 5634: the October 28 experiment report names `trainval_available_20261028.csv`, and the checkpoint metadata/run summary counts agree. 6c32: `checkpoint_selection_macro_ap_20261102.json` names the exact 8k manifest path and SHA-256; the manifest bytes match. E385: the promotion report identifies E385 as the exact source checkpoint for 6c32, and the two state dictionaries are equal. 10k/4c79: the frozen-candidates reproduction command assigns the exact 10k training manifest; the run summary confirms counts but does not itself record the manifest SHA. Soup: checkpoint metadata lists both source hashes and its 8k BN recalibration manifest hash. 15.5k: **NOT VERIFIED** for the seed-42 checkpoint; only the seed-11 protocol names the proposed manifest.

### Manifest image rows and distinct patients

| Manifest/split | Image rows | Patients | One image per patient? |
|---|---:|---:|---|
| 5634 former demo — train | 10,761 | 3,222 | No |
| 5634 former demo — validation | 1,779 | 548 | No |
| 6c32 pre-soup — train | 8,000 | 8,000 | Yes |
| 6c32 pre-soup — validation | 1,000 | 1,000 | Yes |
| E385 8k — train | 8,000 | 8,000 | Yes |
| E385 8k — validation | 1,000 | 1,000 | Yes |
| 10k / 4c79 soup source — train | 10,000 | 10,000 | Yes |
| 10k / 4c79 soup source — validation | 1,000 | 1,000 | Yes |
| 15.5k — train | 15,500 | 15,500 | Yes |
| 15.5k — validation | 2,250 | 2,250 | Yes |
| 2,250 panel | 2,250 | 2,250 | Yes |
| 5k panel | 5,000 | 5,000 | Yes |
| 99 panel | 99 | 99 | Yes |

The soup’s effective source training set is the union of 8k and 10k patients (10k unique patients because 8k is nested within 10k); BN recalibration uses the 8k manifest. The former demo is the only listed training cohort with multiple images per patient.

## 2. Panel membership, overlaps, and common clean sets

Every image name in the 2,250 panel was found in `metadata/train_val_list.txt` (**2,250/2,250**); every image name in the 5k panel was found there (**5,000/5,000**). The official `test_list.txt` was not read. Only panel image files whose names were confirmed in `train_val_list.txt` were loaded for inference.

| Training candidate / source | 2,250 panel overlap | 5k panel overlap | 99 panel overlap |
|---|---:|---:|---:|
| 5634 former demo | 112 | 0 | 3 |
| 6c32 | 0 | 0 | 0 |
| E385 | 0 | 0 | 0 |
| 10k | 0 | 0 | 0 |
| 4c79 alias of 10k | 0 | 0 | 0 |
| Soup, union of 8k+10k and 8k BN calibration | 0 | 0 | 0 |
| 15.5k (manifest link NOT VERIFIED) | 0 | 5000 | 99 |

| Panel | Starting patients/images | Removed for any compared candidate overlap | Scored common-clean rows/patients |
|---|---:|---:|---:|
| 2,250 panel | 2,250 | 112 | 2,138 |
| 5k panel | 5,000 | 0 | 5,000 |

The 112 removed 2,250-panel IDs are exactly the former-demo training overlap. All candidate-associated manifests used for the clean-set construction show zero additional overlap. The 15.5k manifest association is **NOT VERIFIED**, so 2,138 is conditional for the 15.5k candidate. The 5k set includes no 15.5k scoring candidate; its 5,000 patients have zero overlap with the training manifests of the candidates scored there. The 99 panel was read as metadata only to calculate overlap; no 99-panel images were opened or scored.

## 3. Inference metrics

Metrics below are direct scores from the saved per-patient probabilities on the listed clean rows. AUROC/AP are macro averages over labels with both classes / positive support, respectively; all 14 labels were defined on both clean panels. `≥30` uses the labels with at least 30 positive patients on that clean panel. No thresholds were tuned.

### 2,250 clean panel (2,138 patients)

| Candidate | Hash / definition | AUROC all 14 | AP all 14 | AUROC ≥30 support | AP ≥30 support |
|---|---|---:|---:|---:|---:|
| Former demo (5634) | 5634… | 0.7721 | 0.1380 | 0.7409 | 0.1724 |
| 6c32 pre-soup | 6c32… | 0.7936 | 0.1532 | 0.7700 | 0.2093 |
| E385 8k | E385… | 0.7936 | 0.1532 | 0.7700 | 0.2093 |
| 10k / 4c79 source (same file) | 4c79… | 0.7963 | 0.1592 | 0.7742 | 0.2096 |
| Soup | 082b… | 0.8177 | 0.1724 | 0.7926 | 0.2258 |
| Probability mean (6c32, 10k_4c79) | mean(6c32,4c79) | 0.8063 | 0.1723 | 0.7852 | 0.2274 |
| 15.5k | BF14… | 0.8146 | 0.1683 | 0.7887 | 0.2164 |

Labels with ≥30 positives: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Consolidation, Fibrosis, Pleural_Thickening.

### 5k clean panel (5,000 patients)

| Candidate | Hash / definition | AUROC all 14 | AP all 14 | AUROC ≥30 support | AP ≥30 support |
|---|---|---:|---:|---:|---:|
| Former demo (5634) | 5634… | 0.7447 | 0.1296 | 0.7435 | 0.1388 |
| 6c32 pre-soup | 6c32… | 0.7792 | 0.1430 | 0.7755 | 0.1519 |
| E385 8k | E385… | 0.7792 | 0.1430 | 0.7755 | 0.1519 |
| 10k / 4c79 source (same file) | 4c79… | 0.7856 | 0.1404 | 0.7836 | 0.1493 |
| Soup | 082b… | 0.7958 | 0.1504 | 0.7953 | 0.1594 |
| Probability mean (6c32, 10k_4c79) | mean(6c32,4c79) | 0.7955 | 0.1502 | 0.7938 | 0.1595 |

Labels with ≥30 positives: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening.

### Positive support on each clean panel

| Label | 2,250 clean positives | 5k clean positives |
|---|---:|---:|
| Atelectasis | 139 | 359 |
| Cardiomegaly | 48 | 104 |
| Effusion | 117 | 281 |
| Infiltration | 279 | 689 |
| Mass | 85 | 208 |
| Nodule | 112 | 263 |
| Pneumonia | 12 | 34 |
| Pneumothorax | 25 | 69 |
| Consolidation | 38 | 87 |
| Edema | 12 | 33 |
| Emphysema | 20 | 50 |
| Fibrosis | 30 | 78 |
| Pleural_Thickening | 52 | 121 |
| Hernia | 5 | 11 |

Candidate-by-label AUROC/AP/support tables were saved in a private audit workspace and are not included in this package or public repository. They contain no new claims beyond the aggregate metrics reported here.

## 4. Paired patient bootstrap

The delta is soup minus the named comparator. Each resample draws patients with replacement; because both scored panels have exactly one image per patient, a multinomial patient multiplicity vector is equivalent to explicit patient resampling. The weighted AUROC/AP implementation was checked against scikit-learn using explicit sample weights for five resamples per panel. Every reported comparison has 2,000 valid resamples. Intervals are percentile 2.5th–97.5th percentiles; median is the bootstrap median delta.

| Panel | Comparison | AUROC all14 | AP all14 | AUROC ≥30 | AP ≥30 |
|---|---|---|---|---|---|
| 2,250 clean | soup − 6c32 | +0.0242 [+0.0118, +0.0355] | +0.0209 [+0.0054, +0.0415] | +0.0226 [+0.0110, +0.0336] | +0.0166 [+0.0004, +0.0328] |
| 2,250 clean | soup − 10k / 4c79 | +0.0211 [+0.0087, +0.0347] | +0.0126 [-0.0069, +0.0351] | +0.0182 [+0.0099, +0.0269] | +0.0157 [+0.0025, +0.0299] |
| 2,250 clean | soup − probability mean | +0.0113 [+0.0005, +0.0221] | -0.0013 [-0.0203, +0.0176] | +0.0074 [-0.0005, +0.0149] | -0.0017 [-0.0148, +0.0104] |
| 2,250 clean | soup − former demo (5634) | +0.0454 [+0.0278, +0.0630] | +0.0366 [+0.0105, +0.0668] | +0.0517 [+0.0347, +0.0679] | +0.0538 [+0.0342, +0.0769] |
| 2,250 clean | soup − 15.5k | +0.0035 [-0.0117, +0.0168] | +0.0056 [-0.0155, +0.0314] | +0.0039 [-0.0095, +0.0173] | +0.0096 [-0.0066, +0.0266] |
| 5k clean | soup − 6c32 | +0.0167 [+0.0089, +0.0249] | +0.0085 [+0.0011, +0.0174] | +0.0200 [+0.0120, +0.0284] | +0.0083 [+0.0009, +0.0168] |
| 5k clean | soup − 10k / 4c79 | +0.0101 [+0.0018, +0.0191] | +0.0110 [+0.0043, +0.0194] | +0.0117 [+0.0041, +0.0200] | +0.0110 [+0.0045, +0.0192] |
| 5k clean | soup − probability mean | +0.0006 [-0.0049, +0.0057] | +0.0010 [-0.0053, +0.0085] | +0.0017 [-0.0035, +0.0067] | +0.0004 [-0.0055, +0.0080] |
| 5k clean | soup − former demo (5634) | +0.0513 [+0.0337, +0.0685] | +0.0217 [+0.0075, +0.0368] | +0.0518 [+0.0366, +0.0666] | +0.0210 [+0.0060, +0.0357] |

Bootstrap arrays remain in a private audit workspace and are not distributed; this report preserves the precomputed intervals and resample count.

## 5. Saved predictions and clean manifests

- Per-patient prediction CSVs, common-clean manifests, and row-level score tables are excluded from the public release. This report contains aggregate metrics and overlap counts only.

CSV SHA-256: 2,250 predictions `0bd81c0e0357f313854b5d5f825dd322f269cd016ceccb67c7f019f909989c4a`; 5k predictions `c57d4a64d7318e37f33da15322d302b49bf854430fc7383b4da6aea2cf585e70`. All six checkpoint hashes were rechecked after inference and remained unchanged.

## 6. Verified / NOT VERIFIED / limitations

### Verified

- All six requested checkpoint file hashes are listed and matched the expected identities; 10k and soup source `4c79` are the same file/hash.
- Soup source hashes are 6c32 and 4c79. The 6c32/E385 state dictionaries are tensor-identical.
- The 2,250 and 5k panel names are wholly contained in `train_val_list.txt`; no official test list or official test image was accessed.
- Manifest overlaps are listed above. The 2,250 common-clean set excludes 112 former-demo training patients; the 5k comparison set has zero overlap for all compared candidates.
- The six model candidates were run in inference mode only; probability mean is arithmetic mean of sigmoid scores from 6c32 and 10k/4c79. Soup checkpoint hash did not change.
- Patient-level predictions and manifests remain in a private audit workspace and are not distributed; this report publishes aggregate metrics and intervals.

### NOT VERIFIED

- The exact train-manifest binding for the 15.5k seed-42 checkpoint is not recorded in its checkpoint metadata or run summary. Its overlap and score are conditional on the seed-11 protocol’s named manifest being the same cohort assignment.
- The 10k run summary does not embed the manifest path/hash; the saved evaluation reproduction command supplies the exact manifest assignment. No checkpoint-level manifest hash is present.
- The scores are on reused internal NIH development panels, not an untouched external or clinical validation cohort. No claim beyond these reported results is made.

### Computation notes

An initial scoring preflight stopped on a typo in the expected soup SHA string before inference; the hash was corrected against `Get-FileHash`, and the completed inference run rechecked all checkpoint hashes. The first, inefficient bootstrap attempt was interrupted after the prediction files were safely written. A replacement weighted bootstrap initially had AUROC orientation reversed; that output was discarded, the formula was corrected, and the final bootstrap was cross-checked against scikit-learn explicit weights before use. Only the final corrected JSONs are reported above. The per-patient inference script also emitted pandas DataFrame-fragmentation performance warnings while assembling CSV columns; the prediction files were written and their row counts, uniqueness, and metrics were verified.

## Commands with anonymized roots

Replace `PROJECT_ROOT` and `AUDIT_WORKSPACE` with the local project and audit workspace roots. The scoring reads only the named checkpoint, manifest, and train/validation-listed panel image files. It writes outputs only to the workspace audit directory.

```powershell
& 'PROJECT_ROOT\.venv\Scripts\python.exe' 'AUDIT_WORKSPACE\audit_inference_2026-11-03.py'
& 'PROJECT_ROOT\.venv\Scripts\python.exe' 'AUDIT_WORKSPACE\bootstrap_inference_audit_2026-11-03.py'
```

Checkpoint hashes were checked with these exact commands (the checks were repeated after inference):

```powershell
$project = 'PROJECT_ROOT'
$checkpoints = @(
  'data\NexviaProjectData\nih\classifier_available_trainval_3epochs_20261028\best_densenet121.pt',
  'artifacts\best_densenet121_pre_soup_20261102.pt',
  'data\NexviaProjectData\nih\learning_curve_8000_train_1000_val_20261031\densenet121_train8000\best_densenet121.pt',
  'data\NexviaProjectData\nih\learning_curve_10000_train_1000_val_20261031\densenet121_train10000\best_densenet121.pt',
  'reports\frozen_candidates_2250_internal_holdout_2026-11-01\densenet121_weight_soup_8k10k_bn_recalibrated.pt',
  'data\NexviaProjectData\nih\imagenet_densenet121_full15500_seed42_5epochs_20261030\best_densenet121.pt'
)
$checkpoints | ForEach-Object { Get-FileHash -LiteralPath (Join-Path $project $_) -Algorithm SHA256 }
```

No 15.5k inference was run on the 5k panel or the 99 panel. The 99 panel was used only for manifest patient-ID overlap counts. No checkpoint or project file was edited.
