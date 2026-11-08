# Deployed classifier: evaluation on a 999-image NIH test sample

**Checkpoint:** `artifacts/best_densenet121.pt` (DenseNet-121 weight soup, 8k+10k), SHA-256 `082bc9dc…`
**Data:** 999 official-test PNGs from one NIH ChestX-ray14 archive (`images_012.tar.gz`), 549 patients. Sampling protocol, archive checksum, and exclusion record are in the local data folder (`data/nih_test_sample_1000/PROTOCOL.md`, not in this repository).
**Threshold:** fixed 0.5. No threshold tuning, and the test images were not used for training or model selection.

This is a sample of the official test split, not the full split of 25,596 images. The per-label numbers are noisy, especially where there are few positives.

## Macro metrics (13 labels with defined AUROC)

| Metric | Result |
|---|---:|
| Macro AUROC | 0.7967 |
| Macro AUPRC (average precision) | 0.1507 |
| Micro precision at 0.5 | 0.1748 |
| Micro recall at 0.5 | 0.4813 |
| Micro F1 at 0.5 | 0.2565 |

Hernia has no positives in this sample, so its AUROC is undefined and it is excluded from the macro averages.

## Per-label results

| Label | Positives | AUROC | AUPRC | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Atelectasis | 80 | 0.8462 | 0.2968 | 60 | 235 | 20 | 684 |
| Cardiomegaly | 21 | 0.7945 | 0.1617 | 5 | 17 | 16 | 961 |
| Effusion | 131 | 0.8467 | 0.3675 | 105 | 221 | 26 | 647 |
| Infiltration | 122 | 0.7303 | 0.2753 | 80 | 340 | 42 | 537 |
| Mass | 33 | 0.7726 | 0.1053 | 15 | 106 | 18 | 860 |
| Nodule | 33 | 0.7652 | 0.1257 | 16 | 122 | 17 | 844 |
| Pneumonia | 7 | 0.7415 | 0.0238 | 0 | 0 | 7 | 992 |
| Pneumothorax | 98 | 0.7989 | 0.2190 | 0 | 0 | 98 | 901 |
| Consolidation | 36 | 0.8081 | 0.0926 | 8 | 74 | 28 | 889 |
| Edema | 11 | 0.8301 | 0.1053 | 7 | 216 | 4 | 772 |
| Emphysema | 43 | 0.7738 | 0.1046 | 6 | 35 | 37 | 921 |
| Fibrosis | 1 | 0.8848 | 0.0086 | 0 | 2 | 1 | 996 |
| Pleural Thickening | 24 | 0.7644 | 0.0727 | 6 | 86 | 18 | 889 |
| Hernia | 0 | undefined | undefined | 0 | 0 | 0 | 999 |

## Reading the results

- At threshold 0.5 the model predicts no positives for Pneumothorax or Pneumonia, so their confusion counts are zero for true positives. Their AUROC is still informative as a ranking measure, but the 0.5 threshold is not usable for them.
- Labels with fewer than 30 positives (Pneumonia, Edema, Fibrosis, Cardiomegaly, Hernia) have wide uncertainty. Their estimates should not be used to compare models.
- NIH labels are mined from radiology reports and are noisy.
- The scores are uncalibrated model outputs and are not diagnostic probabilities.

## Scope and limits

- The evaluation script's output labels this run as a complete split. That is an artifact of how it was invoked, since the sample size was passed as the official count. The correct scope is "999-image subset of the official test split."
- No confidence intervals yet. Patient-level bootstrap intervals are the next step, and need per-image predictions, which this run did not save.
- Earlier reports on a 500-image test sample and a 1,779-image validation set used a different checkpoint (`5634f216…`). Their numbers do not describe the deployed model and should not be combined with these.

Educational prototype. Not for diagnosis or clinical decisions.
