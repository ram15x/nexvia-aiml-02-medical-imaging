# Limits and bias

This is an educational prototype. Its outputs are not diagnoses and must not guide clinical decisions. A qualified clinician must review all images.

## Classifier

- **Ranking is moderate, calibration is absent.** Macro AUROC on the 999-image NIH sample is 0.797 (see `results.md`). Scores are uncalibrated sigmoid outputs and are not probabilities.
- **At the fixed threshold 0.5 the model misses most positives.** Micro recall at 0.5 is 0.48 and micro precision is 0.17. Pneumothorax (98 positives) and Pneumonia (7 positives) get zero true positives at 0.5, so they are effectively never flagged at that threshold.
- **Some labels have too few positives to judge.** Fibrosis has 1 positive and Hernia has none in this sample, so their figures are not evidence of anything.
- **Labels are report-mined.** NIH ChestX-ray14 labels come from radiology report text and are noisy.
- **Grad-CAM is a coarse aid.** The overlay shows regions that influenced a score. It is not a segmentation mask and does not show the model's reasoning.

## Subgroup performance (age and sex)

Source: `results/nih_sample/subgroup_metrics.csv`, produced by `tools/subgroup_bias.py` with patient-level bootstrap intervals.

- **Sex:** no meaningful gap. Female macro AUROC 0.794 (95% CI 0.742–0.834), male 0.803 (0.766–0.834). The intervals overlap heavily.
- **Age, 60 and over:** a clear gap. Macro AUROC 0.709 (95% CI 0.651–0.770), against 0.822 for under-40s (0.788–0.880) and 0.835 for 40–59 (0.776–0.864). The 60+ interval does not overlap either younger group's interval.
- **Sample size:** the 60+ group has 277 images from 120 patients. The gap is measured on this sample only. Its cause is not known; it could come from training-data age distribution, label noise, or image differences. We did not test these causes.

The age result means the classifier should not be used on older patients without further evaluation.

## Segmentation (spleen)

- **Served checkpoint:** `best_spleen_unet_candidate_20261103.pt`. Its own metadata records a validation mean Dice of 0.833 on the six validation cases it was selected on. That is an optimistic, in-selection figure.
- **Held-out performance is not measured here.** The six holdout cases are not identified in this repository, so no independent Dice for the served model exists in this project. A separate, earlier baseline reported Dice 0.327 on six holdout cases. That figure belongs to the baseline, not the served checkpoint.
- **Segmentation quality is unestablished.** Do not treat the spleen masks as accurate, and do not use them to estimate organ volume clinically.

## Data and deployment

- **Public data only.** NIH ChestX-ray14 (PNG) and MSD Task09 (NIfTI). Real hospital DICOM was not available.
- **Orthanc completion for CT.** When Orthanc reports a CT series as Unknown, completeness depends on an instance count that a person enters. The count is a check, not proof of completeness.
- **No PACS integration.** Results are stored locally and are not written back to Orthanc as DICOM SEG or SR.
- **Review actions are demo records.** The "Approve (demo only)" button records an annotation. It is not clinical sign-off.
