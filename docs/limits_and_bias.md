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

- **Served checkpoint:** `best_spleen_unet_20261109.pt`, a MONAI 3D U-Net trained for 60 epochs on 29 MSD Task09 cases. The case split (29 train, 6 validation, 6 test, seed 42) is recorded in `results/spleen/case_split.json`. The checkpoint is the epoch with the best validation Dice (epoch 51, 0.736). The test cases were not used for training or for choosing the checkpoint.
- **Held-out result:** mean Dice **0.715** on the 6 test cases, from a single run with the default 0.5 threshold and no test-time augmentation (`results/spleen/test_result.json`). Per case it ranges from 0.47 to 0.91. Two cases (spleen_26 and spleen_29) were over-segmented, with predicted volumes about 2.8 and 2.3 times the reference, and scored 0.47 and 0.57. The served API gives the same answer: 0.717 on the same cases, measured in the original image space.
- **How much to trust it:** six test cases is a very small sample, so the figure could move a lot with different cases. Validation Dice swung between 0.28 and 0.74 from one epoch to the next near the end of training. The data is one public dataset of portal-venous CT; it has not been tested on other scanners or on abnormal anatomy.
- **Not suitable for clinical use.** Do not use these masks to estimate organ volume or to support any care decision.
- **Replaced checkpoint:** the earlier served file (`best_spleen_unet_candidate_20261103.pt`, recorded validation Dice 0.833) was selected on its own validation cases and its train/test split was not recorded, so no held-out figure exists for it. It was removed from the served set; it remains in git history.

## Data and deployment

- **Public data only.** NIH ChestX-ray14 (PNG) and MSD Task09 (NIfTI). Real hospital DICOM was not available.
- **Orthanc completion for CT.** When Orthanc reports a CT series as Unknown, completeness depends on an instance count that a person enters. The count is a check, not proof of completeness.
- **No PACS integration.** Results are stored locally and are not written back to Orthanc as DICOM SEG or SR.
- **SHAP is sampled and grid-based.** Values come from 300 sampled coalitions on an 8×8 grid with neutral grey replacement. They describe how this model's score changes when regions are removed. They are not anatomical findings, causes, or proof that the model looks at the right structure.
- **Question answering is not clinical advice.** The language model sees only stored scores, metadata and SHAP regions. It can still misread or overstate them. The server adds a fixed footer and sends urgent-symptom wording to a fixed emergency message, but neither guarantee correctness. Answers are not checked by a clinician, and they are not logged with personal details.
- **Review actions are demo records.** The "Approve (demo only)" button records an annotation. It is not clinical sign-off.
