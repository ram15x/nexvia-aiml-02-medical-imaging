# Model card — AIML-02 medical imaging prototype

**Version:** 4 November 2026  
**Status:** Educational coursework prototype; not for clinical use  
**Owner:** Ramnarayanan M  
**Submission status:** Not submitted

## System summary

This project contains two separate inference paths: a 2D chest X-ray classifier and a 3D spleen CT segmenter. They are not a unified multimodal model. The React/FastAPI demo accepts PNG/JPEG chest images and NIfTI spleen volumes, provides coarse Grad-CAM attribution for chest scores, and allows local demo review annotations. Optional Orthanc routes are implemented, but Orthanc was not verified end to end.

## Classifier

- **Model:** DenseNet-121 classifier for 14 NIH ChestX-ray14 findings. The API checkpoint is the equal-weight soup of the 8k and 10k DenseNet checkpoints with BatchNorm recalibration.
- **Checkpoint:** `artifacts/best_densenet121.pt`; SHA-256 `082bc9dc5ff0399a6eeb07ae4dd9146ab258c39829e2c19a5c3eccc2c1cd8cc8`.
- **Output:** 14 uncalibrated sigmoid scores. They are not calibrated diagnostic probabilities.
- **Evaluation evidence:** On the reused 5,000-patient development panel, macro AUROC/AP was 0.7953/0.1594 across 13 labels with at least 30 positives; the earlier 8k reference (6c32) scored 0.7755/0.1519. On the 2,138-patient common-clean development panel, the soup scored 0.7926/0.2258 across 9 labels with at least 30 positives; 6c32 scored 0.7700/0.2093. Both panels were reused during development and neither is an independent test set. NIH labels are report-mined. Absolute performance is modest and rare-label performance is uneven.
- **Not deployed:** A separate soup/15.5k probability ensemble scored 0.8028 AUROC on the 2,138 panel, but it is an undeployed candidate; its 15.5k training-manifest link is not verified. It is not the API model and is not presented as an API improvement.

## Spleen segmenter

- **Model:** MONAI 3D U-Net for binary spleen masks from Task09 Spleen CT volumes.
- **Checkpoint:** `artifacts/best_spleen_unet_candidate_20261103.pt`; SHA-256 `46622de72ac91c7e17d36da19d1088e90e1ad836a0d6e00533b37b805b5cb7a4`.
- **Recorded training:** 29 labeled volumes; 6 validation volumes; 96³ sliding-window ROI; 134 completed epochs. Epoch 125 was selected using mean Dice on the same six validation cases.
- **Evaluation evidence:** Mean argmax Dice was 0.8327 on those six selected validation cases. This is optimistic, not an independent estimate. A repeated six-case internal holdout scored 0.7883, but those cases had earlier project evaluation history; five appeared in training in separate nested-CV folds. This is not a fresh project-wide test. The 20 unlabeled challenge test volumes were not evaluated.
- **API settings:** Threshold 0.50 and flip test-time augmentation disabled. Dice measures mask overlap, not classification accuracy or clinical benefit.

## Intended use and limits

**Intended use:** educational demonstration of dataset auditing, model training/evaluation, image attribution, segmentation, and a local human-review interface.

**Not intended for:** screening, diagnosis, triage, treatment planning, clinical workflow, or decisions about an individual. Do not upload real patient data. The interface's Approve/Reject controls are demonstration annotations, not radiologist decisions.

Known limitations include reused development panels, report-mined labels, small segmentation cohorts, checkpoint selection on the six validation cases, incomplete subgroup coverage, no external or prospective validation, and unverified live Orthanc behavior. The age/recorded-sex analysis is exploratory; race/ethnicity and several relevant acquisition subgroups are unavailable. Grad-CAM is a coarse attribution map, not a causal or clinically faithful explanation.

## Data and reproducibility

The full NIH and MSD datasets are not included in this project folder. Only small demo examples and trained checkpoints are packaged. The current project copy has no `data/` directory and no configured DVC remote, so it cannot reproduce the training runs without restoring the required datasets and manifests. See [`data_inventory.md`](data_inventory.md).

## Evidence

- [November 3 classifier comparisons](../reports/Nexvia_AIML02_Inference_Comparisons_2026-11-03.md)
- [November 3 closeout](../reports/project1_closeout_2026-11-03.md)
- [API and regression audit](../reports/project1_api_fix_regression_2026-11-03.md)
- [Final browser, CPU, clean-install, and Docker verification](../reports/project1_final_checks_2026-11-03.md)
