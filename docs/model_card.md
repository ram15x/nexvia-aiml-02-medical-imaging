# Model card — Medical Imaging AI Review Platform

**Project:** Medical Imaging AI Review Platform
**Owner:** Ramnarayanan M
**Use:** Educational demonstration of medical-imaging machine learning and review workflows

## System summary

The platform contains two independent inference paths: a 2D NIH ChestX-ray14 classifier and a 3D Task09 spleen CT segmenter. The React/FastAPI application accepts PNG/JPEG radiographs and NIfTI volumes, presents model outputs and coarse Grad-CAM visualizations, and supports local demonstration review events and Orthanc data flows.

## Chest X-ray classifier

- **Architecture:** DenseNet-121 with a 14-output sigmoid classification head.
- **API checkpoint:** `artifacts/best_densenet121.pt` (SHA-256 `082bc9dc5ff0399a6eeb07ae4dd9146ab258c39829e2c19a5c3eccc2c1cd8cc8`). It is an equal-weight average of the 8k and 10k DenseNet checkpoints with BatchNorm recalibration.
- **Output:** 14 uncalibrated scores corresponding to NIH ChestX-ray14 report-mined findings.
- **Development metrics:** Macro AUROC/AP is 0.7953/0.1594 across 13 labels with ≥30 positives on the 5,000-patient panel. On the 2,138-patient panel, the corresponding metrics are 0.7926/0.2258 across 9 labels. The earlier 8k reference (6c32) scored 0.7755/0.1519 and 0.7700/0.2093 on those panels.
- **Evaluation context:** Both panels were reused during development. Report-mined labels have known uncertainty, and AP depends on label prevalence. Grad-CAM is a coarse attribution map rather than a causal explanation or localization annotation.

## Spleen CT segmenter

- **Architecture:** MONAI 3D U-Net for binary spleen masks from Task09 Spleen CT volumes.
- **API checkpoint:** `artifacts/best_spleen_unet_candidate_20261103.pt` (SHA-256 `46622de72ac91c7e17d36da19d1088e90e1ad836a0d6e00533b37b805b5cb7a4`).
- **Training setup:** 29 training volumes; 6 validation volumes; 96³ sliding-window ROI; 134 completed epochs. The checkpoint was selected by mean Dice on the six validation cases.
- **Development results:** Mean argmax Dice was 0.8327 on the six checkpoint-selection cases. The six-case internal holdout result was 0.7883 and the cases have prior project evaluation history. These results describe a small development cohort.
- **API inference settings:** Threshold 0.50 and flip test-time augmentation disabled.

## Intended use

The platform is intended for educational demonstrations of data preparation, classification, attribution, 3D segmentation, API integration, and browser-based model review. It is not intended for screening, diagnosis, triage, treatment planning, or decisions about an individual. Do not upload identifiable patient data.

## Evaluation boundaries

The classifier results come from reused development panels; the segmentation validation result comes from cases used to select the checkpoint. These metrics describe model behavior in the stated cohorts. Scores are uncalibrated, and the work is an educational prototype rather than a clinical evaluation. Chest X-ray classification and spleen CT segmentation are separate workflows.

See [the architecture guide](architecture.md), [data guide](data_inventory.md), and [API reference](api.md).