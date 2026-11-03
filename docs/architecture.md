# Project 1 architecture and verification status

Last reconciled: November 3, 2026. This describes the packaged demo and the latest recorded checks. Results are engineering evidence, not clinical validation.

## End-to-end structure

~~~mermaid
flowchart LR
  X[Chest X-ray PNG/JPEG] --> UI[React review interface]
  UI -->|POST /predict| API[FastAPI]
  API --> CX[DenseNet-121 classifier]
  CX --> P[14 uncalibrated finding scores]
  UI -->|POST /explain| API
  API --> G[Grad-CAM attribution overlay]
  UI -->|POST /reviews| API
  API --> R[Local JSONL review events]
  N[Spleen CT NIfTI] --> UI
  UI -->|POST /segment/spleen| API
  API --> SU[MONAI 3D U-Net]
  SU --> M[Mask and tri-planar previews]
  O[Optional Orthanc DICOM server] -. unverified live integration .-> API
~~~

The chest and spleen paths are separate models and input formats. Chest PNG/JPEG files pass through the shared resize and ImageNet normalization path and produce 14 report-label sigmoid scores. These are uncalibrated scores. Grad-CAM is a coarse feature attribution map; it is not a bounding-box detector or a clinical explanation. Review annotations are saved locally and are not radiologist decisions.

The spleen path accepts a 3D NIfTI CT volume. The packaged model is the candidate in artifacts/best_spleen_unet_candidate_20261103.pt, SHA-256 46622de72ac91c7e17d36da19d1088e90e1ad836a0d6e00533b37b805b5cb7a4. Its checkpoint records a 96×96×96 sliding-window ROI. The API inference setting is threshold 0.50 with flip TTA disabled. The offline CLI and API are separate interfaces; consult their commands and settings independently.

## Runtime components

| Component | Code/configuration | Verification status |
|---|---|---|
| Classifier | api/modeling.py, api/app.py, DenseNet-121 checkpoint at artifacts/best_densenet121.pt | /predict and browser upload passed in the November 3 verification record. A rerun during the current pre-submission pass could not load the model because the available global Python lacks torchvision. |
| Segmentation | segment_spleen.py, api/app.py, candidate checkpoint listed above | Three live HTTP segmentation calls passed in the November 3 record. The current pre-submission rerun is pending the pinned venv; it has not produced fresh Dice values. |
| API | FastAPI routes /health, /predict, /explain, /segment/spleen, /reviews, and optional Orthanc routes | Live HTTP health, prediction, and segmentation checks passed previously; current Docker engine was unavailable to this audit session. |
| Web UI | React app in web/ | Browser upload checks passed previously. Current npm run build did not pass in the unmanaged shell because package dependencies were not installed there. |
| Docker | docker-compose.yml, api/Dockerfile, web/Dockerfile | API/web live HTTP checks passed previously. Docker uses CPU-only PyTorch. This audit session could not connect to the Docker engine, so no new container run is claimed. |
| Orthanc | Optional Compose profile and API connector | NOT VERIFIED end to end. The code/configuration exists; the latest recorded synthetic CT workflow did not reach inference. |
| Data/versioning | dvc.yaml, dvc.lock, DVC pointer files and audit scripts | Historical data audits are documented. Raw image/volume datasets and DVC input pointers are absent; a small local cache with model artifacts remains, and no remote is configured. Training cannot be reproduced from this package alone. |

.env.example documents runtime settings. Relative paths resolve from the project root. The default classifier and segmentation checkpoints are packaged under artifacts/; native API and Compose select the same segmentation candidate. Runtime startup and /health report its path, SHA-256, checkpoint ROI, threshold, and TTA setting.

## Performance evidence and limits

The classifier is the 8k/10k DenseNet-121 weight soup, SHA-256 082bc9dc5ff0399a6eeb07ae4dd9146ab258c39829e2c19a5c3eccc2c1cd8cc8. Its macro AUROC/AP was 0.7953/0.1594 on the reused 5,000-patient panel (13 labels with at least 30 positives) and 0.7926/0.2258 on the reused 2,138-patient panel (9 labels). Both panels were used during development; neither is an independent test set. NIH labels are report-mined, and absolute performance is modest. The ensemble that scored 0.8028 on the 2,138 panel is an undeployed candidate; the 15.5k checkpoint-to-manifest link is NOT VERIFIED.

The segmentation checkpoint was selected at epoch 125 using the same six validation cases that produced mean argmax Dice 0.8327; that value is optimistic. A repeated six-case internal holdout scored 0.7883 and has prior project evaluation history. It is not a fresh project-wide independent test. These Dice values describe segmentation overlap, not classification accuracy.

This is an educational, non-diagnostic prototype. It has no clinical validation, no patient-specific safety controls, and no evidence that a score or mask improves care. Do not use real patient data or use outputs for diagnosis or treatment.

## Evidence references

- ../reports/project1_closeout_2026-11-03.md — model metrics, splits, limitations, and segmentation retrain record.
- data_inventory.md — current local/package data availability, DVC, and GitHub boundary.
- ../reports/project1_api_fix_regression_2026-11-03.md — API settings, input guard, and regression audit.
- ../reports/Nexvia_AIML02_Demo_Segmentation_Audit_2026-11-03.md — API, browser, and Orthanc verification boundary.
- ../reports/verification_screenshots/final_verification_results_2026-11-03.md — earlier Docker, browser, and clean-install checks.

