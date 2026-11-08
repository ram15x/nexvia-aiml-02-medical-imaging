# Project results

This report summarizes the model outputs and application behavior presented for the Medical Imaging AI Review Platform.

## Chest X-ray classification

The deployed classifier is an equal-weight average of the 8k and 10k DenseNet-121 checkpoints with BatchNorm recalibration. It returns 14 sigmoid scores for NIH ChestX-ray14 report-mined findings. The comparison model is the earlier 8k checkpoint, 6c32.

Macro metrics are averaged over labels with at least 30 positive patients in each stated development panel.

| Panel | Model | Macro AUROC | Macro AP | Labels |
|---|---|---:|---:|---:|
| 5,000 patients | 6c32 reference | 0.7755 | 0.1519 | 13 |
| 5,000 patients | API classifier | 0.7953 | 0.1594 | 13 |
| 2,138 patients | 6c32 reference | 0.7700 | 0.2093 | 9 |
| 2,138 patients | API classifier | 0.7926 | 0.2258 | 9 |

The panels were reused during development and are reported as development evidence. Sigmoid scores are uncalibrated; Grad-CAM provides a coarse attribution view.

## Spleen CT segmentation

The deployed MONAI 3D U-Net was trained on 29 labeled volumes and selected using six validation cases. Six reserved internal held-out cases gave mean per-case Dice 0.3272, with one case at 0.0963. These figures summarize a small development cohort.

The API uses a 96³ sliding-window ROI, threshold 0.50, and disabled flip TTA. It returns a binary mask with tri-planar previews.

## Application flow

The React interface uploads a PNG/JPEG or NIfTI volume to FastAPI. FastAPI routes the input to the matching model and returns its scores, visualization, or mask. The optional Orthanc path supports DX synchronization and CT-series ingestion. The API stores demonstration review actions and CT mask metadata locally. Docker Compose starts the web and API services using CPU-based PyTorch.

## Intended use

The system demonstrates machine-learning workflows, API integration, visualization, and model review. It is educational and non-diagnostic; it is not intended to guide individual care.