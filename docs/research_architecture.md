# Model architecture and design notes

This document describes the architecture used in the application and how it fits the two project workflows.

## Classifier

The chest pathway uses a pretrained DenseNet-121 visual encoder with a 14-label output head. The deployed checkpoint is a 50/50 weight average of two DenseNet checkpoints, followed by BatchNorm recalibration. The resulting model is loaded once by the FastAPI service and reused for image prediction and Grad-CAM generation.

The classification sequence is:

1. Read a PNG/JPEG image and apply the classifier's image preprocessing.
2. Pass the image through DenseNet-121.
3. Convert 14 output logits to sigmoid scores.
4. Return scores to the browser; optionally generate a coarse Grad-CAM map for one label.
5. Store a reviewer action as a local demonstration event when requested.

The 8k/10k checkpoint average is the API classifier. An older DenseNet-121 checkpoint, 6c32, is the comparison reference used in the documented development tables.

## Spleen segmenter

The CT workflow is an independent MONAI 3D U-Net. It uses a 96³ sliding-window ROI with threshold 0.50 and no flip TTA in the API. The output is a binary mask with visual previews. The API can accept a NIfTI directly or process a CT series retrieved through Orthanc and converted into the segmentation input format.

## Application composition

The architecture stays modular: React handles upload and review, FastAPI routes inputs to the matching model, model outputs are returned to the browser, and optional local logs record demonstration review and CT result metadata. Docker Compose starts the web and API services. The hosted Gradio page is a compact presentation of the two model paths rather than a copy of the full React application.

## Evaluation approach

The classifier tables use macro AUROC and macro AP across labels with at least 30 positive patients on each panel, plus paired patient bootstrap intervals for model differences. The spleen workflow reports case-level Dice and cohort means. Classifier panels were reused during development, and segmentation checkpoint selection used the six validation cases; present these as development results. NIH labels are report-mined. The system is an educational prototype, not a clinical tool.