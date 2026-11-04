# Classifier architecture

## Deployed model

The application uses DenseNet-121 with a 14-label sigmoid head for NIH ChestX-ray14 findings. The API checkpoint is an equal-weight average of the 8k and 10k DenseNet checkpoints with BatchNorm recalibration. The model accepts a preprocessed chest X-ray and returns 14 scores; the optional Grad-CAM route creates a coarse attribution overlay for a selected label.

## Evaluation summary

The API model was compared with the earlier 8k reference checkpoint, 6c32:

| Development panel | Metric | 6c32 reference | API model | Labels with ≥30 positives |
|---|---|---:|---:|---:|
| 5,000 patients | Macro AUROC | 0.7755 | 0.7953 | 13 |
| 5,000 patients | Macro AP | 0.1519 | 0.1594 | 13 |
| 2,138 patients | Macro AUROC | 0.7700 | 0.7926 | 9 |
| 2,138 patients | Macro AP | 0.2093 | 0.2258 | 9 |

Both panels were reused during candidate comparison. NIH labels are report-mined, and classifier outputs are not calibrated clinical probabilities. These figures describe the project development cohorts.

## Architecture choices

The model keeps a single DenseNet-121 backbone to maintain a compact inference path. An earlier DenseNet+Swin fusion screen scored below its DenseNet reference under the recorded setup, so the deployed API remains the DenseNet weight-average model. The 14 labels are modeled as separate sigmoid outputs; there is no text encoder, BERT component, or external LLM in the classifier path.

Grad-CAM is included as a coarse visual attribution. It does not establish why a prediction was made or provide a lesion boundary. The CT segmentation workflow is a separate MONAI 3D U-Net described in [the system architecture](architecture.md).