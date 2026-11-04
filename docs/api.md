# API guide

FastAPI serves the classifier, attribution, segmentation, review, and Orthanc workflows. Start the application using the root [README](../README.md).

## Routes

| Method and route | Workflow | Input and output |
|---|---|---|
| `GET /health` | Service and model status | Returns readiness and active checkpoint/configuration metadata. |
| `POST /predict` | Chest X-ray classification | PNG/JPEG upload; returns 14 report-mined finding scores. |
| `POST /explain` | Classifier visualization | PNG/JPEG upload plus a `label`; returns a Grad-CAM overlay and class attribution information. |
| `POST /segment/spleen` | 3D spleen segmentation | NIfTI upload; returns a binary mask and preview data. |
| `POST /reviews` | Browser review workflow | Saves a demonstration review event to the configured local JSONL file. |
| `POST /orthanc/sync` | DX synchronization | Processes eligible stable Orthanc DX studies and records classification results. |
| `POST /orthanc/series/{series_id}/segment/spleen` | Orthanc CT segmentation | Converts an eligible CT series to the segmentation input format and returns a result summary, mask URL, and provenance metadata. |
| `GET /orthanc/series/{series_id}/segmentation-results` | CT result list | Lists saved segmentation summaries and mask retrieval links for a series. |
| `GET /orthanc/series/{series_id}/segmentation-results/{result_id}/mask` | CT mask retrieval | Streams the stored NIfTI-GZIP mask and its SHA-256 response header. |

### Orthanc CT series handling

A series marked complete can be processed directly. When Orthanc reports `Unknown`, `expected_instances` may be supplied as an independently obtained count between 2 and 512; the API compares it with the retrieved instance count and rechecks series membership and stability. The expected count is used as a completeness input before processing.

## Outputs and local storage

`NEXVIA_ORTHANC_SEGMENTATION_RESULTS_PATH` selects the JSONL summary file and defaults to `demo/orthanc_segmentations.jsonl`. `NEXVIA_ORTHANC_SEGMENTATION_MASK_DIR` selects the NIfTI-GZIP mask folder and defaults to `demo/orthanc_masks/`. Relative paths resolve from the project root.

Each CT record links a result ID to its Orthanc series and includes the output mask digest, model checkpoint digest, inference settings, series metadata, and completeness basis. Masks are stored in NIfTI-GZIP format. Review and result records are local demonstration outputs; they are not a shared production audit service.

## Response interpretation

Chest outputs are uncalibrated sigmoid scores for 14 report-mined findings. Grad-CAM is a coarse visual attribution. The spleen route returns a binary segmentation mask; Dice measures mask overlap. These outputs support model demonstration and review of the pipeline, not clinical decisions.