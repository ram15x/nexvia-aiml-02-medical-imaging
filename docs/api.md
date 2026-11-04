# API reference

This API serves a local educational prototype. It is not a clinical service. Run the API and UI using the commands in the root README.

## Main routes

| Method and route | Purpose | Input / output |
|---|---|---|
| `GET /health` | Readiness and configured model metadata | Reports classifier and segmentation checkpoint metadata, including the segmentation checkpoint path/hash and configured ROI, threshold, and flip-TTA setting. |
| `POST /predict` | Classify a chest X-ray | Multipart image upload; returns 14 uncalibrated report-mined finding scores. |
| `POST /explain` | Produce a coarse Grad-CAM overlay | Multipart chest X-ray plus a `label` form field; returns the overlay and class attribution information. This is not a detector or a clinical explanation. |
| `POST /segment/spleen` | Segment an uploaded NIfTI spleen CT | Multipart NIfTI upload; returns a binary mask and preview data. |
| `POST /reviews` | Record a demonstration review action | Review event is appended to the configured local JSONL audit file. It does not represent a clinical decision. |
| `POST /orthanc/sync` | Process eligible stable Orthanc DX studies | Pulls eligible studies, classifies DX images, and stores results in the configured JSONL file. |
| `POST /orthanc/series/{series_id}/segment/spleen` | Convert and segment a stable Orthanc CT series | Complete series need no extra parameter. For status `Unknown`, optional query parameter `expected_instances` must be an independently obtained integer from 2 to 512 matching the Orthanc instance count. Missing, inconsistent and incomplete statuses are rejected. The series membership and study stability are rechecked after retrieval. A matching count does not prove completeness. Returns summary, `result_id`, `mask_sha256`, `mask_url`, and completeness provenance; saves the NIfTI mask and JSONL locally. |
| `GET /orthanc/series/{series_id}/segmentation-results` | List stored CT segmentation results | Returns metadata and mask retrieval URLs for results recorded for the series. |
| `GET /orthanc/series/{series_id}/segmentation-results/{result_id}/mask` | Retrieve a saved CT mask | Streams the NIfTI-GZIP file and provides its SHA-256 in the `X-Content-SHA256` response header. The API verifies the stored digest before serving. |

## Local CT result storage

`NEXVIA_ORTHANC_SEGMENTATION_RESULTS_PATH` selects the JSONL metadata file and defaults to `demo/orthanc_segmentations.jsonl`. `NEXVIA_ORTHANC_SEGMENTATION_MASK_DIR` selects the mask directory and defaults to `demo/orthanc_masks/`. Relative paths resolve from the project root. These are local demo outputs and are excluded from Git.

Each saved result links a generated result ID to the Orthanc series and records the output mask SHA-256, segmentation checkpoint SHA-256, threshold, flip-TTA setting, series status, completeness basis, supplied count, summary values, and non-diagnostic disclaimer. `operator_supplied_count_match` records a user's supplied count; it is not independent proof that every slice is correct. The mask is stored as NIfTI-GZIP, not converted into a DICOM SEG or DICOM SR object and not written back into Orthanc. The GET endpoints provide retrieval from the API host's local storage; they are not a shared or durable production storage service.

## Verification boundary

The CT list/retrieval flow was exercised using one generated synthetic 24-slice CT series with explicit completeness metadata. A separate public de-identified 81-slice CT series was reported as Unknown by Orthanc: the default request and a wrong expected count returned HTTP 422, while the independently reported matching count allowed processing, with the saved mask checksum verified on retrieval. This is one interoperability check without a reference mask, not an accuracy or broad scanner-compatibility test. Multi-user access control, retention, backup, DICOM SEG/SR interoperability, and clinical use are not verified. The data citation and check are in `reports/requirements_and_verification.md`.
