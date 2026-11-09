# API

Base URL in Docker: `http://localhost:18080`. All responses carry the disclaimer that outputs are not diagnoses.

| Method and path | Input | Output |
|---|---|---|
| `GET /health` | none | Checkpoint presence and load state for the classifier and the spleen model, with SHA-256 and settings. |
| `POST /predict` | multipart `file`: PNG, JPEG, or single-frame CR/DX DICOM (`.dcm`, or content type `application/dicom`). Max 12 MiB. | 14 findings with uncalibrated scores, model metadata and checkpoint hash. |
| `POST /explain` | multipart `file` (as above) and form `label` (one of the 14 labels) | PNG Grad-CAM overlay for that label. |
| `POST /segment/spleen` | multipart `file`: non-empty 3D NIfTI, at most 45 million voxels, with usable affine | Mask summary (voxel count, estimated volume), geometry check, and tri-planar preview images. |
| `POST /orthanc/series/{series_id}/segment/spleen` | Orthanc CT series ID, optional `expected_instances` | Converts the stable series to NIfTI and segments it. Stores a result record and mask. |
| `GET /orthanc/series/{series_id}/segmentation-results` | series ID | Recorded CT segmentation results for that series. |
| `GET /orthanc/series/{series_id}/segmentation-results/{result_id}/mask` | series and result IDs | The stored NIfTI-GZIP mask. |
| `POST /orthanc/{instance_id}/predict` | Orthanc instance ID | Scores one CR/DX instance and records the result. |
| `POST /orthanc/sync?limit=N` | `limit` 1–64 (default 16) | Consumes stable-study events and processes a bounded batch of CR/DX instances. Resumable. Returns HTTP 409 if a sync is already running. |
| `GET /orthanc/{instance_id}/results` | Orthanc instance ID | Recorded classifier results for that instance. |
| `POST /reviews` | JSON: `case_ref`, `reviewed_findings` (labels), `note`, `reviewer_action` in {`accepted_for_demo`, `needs_review`, `rejected`} | HTTP 201 and an event record appended to the review log, with the model checkpoint hash. |

## Automatic Orthanc pull

Set `ORTHANC_AUTO_SYNC_SECONDS` to a positive number to run `POST /orthanc/sync` on that interval in the background. The Docker configuration defaults to 60 seconds. It uses the same lock and cursor as the manual endpoint, so the two never run at once. Set it to `0` to disable.

## Configuration

Environment variables are listed in `.env.example`. Key ones: `NEXVIA_CHECKPOINT`, `NEXVIA_SEGMENTATION_CHECKPOINT`, `NEXVIA_SEGMENTATION_THRESHOLD`, `NEXVIA_SEGMENTATION_TTA`, `ORTHANC_URL`, `ORTHANC_USERNAME`, `ORTHANC_PASSWORD`, `ORTHANC_AUTO_SYNC_SECONDS`.
