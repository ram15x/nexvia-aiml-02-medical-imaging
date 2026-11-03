# Project 1 requirements and verification

## Portal brief

The Nexvia portal assigns **AI-Assisted Medical Imaging and Diagnosis System**. It asks for public imaging data; preprocessing and DVC; transfer-learning classification; MONAI 3D U-Net segmentation; Grad-CAM; an Orthanc service that pulls studies, runs inference, and stores results; a React review page with scores, overlays, approve/reject actions and an audit log; Docker; documentation of model limits and bias; and a working demo, GitHub repository, architecture/data/API documentation, test results, a 10-minute presentation, and an individual reflection.

The portal specifies one master Google Drive folder for both assigned projects and says the link can be submitted only once. This project has not been submitted. The separate Project 2 folder is not represented as complete here.

## Requirement-by-requirement status

| Portal item | Status | Evidence and boundary |
|---|---|---|
| Public data and DVC | Partial | NIH ChestX-ray14 and MSD Task09 were used. DVC files and audit scripts are included. Raw datasets, patient-level manifests, and a configured remote are absent from this package; training cannot be reproduced from it. See `docs/data_inventory.md`. |
| Classifier and segmentation models | Implemented | DenseNet-121 returns 14 report-mined finding scores; MONAI 3D U-Net returns a spleen mask. They are independent paths with different inputs. See `docs/architecture.md` and model card. |
| Grad-CAM | Implemented; smoke-checked | A coarse attribution overlay is available for the classifier. It is not a bounding-box detector or clinical explanation. |
| Orthanc DICOM service | Partial; synthetic DX path verified | A generated single-frame DX study passed upload, stable-study sync, classifier inference, and local JSONL result storage in an isolated Orthanc/API check. This does not test clinical images. The Orthanc CT-series segmentation path has not passed a live end-to-end check; no DICOM SEG/SR is written back. |
| React viewer and review actions | Implemented; browser upload checks recorded | The UI supports image/volume upload, scores or masks, and demo approve/reject actions. Review events are local JSONL records, not a clinical audit workflow. |
| Docker | API/web checks passed previously | The API and web services returned successful live HTTP responses in recorded checks. Docker uses CPU-only PyTorch. The test suite below also ran in the local CPU Docker test image. |
| Limits and bias | Documented | The model card and data inventory describe reused panels, report-mined labels, missing input data, limited cohorts, and non-diagnostic use. No clinical validation is claimed. |
| Working demo | Local demo implemented; checks recorded | Browser upload and live HTTP API/web checks are recorded. Orthanc verification is limited to synthetic DX as stated above; CT segmentation through Orthanc is unverified. There is no hosted deployment URL. |
| GitHub, README, architecture/data/API docs | Present | The public repository contains the release package and setup documentation. The canonical package for the handoff is `submission_final/`. |
| Test results | Full suite rerun; 56 passed, 2 warnings | The complete packaged suite reported **56 passed, 2 warnings** in the local Docker test image. The image uses Python 3.12 and CPU PyTorch; this is not the pinned Python 3.10/CUDA 12.8 Windows environment. The two warnings are invalid-escape warnings in `test_ct_series.py`; no test failed. |
| 10-minute presentation | Deck included; visual check performed | The 10-slide presentation has been opened in PowerPoint and reviewed in slide sorter. Slide 10 remains a prompt for the student’s own reflection. Rendering to slide images was not performed. |
| Individual reflection and recording | Student action remains | The reflection must describe only what the student personally did and learned. The student will record and present the demo. |

## Test-suite details

Command: `python -m pytest -q`, run from the package root in the existing local Docker API image with the source mounted read-only. The test environment was Python 3.12, torch 2.15.0+cpu, torchvision 0.29.2+cpu, MONAI 1.6.1, and pytest 9.2.0, with the small missing test libraries installed in the disposable environment. No model training, data download, official test data, or checkpoint edit was part of this verification.

The first run exposed a stale assertion that expected segmentation flip TTA to be enabled, although the API configuration disables it. It also exposed two test-support audit scripts missing from the packaged folder. The assertion was corrected to match the API setting, and the missing support scripts were added to the package. After those changes, both the development suite and packaged suite reported 56 passed and 2 warnings. The pinned Windows/CUDA suite was not rerun in this check.

## Orthanc scope

The successful test used generated synthetic DX DICOM, not a patient study. Orthanc accepted the instance; the API observed the stable study, ran chest classification, and returned/stored 14 uncalibrated scores. This verifies the demo’s DX pull/infer/store path for that synthetic case. It does not establish compatibility with all DICOM studies or clinical use. The manual CT series conversion and spleen inference path remains unverified end to end.

## Deck check

The current 10-slide file opened in Microsoft PowerPoint. The slide sorter showed a consistent template and no obvious off-slide shapes. A full slide-image render was not produced, so this is a direct PowerPoint visual inspection, not a rendering/export check. Slide text remains small at sorter scale; review and rehearse each slide at presentation size before recording.
## Latest pre-submission recheck

The complete suite was rerun after the API changes in the existing CPU API image. The package folder was mounted read-only; test-only dependencies were installed in the disposable container and removed with it. No datasets, images, or model weights were downloaded, and no project dependency file or checkpoint was changed.

Exact command (PowerShell, from the package root):

```powershell
$projectMount = "$((Get-Location).Path):/app:ro"
docker run --rm --entrypoint sh -v $projectMount -w /app submission_final-api -lc 'set -e; python -m pip install --disable-pip-version-check --no-input pytest==9.2.0 pandas==2.3.3 scikit-learn==1.7.2 scikit-image==0.25.2; python -m pytest -q -p no:cacheprovider'
```

Result: **56 passed, 2 warnings in 9.03 seconds**. The warnings are Python invalid-escape warnings at `test_ct_series.py:204` for `\(` and `\)`; no test failed. The image used Python 3.12 and CPU PyTorch, so this does not verify the pinned Windows/Python 3.10/CUDA 12.8 environment.

For completeness, a direct host attempt (`python -m pytest -q`) stopped during collection because that host environment lacked `torchvision` and `nibabel`. The first direct attempt in the API image also lacked `pytest`. The disposable test run above supplied the missing test dependencies and completed all tests.

## Orthanc live check and saved synthetic DX evidence

The local Orthanc service answered `GET http://127.0.0.1:18042/system` with HTTP 200 and reported version 1.13.0. The already-saved isolated synthetic DX integration record reports `status=processed`, source `orthanc_stable_study_changes`, 14 finding scores, and classifier checkpoint SHA-256 prefix `082bc9dc5ff0`. Its JSONL file SHA-256 is `4f782294a9dc9d8fe4db88904f5eea3c1c1fd7fe7b10a8dc05984381d27a088e`. This is a synthetic, single-frame DX demonstration; this pass did not upload another study. Orthanc CT-series conversion and spleen segmentation remain unverified through Orthanc.

## PowerPoint visual review

I opened `deliverables/Medical_Imaging_AI_Review_Platform.pptx` in PowerPoint and reviewed all ten slides in Slide Sorter. I also inspected the cover, classifier results, spleen segmentation, limitations, and reflection slides at presentation size. The layout is consistent and I saw no text outside the slide bounds. I did not export slide images or claim a rendered-image review.

