# Medical Imaging AI Review Platform

**Nexvia Technologies · AIML-02**

I built this educational prototype for two separate medical-imaging tasks: chest X-ray finding classification and 3D spleen segmentation from CT. The internship portal describes the assignment as **AI-Assisted Medical Imaging and Diagnosis System**. This repository documents what I built, the checks I ran, and the limits of the results. It is not for diagnosis or treatment.

## Current status

The local release package is assembled for my final presentation. Its full test suite passed 56 tests with 2 warnings in a disposable Python 3.12 CPU container; this does not verify the pinned Windows/CUDA setup. The synthetic Orthanc DX route was demonstrated, while CT-series segmentation through Orthanc remains unverified. I still need to record the presentation and write my own reflection. The portal requires both assigned projects in one Google Drive folder and locks the link after submission; I have not submitted the combined internship deliverables.

## What the prototype does

- **Chest X-ray:** a React page sends a PNG/JPEG to FastAPI. A DenseNet-121 model returns 14 uncalibrated finding scores. Grad-CAM can show a coarse attribution overlay. Review actions are recorded locally for demonstration; they are not clinical decisions.
- **Spleen CT:** a NIfTI volume is sent to FastAPI and segmented by a MONAI 3D U-Net. The app returns a mask and tri-planar previews. This is a separate model path from chest X-ray classification.
- **Orthanc:** a synthetic single-frame DX study was uploaded to Orthanc, picked up by the sync route, classified, and written to the local JSONL results log. That verifies this small demo path only. The Orthanc CT-series-to-segmentation route has not passed a live end-to-end check.

The architecture and API flow are in [docs/architecture.md](docs/architecture.md). The available data and what is missing from this package are in [docs/data_inventory.md](docs/data_inventory.md).

## What the results say

### Chest X-ray classifier

The API uses a DenseNet-121 weight average of the 8k and 10k checkpoints, with BatchNorm recalibration. It was compared with the earlier 8k checkpoint, 6c32, on two reused development panels. Macro metrics below use labels with at least 30 positives on that panel.

| Panel | Model | Macro AUROC | Macro AP | Labels included |
|---|---|---:|---:|---:|
| 5,000 patients | 6c32 reference | 0.7755 | 0.1519 | 13 |
| 5,000 patients | API weight average | 0.7953 | 0.1594 | 13 |
| 2,138 patients | 6c32 reference | 0.7700 | 0.2093 | 9 |
| 2,138 patients | API weight average | 0.7926 | 0.2258 | 9 |

Both panels were reused while comparing candidates. They are **not independent test sets**. Labels are report-mined, performance is modest, and the scores are not calibrated clinical probabilities. An undeployed average of the API model and a 15.5k checkpoint scored 0.8028 macro AUROC on the 2,138-patient panel, but its training-manifest link is **NOT VERIFIED**. It is not the deployed model and is not presented as an API improvement. See [reports/classifier_results.md](reports/classifier_results.md) and [docs/model_card.md](docs/model_card.md).

### Spleen CT segmentation

The packaged U-Net was trained on 29 labeled volumes. Its selected checkpoint reached mean argmax Dice 0.8327 on six validation cases; those same cases selected the checkpoint, so the score is optimistic. A repeated six-case internal holdout scored 0.7883, but those cases had prior project evaluation history, so this is not a fresh project-wide independent test. The old and new runs used different inference ROIs; these numbers do not establish a clean like-for-like improvement. Full per-case scores and limitations are in [reports/project_closeout.md](reports/project_closeout.md).

## Portal requirement check

| Assigned requirement | What is present | Remaining limit |
|---|---|---|
| Public medical-imaging data; DVC | NIH ChestX-ray14 and MSD Task09 were used; DVC configuration and audit scripts are included. | Full images and patient-level manifests are not in this package; no DVC remote is configured. Training is not reproducible from this package alone. |
| DenseNet/EfficientNet classification and MONAI 3D U-Net segmentation | Separate DenseNet-121 and MONAI 3D U-Net inference paths are implemented. | The two model paths are prototypes with the evaluation limits above. |
| DICOM workflow and Orthanc | Synthetic DX upload → stable-study sync → classifier inference → local result storage passed. | The live test used generated synthetic input. Orthanc CT-series segmentation has not passed end-to-end verification. No DICOM SEG/SR is written back. |
| Grad-CAM | Coarse attribution overlay is implemented and smoke-checked. | It is not an object detector or clinical explanation. |
| React review viewer, overlays, scores, Approve/Reject, audit log | Browser upload, scores/overlays, local review actions, and local JSONL audit output are implemented. | Demo-only review records; no radiologist workflow or clinical audit system. |
| Docker | API and web containers passed recorded live HTTP checks. | Docker uses CPU-only PyTorch. Orthanc CT segmentation is not verified. |
| Limits/bias documentation | Model card and data inventory describe report-mined labels, panel reuse, sample limits, and educational use. | No clinical validation or evidence of benefit to care. |
| Working demo and technical documentation | Local UI/API, setup instructions, architecture, data notes, tests, and this deck are included. | A hosted deployment is not included. The demo is local. |
| Presentation and reflection | A 10-slide presentation is included. | I still need to record and present it, and write my own reflection accurately. |

The full item-by-item check against the portal is [reports/requirements_and_verification.md](reports/requirements_and_verification.md).

## Run the local demo

The project includes the classifier and spleen checkpoints under `artifacts/`. Copy `.env.example` to `.env` before running. Keep `.env` private.

### Option A: Docker Compose

Install Docker Desktop and start its Linux engine. From the folder containing `docker-compose.yml`:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Open the web app at `http://localhost:15180` and API health at `http://localhost:18080/health`. Stop the services with `Ctrl+C` or `docker compose down`. Docker runs CPU-only PyTorch; the prior native Windows test used CUDA 12.8 wheels.

### Option B: Windows with Python and Node

Use Python 3.10 and Node.js 20 or later. Run the commands from the project root.

```powershell
Copy-Item .env.example .env
py -3.10 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-windows-tested.txt
python -m pytest -q
```

I reran the complete suite after the API changes: **56 passed, 2 warnings**. The run used the existing Python 3.12 CPU API image, with the missing test packages installed only in a disposable container; the project folder was mounted read-only. The two warnings are invalid-escape warnings in `test_ct_series.py`. A direct run in the host Python 3.10 environment could not collect tests because `torchvision` and `nibabel` were missing there; I left that environment unchanged. This is not the pinned Windows/CUDA environment. See `reports/requirements_and_verification.md` for the exact command and details. The pinned Windows install uses a large CUDA PyTorch wheel; allow several gigabytes before choosing this route.

Start the API in one PowerShell window:

```powershell
& .\.venv\Scripts\Activate.ps1
python -m uvicorn api.app:app --env-file .env --host 127.0.0.1 --port 18080
```

In a second PowerShell window:

```powershell
Set-Location .\web
npm ci
$env:VITE_API_URL = "http://127.0.0.1:18080"
npm run dev -- --host 127.0.0.1 --port 15180
```

Open `http://127.0.0.1:15180`. The backend API is at `http://127.0.0.1:18080`. The shipped checkpoints are already in `artifacts/`; keep their filenames or update the corresponding `.env` setting if you deliberately use a different checkpoint.

## What is not established

- Neither classifier panel is an independent test set. Both were reused for development decisions.
- Segmentation validation selected the checkpoint. The internal holdout was evaluated before and is not project-wide untouched.
- The 15.5k checkpoint’s manifest link is **NOT VERIFIED**.
- Synthetic Orthanc DX sync and classification were exercised. Live Orthanc CT segmentation remains unverified.
- The model does not provide calibrated clinical probabilities, clinical validation, or evidence of improved patient outcomes.
- The prototype is for education only and must not guide diagnosis or treatment.

## Before the combined portal submission

The portal accepts one Google Drive link for **both** assigned projects and locks the submission after it is sent. Project 2 is separate and must be included before I submit. I have not submitted the link. My presentation recording and personal reflection remain mine to complete.
