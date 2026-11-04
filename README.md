# Medical Imaging AI Review Platform

**Nexvia Technologies · AIML-02**

I built this educational prototype for two separate medical-imaging tasks: chest X-ray finding classification and 3D spleen segmentation from CT. The internship portal describes the assignment as **AI-Assisted Medical Imaging and Diagnosis System**. This repository documents what I built, the checks I ran, and the limits of the results. It is not for diagnosis or treatment.

## Current status

The local release package is assembled for handoff and synchronized to the public GitHub `main` branch (latest verified package commit: `99fe134fabb32de756a3659c807e359bb3f197fe`). The complete packaged test suite most recently passed **58 tests with 2 warnings** in a disposable Python 3.12 CPU container. Test-only dependencies were installed inside that temporary container; the package was mounted read-only. This does not verify the pinned Windows/Python 3.10/CUDA 12.8 environment. Synthetic Orthanc DX classification and a synthetic CT series with explicit completeness metadata passed isolated API checks. CT segmentation output is saved locally with a SHA-256 record and can be listed and retrieved through the API. Ordinary CT series may have Orthanc status Unknown and are rejected by the current route; these tests do not establish broad DICOM compatibility or clinical use. There is no hosted deployment. The portal requires both assigned projects in one Google Drive folder and locks the link after submission; I have not submitted the combined internship deliverables.

## What the prototype does

- **Chest X-ray:** a React page sends a PNG/JPEG to FastAPI. A DenseNet-121 model returns 14 uncalibrated finding scores. Grad-CAM can show a coarse attribution overlay. Review actions are recorded locally for demonstration; they are not clinical decisions.
- **Spleen CT:** a NIfTI volume is sent to FastAPI and segmented by a MONAI 3D U-Net. The app returns a mask and tri-planar previews. This is a separate model path from chest X-ray classification.
- **Orthanc:** a synthetic single-frame DX study was uploaded to Orthanc, picked up by the sync route, classified, and written to the local JSONL results log. A synthetic CT series with explicit expected instance count passed the Orthanc-to-segmentation HTTP route. Its NIfTI-GZIP mask is saved locally; result metadata and the mask can be retrieved through separate API routes. These synthetic checks do not establish broad CT compatibility. No DICOM SEG/SR object is written back to Orthanc.

The architecture is in [docs/architecture.md](docs/architecture.md), API routes and payloads in [docs/api.md](docs/api.md), and the available data and what is missing from this package in [docs/data_inventory.md](docs/data_inventory.md).

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
| Public medical-imaging data; DVC | NIH ChestX-ray14 and MSD Task09 were used. The included DVC lock records only a historical 3-epoch subset screen; it is not the deployed classifier training pipeline. | Full images and patient-level manifests are not in this package; no DVC remote is configured. The DVC stage also contains a machine-specific Windows Python path. Training is not reproducible from this package alone. |
| DenseNet/EfficientNet classification and MONAI 3D U-Net segmentation | Separate DenseNet-121 and MONAI 3D U-Net inference paths are implemented. | The two model paths are prototypes with the evaluation limits above. |
| DICOM workflow and Orthanc | Synthetic DX flow and a constrained synthetic CT flow passed. CT masks and provenance summaries are persisted locally and can be retrieved through the API. | This is local NIfTI-GZIP storage, not DICOM SEG/SR write-back. Stable series marked Unknown are rejected, and real CT compatibility is unverified. |
| Grad-CAM | Coarse attribution overlay is implemented and smoke-checked. | It is not an object detector or clinical explanation. |
| React review viewer, overlays, scores, Approve/Reject, audit log | Browser upload, scores/overlays, local review actions, and local JSONL audit output are implemented. | Demo-only review records; no radiologist workflow or clinical audit system. |
| Docker | API and web containers passed recorded live HTTP checks. | Docker uses CPU-only PyTorch. Synthetic Orthanc CT segmentation passed only for an explicitly complete synthetic series; compatibility with ordinary CT series is limited by Orthanc status Unknown. |
| Limits/bias documentation | Model card and data inventory describe report-mined labels, panel reuse, sample limits, and educational use. | No clinical validation or evidence of benefit to care. |
| Working demo and technical documentation | Local UI/API, setup instructions, architecture, data notes, test results, and deck are included. | The app is not hosted or screen-recorded yet. After Project 2, I will include a short walkthrough of the local demo in my 10-minute presentation recording. |
| Presentation and reflection | The 10-minute deck is included. | I will record and present it after Project 2, with a short local demo walkthrough; I still need to write my own reflection. |

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

The complete packaged suite passed **57 tests with 2 warnings** in a disposable Python 3.12 CPU container; the project folder was mounted read-only. That run preceded a newly added asymmetric CT orientation regression test. Its core assertions passed in a direct smoke check using the cached API image, but the complete suite has not been rerun. The warnings are invalid-escape warnings in `test_ct_series.py`. A direct run in the host Python 3.10 environment could not collect tests because `torchvision` and `nibabel` were missing there; that environment was left unchanged. This does not verify the pinned Windows/CUDA environment. See `reports/requirements_and_verification.md` for the exact command and details. The pinned Windows install uses a large CUDA PyTorch wheel; allow several gigabytes before choosing this route.

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
- The 15.5k classifier checkpoint's training-manifest link is **NOT VERIFIED**.
- Synthetic Orthanc DX sync/classification and one synthetic Orthanc CT segmentation request were exercised. The CT mask is persisted as NIfTI-GZIP with a checksum and retrieval route. Ordinary CT series that Orthanc marks Unknown are rejected; broad DICOM compatibility remains unverified.
- The model does not provide calibrated clinical probabilities, clinical validation, or evidence of improved patient outcomes.
- The prototype is for education only and must not guide diagnosis or treatment.

## Before the combined portal submission

The portal accepts one Google Drive link for **both** assigned projects and locks the submission after it is sent. Project 2 is separate and must be included before I submit. I have not submitted the link. After Project 2, I will record and present the 10-minute deck with a short walkthrough of the local demo, then write my own reflection.
