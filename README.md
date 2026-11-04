# Medical Imaging AI Review Platform

**Nexvia Technologies · AIML-02**

I built this educational prototype for two separate medical-imaging tasks: chest X-ray finding classification and 3D spleen segmentation from CT. The internship portal describes the assignment as **AI-Assisted Medical Imaging and Diagnosis System**. This repository documents what I built, the checks I ran, and the limits of the results. It is not for diagnosis or treatment.

## Current status

The local release package is assembled for handoff. The complete packaged test suite most recently passed **69 tests** in a disposable Python 3.12 CPU container. Test-only dependencies were installed inside that temporary container; the package was mounted read-only. This does not verify the pinned Windows/Python 3.10/CUDA 12.8 environment. Synthetic Orthanc DX and CT checks passed, as did an isolated interoperability check on one public de-identified 81-slice CT series. For an Orthanc series marked Unknown, an independently supplied expected slice count is required and must match; this count does not prove completeness. CT segmentation output is saved locally with a SHA-256 record and can be retrieved through the API. These checks do not establish broad DICOM compatibility or clinical use. A separate public Gradio demo is hosted on Hugging Face Spaces; see the Hosted demo section below. It does not include the full local React/FastAPI workflow. The portal requires both assigned projects in one Google Drive folder and locks the link after submission; I have not submitted the combined internship deliverables.

## What the prototype does

- **Chest X-ray:** a React page sends a PNG/JPEG to FastAPI. A DenseNet-121 model returns 14 uncalibrated finding scores. Grad-CAM can show a coarse attribution overlay. Review actions are recorded locally for demonstration; they are not clinical decisions.
- **Spleen CT:** a NIfTI volume is sent to FastAPI and segmented by a MONAI 3D U-Net. The app returns a mask and tri-planar previews. This is a separate model path from chest X-ray classification.
- **Orthanc:** a synthetic single-frame DX study was uploaded to Orthanc, picked up by the sync route, classified, and written to the local JSONL results log. A synthetic CT series and one public de-identified CT series passed the Orthanc-to-segmentation route. For the latter, Orthanc reported Unknown and an independently supplied count matched all 81 slices. Its NIfTI-GZIP mask and provenance are stored locally and retrievable through the API. These checks do not establish broad CT compatibility. No DICOM SEG/SR object is written back to Orthanc.

The architecture is in [docs/architecture.md](docs/architecture.md), API routes and payloads in [docs/api.md](docs/api.md), and the available data and what is missing from this package in [docs/data_inventory.md](docs/data_inventory.md).

## Hosted demo

[Open the hosted demo](https://huggingface.co/spaces/Ram15x/medical-imaging-ai-review). It is a separate, lightweight Gradio deployment that uses the same classifier and spleen checkpoint files as the local package (classifier SHA-256 `082bc9dc…`; segmentation SHA-256 `46622de7…`). It is not a hosted copy of the full React/FastAPI application: Grad-CAM, review/audit actions, and Orthanc workflows are available only in the local project.

I exercised both hosted tabs in a browser with generated, non-patient inputs. The classifier returned 14 scores; the CT path returned sagittal, coronal, and axial previews. This confirms those inference paths ran, not model accuracy or clinical performance. The hosted page warns users not to upload patient data.

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
| Public medical-imaging data; DVC | NIH ChestX-ray14 and MSD Task09 were used. Two retained demo examples are tracked by DVC, with a local backup. The included DVC lock records a historical 3-epoch subset screen. | The demo backup is local to this laptop. Full training images and patient-level manifests are not in the package; the historical stage has a machine-specific Python path and does not reproduce the shipped checkpoints. |
| DenseNet/EfficientNet classification and MONAI 3D U-Net segmentation | Separate DenseNet-121 and MONAI 3D U-Net inference paths are implemented. | The two model paths are prototypes with the evaluation limits above. |
| DICOM workflow and Orthanc | Synthetic DX and CT flows and one public de-identified CT series passed. CT masks and provenance summaries are persisted locally and can be retrieved through the API. | Unknown-status series require an independently supplied matching count; a count alone does not prove completeness. This is local NIfTI-GZIP storage, not DICOM SEG/SR write-back. Broad real-CT compatibility is unverified. |
| Grad-CAM | Coarse attribution overlay is implemented and smoke-checked. | It is not an object detector or clinical explanation. |
| React review viewer, overlays, scores, Approve/Reject, audit log | Browser upload, scores/overlays, local review actions, and local JSONL audit output are implemented. | Demo-only review records; no radiologist workflow or clinical audit system. |
| Docker | API and web containers passed recorded live HTTP checks. | Docker uses CPU-only PyTorch. Orthanc CT interoperability was checked on one synthetic and one public de-identified series; other scanners and series are not established. |
| Limits/bias documentation | Model card and data inventory describe report-mined labels, panel reuse, sample limits, and educational use. | No clinical validation or evidence of benefit to care. |
| Working demo and technical documentation | Local UI/API, setup instructions, architecture, data notes, test results, and deck are included. | A limited Gradio demo is deployed; the full React/FastAPI app remains local. The hosted demo was browser-smoke-tested with generated synthetic inputs and does not include Grad-CAM, review/audit actions, or Orthanc. Ram will record the local demo walkthrough in the 10-minute presentation after Project 2. |
| Presentation and reflection | The 10-minute deck is included. | Slides 3, 4 and 9 contain older deployment, Orthanc and test-status wording; see `deliverables/README.md` before recording. I will record and present it after Project 2 and write my own reflection. |

The full item-by-item check against the portal is [reports/requirements_and_verification.md](reports/requirements_and_verification.md).

## Versioned demo data

The two `.dvc` files in `demo/` record the retained chest and spleen examples. Their roughly 16 MB backup is in the private working project's `dvc-demo-store/`, outside this release. With DVC 3.67.1 installed, a local copy of this release next to that store can use `dvc remote add --local demo_local ../dvc-demo-store` and then `dvc pull -r demo_local demo/chest_xray_example.png.dvc demo/spleen_volume_example.nii.gz.dvc`. The examples and local backup are not published to GitHub. This versioning does not restore the missing original training dataset or reproduce model training. See [docs/data_inventory.md](docs/data_inventory.md).

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

The complete packaged suite passed **69 tests** in a disposable Python 3.12 CPU container with the project mounted read-only. The pinned Windows/CUDA test environment was not verified by that run. See `reports/requirements_and_verification.md` for the command and details. The pinned Windows install uses a large CUDA PyTorch wheel; allow several gigabytes before choosing this route.

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
- Synthetic Orthanc DX and CT checks and one public de-identified CT interoperability check were exercised. A series marked Unknown needs an independently supplied matching count; that does not prove completeness. The CT mask is persisted as NIfTI-GZIP with a checksum and retrieval route. Broad DICOM compatibility remains unverified.
- The model does not provide calibrated clinical probabilities, clinical validation, or evidence of improved patient outcomes.
- The prototype is for education only and must not guide diagnosis or treatment.

## Before the combined portal submission

The portal accepts one Google Drive link for **both** assigned projects and locks the submission after it is sent. Project 2 is separate and must be included before I submit. I have not submitted the link. After Project 2, I will record and present the 10-minute deck with a short walkthrough of the local demo, then write my own reflection.
