# Nexvia AIML-02 demo and segmentation audit — 2026-11-03

> **Historical snapshot:** This report records the first Nov 3 bring-up attempt. Its NOT VERIFIED outcomes describe that attempt only. Later local HTTP, browser, fresh-install, Docker, synthetic Orthanc and one public de-identified CT interoperability check are recorded in `final_checks.md` and `requirements_and_verification.md`. Broad real-world DICOM compatibility remains unverified. Keep this report as the failed-attempt record, not as the current status.

## Summary

- **Fresh venv + pinned install:** NOT VERIFIED. Created a new Python 3.10.1 venv in the writable audit workspace. Offline install stopped because no cached wheel existed for `annotated-doc==0.0.5`. No packages or datasets were downloaded.
- **FastAPI:** The native server started and `/health` returned HTTP 200 once, reporting the checkpoint present. A subsequent live HTTP prediction request timed out / could not connect. The same `/predict` route returned HTTP 200 through an in-process ASGI client, with 14 findings and the expected soup checkpoint SHA-256. This confirms application inference, but it does not substitute for the requested live HTTP upload.
- **Frontend:** Vite 8.3.5 started with a workspace-only config. Chrome could not connect to the local URL (`net::ERR_CONNECTION_TIMED_OUT`); therefore a browser upload was **NOT VERIFIED**.
- **Docker / WSL:** Docker client 29.8.0 was present, but the Docker engine pipe was unavailable. `wsl --status` returned `Wsl/E_ACCESSDENIED`. Orthanc was not tested.
- **Segmentation:** Fresh inference and a prediction/reference overlay completed on validation case `spleen_10`. Geometry and binary labels matched. The visualization-only source-grid Dice was 0.4096 and predicted/reference voxel ratio was 2.16; this is one validation case, not a new performance estimate.
- **15.5k seed-42 manifest:** NOT VERIFIED. Its run directory contains checkpoints, `run_summary.json`, and `training_history.json`, but no saved manifest, config file, or log identifying the patient training manifest. Summary/history hashes are recorded below; neither binds the run to the clean-set manifest.

## A. Demo bring-up evidence

### Fresh environment attempt

Created `AUDIT_WORKSPACE/demo_audit_venv` with Python 3.10.1. Its initial package list contained only pip and setuptools. The offline requirements install failed exactly:

```text
ERROR: Could not find a version that satisfies the requirement annotated-doc==0.0.5 (from versions: none)
ERROR: No matching distribution found for annotated-doc==0.0.5
```

The local pip cache contained no wheels. I did not use an external package source because the task forbids downloads. The lockfiles also pin Linux CPU Torch wheels (`torch==2.15.0+cpu`, `torchvision==0.29.2+cpu`), while this host is Windows.

### Native API

Used the pre-existing `data\NexviaProjectData\aiml02-venv310` runtime as a fallback, not the new venv. Observed versions include Python 3.10.1, FastAPI 0.162.0, Uvicorn 0.54.0, Torch 2.11.0+cu128, MONAI 1.6.1, and nibabel 5.4.2. This differs from the CPU lockfile.

- Native Uvicorn startup logged `Application startup complete` and `Uvicorn running on http://127.0.0.1:8765`.
- One live `GET /health` returned HTTP 200 with `status=checkpoint_present`, `model_available=true`, and the educational-use disclaimer.
- Later live HTTP requests timed out or were refused. The captured errors were `curl: (7) Failed to connect to 127.0.0.1:8765` and `The request was canceled due to the configured HttpClient.Timeout of 5 seconds elapsing.`
- An in-process ASGI request to `/predict` succeeded: HTTP 200, 14 findings, disclaimer present, checkpoint SHA-256 `082bc9dc5ff0399a6eeb07ae4dd9146ab258c39829e2c19a5c3eccc2c1cd8cc8`.
- Input was `00026095_000.png`, listed in the already-audited clean 2,250 panel and `train_val_list.txt`. It is an internal development image, not an official NIH test image.
- The ASGI response JSON and one-off helper remain in the private audit workspace; they are excluded from the compact handoff and public repository.

### Frontend

The ordinary Vite command first failed because Vite attempted to write a temporary config under the read-only project directory:

```text
EPERM: operation not permitted, open '...\web\node_modules\.vite-temp\vite.config.js.timestamp-....mjs'
```

I then used a temporary config in the writable workspace with `cacheDir` there and `--configLoader runner`; Vite reported ready at `http://127.0.0.1:5173/`. Opening that URL in Chrome returned `net::ERR_CONNECTION_TIMED_OUT`. No browser upload was made.

### Docker / WSL checks

`docker version` reported client version 29.8.0, then failed to connect to `npipe:////./pipe/dockerDesktopLinuxEngine` because the pipe was absent. `wsl --status` returned access denied (`Wsl/E_ACCESSDENIED`). The Docker engine / Orthanc workflow is therefore untested.

## B. Segmentation inference and preprocessing comparison

A fresh inference was run with the existing `artifacts\best_spleen_unet.pt` on `Task09_Spleen/imagesTr/spleen_10.nii.gz`, which is in the frozen validation split. Its paired reference is `Task09_Spleen/labelsTr/spleen_10.nii.gz`. No challenge test image was read.

The prediction/reference overlay, its measurements, and binary mask were produced in the private audit workspace. Those image- and case-level artifacts are excluded from the compact handoff and public repository.

The overlay shows the reference in green, prediction in red, and overlap in yellow. It also shows substantial red spill outside the reference on several slices. The case-specific source-grid display Dice is 0.4096; prediction volume is 2.160 times the reference volume. The project’s previously reported six-case held-out mean Dice is 0.3272; this audit did not recompute that metric or alter the model.

| Property | Training code | Inference code | Finding |
|---|---|---|---|
| Orientation | RAS for image and label | RAS for image | Same canonical orientation; output is inverted to source geometry. |
| Spacing | 1.5 × 1.5 × 2.0 mm; bilinear image / nearest label | 1.5 × 1.5 × 2.0 mm; bilinear image | Same target spacing. |
| Intensity | Clip HU to [-57, 164], scale to [0, 1] | Same | Match. |
| Foreground ROI | Foreground crop; training then pads and samples positive/negative 64³ patches | Foreground crop; 64³ sliding-window inference, 0.25 overlap | Compatible crop/ROI; training uses random patches, inference tiles the full volume. |
| Label mapping | Dataset labels 0=background, 1=spleen; 2 output channels | Foreground probability is softmax channel 1; binary threshold 0.80 | Match. |

For `spleen_10`, source image spacing is approximately 0.9766 × 0.9766 × 5.0 mm, input and label are RAS with shape 512 × 512 × 55, labels are {0,1}, output mask is {0,1}, and output affine matches the source within 1e-4. The source grid is therefore substantially resampled for model input, then restored for export. The one-case Dice is visual review only and should not be compared to the resampled-grid evaluation score.

## C. Seed-42 15.5k run directory

Directory: `data\NexviaProjectData\nih\imagenet_densenet121_full15500_seed42_5epochs_20261030`

Files present: `best_densenet121.pt`, `last_checkpoint.pt`, `run_summary.json`, and `training_history.json`. No train manifest, separate config file, or log was present. The run summary records 15,500 train images, 2,250 validation images, five epochs, DenseNet-121, and `test_split_used=false`, but has no manifest path/hash. Thus the 15.5k candidate’s exact training patient assignment remains **NOT VERIFIED** and cannot be compared byte-for-byte with the clean-set manifest.

- `run_summary.json` SHA-256: `837923ad16677c146777dacc5decb7ad81f1154c61022eb937db2074063af498`
- `training_history.json` SHA-256: `a9fa410d9029b31dee9dcf1467bab1fd04018f31feabd6243dd0bb556016a257`

## Reproducible command patterns

Historical commands used local absolute paths. The paths below use project and workspace variables so this report can be copied without embedding machine-specific roots. Set `NEXVIA_AUDIT_WORKSPACE` to the folder containing the saved audit scripts and run the project commands from the project root.

```powershell
$project = (Get-Location).Path
$workspace = $env:NEXVIA_AUDIT_WORKSPACE
$python = Join-Path $project 'data\NexviaProjectData\aiml02-venv310\Scripts\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:NEXVIA_CHECKPOINT = Join-Path $project 'artifacts\best_densenet121.pt'
$env:NEXVIA_SEGMENTATION_CHECKPOINT = 'artifacts/best_spleen_unet_candidate_20261103.pt'
& $python -m uvicorn api.app:app --host 127.0.0.1 --port 8765
Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8765/health' -TimeoutSec 10
curl.exe --fail-with-body -sS -F "file=@$(Join-Path $project 'data\NexviaProjectData\nih\images_224_fresh_15k_20261029\00026095_000.png')" http://127.0.0.1:8765/predict
& $python (Join-Path $workspace 'api_asgi_smoke.py')
docker version
wsl --status
& $python '.\segment_spleen.py' --input '.\data\NexviaProjectData\Task09_Spleen\imagesTr\spleen_10.nii.gz' --checkpoint '.\artifacts\best_spleen_unet.pt' --output-dir (Join-Path $workspace 'segmentation_audit_output') --threshold 0.80 --device auto
```
A duplicate inference invocation returned `FileExistsError` because the first run had correctly populated the output directory; the script refuses to overwrite an existing non-empty directory. The initial `docker version` and `wsl --status` errors are documented above.

## Verified / NOT VERIFIED

**Verified:** The requested soup checkpoint hash is present at `artifacts/best_densenet121.pt`; `/health` returned 200 once; `/predict` returned 200 with 14 findings through ASGI; fresh validation-case segmentation mask and overlay were created; geometry, intensity, spacing, ROI, and label mapping were checked against source code and files; seed-42 folder contents were inspected and summary/history hashed.

**NOT VERIFIED:** A fresh venv with pinned requirements; subsequent live HTTP prediction; Chrome upload through the frontend; Docker/Orthanc availability; the 15.5k seed-42 training manifest binding. The score/demo must remain labeled educational and non-diagnostic. No classifier results were changed, no checkpoint was modified, no new dataset was downloaded, no official test image was accessed, and no training was run.
## Follow-up local bring-up attempt during document closeout

The native API and Vite frontend both reported ready on alternate local ports 8767 and 5174. Browser navigation to `http://127.0.0.1:5174/` failed with `net::ERR_CONNECTION_TIMED_OUT`; subsequent HTTP checks could not connect to `127.0.0.1:8767` (`curl: (7) Failed to connect`). No upload was made and no successful browser screenshot exists. This follow-up does not change the verified in-process ASGI result above; live HTTP and browser upload remain NOT VERIFIED.


