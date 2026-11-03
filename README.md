# AIML-02 — Chest imaging educational prototype

Public source and scrubbed deliverables: https://github.com/ram15x/nexvia-aiml-02-medical-imaging

This coursework project has two separate model paths: NIH ChestX-ray14 finding classification and MSD spleen CT segmentation. It is an educational, non-diagnostic prototype. Project 1 has not been submitted to Nexvia. Data availability and reproducibility status: [data inventory](docs/data_inventory.md).
## Portal requirement status

| Requirement | Project 1 status |
|---|---|
| Public medical-imaging data and DVC | NIH ChestX-ray14 and MSD Task09 were used in development. Full images and patient-level manifests are no longer in this folder; the DVC remote is not configured. Training cannot be reproduced from this package alone. |
| Transfer-learning classifier and 3D segmentation | Separate PyTorch DenseNet-121 and MONAI 3D U-Net paths are implemented. The packaged API checkpoints are identified below and in the model card. |
| Grad-CAM | Implemented and smoke-checked on a sample. The map is coarse attribution, not a detection box or clinical explanation. |
| React scan review | Browser upload, scores/overlays, and local review actions are implemented. The UI records demo approval/rejection actions to a local JSONL audit file; this is not a clinical workflow. |
| Orthanc and DICOM workflow | Integration code and an optional Docker profile exist. The live Orthanc server workflow is **NOT VERIFIED**; CT segmentation does not write DICOM SEG/SR results back. |
| Docker | API and web containers passed recorded live HTTP checks on Nov 3. Orthanc was not part of that run. |
| Limits, API/data documentation, and evidence | README, architecture, data inventory, model card, and audit reports are included. Metrics are development results with the reuse and sample-size limits stated below. |
| Presentation, demo recording, and reflection | The 10-slide deck is included. You will record the demo/presentation and write the reflection yourself; slide 10 remains a prompt for your own account. |

### End-to-end paths

**Chest X-ray:** image upload → FastAPI validation → DenseNet-121 returns 14 uncalibrated finding scores → optional Grad-CAM overlay → React review screen → local JSONL review event. This web path was exercised in the saved browser and Docker checks.

**Spleen CT:** NIfTI upload → FastAPI validation → MONAI 3D U-Net mask inference → mask and tri-planar previews in React. This upload path was exercised in the saved browser and Docker checks. The separate Orthanc/DICOM path is code-only for this handoff and remains **NOT VERIFIED** at runtime.

**Classifier — November 3, 2026:** The earlier 8k reference is DenseNet-121 checkpoint 6c32. On Nov 3, the 8k/10k DenseNet-121 weight soup with BatchNorm recalibration was evaluated against that reference: macro AUROC/AP was 0.7953/0.1594 on the reused 5,000-patient panel (13 labels with ≥30 positives), versus 0.7755/0.1519 for 6c32; on the 2,138-patient common-clean panel (9 such labels), it was 0.7926/0.2258 versus 0.7700/0.2093. No classifier checkpoint changed that day. The paired intervals and panel definitions are summarized in [the model card](docs/model_card.md).

An undeployed inference-only candidate averaged soup and 15.5k sigmoid outputs. On the reused 2,138 panel, macro AUROC on the nine labels with ≥30 positives was 0.8028 versus 0.7926 for soup; paired 2,000-resample patient-bootstrap median delta +0.0102, 95% interval [+0.0042, +0.0168]. This is a provisional panel result, not an improvement to the API classifier. The 15.5k training-manifest link is **NOT VERIFIED**. No classifier checkpoint changed on Nov 3. Full values and reproduction details: [`project1_closeout_2026-11-03.md`](reports/project1_closeout_2026-11-03.md) and [final inference checks](reports/project1_final_checks_2026-11-03.md).

Both classifier panels were reused during development; neither is an independent test set. NIH labels are report-mined. Absolute performance remains modest, and AP depends on prevalence. Checkpoint 5634 is historical context only: the audit links it to an earlier 3,222-patient, 3-epoch run.

**Spleen CT segmentation — November 3, 2026:** A 96³ ROI run trained on 29 volumes for 134 epochs. The best checkpoint was selected at epoch 125 by mean argmax Dice 0.8327 on the same six validation cases; this estimate is optimistic. A repeated six-case internal holdout scored 0.7883. The six cases were excluded from the Nov 3 final run’s training and selection, but had prior project evaluations; five appeared as training cases in separate nested-CV folds. This is not a fresh project-wide independent test, and whether earlier scores influenced configuration choices is NOT RECORDED. Per-case metrics, the full validation curve, largest-component exploratory results, checkpoint SHA-256, and exact commands are in the [final-day closeout report](reports/project1_closeout_2026-11-03.md).

The segmentation candidate is packaged as `artifacts/best_spleen_unet_candidate_20261103.pt` (SHA-256 `46622de72ac91c7e17d36da19d1088e90e1ad836a0d6e00533b37b805b5cb7a4`). The protected `artifacts/best_spleen_unet.pt` was not modified. `.env` uses a project-relative checkpoint path, and Docker Compose points to the candidate under its `/app/artifacts` mount; the Docker API and web containers were started and exercised over live HTTP with Docker Compose on Nov 3, 2026; Orthanc itself remains **NOT VERIFIED**. The API uses threshold 0.50 with flip TTA disabled, replacing its previous 0.80 plus axis-0 flip probability average. A live Uvicorn check returned HTTP 200 for `/predict` and three `/segment/spleen` uploads. The segmentation helper took 2.990–7.294 seconds and complete HTTP requests took 3.715–9.518 seconds; the earlier in-process test-client stall did not recur, and its specific cause is **NOT VERIFIED**. A near-constant grayscale variance guard (variance ≤1.0 on 8-bit grayscale intensity) returns HTTP 422; a blank image was rejected, and a real image was accepted. Browser upload and the fresh pinned Windows environment have since been verified. Orthanc runtime remains **NOT VERIFIED**. Full request measurements and regression checks: [API fix and regression audit](reports/project1_api_fix_regression_2026-11-03.md).

**Historical segmentation baseline:** The old model's validation Dice was 0.3170 at threshold 0.80, selected on those same six cases; exploratory largest-component mean was 0.4200. The separate old 0.3272 score came from another six-case internal holdout. These are historical results and must not be substituted for the new argmax run. The new 0.8327 result is not a protocol-matched improvement over the old threshold-selected score.

**Demo boundary:** The live `/predict` response returned 14 findings with soup SHA-256 `082bc9dc5ff0399a6eeb07ae4dd9146ab258c39829e2c19a5c3eccc2c1cd8cc8`. HTTP segmentation scores on three existing validation scans were 0.904689, 0.851324, and 0.722703 for spleen_10, spleen_63, and spleen_17. These are validation cases, not independent test performance. Browser upload, Docker API/web runtime, and a fresh pinned Windows environment were verified on Nov 3, 2026. Orthanc runtime and PowerPoint rendering remain **NOT VERIFIED**. `requirements-windows-tested.txt` installs the tested Windows CUDA 12.8 environment when used with its embedded PyTorch wheel index. See [the demo and segmentation audit](reports/Nexvia_AIML02_Demo_Segmentation_Audit_2026-11-03.md) and the final-day closeout.

Historical classifier and segmentation experiments are retained in the local `reports/` archive. The public GitHub release includes the current closeout and verification reports only; patient-level prediction tables, raw inputs, environment dumps, and obsolete experiment artifacts are excluded. Earlier candidate metrics are not evidence of independent generalization. The current conclusions and caveats are summarized in this README and `docs/model_card.md`.
## What is in the project

- `data_audit.py`: metadata checks, safe filenames, patient-level train/validation assignment, official test split isolation, and MSD Task09 case manifest checks.
- `train_classifier.py`: multi-label DenseNet-121 training and validation metrics. It never reads test; subset runs are engineering pilots.
- `gradcam.py`: coarse attribution overlays from a trained checkpoint; not a clinical explanation.
- `train_segmentation.py`, `select_segmentation_threshold.py`, and `evaluate_segmentation.py`: MONAI 3D U-Net/SegResNet training, validation-only checkpoint/threshold selection, and a separate internal holdout evaluator. The U-Net remains selected; SegResNet was screened and rejected. The baseline checkpoint is `artifacts/best_spleen_unet.pt`.
- `segment_spleen.py`: separate NIfTI inference; optionally averages direct and axis-0-flipped probabilities, reverses orientation/spacing/crop transforms, writes a binary mask, and verifies source shape and affine.
- `api/app.py`: local chest-image and NIfTI spleen inference, health, Grad-CAM, and optional Orthanc route code. Live Orthanc behavior is NOT VERIFIED in the final-day closeout.
  The `/health` model metadata includes optional `promotion_evaluation`, copied from checkpoint metadata when present; otherwise it is `null`. This pass-through is not a metric calculation or model-promotion action. It was added to expose any saved promotion-evaluation record at runtime. The current soup checkpoint does not contain that key.
- `web/`: React review and CT panels. Browser uploads passed the Nov 3 verification; the Orthanc workflow remains NOT VERIFIED.
- `docker-compose.yml`: local-only API and web services, with an optional Orthanc profile.
- DVC: historical pipeline configuration and reports remain. Raw input data/pointers and a configured DVC remote are absent; a small `.dvc/cache` with model artifacts remains. Training cannot be reproduced until inputs and a remote are restored.
- `docs/`: architecture, model card, data inventory, presentation outline, and research notes. The local `reports/` archive contains historical experiments; only the current, scrubbed audits are in the public release.
- `scripts/compare_spleen_soft_vote.py`, `scripts/compare_spleen_flip_tta.py`: validation-only model blend and flip-TTA comparisons. Their small validation set does not provide independent performance estimates.
- `deliverables/`: the 10-slide presentation and a short local demo recording for AIML-02.

## Data handling

### Local data status (November 3 cleanup)

The full NIH and MSD image/volume datasets, temporary training workspace, and local training environments were removed to free storage. Model checkpoints, code, selected reports, and a small DVC cache containing a model artifact remain. Two small demo examples remain in the private working folder under `demo/sample_inputs/`; their exact source split is NOT RECORDED, so do not treat them as independent test cases. They are excluded from the public GitHub repository and the compact submission bundle. The Docker demo uses uploaded inputs and packaged checkpoints and does not need the training datasets. Training or reproducing the reported experiments requires restoring the official datasets and patient-level manifests. The full NIH official test split is not packaged.

DVC configuration and historical reports remain. The raw data pointers and workstation-only remote are absent; a small `.dvc/cache` remains with model artifacts, not the training images or volumes. DVC runs cannot be reproduced from this compact working copy until the data, manifests, and remote are restored.

For a complete NIH download, run `data_audit.py nih` with the official metadata, split files, and image root, then run `train_classifier.py` as shown below. Do not use NIH's Requester Pays bucket unless you explicitly approve possible charges.

```powershell
python -m pip install -r requirements.txt
python data_audit.py nih `
  --metadata data\NexviaProjectData\nih\Data_Entry_2017_v2020.csv `
  --train-val-list data\NexviaProjectData\nih\train_val_list.txt `
  --test-list data\NexviaProjectData\nih\test_list.txt `
  --image-root data\NexviaProjectData\nih\images `
  --output reports\nih --check-files
```

Install a compatible CUDA-enabled PyTorch/torchvision pair for the chosen training runtime, then install `requirements-training.txt`.

```powershell
python train_classifier.py `
  --manifest reports\nih\nih_manifest.csv `
  --image-root data\NexviaProjectData\nih\images `
  --output reports\nih-classifier `
  --epochs 3 --batch-size 8 --checkpoint-every-steps 1000
```

`--architecture densenet121_swin_t` runs the screened late-fusion comparison candidate. `--architecture densenet121_label_attention` and `densenet121_label_graph` compare a per-finding spatial-attention head without and with a positive-PMI label graph built from train targets only. `densenet121_label_residual` and `densenet121_label_graph_residual` preserve a global DenseNet head and add per-label attention corrections, with the graph optional. The direct attention-only screen underperformed; residual screens are mixed and none has been promoted. Keep the same seed, patient split, and training budget for baseline and each ablation. Do not rerun the already-used 967-image test subset for model selection.

If a runtime disconnects, resume from `reports\nih-classifier\last_checkpoint.pt` with the same manifest, output directory, and training settings by adding `--resume-from reports\nih-classifier\last_checkpoint.pt`. The resume file stores the model, optimizer, scaler, epoch/sample order, validation history, and a hash of the manifest. Do not resume against different data or settings.

For MSD, the official `dataset.json`, all 41 image-label pairs, and all 20 unlabeled official test images have been audited. The training list did not define this project's validation/test protocol, so a seed-42 case-level split was frozen before training. The baseline and foreground-only-Dice runs peaked at validation Dice 0.2603 and 0.2599 at threshold 0.5. Validation-only threshold selection chose 0.80. On the six reserved labeled cases, the selected model scored mean case Dice 0.3272 (range 0.0963–0.4751). This result is weak and variable. This historical six-case internal-holdout result is documented in the closeout. Its original `heldout_evaluation.json` run artifact was removed during the November 3 storage cleanup. It is distinct from the 0.3170 threshold-selection validation result. The official unlabeled challenge images were not evaluated.

To produce a binary mask from one compatible Task09 Spleen `.nii` or `.nii.gz` CT volume, use a Python environment with the pinned segmentation dependencies. This offline path is separate from the chest X-ray API and only accepts NIfTI volumes:

```powershell
python segment_spleen.py `
  --input data\NexviaProjectData\Task09_Spleen\imagesTr\spleen_20.nii.gz `
  --checkpoint artifacts\best_spleen_unet.pt `
  --output-dir reports\spleen-inference-case20
```

The default foreground threshold 0.80 was chosen on six validation cases and may not generalize. The script refuses to reuse a non-empty output directory and checks that the written mask is binary and matches the source NIfTI's shape and affine. A smoke run on `spleen_20` (a training case) verified those geometry checks; it was not scored or used to tune the model. The U-Net result must not be used for care decisions.

## Local demo

The project root contains the model files used by the demo: `artifacts/best_densenet121.pt` for chest X-ray classification and `artifacts/best_spleen_unet_candidate_20261103.pt` for spleen CT segmentation. Do not replace either with a historical candidate when following these steps. Copy `.env.example` to `.env`; keep `.env` local and do not commit it.

Run the commands below from the project package root: the folder containing `api/`, `web/`, `artifacts/`, and the test files. In the development checkout this folder is `submission_final/`; after extracting the submission package it is the folder you opened.

### Option A — Docker Compose

Docker Desktop and its Linux engine must be running. From the project root:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Open the UI at `http://localhost:15180` and health at `http://localhost:18080/health`. Stop the stack with `Ctrl+C` or `docker compose down`. The API and web containers passed live HTTP checks on November 3, 2026, as recorded in `reports/project1_final_checks_2026-11-03.md`; Orthanc was not part of that verified flow. This option uses CPU-only PyTorch in Docker. A first build may pull base-image layers; check their size before pulling if downloads over 500 MB require approval.

### Option B — Native Windows

Use Python 3.10 and Node.js 20 or later. Run these commands from the project root in PowerShell:

```powershell
Copy-Item .env.example .env
py -3.10 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-windows-tested.txt
python -m pytest -q
```

The test command must run inside this activated venv. A complete suite on October 31 recorded 43 passing tests; that was an earlier test revision. The exact expected count for this packaged revision is **NOT VERIFIED** until the full suite is run with the pinned environment. The November 3 install record says the pinned CUDA PyTorch wheel is 2.75 GB. Do not start that install if a download of that size has not been approved.

Start the backend in one PowerShell window, still at the project root:

```powershell
& .\.venv\Scripts\Activate.ps1
python -m uvicorn api.app:app --env-file .env --host 127.0.0.1 --port 18080
```

Start the frontend in a second window:

```powershell
Set-Location .\web
npm ci
$env:VITE_API_URL = "http://127.0.0.1:18080"
npm run dev -- --host 127.0.0.1 --port 15180
```

Open `http://127.0.0.1:15180`; API health is `http://127.0.0.1:18080/health`. The sample checkpoints are already in `artifacts/`; if using replacement files, place the classifier at `artifacts/best_densenet121.pt` and the spleen model at `artifacts/best_spleen_unet_candidate_20261103.pt`, or set their variables in `.env`. Relative model and result paths resolve from the project root. The classifier returns uncalibrated scores, and the Grad-CAM display is a coarse attribution map, not a detection box or clinical explanation. Orthanc remains unverified; do not use this prototype for care decisions.

## Known limitations

- The recorded Nov 3 Docker and clean Windows checks passed at that time. In this pre-submission rerun, `npm ci` installed the 19 locked web packages and `npm run build` passed. Docker access was denied by the local engine pipe. The available global Python lacks torchvision, MONAI, NiBabel, and pydicom, so the current native inference and full-suite checks are not a fresh pass.
- Docker Compose API and web containers were verified with live HTTP checks on Nov 3, 2026; the optional Orthanc container and live Orthanc workflow were not tested.
- PowerPoint rendering of the deck was not verified; the deck received a text and shape-bounds audit only.
- Cold-start latency can be about 12 seconds: an earlier fresh-venv check took 12.629 seconds to the first health response. In the A1 rerun, initial health took 4.254 seconds before model load, and the first prediction took 3.691 seconds. CPU spleen segmentation took about 28.4 seconds in the helper alone; complete request time is longer.
- The tested native Windows environment used CUDA 12.8 PyTorch wheels; Docker uses CPU-only PyTorch. Docker segmentation is configured for CPU, but the device was not logged at runtime.
- The classifier development panels were reused during model comparison and are not an independent test set. The segmentation validation and internal holdout cases also have the reuse limitations described in the closeout report.
- This is an educational prototype, not for clinical use or care decisions.

## Verification and remaining work

The saved [Nov 3 final checks](reports/project1_final_checks_2026-11-03.md) and [API regression audit](reports/project1_api_fix_regression_2026-11-03.md) document passing browser uploads, fresh pinned Windows live HTTP requests, Docker API/web checks, CPU-versus-CUDA mask agreement, and the package quick-start. These are dated checks, not a fresh Nov 4 rerun.

**Current recheck:** this machine currently has no project Python environment with `pytest` and `torch`, and Docker access returned permission denied for the Docker engine pipe. I therefore could not repeat the full Python suite or live API requests in this session. The latest complete Python suite result remains 43 passed on Oct 31; a complete run after the Nov 3 API changes is **NOT VERIFIED**. Do not infer a current pass from the older result.

**Presentation:** v21 opens in PowerPoint, contains 10 slides, and its XML parts parse. I did not complete a rendered review of every slide. Slide 10 intentionally leaves the personal reflection to you. Orthanc runtime and clinical performance remain **NOT VERIFIED**.

Before the portal's one-time submission, finish your own reflection and recording, ensure the reviewer can access the public GitHub repository and combined Drive folder, and retain the limitations above. The portal requires both assigned projects in the same Drive folder; this Project 1 README does not claim Project 2 is complete.
## Model card

See [docs/model_card.md](docs/model_card.md) for intended use, separate model paths, measured results, limits, and unmeasured subgroup risks.


Current closeout: [Project 1 closeout — November 3, 2026](reports/project1_closeout_2026-11-03.md).










