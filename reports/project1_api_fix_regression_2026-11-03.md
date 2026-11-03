# Project 1 API fixes and regression audit — 2026-11-03

Scope: deployment-path and input-guard fixes plus inference-only regression checks. No training, downloads, classifier/segmentation checkpoint edits other than copying the already-trained segmentation candidate to a new artifact filename, or official test data use occurred in this audit.

## F1. Checkpoint packaging and paths — PASS

- Packaged candidate: `artifacts/best_spleen_unet_candidate_20261103.pt`.
- SHA-256: `46622DE72AC91C7E17D36DA19D1088E90E1AD836A0D6E00533B37B805B5CB7A4`, matching the requested `46622DE7...`.
- `.env:8` selects the artifact by project-relative path: `NEXVIA_SEGMENTATION_CHECKPOINT=artifacts/best_spleen_unet_candidate_20261103.pt`.
- `docker-compose.yml:10` points to `/app/artifacts/best_spleen_unet_candidate_20261103.pt` inside the existing `./artifacts:/app/artifacts:ro` container mount. This is a container path, not a host-specific path.
- Protected checkpoint hashes were re-read after the fixes: old segmentation `414266BA20B24D9B4987FAE348BC26E33D14CDD1CF12D1077B35623937C55750`; soup `082BC9DC5FF0399A6EEB07AE4DD9146AB258C39829E2C19A5C3ECCC2C1CD8CC8`; 15.5k `BF144A8A0844F6256B7CDA7D0AAB5AA355F75036BFD409AD2FD1A9B23CC40831`. These match the prior recorded values; none was modified.
- The API code still has its old-checkpoint default if `NEXVIA_SEGMENTATION_CHECKPOINT` is absent (`api/app.py:46-48`). The tested configuration provided the environment setting. Starting the API without that setting would not select the candidate; retain the `.env`/Compose setting when launching.

## F2. Clean-copy in-process check — PARTIAL PASS

A fresh runtime-focused copy was made under `<CLEAN_COPY>` with the API source, classifier and candidate segmentation artifacts, runtime config, and required base model code. It was not a byte-for-byte clone of the full repository: large datasets, training history, and unrelated project files were omitted. The classifier and segmenter checkpoint paths resolved inside the copy.

- API module import from the copy succeeded. In-process classifier helper returned 14 scores in 2.981 seconds using the unchanged soup checkpoint.
- In-process segmentation helper returned a `(512, 512, 55)` mask with 60,756 foreground voxels in 16.491 seconds using the copied candidate checkpoint.
- A blank PNG passed through the copied API decoder was rejected with status 422 and the expected explanatory detail.
- The copy used the machine's existing Python environment and input images/volume bytes supplied from the project data. This does **not** verify a clean dependency installation or a fully self-contained deployment.
- The requested fixed-string scan for the two machine-specific path markers returned **0 hits** in the clean-copy text/config/code scope and **0 hits** in the project text/config/code scope. Binary checkpoints/images, Git metadata, virtual environments, generated build/presentation work, bulk data, and this report (which describes the scan) were excluded. There are no path hits to list within the scanned scope.

## F3. Constant-image guard — PASS

`api/app.py:372-380` converts the decoded input to 8-bit grayscale, computes variance, and returns HTTP 422 when variance is `<= 1.0`, with: `Image has near-constant intensity (grayscale variance <= 1.0); upload a non-blank PNG or JPEG.`

A blank PNG returned 422. A real chest radiograph returned HTTP 200 and 14 output scores. This is an input-quality check, not a diagnostic-quality check.

## F4. Live HTTP run and segmentation stall — PASS for observed requests; cause NOT VERIFIED

The successful run used one supervisor command that started Uvicorn, waited for `/health`, posted the images, measured requests, and terminated the child server. The execution form was `python run_live_api_f4.py <PROJECT_ROOT> <AUDIT_WORKSPACE>`. The HTTP results came from the live local server, not an in-process ASGI test client.

| Request | HTTP | Helper time | Full route time | Response body | Device |
|---|---:|---:|---:|---:|---|
| `/health` | 200 | — | ready after 5.074 s | health JSON | — |
| `/predict` real image | 200 | — | 3.012 s | 1,192 B | — |
| `/predict` blank image | 422 | — | 0.002 s | 107 B | — |
| `/segment/spleen` spleen_10 | 200 | 7.294 s | 9.518 s | 369,888 B | CUDA |
| `/segment/spleen` spleen_63 | 200 | 2.990 s | 3.715 s | 437,144 B | CUDA |
| `/segment/spleen` spleen_17 | 200 | 4.067 s | 5.492 s | 415,067 B | CUDA |

The previous in-process TestClient stall did not recur in this live-server run. Its root cause is **NOT VERIFIED**. The measured route-minus-helper overhead was about 2.224, 0.724, and 1.425 seconds, respectively; this audit did not isolate the individual sources of that overhead. Uvicorn was terminated after the requests; the supervisor reported exit code 1 after termination, and no server process was left running.

## F5. Inference regression checks — PASS within stated scope

### Six segmentation validation cases

The new candidate was scored with the saved inference pipeline and CUDA. Every value matched the saved final-inference result exactly (difference 0 at recorded precision); mean Dice was 0.8326745542.

| Case | Dice |
|---|---:|
| spleen_10 | 0.9030452456 |
| spleen_63 | 0.8495489063 |
| spleen_46 | 0.8111235671 |
| spleen_17 | 0.7153474652 |
| spleen_18 | 0.8460353011 |
| spleen_2 | 0.8709468398 |
| **Mean** | **0.8326745542** |

This is a checkpoint-selection validation set, so the mean is optimistic and is not an independent test estimate.

### Classifier soup on the 2,138-patient panel

The soup checkpoint hash was unchanged. Recomputed all-14 macro AUROC was 0.8176906965 versus saved 0.8176960471 (delta -0.0000053506). On the nine labels with at least 30 positives, macro AUROC was 0.7925965892 versus 0.7925995055 (delta -0.0000029163). Both are within 0.001. Recomputed macro AP differed from the saved results by +0.000015364 (all 14) and +0.000028694 (nine-label subset); maximum absolute score difference was 0.000217766.

### Actual API segmentation route on three cases

| Case | HTTP route Dice | Earlier argmax-table Dice | Difference |
|---|---:|---:|---:|
| spleen_10 | 0.9046886942 | 0.9030452456 | +0.0016434487 |
| spleen_63 | 0.8513238122 | 0.8495489063 | +0.0017749059 |
| spleen_17 | 0.7227027968 | 0.7153474652 | +0.0073553316 |

All three returned HTTP 200. The maximum difference is below 0.05. The route output and saved argmax evaluation are separate code paths and their scores are close but not numerically identical; this audit records the difference without attributing it to a single cause.

## F6. Holdout influence — NOT FULLY VERIFIED

The recorded final run selected epoch 125 by the six validation cases. The threshold was argmax as fixed in the user-provided run instructions. The 96³ ROI, sampling, batch, AMP, optimizer, and loss choices were also specified before that run in those instructions. The six internal holdout cases already had prior evaluation history. The run records do not establish whether those earlier results influenced the broader configuration choices; that influence is **NOT VERIFIED**. No recorded epoch or threshold selection was based on the six internal holdout scores.

## F7. Documentation updates

Updated the root README, the November 3 closeout, the demo/segmentation audit, the inference-comparisons report, and the final-checks report. Updated slide 4 and slide 8 speaker notes in `deliverables/AIML02_Project1_Presentation_2026-11-03_v20.pptx`. The notes cover threshold 0.50/no flip TTA, candidate packaging, Docker runtime status, the live HTTP timing results, the unresolved earlier stall cause, and the variance guard. Every XML/relationship member in the final PPTX package parsed successfully. Visual rendering in PowerPoint remains **NOT VERIFIED** because PowerPoint COM was unavailable with an access-denied error.

The final-checks report's former statement that Compose still pointed to the old segmentation checkpoint was corrected to the current candidate path and live-HTTP status. No model or metric values were changed by that documentation correction.

## Docker status

`docker compose config --quiet` exited 0. `docker version` reported client 29.8.0 but could not connect to the Docker Desktop Linux engine named pipe (`npipe:////./pipe/dockerDesktopLinuxEngine`; file not found). Compose syntax/configuration passes; container execution is **NOT VERIFIED**.

## Exact checks and evidence files

- Hashes: PowerShell `Get-FileHash -Algorithm SHA256 <path>` on each checkpoint and changed deliverable.
- Compose parse: `docker compose config --quiet` (exit 0).
- Live HTTP: `python run_live_api_f4.py <PROJECT_ROOT> <AUDIT_WORKSPACE>`; recorded response and helper timing output in the external audit workspace as `f4_live_http_result.json` and `f4_helper_timings.jsonl`.
- Clean-copy in-process: `python run_clean_copy_f2.py <PROJECT_ROOT> <CLEAN_COPY>`; recorded output as `clean_copy_f2_result.json` in the external audit workspace.
- Regression: `python f5_segmentation_reproduction.py <PROJECT_ROOT>` and the saved classifier reproduction command; results are recorded in `f5_segmentation_reproduction.json` and `classifier_soup_reproduction_2026-11-03.json` in the external audit workspace.
- Path scan: ripgrep fixed-string checks for each requested machine-specific path marker, with the exclusions listed above; zero matches in both scanned trees.

The scripts and raw JSON measurements are retained outside the project audit copy, in the local audit workspace. Their host-specific location is intentionally not embedded here. Docker runtime, clean dependency installation, browser upload, Orthanc, and PowerPoint rendering remain **NOT VERIFIED**.

