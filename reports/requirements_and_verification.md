# Project 1 requirements and verification

## Portal brief

The Nexvia portal assigns **AI-Assisted Medical Imaging and Diagnosis System**. It asks for public imaging data; preprocessing and DVC; transfer-learning classification; MONAI 3D U-Net segmentation; Grad-CAM; an Orthanc service that pulls studies, runs inference, and stores results; a React review page with scores, overlays, approve/reject actions and an audit log; Docker; documentation of model limits and bias; and a working demo, GitHub repository, architecture/data/API documentation, test results, a 10-minute presentation, and an individual reflection.

The portal specifies one master Google Drive folder for both assigned projects and says the link can be submitted only once. This project has not been submitted. The separate Project 2 folder is not represented as complete here.

## Requirement-by-requirement status

| Portal item | Status | Evidence and boundary |
|---|---|---|
| Public data and DVC | Partially verified | NIH ChestX-ray14 and MSD Task09 were used. Two retained demo inputs are now DVC-tracked with a local remote, whose file hashes were checked. Raw training datasets and patient-level manifests remain absent; the historical training run cannot be reproduced from this package. See `docs/data_inventory.md`. |
| Classifier and segmentation models | Implemented | DenseNet-121 returns 14 report-mined finding scores; MONAI 3D U-Net returns a spleen mask. They are independent paths with different inputs. See `docs/architecture.md` and model card. |
| Grad-CAM | Implemented; smoke-checked | A coarse attribution overlay is available for the classifier. It is not a bounding-box detector or clinical explanation. |
| Orthanc DICOM service | Partial; synthetic DX and constrained synthetic CT paths verified | A generated DX study passed upload, stable-study sync, classifier inference, and local JSONL result storage. On Nov 4, an isolated synthetic 24-slice CT series with explicit expected instance count reached Complete/stable and passed the CT segmentation route. A stable synthetic series without completeness metadata remained Unknown and was rejected. No real CT compatibility is established, and no DICOM SEG/SR is written back. |
| React viewer and review actions | Implemented; browser upload checks recorded | The UI supports image/volume upload, scores or masks, and demo approve/reject actions. Review events are local JSONL records, not a clinical audit workflow. |
| Docker | API/web checks passed previously | The API and web services returned successful live HTTP responses in recorded checks. Docker uses CPU-only PyTorch. The test suite below also ran in the local CPU Docker test image. |
| Limits and bias | Documented | The model card and data inventory describe reused panels, report-mined labels, missing input data, limited cohorts, and non-diagnostic use. No clinical validation is claimed. |
| Working demo | Local demo implemented; browser upload and live HTTP checks recorded | No hosted deployment URL or screen recording is included yet. Ram will include a short local app walkthrough in the 10-minute presentation after Project 2. Orthanc verification uses synthetic DX and one synthetic CT series with explicit completeness metadata; ordinary CT series reported as Unknown are rejected. |
| GitHub, README, architecture/data/API docs | Present; latest sync pending verification | An earlier 64-file package matched the public branch. DVC pointer and documentation changes were made afterwards; check the branch for those changes before relying on the public copy. |
| Test results | **58 passed, 2 warnings** in a disposable Python 3.12 CPU container | Latest full packaged suite passed. Test-only dependencies were installed inside the disposable container; the project mount was read-only. This does not verify the pinned Windows/Python 3.10/CUDA 12.8 environment. |
| 10-minute presentation | Deck included; visual check performed | The 10-slide deck was opened in PowerPoint and reviewed in Slide Sorter; it has not been rendered to slide images. Ram will record and present it after Project 2 with a short local demo walkthrough. Slide 10 remains a prompt for Ram’s own reflection. |
| Individual reflection and recording | Ram’s action remains | After Project 2, Ram will record and present the deck with a short walkthrough of the local demo. He must write his personal reflection in his own words. |

The earlier 57-test run and 38-test focused run below are historical results and have been superseded by the complete 58-test run recorded under “Latest regression verification.”

## Test-suite details

Command: `python -m pytest -q`, run from the package root in the existing local Docker API image with the source mounted read-only. The test environment was Python 3.12, torch 2.15.0+cpu, torchvision 0.29.2+cpu, MONAI 1.6.1, and pytest 9.2.0, with the small missing test libraries installed in the disposable environment. No model training, data download, official test data, or checkpoint edit was part of this verification.

The first run exposed a stale assertion that expected segmentation flip TTA to be enabled, although the API configuration disables it. It also exposed two test-support audit scripts missing from the packaged folder. The assertion was corrected to match the API setting, and the missing support scripts were added to the package. After those changes, both the development suite and packaged suite reported 56 passed and 2 warnings. The pinned Windows/CUDA suite was not rerun in this check.

## Orthanc scope

The earlier test used generated synthetic DX DICOM, not a patient study. Orthanc accepted the instance; the API observed the stable study, ran chest classification, and returned/stored 14 uncalibrated scores. This verifies the DX pull/infer/store path for that synthetic case only and does not establish broad DICOM compatibility or clinical use. At that point CT had not been tested end to end; the Nov 4 CT check is recorded below.

## Deck check

The current 10-slide file opened in Microsoft PowerPoint. The slide sorter showed a consistent template and no obvious off-slide shapes. A full slide-image render was not produced, so this is a direct PowerPoint visual inspection, not a rendering/export check. Slide text remains small at sorter scale; review and rehearse each slide at presentation size before recording.
## Latest pre-submission recheck

The complete suite was rerun after the API changes in the existing CPU API image. The package folder was mounted read-only; test-only dependencies were installed in the disposable container and removed with it. No datasets, images, or model weights were downloaded, and no project dependency file or checkpoint was changed.

Exact command (PowerShell, from the package root):

```powershell
$projectMount = "$((Get-Location).Path):/app:ro"
docker run --rm --entrypoint sh -v $projectMount -w /app submission_final-api -lc 'set -e; python -m pip install --disable-pip-version-check --no-input pytest==9.2.0 pandas==2.3.3 scikit-learn==1.7.2 scikit-image==0.25.2; python -m pytest -q -p no:cacheprovider'
```

Result at that earlier check: **56 passed, 2 warnings in 9.03 seconds**. The warnings are Python invalid-escape warnings at `test_ct_series.py:204` for `\(` and `\)`; no test failed. The image used Python 3.12 and CPU PyTorch, so this does not verify the pinned Windows/Python 3.10/CUDA 12.8 environment.

## Latest regression verification

After adding the Orthanc CT mask persistence and retrieval routes, the complete packaged suite was rerun in the same disposable CPU container, with the project mounted read-only. The already-running synthetic Orthanc and API containers were used; no model files or project data were downloaded. Small test-only Python packages were installed inside the temporary API container to supply `pytest`, `pandas`, `scikit-learn`, and `scikit-image`.

Commands:

```powershell
docker exec nexvia-final-check-api python -m pip install --disable-pip-version-check --no-input -q pytest==9.2.0 pandas==2.3.3 scikit-learn==1.7.2 scikit-image==0.25.2
docker exec nexvia-final-check-api python -m pytest -q -p no:cacheprovider
```

Historical result: **57 passed, 2 warnings in 9.83 seconds**. Both warnings were invalid-escape `SyntaxWarning`s in `test_ct_series.py`; no tests failed. This older result was superseded by the latest full run below.

For completeness, a direct host attempt (`python -m pytest -q`) stopped during collection because that host environment lacked `torchvision` and `nibabel`. The first direct attempt in the API image also lacked `pytest`. The disposable test run above supplied the missing test dependencies and completed all tests.

## Orthanc live check and saved synthetic DX evidence

The local Orthanc service answered `GET http://127.0.0.1:18042/system` with HTTP 200 and reported version 1.13.0. The already-saved isolated synthetic DX integration record reports `status=processed`, source `orthanc_stable_study_changes`, 14 finding scores, and classifier checkpoint SHA-256 prefix `082bc9dc5ff0`. Its JSONL file SHA-256 is `4f782294a9dc9d8fe4db88904f5eea3c1c1fd7fe7b10a8dc05984381d27a088e`. This is a synthetic, single-frame DX demonstration; this pass did not upload another study. The later synthetic CT check is recorded below.

## PowerPoint visual review

I opened `deliverables/Medical_Imaging_AI_Review_Platform.pptx` in PowerPoint and reviewed all ten slides in Slide Sorter. I also inspected the cover, classifier results, spleen segmentation, limitations, and reflection slides at presentation size. The layout is consistent and I saw no text outside the slide bounds. I did not export slide images or claim a rendered-image review.


## Orthanc CT synthetic verification (Nov 4)

This was an inference-only isolated Docker test. No project file, checkpoint, dataset, or model weight was changed or downloaded. The submission folder was mounted read-only into the already-cached API image. Orthanc image tag 26.9.1 reported Orthanc version 1.13.0, local image ID sha256:d2705f2c56547e55ce9bc2250fd22d05d326493b3f3d9fa191ecb7db8ca6b5a2; API image submission_final-api:latest had local image ID sha256:8728a0cd350459d94c79752a94ed1e3dea970b336e9d7029946c705b81444710. The mounted api/app.py SHA-256 was 853a33f8fa00a95f2085e91f6e6f0d7a3389b529d4751bb9b84ea66e859d507c; api/ct_series.py SHA-256 was ba6e3156ecb8ebadf096403da05c8866ea359df736d22e223d62ab4dee3679eb. Docker reported torch.cuda.is_available() = False.

A generated 24-slice synthetic CT series with 0.7x0.9 mm pixel spacing, 2.5 mm slice spacing, known orientation/origin, shuffled upload order, ImagesInAcquisition=24, and NumberOfTemporalPositions=1 was uploaded to Orthanc. Orthanc reported Status=Complete, ExpectedNumberOfInstances=24, 24 stored instances, and IsStable=true. The first route request, before study stability, correctly returned HTTP 422. After it became stable, POST `/orthanc/series/{id}/segment/spleen` returned HTTP 200; the response reported shape [64,64,24], 24 DICOM instances, and `stored=true`. The temporary API JSONL summary recorded 31,624 predicted voxels and 49.81 mL. An independent converter check matched the expected RAS affine for this phantom ([-0.9, 0, 0, -10; 0, -0.7, 0, -20; 0, 0, 2.5, 30; 0, 0, 0, 1]) and rescaled intensities ranged from -1000 to 173 HU. The generated phantom check confirms the affine header and HU rescaling. A separate asymmetric-pixel-to-world unit test was added to verify row/column mapping and RAS coordinates; this tests the converter logic only. Real scanner compatibility and clinical input geometry remain NOT VERIFIED. These mask values are outputs on a synthetic phantom and are not accuracy results. In that first check, results were temporary and the mask was discarded; the follow-up below added persistent local storage and retrieval.

## Orthanc CT result persistence and retrieval

The API now saves the CT segmentation result as a local NIfTI-GZIP file and appends JSONL metadata including its SHA-256, model checkpoint SHA-256, threshold, flip-TTA setting, summary, and disclaimer. `GET /orthanc/series/{series_id}/segmentation-results` lists recorded results; `GET /orthanc/series/{series_id}/segmentation-results/{result_id}/mask` verifies the file checksum before returning it. Defaults are `demo/orthanc_segmentations.jsonl` and `demo/orthanc_masks/`. These are git-ignored local demo outputs, not persistent production storage and not DICOM SEG/SR write-back.

The follow-up used a generated synthetic 24-slice CT in Orthanc series `364bb107-2c18b2b8-46abdfe0-560a7247-9728e2c7`. The POST segmentation route returned HTTP 200; the results-list route returned HTTP 200; mask retrieval returned HTTP 200 with a 5,537-byte payload. The downloaded payload SHA-256 matched the JSONL record and the `X-Content-SHA256` response header. Docker inference ran with CPU PyTorch. The synthetic mask is not an accuracy result. This verifies only the local API storage/retrieval path for that generated input; real CT compatibility, DICOM SEG/SR interoperability, production retention, and clinical use remain unverified.

A second stable synthetic CT series with 24 instances but no expected-count metadata was reported by Orthanc as Status=Unknown. The route returned HTTP 422 with Orthanc series is not confirmed complete (status: Unknown) and did not run segmentation. This matches Orthanc's documented behavior: generic CT usually has no standard total-instance count, so its completeness may remain Unknown (https://orthanc.uclouvain.be/book/faq/series-completion.html). Thus CT integration is verified only for an explicitly complete synthetic input. Normal clinical CT compatibility, incomplete-series behavior beyond this guard, DICOM SEG/SR writeback, and clinical utility remain unverified.

The DICOM generator was streamed inline to the API container and not saved as a file; generator source and SHA-256 are therefore NOT RECORDED.

Exact route invocation from PowerShell:
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:28000/orthanc/series/0467c267-b4eb5cf0-8215cc60-210bdce6-f006ea65/segment/spleen' -TimeoutSec 360

## Latest full-suite result (Nov 4)

The full package test suite passed **58 tests with 2 warnings in 8.97 seconds**. This ran in a disposable container based on the cached `submission_final-api:latest` image, under Python 3.12 with CPU PyTorch. The project was mounted read-only and `PYTHONDONTWRITEBYTECODE=1` prevented bytecode output in the mount. The only temporary installs were `pytest==9.0.2`, `pandas==2.3.3`, and `scikit-learn==1.7.2`; the container was removed after the run. No model, dataset, checkpoint, or dependency file was downloaded or edited. The two warnings were `SyntaxWarning`s for invalid escape sequences `\(` and `\)` in `test_ct_series.py`; there were no failures. This result does not verify the pinned Windows/Python 3.10/CUDA 12.8 environment.

Exact command:

```powershell
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -e NEXVIA_ORTHANC_SEGMENTATION_MASK_DIR=/tmp/nexvia-test-masks -v 'C:\Users\ramna\OneDrive\Desktop\Projects\Nexvia Internship\Project-1:/app:ro' -w /app/submission_final --entrypoint python submission_final-api:latest -c "import subprocess,sys; subprocess.check_call([sys.executable,'-m','pip','install','--no-cache-dir','pytest==9.0.2','pandas==2.3.3','scikit-learn==1.7.2']); subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider'],check=True)"
```

## DVC/reproduction boundary

The release does not contain the NIH images, the DVC manifest, a DVC remote, or the recorded machine-local environment. DVC was not installed in the verification environment, so the pipeline was not rerun. The committed lock describes a historical 3-epoch subset screen. Its api/modeling.py and train_classifier.py MD5 hashes differ from the current release files; requirements-training.txt still matches. The 28,486,934-byte lock output differs from the 28,496,134-byte packaged classifier, and the segmentation model is outside this DVC stage. It does not establish reproduction of the deployed classifier or spleen model; the stage remains non-reproducible from this package.
