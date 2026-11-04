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
| Orthanc DICOM service | Limited interoperability verified | A generated DX study passed sync/classification/storage. A synthetic CT and one public de-identified 81-slice CT were processed. The real series was Unknown in Orthanc and required an independently supplied matching count; a count does not prove completeness. The mask was retrieved with a verified checksum. No broad CT compatibility or accuracy is established, and no DICOM SEG/SR is written back. |
| React viewer and review actions | Implemented; browser upload checks recorded | The UI supports image/volume upload, scores or masks, and demo approve/reject actions. Review events are local JSONL records, not a clinical audit workflow. |
| Docker | API/web checks passed previously | The API and web services returned successful live HTTP responses in recorded checks. Docker uses CPU-only PyTorch. The test suite below also ran in the local CPU Docker test image. |
| Limits and bias | Documented | The model card and data inventory describe reused panels, report-mined labels, missing input data, limited cohorts, and non-diagnostic use. No clinical validation is claimed. |
| Working demo | Local demo implemented; browser upload and live HTTP checks recorded | No hosted deployment URL or screen recording is included yet. Ram will include a short local app walkthrough in the 10-minute presentation after Project 2. The Orthanc check includes one public de-identified CT series; Unknown requires an independently supplied matching count. |
| GitHub, README, architecture/data/API docs | Present | The release includes the README, architecture, data and API documentation, DVC pointers, and a checksum manifest. Compare the public branch against the manifest when updating the release. |
| Test results | **69 passed** in a disposable Python 3.12 CPU container | Latest full staged suite passed without warnings. Test-only dependencies were installed inside the disposable container; the project mount was read-only. This does not verify the pinned Windows/Python 3.10/CUDA 12.8 environment. |
| 10-minute presentation | Deck included; visual check performed | The 10-slide deck was opened in PowerPoint and reviewed in Slide Sorter; it has not been rendered to slide images. Ram will record and present it after Project 2 with a short local demo walkthrough. Slide 10 remains a prompt for Ram’s own reflection. |
| Individual reflection and recording | Ram’s action remains | After Project 2, Ram will record and present the deck with a short walkthrough of the local demo. He must write his personal reflection in his own words. |

The earlier 57-test, 58-test and 38-test focused runs below are historical results and have been superseded by the 69-test run recorded below.

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

The follow-up used a generated synthetic 24-slice CT in Orthanc series `364bb107-2c18b2b8-46abdfe0-560a7247-9728e2c7`. The POST segmentation route returned HTTP 200; the results-list route returned HTTP 200; mask retrieval returned HTTP 200 with a 5,537-byte payload. The downloaded payload SHA-256 matched the JSONL record and the `X-Content-SHA256` response header. Docker inference ran with CPU PyTorch. The synthetic mask is not an accuracy result. At that time real CT had not yet been checked; the later single-series interoperability check is described below. Broad CT compatibility, DICOM SEG/SR interoperability, production retention, and clinical use remain unverified.

A second stable synthetic CT series with 24 instances but no expected-count metadata was reported by Orthanc as Status=Unknown. In that earlier check the route returned HTTP 422 and did not run segmentation. This matches Orthanc's documented behavior: generic CT usually has no standard total-instance count, so its completeness may remain Unknown (https://orthanc.uclouvain.be/book/faq/series-completion.html). The later guarded count option and one real-series check are described below. Broad clinical CT compatibility, DICOM SEG/SR writeback, and clinical utility remain unverified.

The DICOM generator was streamed inline to the API container and not saved as a file; generator source and SHA-256 are therefore NOT RECORDED.

Exact route invocation from PowerShell:
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:28000/orthanc/series/0467c267-b4eb5cf0-8215cc60-210bdce6-f006ea65/segment/spleen' -TimeoutSec 360

## Final rebuild and browser check

The API and web images were built from the release Dockerfiles on November 4. The initial API attempt with `--network=none` failed because its package-install layer was not cached. An online rebuild then succeeded; the official CPU Torch wheel was 209,530,490 bytes. The frontend build ran `npm ci` and `npm run build` successfully using its cached npm layer. The API build was tested in a disposable container with the release mounted read-only and temporary test-only installations of `pytest==9.0.2`, `pandas==2.3.3`, and `scikit-learn==1.7.2`: **58 passed, 2 invalid-escape warnings in 13.06 seconds**. No checkpoint was changed and no dataset was downloaded.

In Chrome, the rebuilt application accepted the retained chest example and showed 14 scores, a Grad-CAM overlay and a recorded demo review. It accepted the retained spleen volume and showed an axial overlay and mask download; the example had no reference mask, so this is a workflow check, not an accuracy measurement. The first request from `127.0.0.1` failed while the frontend was configured to call `localhost`. The frontend now uses the current loopback hostname when both the page and configured API are loopback addresses; chest prediction worked at both addresses after rebuilding. The CT screen also previously displayed historical Dice values from the old segmentation model beside the new model output; those values were removed from the screen. The model weights and evaluation results did not change. These frontend changes were made after the Python suite run; the frontend production build and browser checks were repeated afterwards.

The DVC demo examples were restored into a fresh GitHub clone from the existing local store, and both restored MD5s matched the pointer files. This revealed that the restored chest PNG appeared as an untracked file. The release now includes `demo/.gitignore` and `.dvc/.gitignore` to keep the example and DVC cache out of ordinary Git commits. The historical training data and manifests remain unavailable.

## Independent arithmetic check of saved results

The saved patient-level predictions and their manifests were checked without running inference. All image and patient IDs matched in order; each patient appeared once per panel; every saved target matched the finding labels in its manifest. A separate NumPy implementation recomputed AUROC and average precision, grouping tied scores. The resulting macro values matched the saved JSON results within 2.3e-16. On the 2,138-patient panel, the 8k reference and soup respectively had AUROC/AP 0.770010/0.209277 and 0.792600/0.225797 over nine labels with at least 30 positives. On the 5,000-patient panel, they had 0.775480/0.151913 and 0.795313/0.159381 over 13 such labels. This checks the saved data and arithmetic; it does not independently establish the source labels, model inference, independence of the reused panels, or clinical performance.

Private evidence under `AUDIT_WORKSPACE`: `verify_saved_metrics_tmp.py` SHA-256 `12a76545a7983e4b98593658f75672f0463cc6f5c0c47069b65497a057cbbef2`; `inference_audit_2026-11-03/panel_2250_patient_predictions.csv` SHA-256 `0bd81c0e0357f313854b5d5f825dd322f269cd016ceccb67c7f019f909989c4a`; `panel_5k_patient_predictions.csv` SHA-256 `c57d4a64d7318e37f33da15322d302b49bf854430fc7383b4da6aea2cf585e70`; `panel_2250_common_clean_manifest.csv` SHA-256 `34febfbfbedf4bfc78210c817d40fc0ed04f328d8cc1486c583bdd83470a741d`; `panel_5k_common_clean_manifest.csv` SHA-256 `aaca5bf8c3aea88b87f8828b16c486f175fc7e417c0a259a640f0979b9bb0dd7`. The CSV files contain patient-level records and are excluded from the public package. To repeat locally, mount the audit workspace read-only at `/evidence` in the API image and run `python /evidence/verify_saved_metrics_tmp.py`.

The six saved validation Dice values average to 0.8326745542, and the six repeated internal-holdout values average to 0.7882745320, matching the saved summaries. The audited best checkpoint and shipped segmentation checkpoint both have SHA-256 `46622de72ac91c7e17d36da19d1088e90e1ad836a0d6e00533b37b805b5cb7a4`. Source: `AUDIT_WORKSPACE/work/segmentation_retrain_20261103_1132/final_inference/final_inference_metrics.json`, SHA-256 `8fea12f08441d40cff2f445610e0abee05efad4e73122ec6911a4b728abdd7c12`. This verifies the reported averages and checkpoint identity, not an independent recomputation of the six-case scores. A later check of three saved source-grid masks against separately obtained reference masks is described below. Validation selected the checkpoint, and the holdout had prior project use.

### Three saved segmentation masks checked against reference files

Three small `labelsTr` files were obtained from the public [MedOtter MSD Spleen mirror](https://huggingface.co/datasets/MedOtter/msd-spleen) at revision `6ba3d36fc2521cd079658f7787ccbb9f7d4185d7`; total download was 69,972 bytes. Their SHA-256 values matched the mirror's LFS metadata. The retained local demo image also exactly matched the mirror's `imagesTr/spleen_10.nii.gz` SHA-256 `a0ff3c1e61e150220fdea277c7999e8161b82c6eef4c2c737980b745062c08a3`. For all three cases, the reference and saved prediction had matching shape and affine, and their foreground voxel counts matched the archived report.

Comparing the saved source-grid masks against those reference files reproduced the archived Dice values exactly: `spleen_10` **0.904688694**, `spleen_17` **0.722702797**, and `spleen_63` **0.851323812**. These are saved API-path predictions with threshold 0.50 and no flip TTA. They are distinct from the resampled-grid argmax scores used in the six-case validation mean above. The three cases were part of the validation set used for checkpoint selection; this is not a fresh test or a claim of improvement. The original reference-file hashes were not retained, so binary identity with the original labels is not independently established; the mirror's hashes, geometry, foreground counts and reproduced scores are the evidence available. No model inference, training or official test images were used in this check.

The private evidence is `AUDIT_WORKSPACE/msd_mask_verification/verify.py`, SHA-256 `fc3e651764c1b7133f3a06e4a3619690e88d00ff7f07975ab5f47ca11523d4dd`, the three mirror labels in that folder, and `AUDIT_WORKSPACE/work/api_path_argmax_config_check_20261103/report.json`, SHA-256 `5603ea08eb9ce37e7cce230fa30ebbd056790b88b05f7fa3392dd5cf05d0f77a`. The labels remain private and are not in the public package. Mirror label SHA-256 values for cases 10, 17 and 63 respectively are `d371db0dc59819e614732a563b6ede1cc0d2b7bd23ba7ae03cf138d08e63d693`, `74aa8723e719d165f24a3e45357389010ef1eed4dd4c6ecda5d0a81eae0bc047`, and `d9f3d1556d24975dbe956b2aa6edeccda661d348be9e5acb9f46cd04666fa98e`. MSD publication: Antonelli et al., *The Medical Segmentation Decathlon*, Nature Communications 13, 4128 (2022), https://doi.org/10.1038/s41467-022-30695-9.

## Public CT interoperability check and guarded count option

One public de-identified abdominal CT series from TCIA was used only for an isolated local interoperability check, not for training or an accuracy claim. TCIA metadata reported **81 instances and 42,766,514 uncompressed bytes**. The downloaded ZIP was 23,645,076 bytes, SHA-256 `14a1b3a0fdc9faff4fda58a884d592d9c09ba1c3a310a2bdef8c8e448b544c16`. It contained 81 DICOM files and a license file. The images are not included in the release.

Orthanc received all 81 instances and marked the study stable but the series status `Unknown`. The original route returned HTTP 422. After the guarded change, the default request still returned 422; a supplied count of 80 returned 422; the independently reported count of 81 returned HTTP 200. Retrieval of the locally stored mask returned HTTP 200 and its SHA-256 matched the recorded digest and response header. The result record contained status `Unknown`, basis `operator_supplied_count_match`, and count 81. The same flow was exercised through the React UI. The input had no reference mask; the output is not a Dice or accuracy result and does not establish broad scanner compatibility.

For an `Unknown` series, the operator must provide a count from an independent source. The adapter compares counts, requires study stability, checks that membership and status did not change during retrieval, and still applies geometry, intensity and size checks. A count match is not proof that the series is complete or correct. `Incomplete`, `Missing`, and `Inconsistent` statuses remain rejected. The default without a count remains rejection. Other modalities, compressed formats, acquisitions, clinical use and DICOM SEG/SR write-back are not established.

The full staged suite after the change reported **69 passed in 8.14 seconds**, with no warnings, in the disposable Python 3.12 CPU container using the test-only dependency versions above. Browser checks also confirmed the existing chest prediction, Grad-CAM, demo review and NIfTI segmentation paths. The final API Docker image was rebuilt; its copied API source hashes matched the staged files. An initial automated request arrived before startup and was refused; after `/health` returned HTTP 200, the real-CT route and mask retrieval passed again on that rebuilt image. The temporary CT ZIP and isolated in-memory Orthanc instance were removed after the check. The models were not retrained or modified. The private audit scripts are `AUDIT_WORKSPACE/tcia_ct_verify/check_orthanc.py` (SHA-256 `1ea7b6989c644e1afce689f6d56e34b1b9cfa01616c60b855101c1f2d55f4c97`) and `check_candidate.py` (SHA-256 `672e9ae9fe801f820c4094744c2624a21cc483ac2076b07cb44590eb5c4e5b65`); `sanitized_result.json` has SHA-256 `7b25f6f2156898e95af72509c22f922ba97b32f3cab6e439682b80d03675fe82`.

Data citation: Cancer Moonshot Biobank. (2022). *Cancer Moonshot Biobank – Colorectal Cancer Collection (CMB-CRC)* (Version 11) [dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/DJG7-GZ87. The checked series metadata lists the Creative Commons Attribution 4.0 International license: https://creativecommons.org/licenses/by/4.0/. Data used in this interoperability check were generated by the National Cancer Institute's Cancer Moonshot Biobank. TCIA API documentation: https://wiki.cancerimagingarchive.net/display/Public/NBIA+Search+REST+API+Guide.

## Earlier full-suite result (Nov 4)

The full package test suite passed **58 tests with 2 warnings in 8.97 seconds**. This ran in a disposable container based on the cached `submission_final-api:latest` image, under Python 3.12 with CPU PyTorch. The project was mounted read-only and `PYTHONDONTWRITEBYTECODE=1` prevented bytecode output in the mount. The only temporary installs were `pytest==9.0.2`, `pandas==2.3.3`, and `scikit-learn==1.7.2`; the container was removed after the run. No model, dataset, checkpoint, or dependency file was downloaded or edited. The two warnings were `SyntaxWarning`s for invalid escape sequences `\(` and `\)` in `test_ct_series.py`; there were no failures. This result does not verify the pinned Windows/Python 3.10/CUDA 12.8 environment.

Exact command:

```powershell
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -e NEXVIA_ORTHANC_SEGMENTATION_MASK_DIR=/tmp/nexvia-test-masks -v '<local-path>' -w /app/submission_final --entrypoint python submission_final-api:latest -c "import subprocess,sys; subprocess.check_call([sys.executable,'-m','pip','install','--no-cache-dir','pytest==9.0.2','pandas==2.3.3','scikit-learn==1.7.2']); subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider'],check=True)"
```

## Historical DVC training-stage boundary

The release does not contain the NIH training images, their historical DVC manifest, a shared DVC remote, or the recorded machine-local environment. That historical pipeline was not rerun. The committed lock describes a 3-epoch subset screen. Its api/modeling.py and train_classifier.py MD5 hashes differ from the current release files; requirements-training.txt still matches. The 28,486,934-byte lock output differs from the 28,496,134-byte packaged classifier, and the segmentation model is outside this DVC stage. It does not establish reproduction of the deployed classifier or spleen model; the stage remains non-reproducible from this package. The separate, later DVC restore of two retained demo examples is described above and in `docs/data_inventory.md`.
