# Test results

Run with `python -m pytest -q` from the repository root, using the packages in `requirements.txt`.

| Suite | Tests | What it covers |
|---|---:|---|
| API, classifier, CT conversion, data audit, segmentation split, DICOM upload, Orthanc auto-pull | 72 | Endpoint behaviour, preprocessing, DICOM decoding, geometry checks, sync lock and interval worker |
| SHAP attribution and guarded question answering (`test_shap_and_qa.py`) | 10 | KernelSHAP additivity and repeatability on the served checkpoint, overlay output, missing key (503), urgent-symptom bypass, prompt guardrails, key never returned, upstream failure (502), bounded result store |
| Real DICOM sample (`test_real_dicom_sample.py`) | 3 | DICOM and PNG scores match on NIH images; skipped when the sample images are absent |

Current result on `e553b94` with the NIH sample images present: 85 passed, 0 failed. Without the images the real-sample suite is skipped, leaving 82 passed and 3 skipped.

Tests that depend on data or checkpoints skip with a reason when those files are absent. The real-sample DICOM tests need the NIH sample images (`dvc pull`).

## Not covered by these tests

- Live Orthanc: the sync and CT routes have no tests against a running or mocked Orthanc server. Only the interval timer is tested (`test_orthanc_auto_sync.py`).
- Docker: the images are not built in CI. Docker builds and the browser flow must be checked on a machine with Docker.
- Spleen accuracy: no held-out evaluation is possible from this repository (see `limits_and_bias.md`).
