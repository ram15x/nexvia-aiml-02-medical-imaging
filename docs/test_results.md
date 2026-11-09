# Test results

Run with `python -m pytest -q` from the repository root, using the packages in `requirements.txt`.

| Suite | Tests | What it covers |
|---|---:|---|
| API, classifier, CT conversion, data audit, segmentation split, DICOM upload, Orthanc auto-pull | 72 | Endpoint behaviour, preprocessing, DICOM decoding, geometry checks, sync lock and interval worker |
| SHAP attribution and guarded question answering (`test_shap_and_qa.py`) | 10 | KernelSHAP additivity and repeatability on the served checkpoint, overlay output, missing key (503), urgent-symptom bypass, prompt guardrails, key never returned, upstream failure (502), bounded result store |
| Hosted-demo guards (`test_guards.py`): rate limits, Orthanc switch, proxy header handling | 8 | Limits apply per client and only to POST, `/qa` has its own hourly cap, `/orthanc/*` answers 404 when disabled, and every guard is off by default |
| Frozen classifier baseline (`test_baseline.py`) | 3 | The committed metrics match `results/nih_sample/baseline.json`, and a macro or per-label AUROC drop is reported as a regression |
| Real DICOM sample (`test_real_dicom_sample.py`) | 3 | DICOM and PNG scores match on NIH images; skipped when the sample images are absent |

Current result with a clean venv on the pinned API dependencies (Python 3.13, CPU, NIH sample images and demo image absent): 92 passed, 4 skipped, 0 failed. The skips are three real-sample DICOM tests (`dvc pull` supplies the images) and one SHAP test that needs the demo image. With those files present the earlier full run was 85 of 85 before the guard and baseline tests were added.

Tests that depend on data or checkpoints skip with a reason when those files are absent. The real-sample DICOM tests need the NIH sample images (`dvc pull`).

## Not covered by these tests

- Live Orthanc: the sync and CT routes have no tests against a running or mocked Orthanc server. Only the interval timer is tested (`test_orthanc_auto_sync.py`).
- Docker: the images are not built in CI. `scripts/smoke_docker.ps1` builds the stack and checks it on a machine with Docker; the browser flow was checked by driving the built UI against the running API (predict, Grad-CAM, review log, spleen segmentation).
- Spleen accuracy: not part of the pytest suite. The held-out evaluation is a separate run (`evaluate_segmentation.py`) whose result is in `results/spleen/test_result.json`; it needs the MSD Task09 volumes, which are not in this repository.
