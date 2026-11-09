# Test results

Run with `python -m pytest -q` from the repository root, using the packages in `requirements.txt`.

| Suite | Tests | What it covers |
|---|---:|---|
| API, classifier, CT conversion, data audit, segmentation split, DICOM upload, Orthanc auto-pull, real DICOM sample | 72 | Endpoint behaviour, preprocessing, DICOM decoding, geometry checks, sync lock and interval worker |

Current result: 72 passed, 0 failed (last run on the commit that added this document).

Tests that depend on data or checkpoints skip with a reason when those files are absent. The real-sample DICOM tests need the NIH sample images (`dvc pull`).

## Not covered by these tests

- Live Orthanc: the sync and CT routes have no tests against a running or mocked Orthanc server. Only the interval timer is tested (`test_orthanc_auto_sync.py`).
- Docker: the images are not built in CI. Docker builds and the browser flow must be checked on a machine with Docker.
- Spleen accuracy: no held-out evaluation is possible from this repository (see `limits_and_bias.md`).
