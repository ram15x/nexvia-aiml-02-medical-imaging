"""Real-image DICOM check: NIH sample PNGs converted to DICOM must score identically.

Skipped when the NIH sample images are not present (they are tracked by DVC).
"""

from pathlib import Path

import pytest

pytest.importorskip("pydicom")
from PIL import Image  # noqa: E402

SAMPLE = Path("data/nih_test_sample_1000/images")
IMAGES = ["00028173_004.png", "00028173_005.png", "00028173_009.png"]


@pytest.mark.skipif(not all((SAMPLE / name).is_file() for name in IMAGES),
                    reason="NIH sample images not present (run dvc pull)")
@pytest.mark.parametrize("name", IMAGES)
def test_nih_png_and_dicom_score_identically(name, tmp_path):
    from fastapi.testclient import TestClient
    import api.app as app_module
    from tools.png_to_dicom_dx import png_to_dicom

    if not app_module.CHECKPOINT.is_file():
        pytest.skip("classifier checkpoint not installed")
    png = SAMPLE / name
    dcm = png_to_dicom(png, tmp_path / name.replace(".png", ".dcm"), patient_id=name.split("_")[0])
    client = TestClient(app_module.app)
    with open(png, "rb") as fh:
        png_scores = client.post("/predict", files={"file": (name, fh, "image/png")}).json()["findings"]
    with open(dcm, "rb") as fh:
        response = client.post("/predict", files={"file": (dcm.name, fh, "application/dicom")})
    assert response.status_code == 200, response.text
    dicom_scores = response.json()["findings"]
    for a, b in zip(png_scores, dicom_scores):
        assert a["label"] == b["label"]
        assert abs(a["score"] - b["score"]) < 1e-6
