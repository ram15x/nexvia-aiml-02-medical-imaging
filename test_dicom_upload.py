"""DICOM upload path: a DX object converted from a PNG must score like the PNG."""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("pydicom")
from tools.png_to_dicom_dx import png_to_dicom  # noqa: E402


@pytest.fixture()
def gray_png(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    arr = (rng.random((256, 256)) * 255).astype(np.uint8)
    path = tmp_path / "xray.png"
    Image.fromarray(arr, mode="L").save(path)
    return path


def test_dicom_round_trip_preserves_pixels(gray_png, tmp_path):
    from api.orthanc import dicom_bytes_to_image

    dcm = png_to_dicom(gray_png, tmp_path / "xray.dcm", patient_id="TEST-0001")
    image = dicom_bytes_to_image(dcm.read_bytes())
    original = np.asarray(Image.open(gray_png).convert("L"))
    decoded = np.asarray(image.convert("L"))
    assert decoded.shape == original.shape
    assert np.array_equal(decoded, original)


def test_predict_accepts_dicom(gray_png, tmp_path):
    from fastapi.testclient import TestClient
    import api.app as app_module

    if not app_module.CHECKPOINT.is_file():
        pytest.skip("classifier checkpoint not installed")
    dcm = png_to_dicom(gray_png, tmp_path / "xray.dcm", patient_id="TEST-0002")
    client = TestClient(app_module.app)
    with open(gray_png, "rb") as fh:
        png_scores = client.post("/predict", files={"file": ("x.png", fh, "image/png")}).json()
    with open(dcm, "rb") as fh:
        response = client.post("/predict", files={"file": ("x.dcm", fh, "application/dicom")})
    assert response.status_code == 200, response.text
    dicom_scores = response.json()
    assert [f["label"] for f in dicom_scores["findings"]] == [f["label"] for f in png_scores["findings"]]
    for a, b in zip(png_scores["findings"], dicom_scores["findings"]):
        assert abs(a["score"] - b["score"]) < 1e-5


def test_predict_rejects_non_radiograph_dicom(tmp_path, gray_png):
    from fastapi.testclient import TestClient
    import pydicom
    import api.app as app_module

    dcm = png_to_dicom(gray_png, tmp_path / "ct.dcm", patient_id="TEST-0003")
    ds = pydicom.dcmread(dcm)
    ds.Modality = "CT"
    ds.save_as(dcm)
    client = TestClient(app_module.app)
    with open(dcm, "rb") as fh:
        response = client.post("/predict", files={"file": ("ct.dcm", fh, "application/dicom")})
    assert response.status_code == 422
