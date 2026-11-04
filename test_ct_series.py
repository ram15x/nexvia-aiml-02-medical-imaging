from __future__ import annotations

from io import BytesIO

import nibabel as nib
import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from api.ct_series import CTSeriesError, _series_volume, fetch_orthanc_ct_series_to_nifti
from api import app as api_module


def make_ct_slice(position_z: float, pixel_value: int, *, series_uid: str,
                  orientation=(1, 0, 0, 0, 1, 0), pixels=None) -> FileDataset:
    sop_uid = generate_uid()
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = sop_uid
    dataset = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = sop_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.Modality = "CT"
    dataset.Rows = 2
    dataset.Columns = 3
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.ImageOrientationPatient = list(orientation)
    dataset.ImagePositionPatient = [10.0, 20.0, position_z]
    dataset.PixelSpacing = [0.8, 0.5]
    dataset.RescaleSlope = 2
    dataset.RescaleIntercept = -1000
    if pixels is None:
        pixels = np.full((2, 3), pixel_value, dtype=np.int16)
    else:
        pixels = np.asarray(pixels, dtype=np.int16)
        assert pixels.shape == (2, 3)
    dataset.PixelData = pixels.tobytes()
    return dataset


def serialize(dataset: FileDataset) -> bytes:
    stream = BytesIO()
    dataset.save_as(stream, enforce_file_format=True)
    return stream.getvalue()


def test_ct_series_sorts_slices_rescales_hu_and_preserves_ras_affine():
    series_uid = generate_uid()
    datasets = [
        make_ct_slice(2.0, 12, series_uid=series_uid),
        make_ct_slice(0.0, 10, series_uid=series_uid),
        make_ct_slice(1.0, 11, series_uid=series_uid),
    ]
    volume, affine = _series_volume(datasets)
    assert volume.shape == (3, 2, 3)  # NIfTI (column, row, slice)
    assert volume[0, 0, :].tolist() == pytest.approx([-980, -978, -976])
    np.testing.assert_allclose(affine, [
        [-0.5, 0, 0, -10],
        [0, -0.8, 0, -20],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], rtol=0, atol=1e-5)


def test_ct_series_maps_asymmetric_pixels_to_expected_ras_coordinates():
    series_uid = generate_uid()
    orientation = (0, 1, 0, -1, 0, 0)
    pixels0 = np.array([[10, 11, 12], [20, 21, 22]], dtype=np.int16)
    pixels1 = pixels0 + 100
    datasets = [
        make_ct_slice(1.0, 0, series_uid=series_uid, orientation=orientation, pixels=pixels1),
        make_ct_slice(0.0, 0, series_uid=series_uid, orientation=orientation, pixels=pixels0),
    ]

    volume, affine = _series_volume(datasets)

    # NIfTI axes are (column, row, slice); this catches row/column swaps
    # using distinct values at every pixel and a non-identity DICOM orientation.
    assert volume.shape == (3, 2, 2)
    assert volume[2, 1, 1] == pytest.approx((22 + 100) * 2 - 1000)
    voxel = np.array([2.0, 1.0, 1.0, 1.0])
    expected_ras = np.array([-9.2, -21.0, 1.0, 1.0])
    np.testing.assert_allclose(affine @ voxel, expected_ras, rtol=0, atol=1e-5)


def test_ct_series_rejects_irregular_slice_spacing():
    series_uid = generate_uid()
    datasets = [
        make_ct_slice(0.0, 10, series_uid=series_uid),
        make_ct_slice(1.0, 11, series_uid=series_uid),
        make_ct_slice(3.0, 12, series_uid=series_uid),
    ]
    with pytest.raises(CTSeriesError, match="Irregularly spaced"):
        _series_volume(datasets)


def test_ct_series_rejects_mixed_orientation():
    series_uid = generate_uid()
    datasets = [
        make_ct_slice(0.0, 10, series_uid=series_uid),
        make_ct_slice(1.0, 11, series_uid=series_uid,
                      orientation=(0, 1, 0, 1, 0, 0)),
    ]
    with pytest.raises(CTSeriesError, match="inconsistent orientation"):
        _series_volume(datasets)


def test_ct_series_rejects_multiframe():
    series_uid = generate_uid()
    datasets = [
        make_ct_slice(0.0, 10, series_uid=series_uid),
        make_ct_slice(1.0, 11, series_uid=series_uid),
    ]
    datasets[1].NumberOfFrames = 2
    with pytest.raises(CTSeriesError, match="multiframe"):
        _series_volume(datasets)


def test_orthanc_fetch_writes_nifti_and_requires_stable_series(monkeypatch, tmp_path):
    monkeypatch.setenv("ORTHANC_PASSWORD", "test-password")
    series_uid = generate_uid()
    instance_bytes = [
        serialize(make_ct_slice(z, 10 + int(z), series_uid=series_uid))
        for z in range(2)
    ]
    ids = ["instance-00000001", "instance-00000002"]

    class JsonResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    class FileResponse:
        def __init__(self, content):
            self.content = content
            self.headers = {"Content-Length": str(len(content))}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield self.content

        def close(self):
            pass

    def mock_get(url, **kwargs):
        if url.endswith("/series/series-00000001"):
            return JsonResponse({"Status": "Complete", "Instances": ids,
                                 "ParentStudy": "study-00000001"})
        if url.endswith("/studies/study-00000001"):
            return JsonResponse({"IsStable": True})
        for index, instance_id in enumerate(ids):
            if url.endswith(f"/instances/{instance_id}/file"):
                return FileResponse(instance_bytes[index])
        raise AssertionError(f"Unexpected Orthanc URL: {url}")

    monkeypatch.setattr("api.ct_series.requests.get", mock_get)
    output_path = tmp_path / "converted.nii.gz"
    metadata = fetch_orthanc_ct_series_to_nifti("series-00000001", output_path)
    image = nib.load(str(output_path))
    assert metadata["instance_count"] == 2
    assert metadata["voxel_count"] == 12
    assert image.shape == (3, 2, 2)
    np.testing.assert_allclose(image.affine, np.asarray(metadata["affine"]), atol=1e-5)


def test_orthanc_fetch_rejects_unstable_series(monkeypatch):
    monkeypatch.setenv("ORTHANC_PASSWORD", "test-password")

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    def mock_get(url, **kwargs):
        if url.endswith("/series/series-00000001"):
            return Response({"Status": "Complete",
                             "Instances": ["instance-00000001", "instance-00000002"],
                             "ParentStudy": "study-00000001"})
        return Response({"IsStable": False})

    monkeypatch.setattr("api.ct_series.requests.get", mock_get)
    with pytest.raises(CTSeriesError, match="stable"):
        fetch_orthanc_ct_series_to_nifti("series-00000001", __import__("pathlib").Path("never.nii.gz"))


@pytest.mark.parametrize("status", ["Unknown", "Missing", "Inconsistent", "Incomplete", None])
def test_orthanc_fetch_rejects_unconfirmed_series_status_before_download(monkeypatch, tmp_path, status):
    monkeypatch.setenv("ORTHANC_PASSWORD", "test-password")
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            payload = {"Instances": ["instance-00000001", "instance-00000002"],
                    "ParentStudy": "study-00000001"}
            if status is not None:
                payload["Status"] = status
            return payload

    def mock_get(url, **kwargs):
        calls.append(url)
        return Response()

    monkeypatch.setattr("api.ct_series.requests.get", mock_get)
    output_path = tmp_path / "must-not-exist.nii.gz"
    expected_status = status if status is not None else "missing"
    with pytest.raises(CTSeriesError, match=f"not confirmed complete \(status: {expected_status}\)"):
        fetch_orthanc_ct_series_to_nifti("series-00000001", output_path)

    assert len(calls) == 1
    assert calls[0].endswith("/series/series-00000001")
    assert not output_path.exists()


def test_orthanc_segmentation_records_compact_non_phi_result(monkeypatch, tmp_path):
    checkpoint = tmp_path / "spleen.pt"
    checkpoint.write_bytes(b"test checkpoint marker")
    log_path = tmp_path / "orthanc_segmentations.jsonl"
    monkeypatch.setattr(api_module, "SEGMENTATION_CHECKPOINT", checkpoint)
    monkeypatch.setattr(api_module, "ORTHANC_SEGMENTATION_RESULTS_PATH", log_path)

    import api.ct_series as ct_module
    monkeypatch.setattr(ct_module, "fetch_orthanc_ct_series_to_nifti", lambda series_id, path: {
        "orthanc_series_id": series_id, "instance_count": 2,
        "input_shape": [3, 2, 2], "conversion": "geometry-preserving test conversion",
    })
    monkeypatch.setattr(api_module, "_process_spleen_volume", lambda input_path, output_dir: {
        "input_shape": [3, 2, 2], "output_geometry_matches_input": True,
        "predicted_spleen_voxels": 3, "estimated_predicted_volume_ml": 0.02,
        "inference_augmentation": "test", "preview_planes": {},
        "output_mask_filename": "spleen-mask.nii.gz",
        "output_mask_nii_gz_base64": "", "disclaimer": "Not for clinical use.",
    })
    response = api_module.segment_orthanc_ct_series("series-00000001")
    assert response["stored"] is True
    assert response["dicom_instance_count"] == 2
    saved = log_path.read_text(encoding="utf-8")
    assert "orthanc_series_id" in saved
    assert "PatientName" not in saved
    assert "PatientID" not in saved
