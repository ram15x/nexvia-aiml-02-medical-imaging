from io import BytesIO
import base64
import hashlib
import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import requests
import torch
from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import UploadFile
from torch import nn

from api import app as api_module
from api.modeling import ARCHITECTURE_NAMES, build_classifier, gradcam_target, image_to_tensor
from api import orthanc as orthanc_module


class FixedLogitModel(nn.Module):
    def forward(self, images):
        return torch.zeros((images.shape[0], 14), dtype=images.dtype)


def test_health_reports_segmentation_checkpoint_and_inference_settings(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "spleen_candidate.pt"
    torch.save(
        {"patch_size": [96, 96, 96], "architecture": "monai_unet_3d"},
        checkpoint_path,
    )
    monkeypatch.setattr(api_module, "SEGMENTATION_CHECKPOINT", checkpoint_path)
    monkeypatch.setattr(api_module, "SEGMENTATION_THRESHOLD", 0.50)
    monkeypatch.setattr(api_module, "SEGMENTATION_TTA", False)

    health = api_module.health()
    info = health["segmentation"]

    assert info["checkpoint_path"] == str(checkpoint_path)
    assert info["checkpoint_sha256"] == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert info["roi_size"] == [96, 96, 96]
    assert info["threshold"] == 0.50
    assert info["flip_tta"] is False
    assert info["architecture"] == "monai_unet_3d"


def png_bytes(size=(24, 16)):
    stream = BytesIO()
    Image.fromarray(np.full((size[1], size[0], 3), 127, dtype=np.uint8)).save(stream, format="PNG")
    return stream.getvalue()


def dicom_bytes(modality="DX", photometric="MONOCHROME2"):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    sop_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = sop_uid
    dataset.Modality = modality
    dataset.Rows = 4
    dataset.Columns = 4
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = np.arange(16, dtype=np.uint16).tobytes()
    stream = BytesIO()
    dataset.save_as(stream, enforce_file_format=True)
    return stream.getvalue()


@pytest.mark.parametrize("modality", ["CT", "MR", "missing"])
def test_orthanc_chest_classifier_rejects_non_radiograph_modalities(monkeypatch, modality):
    monkeypatch.setenv("ORTHANC_PASSWORD", "test-password")
    content = dicom_bytes(modality="" if modality == "missing" else modality)

    class Response:
        headers = {"Content-Length": str(len(content))}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield content

        def close(self):
            pass

    monkeypatch.setattr(orthanc_module.requests, "get", lambda *args, **kwargs: Response())
    with pytest.raises(orthanc_module.UnsupportedDicomModality):
        orthanc_module.fetch_orthanc_image("instance-12345678")


def test_orthanc_chest_classifier_accepts_cr_dx_radiograph(monkeypatch):
    monkeypatch.setenv("ORTHANC_PASSWORD", "test-password")
    content = dicom_bytes(modality="DX")

    class Response:
        headers = {"Content-Length": str(len(content))}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield content

        def close(self):
            pass

    monkeypatch.setattr(orthanc_module.requests, "get", lambda *args, **kwargs: Response())
    image = orthanc_module.fetch_orthanc_image("instance-12345678")
    assert image.mode == "RGB"
    assert image.size == (4, 4)


def test_single_orthanc_prediction_returns_415_for_ct(monkeypatch):
    def reject_ct(_instance_id):
        raise orthanc_module.UnsupportedDicomModality("CT")

    monkeypatch.setattr(orthanc_module, "fetch_orthanc_image", reject_ct)
    with pytest.raises(HTTPException) as error:
        api_module.predict_orthanc_instance("instance-12345678")
    assert error.value.status_code == 415
    assert "CR/DX" in error.value.detail


def test_supported_model_registry_has_fresh_14_label_heads():
    assert set(ARCHITECTURE_NAMES) == {
        "densenet121", "efficientnet_b0", "swin_t", "densenet121_swin_t",
        "densenet121_evidence_concentration_gate",
        "densenet121_label_attention", "densenet121_label_graph",
        "densenet121_label_residual", "densenet121_label_graph_residual",
    }
    model = build_classifier(
        "densenet121_label_graph_residual", 14, pretrained=False,
        label_adjacency=np.eye(14, dtype=np.float32),
    ).eval()
    with torch.inference_mode():
        output = model(torch.zeros((1, 3, 64, 64)))
    assert output.shape == (1, 14)
    assert torch.isfinite(output).all()


def test_hybrid_gradcam_targets_cnn_branch():
    model = build_classifier("densenet121_swin_t", 14, pretrained=False)
    layer, channels_last = gradcam_target(model, "densenet121_swin_t")
    assert layer is model.cnn.features
    assert channels_last is False


def test_shared_preprocessing_returns_finite_normalized_tensor():
    tensor = image_to_tensor(Image.fromarray(np.full((20, 30), 255, dtype=np.uint8)))
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_upload_rejects_content_type_mismatch():
    with pytest.raises(HTTPException) as error:
        api_module._decode_uploaded_image(png_bytes(), "image/jpeg")
    assert error.value.status_code == 415


def test_spleen_route_rejects_non_nifti_before_running_model():
    upload = UploadFile(filename="scan.dcm", file=BytesIO(b"not nifti"))
    with pytest.raises(HTTPException) as error:
        api_module.segment_spleen(upload)
    assert error.value.status_code == 415


def test_spleen_route_returns_preview_and_removes_temporary_artifacts(monkeypatch, tmp_path):
    import nibabel as nib

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"mock checkpoint")
    monkeypatch.setattr(api_module, "SEGMENTATION_CHECKPOINT", checkpoint)
    observed = {}

    def fake_inference(input_path, checkpoint_path, output_dir, threshold, device, use_flip_tta=False):
        source = nib.load(str(input_path))
        data = np.asanyarray(source.dataobj)
        mask_path = output_dir / "mask.nii.gz"
        nib.save(nib.Nifti1Image((data > 0).astype(np.uint8), source.affine), str(mask_path))
        observed["input_path"] = input_path
        observed["use_flip_tta"] = use_flip_tta
        return {
            "input": str(input_path), "mask": str(mask_path),
            "predicted_spleen_voxels": int(data.size),
            "estimated_predicted_volume_ml": 0.01,
            "output_geometry_matches_input": True,
            "architecture": "monai_unet_3d", "threshold": threshold,
        }

    monkeypatch.setitem(sys.modules, "segment_spleen", SimpleNamespace(run_inference=fake_inference))
    image = nib.Nifti1Image(np.ones((12, 10, 8), dtype=np.float32), np.eye(4))
    upload = UploadFile(filename="scan.nii", file=BytesIO(image.to_bytes()))

    result = api_module.segment_spleen(upload)

    assert result["output_geometry_matches_input"] is True
    assert set(result["preview_planes"]) == {"axial", "coronal", "sagittal"}
    assert result["preview_planes"]["axial"]["png_base64"]
    assert result["output_mask_filename"] == "spleen-mask.nii.gz"
    assert result["output_mask_nii_gz_base64"]
    mask_bytes = gzip.decompress(base64.b64decode(result["output_mask_nii_gz_base64"]))
    downloaded_mask = nib.Nifti1Image.from_bytes(mask_bytes)
    assert downloaded_mask.shape == image.shape
    assert np.allclose(downloaded_mask.affine, image.affine)
    for preview in result["preview_planes"].values():
        rendered = Image.open(BytesIO(base64.b64decode(preview["png_base64"])))
        assert rendered.size[0] > 0 and rendered.size[1] > 0
    assert result["upload_retained"] is False
    assert observed["use_flip_tta"] is False
    assert not observed["input_path"].exists()


def test_orthanc_ct_mask_is_persisted_and_retrievable(monkeypatch, tmp_path):
    from api import ct_series as ct_series_module

    series_id = "series-12345678"
    mask_bytes = gzip.compress(b"synthetic NIfTI mask payload")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    results_path = tmp_path / "results.jsonl"
    mask_dir = tmp_path / "masks"

    monkeypatch.setattr(api_module, "SEGMENTATION_CHECKPOINT", checkpoint)
    monkeypatch.setattr(api_module, "ORTHANC_SEGMENTATION_RESULTS_PATH", results_path)
    monkeypatch.setattr(api_module, "ORTHANC_SEGMENTATION_MASK_DIR", mask_dir)
    monkeypatch.setattr(
        ct_series_module,
        "fetch_orthanc_ct_series_to_nifti",
        lambda *_args, **_kwargs: {
            "conversion": "synthetic geometry check",
            "instance_count": 24,
            "input_shape": [8, 8, 24],
        },
    )
    monkeypatch.setattr(
        api_module,
        "_process_spleen_volume",
        lambda *_args, **_kwargs: {
            "output_mask_nii_gz_base64": base64.b64encode(mask_bytes).decode("ascii"),
            "predicted_spleen_voxels": 12,
            "estimated_predicted_volume_ml": 0.03,
            "disclaimer": "Synthetic test only.",
        },
    )

    response = api_module.segment_orthanc_ct_series(series_id)

    assert response["stored"] is True
    assert response["mask_sha256"] == hashlib.sha256(mask_bytes).hexdigest()
    assert response["mask_url"].endswith("/mask")
    saved = api_module.list_orthanc_segmentation_results(series_id)["results"]
    assert len(saved) == 1
    assert saved[0]["mask_sha256"] == response["mask_sha256"]
    assert "PatientName" not in json.loads(results_path.read_text(encoding="utf-8").splitlines()[0])

    downloaded = api_module.download_orthanc_segmentation_mask(series_id, response["result_id"])
    assert Path(downloaded.path).read_bytes() == mask_bytes
    assert downloaded.headers["x-content-sha256"] == response["mask_sha256"]

    (mask_dir / f"{response['result_id']}.nii.gz").write_bytes(b"changed")
    with pytest.raises(HTTPException) as error:
        api_module.download_orthanc_segmentation_mask(series_id, response["result_id"])
    assert error.value.status_code == 500


def test_score_response_identifies_model_and_returns_all_findings(monkeypatch):
    monkeypatch.setattr(api_module, "_load_model", lambda: FixedLogitModel())
    monkeypatch.setattr(api_module, "_model_metadata", {
        "architecture": "densenet121", "model_name": "DenseNet-121", "best_epoch": 6,
    })
    image = Image.open(BytesIO(png_bytes()))
    body = api_module._score_image(image)
    assert body["model"] == "DenseNet-121"
    assert len(body["findings"]) == 14
    assert body["model_metadata"]["best_epoch"] == 6
    assert all(item["score"] == 0.5 for item in body["findings"])


def test_orthanc_sync_resumes_stable_study_across_bounded_calls(monkeypatch, tmp_path):
    monkeypatch.setattr(api_module, "ORTHANC_RESULTS_PATH", tmp_path / "orthanc.jsonl")
    monkeypatch.setattr(api_module, "ORTHANC_SYNC_STATE_PATH", tmp_path / "orthanc_state.json")
    monkeypatch.setattr(api_module, "_score_image", lambda image: {"findings": [], "disclaimer": "demo"})
    study_id = "study-12345678"
    instances = [
        {"study_id": study_id, "series_id": "series-12345678", "instance_id": "instance-12345678"},
        {"study_id": study_id, "series_id": "series-12345678", "instance_id": "instance-22345678"},
    ]
    cursors = []

    def changes(since, limit):
        cursors.append(since)
        if since == 0:
            return {"Changes": [{"Seq": 1, "ChangeType": "StableStudy", "ID": study_id}],
                    "Last": 1, "Done": True}
        return {"Changes": [], "Last": since, "Done": True}

    orthanc_stub = SimpleNamespace(
        UnsupportedDicomModality=orthanc_module.UnsupportedDicomModality,
        get_orthanc_changes=changes,
        list_stable_study_instances=lambda received_id: instances if received_id == study_id else [],
        fetch_orthanc_image=lambda instance_id: Image.new("RGB", (4, 4)),
    )
    monkeypatch.setitem(sys.modules, "api.orthanc", orthanc_stub)

    first = api_module.sync_orthanc_studies(limit=1)
    second = api_module.sync_orthanc_studies(limit=1)
    third = api_module.sync_orthanc_studies(limit=1)

    assert first["processed"] == 1
    assert first["skipped"] == 0
    assert first["pending_studies"] == 1
    assert second["processed"] == 1
    assert second["pending_studies"] == 0
    assert third["processed"] == 0
    assert cursors == [0, 1, 1]
    assert len((tmp_path / "orthanc.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_orthanc_sync_keeps_study_pending_after_transient_network_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(api_module, "ORTHANC_RESULTS_PATH", tmp_path / "orthanc.jsonl")
    monkeypatch.setattr(api_module, "ORTHANC_SYNC_STATE_PATH", tmp_path / "orthanc_state.json")
    monkeypatch.setattr(api_module, "_score_image", lambda image: {"findings": [], "disclaimer": "demo"})
    study_id = "study-12345678"
    item = {"study_id": study_id, "series_id": "series-12345678", "instance_id": "instance-12345678"}
    failures = {"remaining": 1}

    def fetch(instance_id):
        if failures["remaining"]:
            failures["remaining"] -= 1
            raise requests.Timeout("temporary network failure")
        return Image.new("RGB", (4, 4))

    orthanc_stub = SimpleNamespace(
        UnsupportedDicomModality=orthanc_module.UnsupportedDicomModality,
        get_orthanc_changes=lambda since, limit: {
            "Changes": ([{"Seq": 1, "ChangeType": "StableStudy", "ID": study_id}] if since == 0 else []),
            "Last": 1 if since == 0 else since,
            "Done": True,
        },
        list_stable_study_instances=lambda received_id: [item],
        fetch_orthanc_image=fetch,
    )
    monkeypatch.setitem(sys.modules, "api.orthanc", orthanc_stub)

    first = api_module.sync_orthanc_studies(limit=1)
    second = api_module.sync_orthanc_studies(limit=1)

    assert first["processed"] == 0
    assert first["pending_studies"] == 1
    assert first["temporary_error"] == "Timeout"
    assert second["processed"] == 1
    assert second["pending_studies"] == 0
    assert len((tmp_path / "orthanc.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_orthanc_sync_logs_ct_as_unsupported_instead_of_scoring_as_xray(monkeypatch, tmp_path):
    monkeypatch.setattr(api_module, "ORTHANC_RESULTS_PATH", tmp_path / "orthanc.jsonl")
    monkeypatch.setattr(api_module, "ORTHANC_SYNC_STATE_PATH", tmp_path / "orthanc_state.json")
    study_id = "study-12345678"
    item = {"study_id": study_id, "series_id": "series-12345678", "instance_id": "instance-12345678"}

    def reject_ct(_instance_id):
        raise orthanc_module.UnsupportedDicomModality("CT")

    orthanc_stub = SimpleNamespace(
        UnsupportedDicomModality=orthanc_module.UnsupportedDicomModality,
        get_orthanc_changes=lambda since, limit: {
            "Changes": ([{"Seq": 1, "ChangeType": "StableStudy", "ID": study_id}] if since == 0 else []),
            "Last": 1 if since == 0 else since,
            "Done": True,
        },
        list_stable_study_instances=lambda received_id: [item] if received_id == study_id else [],
        fetch_orthanc_image=reject_ct,
    )
    monkeypatch.setitem(sys.modules, "api.orthanc", orthanc_stub)

    result = api_module.sync_orthanc_studies(limit=1)
    record = json.loads((tmp_path / "orthanc.jsonl").read_text(encoding="utf-8"))

    assert result["processed"] == 0
    assert result["skipped"] == 1
    assert record["status"] == "skipped"
    assert record["reason"] == "unsupported_modality"
    assert record["modality"] == "CT"


def test_orthanc_sync_rejects_excessive_batch_limit():
    with pytest.raises(HTTPException) as error:
        api_module.sync_orthanc_studies(limit=65)
    assert error.value.status_code == 422


def test_orthanc_changes_request_is_cursor_bounded_and_validated(monkeypatch):
    monkeypatch.setenv("ORTHANC_PASSWORD", "test-password")
    seen = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"Changes": [{"Seq": 8, "ChangeType": "StableStudy",
                                 "ID": "study-12345678"}], "Last": 8, "Done": True}

    def fake_get(url, *, params, auth, timeout):
        seen.update({"url": url, "params": params, "auth": auth, "timeout": timeout})
        return Response()

    monkeypatch.setattr(orthanc_module.requests, "get", fake_get)
    payload = orthanc_module.get_orthanc_changes(since=7, limit=500)

    assert payload["Last"] == 8
    assert seen["url"].endswith("/changes")
    assert seen["params"] == {"since": 7, "limit": 500}
    assert seen["auth"] == ("orthanc", "test-password")


def test_orthanc_lists_only_instances_below_stable_study(monkeypatch):
    monkeypatch.setenv("ORTHANC_PASSWORD", "test-password")
    study_id = "study-12345678"
    series_id = "series-12345678"
    instance_id = "instance-12345678"

    class Response:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self.body

    def fake_get(url, *, auth, timeout):
        if url.endswith(f"/studies/{study_id}"):
            return Response({"IsStable": True, "Series": [series_id]})
        if url.endswith(f"/series/{series_id}"):
            return Response({"Instances": [instance_id]})
        raise AssertionError(f"Unexpected Orthanc URL: {url}")

    monkeypatch.setattr(orthanc_module.requests, "get", fake_get)
    result = orthanc_module.list_stable_study_instances(study_id)

    assert result == [{"study_id": study_id, "series_id": series_id,
                       "instance_id": instance_id}]
