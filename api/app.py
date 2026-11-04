"""Local educational inference and review API for AIML-02.

This service is a coursework demo. It is not a medical device and must not be
used to diagnose, triage, or guide care. Run it on localhost only.
"""

from __future__ import annotations

import json
import hashlib
import base64
from contextlib import asynccontextmanager
import io
import logging
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, StreamingResponse
from torch import nn
from .modeling import ARCHITECTURE_NAMES, build_classifier, gradcam_target, image_to_tensor

DISCLAIMER = (
    "Educational demonstration only. Model scores are not a diagnosis and must "
    "not be used for clinical decisions. A qualified clinician must review all images."
)
LABELS = (
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
)
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_UPLOAD_PIXELS = 40_000_000
MAX_NIFTI_UPLOAD_BYTES = 128 * 1024 * 1024
MAX_NIFTI_VOXELS = 45_000_000
PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def _project_path(variable: str, default: str) -> Path:
    configured = os.environ.get(variable)
    path = Path(configured) if configured else Path(default)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _environment_bool(variable: str, default: bool = False) -> bool:
    value = os.environ.get(variable)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{variable} must be a boolean value.")


CHECKPOINT = _project_path("NEXVIA_CHECKPOINT", "artifacts/best_densenet121.pt")
SEGMENTATION_CHECKPOINT = _project_path(
    "NEXVIA_SEGMENTATION_CHECKPOINT",
    "artifacts/best_spleen_unet_candidate_20261103.pt",
)
SEGMENTATION_THRESHOLD = float(os.environ.get("NEXVIA_SEGMENTATION_THRESHOLD", "0.50"))
if not 0.0 < SEGMENTATION_THRESHOLD < 1.0:
    raise ValueError("NEXVIA_SEGMENTATION_THRESHOLD must be between 0 and 1.")
SEGMENTATION_TTA = _environment_bool("NEXVIA_SEGMENTATION_TTA", False)
SEGMENTATION_DEFAULT_ROI = (64, 64, 64)
AUDIT_PATH = _project_path("NEXVIA_AUDIT_PATH", "demo/review_events.jsonl")
ORTHANC_RESULTS_PATH = Path(
    _project_path("NEXVIA_ORTHANC_RESULTS_PATH", "demo/orthanc_predictions.jsonl")
)
ORTHANC_SYNC_STATE_PATH = Path(
    _project_path("NEXVIA_ORTHANC_SYNC_STATE_PATH", "demo/orthanc_sync_state.json")
)
ORTHANC_SEGMENTATION_RESULTS_PATH = Path(
    _project_path("NEXVIA_ORTHANC_SEGMENTATION_RESULTS_PATH", "demo/orthanc_segmentations.jsonl")
)
ORTHANC_SEGMENTATION_MASK_DIR = _project_path(
    "NEXVIA_ORTHANC_SEGMENTATION_MASK_DIR", "demo/orthanc_masks"
)
_model: nn.Module | None = None
_load_error: str | None = None
_model_metadata: dict[str, Any] = {}
_model_lock = threading.RLock()
_audit_lock = threading.Lock()
_orthanc_results_lock = threading.Lock()
_orthanc_sync_lock = threading.Lock()
_segmentation_lock = threading.Lock()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _segmentation_runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "checkpoint_path": str(SEGMENTATION_CHECKPOINT),
        "checkpoint_sha256": None,
        "roi_size": None,
        "threshold": SEGMENTATION_THRESHOLD,
        "flip_tta": SEGMENTATION_TTA,
        "status": "missing",
    }
    if not SEGMENTATION_CHECKPOINT.is_file():
        return info

    info["status"] = "available"
    info["checkpoint_sha256"] = _sha256_file(SEGMENTATION_CHECKPOINT)
    try:
        checkpoint = torch.load(
            SEGMENTATION_CHECKPOINT, map_location="cpu", weights_only=True
        )
        roi = checkpoint.get("patch_size", SEGMENTATION_DEFAULT_ROI)
        info["roi_size"] = [int(value) for value in roi]
        info["architecture"] = checkpoint.get("architecture")
    except Exception as exc:
        info["metadata_error"] = type(exc).__name__
    return info


@asynccontextmanager
async def _application_lifespan(_: FastAPI):
    log_segmentation_configuration()
    yield


app = FastAPI(
    title="Nexvia AIML-02 Educational Demo",
    version="0.1.0",
    lifespan=_application_lifespan,
)
cors_origins = [
    origin.strip()
    for origin in os.environ.get(
        "NEXVIA_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def log_segmentation_configuration() -> None:
    logger.info(
        "Spleen segmentation configuration: %s",
        json.dumps(_segmentation_runtime_info(), sort_keys=True),
    )


def _load_model() -> nn.Module:
    global _model, _load_error, _model_metadata
    with _model_lock:
        if _model is not None:
            return _model
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT}")
        try:
            checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
            architecture = checkpoint.get("architecture")
            if architecture not in ARCHITECTURE_NAMES:
                raise ValueError("Unsupported checkpoint architecture")
            if tuple(checkpoint.get("labels", ())) != LABELS:
                raise ValueError("Checkpoint label order does not match the API")
            model = build_classifier(
                architecture, len(LABELS), pretrained=False,
                label_adjacency=checkpoint.get("label_adjacency"),
            )
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.eval()
            _model = model
            _model_metadata = {
                "architecture": architecture,
                "model_name": checkpoint.get("model_name", ARCHITECTURE_NAMES[architecture]),
                "best_epoch": checkpoint.get("best_epoch"),
                "validation_metrics": checkpoint.get("validation_metrics"),
                "promotion_evaluation": checkpoint.get("promotion_evaluation"),
                "label_graph_source": checkpoint.get("label_graph_source"),
                "checkpoint_sha256": _sha256_file(CHECKPOINT),
            }
            _load_error = None
            return model
        except Exception as exc:
            _load_error = f"{type(exc).__name__}: {exc}"
            raise


def _image_tensor(image: Image.Image) -> torch.Tensor:
    return image_to_tensor(image).unsqueeze(0)


@app.get("/health")
def health() -> dict[str, Any]:
    available = CHECKPOINT.is_file()
    status = "ready" if _model is not None else ("checkpoint_present" if available else "model_unavailable")
    return {
        "status": status,
        "model_available": available,
        "model_loaded": _model is not None,
        "model": _model_metadata or None,
        "load_error": _load_error,
        "segmentation": _segmentation_runtime_info(),
        "disclaimer": DISCLAIMER,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    return _score_image(await _read_uploaded_image(file))


def _process_spleen_volume(input_path: Path, output_dir: Path) -> dict[str, Any]:
    import nibabel as nib

    source = nib.load(str(input_path))
    shape = tuple(int(size) for size in source.shape)
    if len(shape) != 3 or any(size < 2 for size in shape):
        raise ValueError("Input must be a non-empty 3D NIfTI volume.")
    voxel_count = int(np.prod(shape, dtype=np.int64))
    if voxel_count > MAX_NIFTI_VOXELS:
        raise ValueError("NIfTI volume exceeds the 45 million voxel demo limit.")
    if not np.isfinite(source.affine).all() or abs(float(np.linalg.det(source.affine[:3, :3]))) < 1e-8:
        raise ValueError("NIfTI affine does not define usable voxel geometry.")
    output_dir.mkdir()
    from segment_spleen import run_inference
    with _segmentation_lock:
        result = run_inference(
            input_path,
            SEGMENTATION_CHECKPOINT,
            output_dir,
            SEGMENTATION_THRESHOLD,
            "auto",
            use_flip_tta=SEGMENTATION_TTA,
        )

    mask_img = nib.load(result["mask"])
    mask = np.asanyarray(mask_img.dataobj).copy()
    volume = np.asanyarray(source.dataobj).copy()
    source_orientation = nib.orientations.io_orientation(source.affine)
    ras_orientation = nib.orientations.axcodes2ornt(("R", "A", "S"))
    to_ras = nib.orientations.ornt_transform(source_orientation, ras_orientation)
    volume = nib.orientations.apply_orientation(volume, to_ras)
    mask = nib.orientations.apply_orientation(mask, to_ras)
    plane_axes = {"sagittal": 0, "coronal": 1, "axial": 2}
    previews: dict[str, dict[str, Any]] = {}
    for plane_name, axis in plane_axes.items():
        counts = np.count_nonzero(mask, axis=tuple(d for d in range(3) if d != axis))
        slice_index = int(np.argmax(counts)) if int(counts.max()) else volume.shape[axis] // 2
        image_slice = np.take(volume, slice_index, axis=axis).astype(np.float32)
        mask_slice = np.take(mask, slice_index, axis=axis) > 0
        gray = np.uint8(np.clip((image_slice + 57.0) / (164.0 + 57.0), 0, 1) * 255)
        rgb = np.repeat(gray[..., None], 3, axis=-1)
        rgb[mask_slice] = (0.65 * rgb[mask_slice] + 0.35 * np.array([255, 48, 48])).astype(np.uint8)
        preview = Image.fromarray(rgb).transpose(Image.Transpose.ROTATE_90)
        png = io.BytesIO()
        preview.save(png, format="PNG", optimize=True)
        previews[plane_name] = {
            "png_base64": base64.b64encode(png.getvalue()).decode("ascii"),
            "slice_index": slice_index,
            "axis": axis,
        }
    output_mask_path = Path(result["mask"])
    result.pop("input", None)
    result.pop("mask", None)
    result.update({
        "preview_planes": previews,
        "preview_orientation": "RAS",
        "output_mask_filename": "spleen-mask.nii.gz",
        "output_mask_nii_gz_base64": base64.b64encode(output_mask_path.read_bytes()).decode("ascii"),
        "disclaimer": "Experimental educational output only. No reference mask is available for this case, so its accuracy is unknown. Not for clinical use.",
        "upload_retained": False,
    })
    source.uncache()
    mask_img.uncache()
    del volume, mask, image_slice, mask_slice, previews
    return result


@app.post("/segment/spleen")
def segment_spleen(file: UploadFile = File(...)) -> dict[str, Any]:
    """Run the experimental 3D spleen model on a bounded NIfTI CT upload."""
    filename = (file.filename or "").lower()
    if not filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(status_code=415, detail="Upload a .nii or .nii.gz CT volume.")
    if not SEGMENTATION_CHECKPOINT.is_file():
        raise HTTPException(status_code=503, detail="The spleen segmentation checkpoint is not installed.")
    content = file.file.read(MAX_NIFTI_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded NIfTI file is empty.")
    if len(content) > MAX_NIFTI_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="NIfTI upload exceeds the 128 MiB demo limit.")
    suffix = ".nii.gz" if filename.endswith(".nii.gz") else ".nii"
    with tempfile.TemporaryDirectory(prefix="nexvia-spleen-") as scratch:
        scratch_path = Path(scratch)
        input_path = scratch_path / f"upload{suffix}"
        input_path.write_bytes(content)
        try:
            return _process_spleen_volume(input_path, scratch_path / "output")
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not process this NIfTI volume: {type(exc).__name__}.") from exc


@app.post("/orthanc/series/{series_id}/segment/spleen")
def segment_orthanc_ct_series(series_id: str) -> dict[str, Any]:
    """Fetch a stable single-frame CT series from Orthanc and run the spleen model."""
    valid_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-"
    if not 8 <= len(series_id) <= 80 or any(character not in valid_chars for character in series_id):
        raise HTTPException(status_code=422, detail="Invalid Orthanc series ID")
    if not SEGMENTATION_CHECKPOINT.is_file():
        raise HTTPException(status_code=503, detail="The spleen segmentation checkpoint is not installed.")
    with tempfile.TemporaryDirectory(prefix="nexvia-orthanc-ct-") as scratch:
        scratch_path = Path(scratch)
        input_path = scratch_path / "orthanc-ct.nii.gz"
        try:
            try:
                from .ct_series import fetch_orthanc_ct_series_to_nifti
            except ImportError:
                from ct_series import fetch_orthanc_ct_series_to_nifti
            source = fetch_orthanc_ct_series_to_nifti(series_id, input_path)
            result = _process_spleen_volume(input_path, scratch_path / "output")
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail="Could not retrieve the CT series from Orthanc.") from exc
        except Exception as exc:
            detail = str(exc) if exc.__class__.__name__ == "CTSeriesError" else type(exc).__name__
            raise HTTPException(status_code=422, detail=f"Could not process this Orthanc CT series: {detail}") from exc

        record_id = str(uuid.uuid4())
        try:
            mask_bytes = base64.b64decode(result["output_mask_nii_gz_base64"], validate=True)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=500, detail="Segmentation mask could not be stored.") from exc

        ORTHANC_SEGMENTATION_MASK_DIR.mkdir(parents=True, exist_ok=True)
        mask_filename = f"{record_id}.nii.gz"
        mask_path = ORTHANC_SEGMENTATION_MASK_DIR / mask_filename
        pending_mask_path = ORTHANC_SEGMENTATION_MASK_DIR / f".{record_id}.tmp"
        mask_path_sha256 = hashlib.sha256(mask_bytes).hexdigest()
        try:
            pending_mask_path.write_bytes(mask_bytes)
            os.replace(pending_mask_path, mask_path)
        except OSError as exc:
            pending_mask_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="Segmentation mask could not be stored.") from exc

        mask_url = f"/orthanc/series/{series_id}/segmentation-results/{record_id}/mask"
        runtime = _segmentation_runtime_info()
        result.update({"result_id": record_id, "source": "orthanc_ct_series",
                       "orthanc_series_id": series_id, "conversion": source["conversion"],
                       "dicom_instance_count": source["instance_count"], "stored": True,
                       "mask_sha256": mask_path_sha256, "mask_url": mask_url})
        record = {
            "result_id": record_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source": "orthanc_ct_series",
            "orthanc_series_id": series_id,
            "dicom_instance_count": source["instance_count"],
            "input_shape": source["input_shape"],
            "predicted_spleen_voxels": result["predicted_spleen_voxels"],
            "estimated_predicted_volume_ml": result["estimated_predicted_volume_ml"],
            "mask_filename": mask_filename,
            "mask_sha256": mask_path_sha256,
            "checkpoint_sha256": runtime["checkpoint_sha256"],
            "threshold": SEGMENTATION_THRESHOLD,
            "flip_tta": SEGMENTATION_TTA,
            "disclaimer": result["disclaimer"],
        }
        try:
            ORTHANC_SEGMENTATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _orthanc_results_lock, ORTHANC_SEGMENTATION_RESULTS_PATH.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            mask_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="Segmentation result could not be stored.") from exc
        return result


@app.get("/orthanc/series/{series_id}/segmentation-results")
def list_orthanc_segmentation_results(series_id: str) -> dict[str, Any]:
    """Return saved CT result summaries and links to their local NIfTI masks."""
    valid_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-"
    if not 8 <= len(series_id) <= 80 or any(character not in valid_chars for character in series_id):
        raise HTTPException(status_code=422, detail="Invalid Orthanc series ID")
    if not ORTHANC_SEGMENTATION_RESULTS_PATH.exists():
        return {"orthanc_series_id": series_id, "results": []}
    try:
        with _orthanc_results_lock, ORTHANC_SEGMENTATION_RESULTS_PATH.open("r", encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Could not read stored CT segmentation results.") from exc
    results = []
    for record in records:
        if record.get("orthanc_series_id") != series_id:
            continue
        results.append({
            key: record.get(key)
            for key in (
                "result_id", "timestamp_utc", "dicom_instance_count", "input_shape",
                "predicted_spleen_voxels", "estimated_predicted_volume_ml",
                "mask_sha256", "checkpoint_sha256", "threshold", "flip_tta", "disclaimer",
            )
        } | {"mask_url": f"/orthanc/series/{series_id}/segmentation-results/{record.get('result_id')}/mask"})
    return {"orthanc_series_id": series_id, "results": results}


@app.get("/orthanc/series/{series_id}/segmentation-results/{result_id}/mask")
def download_orthanc_segmentation_mask(series_id: str, result_id: str) -> FileResponse:
    """Download a previously saved CT segmentation mask as NIfTI-GZIP."""
    valid_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-"
    if not 8 <= len(series_id) <= 80 or any(character not in valid_chars for character in series_id):
        raise HTTPException(status_code=422, detail="Invalid Orthanc series ID")
    try:
        normalized_result_id = str(uuid.UUID(result_id))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid segmentation result ID") from exc
    stored = list_orthanc_segmentation_results(series_id)["results"]
    match = next((item for item in stored if item.get("result_id") == normalized_result_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail="Stored segmentation result not found.")
    mask_path = ORTHANC_SEGMENTATION_MASK_DIR / f"{normalized_result_id}.nii.gz"
    if not mask_path.is_file():
        raise HTTPException(status_code=404, detail="Stored segmentation mask not found.")
    actual_sha256 = _sha256_file(mask_path)
    if actual_sha256 != match.get("mask_sha256"):
        raise HTTPException(status_code=500, detail="Stored segmentation mask failed its integrity check.")
    return FileResponse(
        mask_path,
        media_type="application/gzip",
        filename=f"spleen-mask-{normalized_result_id}.nii.gz",
        headers={"X-Content-SHA256": actual_sha256},
    )


@app.post("/explain")
async def explain(file: UploadFile = File(...), label: str = Form(...)) -> StreamingResponse:
    """Return a Grad-CAM overlay for one selected finding in the coursework demo."""
    if label not in LABELS:
        raise HTTPException(status_code=422, detail="Select one of the 14 supported findings.")
    image = await _read_uploaded_image(file)
    try:
        model = _load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="No trained checkpoint is installed.") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not load model: {type(exc).__name__}") from exc

    from io import BytesIO
    import torch.nn.functional as F

    activations: list[torch.Tensor] = []

    def capture_features(_module, _inputs, output):
        output.retain_grad()
        activations.append(output)

    architecture = _model_metadata["architecture"]
    target_layer, channels_last = gradcam_target(model, architecture)
    with _model_lock:
        hook = target_layer.register_forward_hook(capture_features)
        try:
            with torch.enable_grad():
                tensor = _image_tensor(image).requires_grad_(True)
                logits = model(tensor)
                index = LABELS.index(label)
                activation = activations[-1]
                gradient = torch.autograd.grad(logits[0, index], activation)[0]
            if channels_last:
                activation = activation.permute(0, 3, 1, 2)
                gradient = gradient.permute(0, 3, 1, 2)
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            heatmap = F.relu((weights * activation).sum(dim=1, keepdim=True))
            heatmap = F.interpolate(heatmap, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
            heatmap = heatmap.detach().cpu().numpy()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not create attribution overlay: {type(exc).__name__}") from exc
        finally:
            hook.remove()

    peak = float(heatmap.max())
    if peak > 0:
        heatmap /= peak
    original = image.convert("RGB")
    resized_heat = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize(
        original.size, Image.Resampling.BILINEAR)
    heat = np.asarray(resized_heat, dtype=np.float32) / 255.0
    base = np.asarray(original, dtype=np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255.0
    alpha = (heat * 0.4)[..., None]
    overlay = Image.fromarray(np.uint8(np.clip(base * (1 - alpha) + red * alpha, 0, 255)))
    buffer = BytesIO()
    overlay.save(buffer, format="PNG")
    buffer.seek(0)
    score = float(torch.sigmoid(logits[0, index]).detach().cpu())
    return StreamingResponse(buffer, media_type="image/png", headers={
        "Cache-Control": "no-store",
        "X-Attribution-Label": label,
        "X-Model-Score": f"{score:.6f}",
        "X-Model-Name": _model_metadata.get("model_name", "unknown"),
        "X-Attribution-Note": "Coarse feature attribution; not a clinical explanation.",
    })


def _decode_uploaded_image(content: bytes, content_type: str) -> Image.Image:
    if content_type not in {"image/png", "image/jpeg"}:
        raise HTTPException(status_code=415, detail="Upload a PNG or JPEG image.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 12 MiB demo limit.")
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    try:
        from io import BytesIO

        image = Image.open(BytesIO(content))
        if image.width * image.height > MAX_UPLOAD_PIXELS:
            raise HTTPException(status_code=413, detail="Image dimensions exceed the 40 megapixel demo limit.")
        expected_format = {"image/png": "PNG", "image/jpeg": "JPEG"}[content_type]
        if image.format != expected_format:
            raise HTTPException(status_code=415, detail="File content does not match its PNG/JPEG content type.")
        image.verify()
        decoded = Image.open(BytesIO(content))
        decoded.load()
        gray = np.asarray(decoded.convert("L"), dtype=np.float32)
        if float(gray.var()) <= 1.0:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Image has near-constant intensity (grayscale variance <= 1.0); "
                    "upload a non-blank PNG or JPEG."
                ),
            )
        return decoded
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="File is not a readable image.") from exc


async def _read_uploaded_image(file: UploadFile) -> Image.Image:
    if file.content_type not in {"image/png", "image/jpeg"}:
        raise HTTPException(status_code=415, detail="Upload a PNG or JPEG image.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    return _decode_uploaded_image(content, file.content_type)


def _score_image(image: Image.Image) -> dict[str, Any]:
    try:
        model = _load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="No trained checkpoint is installed.") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not load model: {type(exc).__name__}") from exc
    with _model_lock, torch.inference_mode():
        scores = torch.sigmoid(model(_image_tensor(image))).squeeze(0).tolist()
    return {
        "case_ref": str(uuid.uuid4()),
        "model": _model_metadata.get("model_name", "Unknown classifier"),
        "model_metadata": _model_metadata,
        "score_semantics": "Uncalibrated sigmoid scores; not diagnostic probabilities.",
        "findings": [{"label": label, "score": round(float(score), 6)}
                     for label, score in zip(LABELS, scores)],
        "disclaimer": DISCLAIMER,
    }


@app.post("/orthanc/{instance_id}/predict")
def predict_orthanc_instance(instance_id: str) -> dict[str, Any]:
    """Fetch one Orthanc instance, score it, and persist the educational result."""
    try:
        from .orthanc import UnsupportedDicomModality, fetch_orthanc_image
    except ImportError:
        from orthanc import UnsupportedDicomModality, fetch_orthanc_image
    try:
        image = fetch_orthanc_image(instance_id)
    except UnsupportedDicomModality as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not read Orthanc instance: {type(exc).__name__}") from exc
    prediction = _score_image(image)
    record = {
        "result_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "orthanc_instance_id": instance_id,
        **prediction,
    }
    ORTHANC_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _orthanc_results_lock, ORTHANC_RESULTS_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {**prediction, "result_id": record["result_id"], "stored": True}


@app.post("/orthanc/sync")
def sync_orthanc_studies(limit: int = 16) -> dict[str, Any]:
    """Consume stable-study events and process a bounded resumable instance batch."""
    if not 1 <= limit <= 64:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 64")
    if not _orthanc_sync_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="An Orthanc sync is already running")
    try:
        try:
            from .orthanc import (
                UnsupportedDicomModality, fetch_orthanc_image, get_orthanc_changes,
                list_stable_study_instances,
            )
        except ImportError:
            from orthanc import (
                UnsupportedDicomModality, fetch_orthanc_image, get_orthanc_changes,
                list_stable_study_instances,
            )

        ORTHANC_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        ORTHANC_SYNC_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            if ORTHANC_SYNC_STATE_PATH.exists():
                state = json.loads(ORTHANC_SYNC_STATE_PATH.read_text(encoding="utf-8"))
                if (not isinstance(state, dict)
                        or not isinstance(state.get("last_seq"), int)
                        or state["last_seq"] < 0
                        or not isinstance(state.get("pending_studies"), list)
                        or any(not isinstance(item, str) for item in state["pending_studies"])):
                    raise ValueError("Invalid Orthanc sync state")
            else:
                # Cursor zero intentionally backfills the retained Orthanc change log.
                state = {"last_seq": 0, "pending_studies": []}
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(status_code=500, detail="Orthanc sync state is unreadable") from exc

        try:
            changes = get_orthanc_changes(state["last_seq"], 500)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Could not read Orthanc changes: {type(exc).__name__}") from exc

        pending = list(dict.fromkeys(state["pending_studies"]))
        queued = 0
        for change in changes["Changes"]:
            if change.get("ChangeType") != "StableStudy":
                continue
            study_id = change.get("ID")
            if (isinstance(study_id, str)
                    and 8 <= len(study_id) <= 80
                    and all(character.isalnum() or character == "-" for character in study_id)
                    and study_id not in pending):
                pending.append(study_id)
                queued += 1
        state = {"last_seq": changes.get("Last", state["last_seq"]),
                 "pending_studies": pending}

        def save_state() -> None:
            temporary = ORTHANC_SYNC_STATE_PATH.with_suffix(
                ORTHANC_SYNC_STATE_PATH.suffix + ".tmp"
            )
            temporary.write_text(json.dumps(state, separators=(",", ":")), encoding="utf-8")
            temporary.replace(ORTHANC_SYNC_STATE_PATH)

        # Persist the cursor and queued studies before inference so a restart resumes safely.
        save_state()
        with _orthanc_results_lock:
            if ORTHANC_RESULTS_PATH.exists():
                records = [json.loads(line) for line in ORTHANC_RESULTS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
            else:
                records = []
        seen = {item.get("orthanc_instance_id") for item in records}
        processed = skipped = 0
        candidates_seen = 0
        transient_error: str | None = None
        remaining_studies: list[str] = []
        with _orthanc_results_lock, ORTHANC_RESULTS_PATH.open("a", encoding="utf-8") as stream:
            for study_index, study_id in enumerate(pending):
                if processed + skipped >= limit:
                    remaining_studies.extend(pending[study_index:])
                    break
                try:
                    candidates = list_stable_study_instances(study_id)
                except requests.HTTPError as exc:
                    if getattr(exc.response, "status_code", None) == 404:
                        continue
                    transient_error = type(exc).__name__
                    remaining_studies.extend(pending[study_index:])
                    break
                except (requests.RequestException, OSError) as exc:
                    transient_error = type(exc).__name__
                    remaining_studies.extend(pending[study_index:])
                    break
                candidates_seen += len(candidates)
                study_pending = False
                for item in candidates:
                    instance_id = item["instance_id"]
                    if instance_id in seen:
                        continue
                    if processed + skipped >= limit:
                        study_pending = True
                        break
                    record: dict[str, Any] = {
                        "result_id": str(uuid.uuid4()),
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "orthanc_instance_id": instance_id,
                        "orthanc_study_id": item["study_id"],
                        "orthanc_series_id": item["series_id"],
                        "source": "orthanc_stable_study_changes",
                    }
                    try:
                        image = fetch_orthanc_image(instance_id)
                        prediction = _score_image(image)
                        record.update(prediction)
                        record["status"] = "processed"
                        processed += 1
                    except UnsupportedDicomModality as exc:
                        record.update({
                            "status": "skipped",
                            "reason": "unsupported_modality",
                            "modality": exc.modality,
                            "details": str(exc),
                            "disclaimer": DISCLAIMER,
                        })
                        skipped += 1
                    except requests.HTTPError as exc:
                        status = getattr(exc.response, "status_code", None)
                        if status != 404:
                            transient_error = type(exc).__name__
                            study_pending = True
                            break
                        record.update({"status": "skipped", "reason": "HTTPError404",
                                       "disclaimer": DISCLAIMER})
                        skipped += 1
                    except (requests.RequestException, OSError) as exc:
                        transient_error = type(exc).__name__
                        study_pending = True
                        break
                    except HTTPException as exc:
                        if exc.status_code >= 500:
                            transient_error = f"HTTPException{exc.status_code}"
                            study_pending = True
                            break
                        record.update({"status": "skipped", "reason": f"HTTPException{exc.status_code}",
                                       "disclaimer": DISCLAIMER})
                        skipped += 1
                    except Exception as exc:
                        record.update({"status": "skipped", "reason": type(exc).__name__,
                                       "disclaimer": DISCLAIMER})
                        skipped += 1
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    seen.add(instance_id)
                if study_pending:
                    remaining_studies.append(study_id)
                    remaining_studies.extend(pending[study_index + 1:])
                    break

        state["pending_studies"] = list(dict.fromkeys(remaining_studies))
        save_state()
        more_may_remain = bool(state["pending_studies"]) or not bool(changes.get("Done", True))
        return {
            "stable_studies_queued": queued,
            "candidates_seen": candidates_seen,
            "processed": processed,
            "skipped": skipped,
            "pending_studies": len(state["pending_studies"]),
            "last_sequence": state["last_seq"],
            "limit": limit,
            "more_may_remain": more_may_remain,
            "temporary_error": transient_error,
            "results_path": str(ORTHANC_RESULTS_PATH),
        }
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Could not update Orthanc result log") from exc
    finally:
        _orthanc_sync_lock.release()


@app.get("/orthanc/{instance_id}/results")
def list_orthanc_results(instance_id: str) -> dict[str, Any]:
    """Return previously stored demo predictions for one Orthanc instance."""
    valid_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-"
    if not 8 <= len(instance_id) <= 80 or any(character not in valid_chars for character in instance_id):
        raise HTTPException(status_code=422, detail="Invalid Orthanc instance ID")
    if not ORTHANC_RESULTS_PATH.exists():
        return {"orthanc_instance_id": instance_id, "results": []}
    try:
        with _orthanc_results_lock, ORTHANC_RESULTS_PATH.open("r", encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Could not read stored Orthanc results") from exc
    results = [record for record in records if record.get("orthanc_instance_id") == instance_id]
    return {"orthanc_instance_id": instance_id, "results": results}


class ReviewEvent(BaseModel):
    case_ref: str = Field(min_length=1, max_length=80)
    reviewed_findings: list[str] = Field(default_factory=list, max_length=14)
    note: str = Field(default="", max_length=500)
    reviewer_action: str = Field(pattern="^(accepted_for_demo|needs_review|rejected)$")


@app.post("/reviews", status_code=201)
def record_review(event: ReviewEvent) -> dict[str, str]:
    unknown = set(event.reviewed_findings) - set(LABELS)
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown finding labels: {sorted(unknown)}")
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "event_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **event.model_dump(),
        "disclaimer": "Demo review annotation only; not clinical sign-off.",
    }
    with _audit_lock, AUDIT_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"event_id": record["event_id"], "status": "recorded"}

