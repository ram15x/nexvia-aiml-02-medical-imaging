"""List bounded Orthanc study instances and render supported images for the demo."""

from __future__ import annotations

import os
import re
from io import BytesIO

import numpy as np
import requests
from PIL import Image

MAX_DICOM_BYTES = 64 * 1024 * 1024
MAX_DICOM_PIXELS = 40_000_000


class UnsupportedDicomModality(ValueError):
    """Raised when a DICOM object does not match the chest-radiograph model."""

    def __init__(self, modality: str | None):
        self.modality = modality or "missing"
        super().__init__(f"Chest classifier accepts CR/DX radiographs; received {self.modality}.")


def _client() -> tuple[str, tuple[str, str]]:
    base = os.environ.get("ORTHANC_URL", "http://orthanc:8042").rstrip("/")
    username = os.environ.get("ORTHANC_USERNAME", "orthanc")
    password = os.environ.get("ORTHANC_PASSWORD")
    if not password:
        raise RuntimeError("Set ORTHANC_PASSWORD before enabling the Orthanc demo")
    return base, (username, password)


def get_orthanc_changes(since: int, limit: int) -> dict:
    """Read one page of Orthanc's persistent changes log."""
    if since < 0 or limit < 1:
        raise ValueError("since must be non-negative and limit must be positive")
    base, auth = _client()
    response = requests.get(
        f"{base}/changes", params={"since": since, "limit": limit},
        auth=auth, timeout=(5, 20),
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("Changes"), list):
        raise ValueError("Unexpected Orthanc changes response")
    last = payload.get("Last", since)
    if not isinstance(last, int) or last < since:
        raise ValueError("Orthanc returned an invalid changes cursor")
    for change in payload["Changes"]:
        if not isinstance(change, dict):
            raise ValueError("Orthanc returned an invalid change record")
        sequence = change.get("Seq")
        if not isinstance(sequence, int) or sequence <= since or sequence > last:
            raise ValueError("Orthanc returned an invalid change sequence")
    return payload


def list_stable_study_instances(study_id: str) -> list[dict[str, str]]:
    """List the instances in one completed/stable Orthanc study."""
    if not re.fullmatch(r"[A-Za-z0-9-]{8,80}", study_id):
        raise ValueError("Invalid Orthanc study ID")
    base, auth = _client()
    response = requests.get(
        f"{base}/studies/{study_id}", auth=auth, timeout=(5, 20)
    )
    response.raise_for_status()
    detail = response.json()
    if not isinstance(detail, dict):
        raise ValueError("Unexpected Orthanc study response")
    if detail.get("IsStable") is False:
        return []
    found: list[dict[str, str]] = []
    for series_id in detail.get("Series", []):
        if not isinstance(series_id, str) or not re.fullmatch(r"[A-Za-z0-9-]{8,80}", series_id):
            continue
        series_response = requests.get(
            f"{base}/series/{series_id}", auth=auth, timeout=(5, 20)
        )
        series_response.raise_for_status()
        series = series_response.json()
        if not isinstance(series, dict):
            raise ValueError("Unexpected Orthanc series response")
        for instance_id in series.get("Instances", []):
            if isinstance(instance_id, str) and re.fullmatch(r"[A-Za-z0-9-]{8,80}", instance_id):
                found.append({"study_id": study_id, "series_id": series_id,
                              "instance_id": instance_id})
    return found


def fetch_orthanc_image(instance_id: str) -> Image.Image:
    if not re.fullmatch(r"[A-Za-z0-9-]{8,80}", instance_id):
        raise ValueError("Invalid Orthanc instance ID")
    base, auth = _client()
    response = requests.get(
        f"{base}/instances/{instance_id}/file",
        auth=auth, timeout=(5, 30), stream=True,
    )
    chunks = bytearray()
    try:
        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > MAX_DICOM_BYTES:
            raise ValueError("DICOM instance exceeds the 64 MiB demo limit")
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            if len(chunks) + len(chunk) > MAX_DICOM_BYTES:
                raise ValueError("DICOM instance exceeds the 64 MiB demo limit")
            chunks.extend(chunk)
    finally:
        response.close()
    import pydicom

    dataset = pydicom.dcmread(BytesIO(chunks))
    modality = str(getattr(dataset, "Modality", "")).strip().upper()
    if modality not in {"CR", "DX"}:
        raise UnsupportedDicomModality(modality or None)
    if int(getattr(dataset, "NumberOfFrames", 1)) != 1:
        raise ValueError("Only single-frame grayscale DICOM images are supported")
    if int(getattr(dataset, "SamplesPerPixel", 1)) != 1:
        raise ValueError("Only single-frame grayscale DICOM images are supported")
    rows = int(getattr(dataset, "Rows", 0))
    columns = int(getattr(dataset, "Columns", 0))
    if rows <= 0 or columns <= 0 or rows * columns > MAX_DICOM_PIXELS:
        raise ValueError("DICOM image dimensions are missing or exceed the 40 megapixel demo limit")
    pixels = np.asarray(dataset.pixel_array)
    if pixels.ndim != 2:
        raise ValueError("Only single-frame grayscale DICOM images are supported")
    photometric = str(getattr(dataset, "PhotometricInterpretation", "")).upper()
    if photometric not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError("Only MONOCHROME1 or MONOCHROME2 radiographs are supported")
    pixels = pixels.astype(np.float32)
    pixels = pixels * float(getattr(dataset, "RescaleSlope", 1))
    pixels = pixels + float(getattr(dataset, "RescaleIntercept", 0))
    center = getattr(dataset, "WindowCenter", None)
    width = getattr(dataset, "WindowWidth", None)
    if isinstance(center, (list, tuple)):
        center = center[0]
    if isinstance(width, (list, tuple)):
        width = width[0]
    if center is not None and width is not None and float(width) > 1:
        low = float(center) - float(width) / 2
        high = float(center) + float(width) / 2
    else:
        low, high = np.percentile(pixels, [0.5, 99.5]).tolist()
        if high <= low:
            raise ValueError("DICOM pixel values have no usable intensity range")
    pixels = np.clip((pixels - low) / (high - low), 0, 1)
    if photometric == "MONOCHROME1":
        pixels = 1 - pixels
    gray = np.uint8(np.round(pixels * 255))
    return Image.fromarray(gray).convert("RGB")
