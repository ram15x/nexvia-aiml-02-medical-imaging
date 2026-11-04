"""Bounded conversion of a stable, single-frame CT DICOM series to NIfTI.

The DICOM patient coordinate geometry is preserved in the NIfTI affine. This
is an educational adapter for the existing NIfTI MONAI inference path; it does
not establish that the spleen model generalizes to arbitrary CT examinations.
"""

from __future__ import annotations

from io import BytesIO
import re
from pathlib import Path

import numpy as np
import requests

MAX_SERIES_INSTANCES = 512
MAX_SERIES_BYTES = 128 * 1024 * 1024
MAX_INSTANCE_BYTES = 32 * 1024 * 1024
MAX_SERIES_VOXELS = 45_000_000


class CTSeriesError(ValueError):
    """The DICOM series cannot be safely mapped to one regular CT volume."""


def _orthanc_client() -> tuple[str, tuple[str, str]]:
    import os

    base = os.environ.get("ORTHANC_URL", "http://orthanc:8042").rstrip("/")
    username = os.environ.get("ORTHANC_USERNAME", "orthanc")
    password = os.environ.get("ORTHANC_PASSWORD")
    if not password:
        raise RuntimeError("Set ORTHANC_PASSWORD before enabling the Orthanc demo")
    return base, (username, password)


def _read_instance(instance_id: str, base: str, auth: tuple[str, str], remaining: int):
    if not re.fullmatch(r"[A-Za-z0-9-]{8,80}", instance_id):
        raise CTSeriesError("Orthanc returned an invalid instance ID")
    response = requests.get(
        f"{base}/instances/{instance_id}/file", auth=auth,
        timeout=(5, 30), stream=True,
    )
    payload = bytearray()
    try:
        response.raise_for_status()
        header = response.headers.get("Content-Length")
        if header is not None and int(header) > min(MAX_INSTANCE_BYTES, remaining):
            raise CTSeriesError("CT series exceeds the bounded DICOM transfer limit")
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            if len(payload) + len(chunk) > min(MAX_INSTANCE_BYTES, remaining):
                raise CTSeriesError("CT series exceeds the bounded DICOM transfer limit")
            payload.extend(chunk)
    finally:
        response.close()
    import pydicom

    try:
        return pydicom.dcmread(BytesIO(payload)), len(payload)
    except Exception as exc:
        raise CTSeriesError("An Orthanc instance is not readable DICOM") from exc


def _series_volume(datasets: list) -> tuple[np.ndarray, np.ndarray]:
    """Return float32 HU volume in (column,row,slice) order and RAS affine."""
    if len(datasets) < 2:
        raise CTSeriesError("CT segmentation requires at least two slices in one series")
    first = datasets[0]
    try:
        reference_orientation = np.asarray(first.ImageOrientationPatient, dtype=np.float64)
        reference_spacing = np.asarray(first.PixelSpacing, dtype=np.float64)
        rows, columns = int(first.Rows), int(first.Columns)
    except (AttributeError, TypeError, ValueError) as exc:
        raise CTSeriesError("CT series is missing required image-plane geometry") from exc
    if reference_orientation.shape != (6,) or reference_spacing.shape != (2,):
        raise CTSeriesError("CT series has invalid image-plane geometry")
    if not np.isfinite(reference_orientation).all() or not np.isfinite(reference_spacing).all():
        raise CTSeriesError("CT series geometry must be finite")
    if rows <= 0 or columns <= 0 or reference_spacing.min() <= 0:
        raise CTSeriesError("CT series dimensions or pixel spacing are invalid")
    if len(datasets) * rows * columns > MAX_SERIES_VOXELS:
        raise CTSeriesError("CT series exceeds the 45 million voxel demo limit")

    row_direction = reference_orientation[:3]
    column_direction = reference_orientation[3:]
    if (not np.isclose(np.linalg.norm(row_direction), 1.0, atol=1e-3)
            or not np.isclose(np.linalg.norm(column_direction), 1.0, atol=1e-3)
            or not np.isclose(float(row_direction @ column_direction), 0.0, atol=1e-3)):
        raise CTSeriesError("CT series orientation vectors are not orthonormal")
    normal = np.cross(row_direction, column_direction)
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm < 0.999 or norm > 1.001:
        raise CTSeriesError("CT series orientation vectors are not orthonormal")
    normal /= norm

    positions = []
    for dataset in datasets:
        if str(getattr(dataset, "Modality", "")).upper() != "CT":
            raise CTSeriesError("Every instance in the selected series must be CT")
        if int(getattr(dataset, "NumberOfFrames", 1)) != 1:
            raise CTSeriesError("Enhanced/multiframe CT is not supported by this demo")
        if str(getattr(dataset, "SeriesInstanceUID", "")) != str(getattr(first, "SeriesInstanceUID", "")):
            raise CTSeriesError("Selected instances do not belong to one DICOM series")
        if int(getattr(dataset, "Rows", 0)) != rows or int(getattr(dataset, "Columns", 0)) != columns:
            raise CTSeriesError("CT slices have inconsistent matrix dimensions")
        if not np.allclose(np.asarray(dataset.ImageOrientationPatient, dtype=np.float64),
                           reference_orientation, rtol=0, atol=1e-4):
            raise CTSeriesError("CT slices have inconsistent orientation")
        if not np.allclose(np.asarray(dataset.PixelSpacing, dtype=np.float64),
                           reference_spacing, rtol=0, atol=1e-4):
            raise CTSeriesError("CT slices have inconsistent pixel spacing")
        try:
            position = np.asarray(dataset.ImagePositionPatient, dtype=np.float64)
        except (AttributeError, TypeError, ValueError) as exc:
            raise CTSeriesError("A CT slice is missing ImagePositionPatient") from exc
        if position.shape != (3,) or not np.isfinite(position).all():
            raise CTSeriesError("A CT slice has invalid ImagePositionPatient")
        positions.append(position)

    order = np.argsort([float(position @ normal) for position in positions])
    datasets = [datasets[int(index)] for index in order]
    positions_array = np.asarray([positions[int(index)] for index in order])
    projected = positions_array @ normal
    offsets = np.diff(projected)
    if np.any(offsets <= 1e-3):
        raise CTSeriesError("CT slices have duplicate or reversed physical positions")
    slice_spacing = float(np.median(offsets))
    if not np.allclose(offsets, slice_spacing, rtol=0.05, atol=0.1):
        raise CTSeriesError("Irregularly spaced CT slices cannot be safely stacked by this demo")
    # Reject gantry-tilt/shear rather than silently flattening slice positions.
    expected_steps = np.outer(projected - projected[0], normal)
    actual_steps = positions_array - positions_array[0]
    if not np.allclose(actual_steps, expected_steps, rtol=0, atol=0.5):
        raise CTSeriesError("Sheared/tilted CT slice geometry is unsupported by this adapter")

    try:
        from pydicom.pixels import apply_modality_lut
        slices = [np.asarray(apply_modality_lut(ds.pixel_array, ds), dtype=np.float32)
                  for ds in datasets]
    except Exception as exc:
        raise CTSeriesError("CT pixel data could not be decoded/rescaled") from exc
    volume = np.stack(slices, axis=0).transpose(2, 1, 0)
    # DICOM uses LPS patient coordinates; NIfTI's affine is written in RAS.
    lps_to_ras = np.diag([-1.0, -1.0, 1.0])
    affine = np.eye(4, dtype=np.float64)
    affine[:3, 0] = lps_to_ras @ row_direction * reference_spacing[1]
    affine[:3, 1] = lps_to_ras @ column_direction * reference_spacing[0]
    affine[:3, 2] = lps_to_ras @ normal * slice_spacing
    affine[:3, 3] = lps_to_ras @ positions_array[0]
    if not np.isfinite(affine).all() or abs(float(np.linalg.det(affine[:3, :3]))) < 1e-8:
        raise CTSeriesError("CT series does not define valid 3D patient geometry")
    return volume, affine


def fetch_orthanc_ct_series_to_nifti(
    series_id: str, output_path: Path, expected_instances: int | None = None
) -> dict:
    """Fetch one stable CT series and write a geometry-preserving NIfTI volume."""
    if not re.fullmatch(r"[A-Za-z0-9-]{8,80}", series_id):
        raise CTSeriesError("Invalid Orthanc series ID")
    base, auth = _orthanc_client()
    response = requests.get(f"{base}/series/{series_id}", auth=auth, timeout=(5, 20))
    response.raise_for_status()
    info = response.json()
    if not isinstance(info, dict) or not isinstance(info.get("Instances"), list):
        raise CTSeriesError("Orthanc returned an invalid series record")
    instances = info["Instances"]
    if not 2 <= len(instances) <= MAX_SERIES_INSTANCES:
        raise CTSeriesError("CT series must contain 2 to 512 single-frame instances")
    if expected_instances is not None:
        if isinstance(expected_instances, bool) or not isinstance(expected_instances, int) or not 2 <= expected_instances <= MAX_SERIES_INSTANCES:
            raise CTSeriesError("Expected instance count must be an integer between 2 and 512")
        if len(instances) != expected_instances:
            raise CTSeriesError("Orthanc instance count does not match the supplied expected count")
    status = info.get("Status")
    if status == "Complete":
        completeness_basis = "orthanc_complete"
    elif status == "Unknown" and expected_instances is not None:
        completeness_basis = "operator_supplied_count_match"
    else:
        display_status = status if isinstance(status, str) and status else "missing"
        raise CTSeriesError(
            f"Orthanc series is not confirmed complete (status: {display_status})"
        )
    study_id = info.get("ParentStudy")
    if not isinstance(study_id, str):
        raise CTSeriesError("Orthanc series has no parent study")
    study_response = requests.get(f"{base}/studies/{study_id}", auth=auth, timeout=(5, 20))
    study_response.raise_for_status()
    study = study_response.json()
    if not isinstance(study, dict) or study.get("IsStable") is not True:
        raise CTSeriesError("Wait until Orthanc marks the study stable before inference")

    datasets = []
    bytes_read = 0
    for instance_id in instances:
        dataset, transferred_bytes = _read_instance(
            instance_id, base, auth, MAX_SERIES_BYTES - bytes_read
        )
        bytes_read += transferred_bytes
        if bytes_read > MAX_SERIES_BYTES:
            raise CTSeriesError("CT series exceeds the 128 MiB demo payload limit")
        datasets.append(dataset)
    if completeness_basis == "operator_supplied_count_match":
        latest_response = requests.get(f"{base}/series/{series_id}", auth=auth, timeout=(5, 20))
        latest_response.raise_for_status()
        latest = latest_response.json()
        if (not isinstance(latest, dict) or not isinstance(latest.get("Instances"), list)
                or latest.get("Status") not in {"Unknown", "Complete"}
                or set(latest["Instances"]) != set(instances)):
            raise CTSeriesError("Orthanc series changed during retrieval; retry after it is stable")
        stable_response = requests.get(f"{base}/studies/{study_id}", auth=auth, timeout=(5, 20))
        stable_response.raise_for_status()
        latest_study = stable_response.json()
        if not isinstance(latest_study, dict) or latest_study.get("IsStable") is not True:
            raise CTSeriesError("Orthanc study changed during retrieval; retry after it is stable")
    volume, affine = _series_volume(datasets)
    try:
        import nibabel as nib
        nib.save(nib.Nifti1Image(volume, affine), str(output_path))
        if output_path.stat().st_size > MAX_SERIES_BYTES:
            output_path.unlink(missing_ok=True)
            raise CTSeriesError("Converted NIfTI exceeds the 128 MiB temporary-volume limit")
        check = nib.load(str(output_path))
        if check.shape != volume.shape or not np.allclose(check.affine, affine, rtol=0, atol=1e-5):
            raise CTSeriesError("Converted NIfTI geometry did not round-trip")
    except CTSeriesError:
        raise
    except Exception as exc:
        raise CTSeriesError("CT series could not be written as a NIfTI volume") from exc
    return {
        "orthanc_series_id": series_id,
        "instance_count": len(datasets),
        "orthanc_series_status": status,
        "completeness_basis": completeness_basis,
        "expected_instances": expected_instances,
        "input_shape": list(volume.shape),
        "affine": affine.tolist(),
        "voxel_count": int(volume.size),
        "transfer_bytes": bytes_read,
        "conversion": "DICOM LPS geometry mapped to NIfTI RAS; modality LUT applied",
    }
