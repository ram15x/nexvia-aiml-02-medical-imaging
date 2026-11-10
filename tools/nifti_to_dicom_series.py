"""Write a NIfTI CT volume as a single-frame DICOM CT series (one file per axial slice).

Used to build a realistic test series for the Orthanc live check from a public MSD volume.
Geometry follows DICOM LPS conventions (NIfTI is RAS, so x and y are negated).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid


def nifti_to_dicom_series(nifti_path: Path, out_dir: Path, *, patient_id: str = "TEST-CT") -> int:
    img = nib.load(str(nifti_path))
    data = np.asarray(img.dataobj)
    aff = img.affine
    out_dir.mkdir(parents=True, exist_ok=True)
    ras_i, ras_j = aff[:3, 0], aff[:3, 1]
    sp_i, sp_j = float(np.linalg.norm(ras_i)), float(np.linalg.norm(ras_j))
    lps = np.array([-1.0, -1.0, 1.0])
    iop = list((ras_i / sp_i * lps)) + list((ras_j / sp_j * lps))
    study_uid, series_uid = generate_uid(), generate_uid()
    for k in range(data.shape[2]):
        pos = (aff @ np.array([0, 0, k, 1.0]))[:3] * lps
        sop = generate_uid()
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = sop
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = CTImageStorage, sop
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
        ds.PatientID, ds.PatientName = patient_id, "Synthetic^Test"
        ds.Modality = "CT"
        ds.InstanceNumber = k + 1
        px = np.clip(np.rint(data[:, :, k].T), -32768, 32767).astype(np.int16)
        ds.Rows, ds.Columns = px.shape
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit, ds.PixelRepresentation = 15, 1
        ds.ImageOrientationPatient = [float(v) for v in iop]
        ds.ImagePositionPatient = [float(v) for v in pos]
        ds.PixelSpacing = [sp_j, sp_i]
        ds.RescaleSlope, ds.RescaleIntercept = 1, 0
        ds.PixelData = px.tobytes()
        ds.save_as(out_dir / f"slice_{k + 1:04d}.dcm", enforce_file_format=True)
    return data.shape[2]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("nifti", type=Path); ap.add_argument("out_dir", type=Path)
    a = ap.parse_args()
    print(nifti_to_dicom_series(a.nifti, a.out_dir), "slices written")
