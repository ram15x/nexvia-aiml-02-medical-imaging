"""Convert one grayscale chest-radiograph PNG into a single-frame DICOM DX object.

Used to create DICOM test inputs from the public NIH sample. The output carries
no patient name; the patient ID is taken from the caller's (already de-identified)
sample identifier. Pixel values are preserved as 8-bit MONOCHROME2.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

DX_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.1.1"  # Digital X-Ray Image Storage - For Presentation


def png_to_dicom(png_path: Path, out_path: Path, patient_id: str, study_date: dt.date | None = None) -> Path:
    gray = np.asarray(Image.open(png_path).convert("L"), dtype=np.uint8)
    study_date = study_date or dt.date(2026, 11, 1)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = DX_SOP_CLASS
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = DX_SOP_CLASS
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "DX"
    ds.PatientID = patient_id
    ds.PatientName = "ANONYMISED"
    ds.StudyDate = study_date.strftime("%Y%m%d")
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyDescription = "Educational test conversion"
    ds.ViewPosition = "PA"
    ds.ImageType = ["ORIGINAL", "PRIMARY"]
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.Rows, ds.Columns = gray.shape
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = gray.tobytes()
    pydicom.dcmwrite(out_path, ds, enforce_file_format=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("png", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--patient-id", required=True)
    args = parser.parse_args()
    png_to_dicom(args.png, args.out, args.patient_id)
    print(args.out)


if __name__ == "__main__":
    main()
