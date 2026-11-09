# DICOM

## What the project does with DICOM

- **Upload:** `POST /predict` accepts a single-frame chest radiograph in DICOM. Only modality CR or DX, MONOCHROME1 or MONOCHROME2, and one frame are accepted. Other modalities return HTTP 422.
- **Orthanc:** the connector reads stable studies and CR/DX instances, and processes stable CT series through a separate route.
- **Conversion:** `api/orthanc.py` applies the rescale slope and intercept, the stored window if present, and otherwise a fixed full range for 8-bit data or a percentile range for higher bit depths. MONOCHROME1 is inverted.
- **Geometry (CT):** `api/ct_series.py` checks spacing, orientation and instance count before writing NIfTI. It rejects series it cannot map safely.

## What we tested, and how

| Test | Input | Result |
|---|---|---|
| Round trip, generated image | 256×256 random grayscale PNG converted to DX DICOM | Pixels identical after decode. Tests: `test_dicom_upload.py`. |
| Upload scoring, generated image | same image as PNG and as DICOM | Scores identical to 1e-5. |
| Rejection | DICOM with modality CT sent to `/predict` | HTTP 422. |
| Real radiographs | three NIH ChestX-ray14 images from the sample, converted to DX DICOM | Scores identical to 0.0 difference, all 14 labels, all three images. Test: `test_real_dicom_sample.py`. |
| Orthanc CT, public series | one de-identified 81-slice TCIA CT series, run through Orthanc | Reported as processed in an earlier run on the developer's Docker, with an operator-supplied instance count because Orthanc reported the series status as Unknown. This repository does not contain that run's output, so it is not re-verified here. |

The DX conversion tool, `tools/png_to_dicom_dx.py`, writes anonymised test objects. It sets the patient name to `ANONYMISED` and carries only the sample identifier as patient ID.

## Limits

- Real patient DICOM from a hospital was not available. The public inputs are PNG (NIH) and NIfTI (MSD). DICOM here means files we generated from public images, plus one public CT series.
- Orthanc completion status for CT is not independently verified. See `limits_and_bias.md`.
