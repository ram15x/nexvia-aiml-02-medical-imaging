"""Build the inputs for tools/live_orthanc_check.py.

Writes, under --out: dx/*.dcm (NIH chest radiographs as DICOM DX), <ct-name>/*.dcm (an MSD spleen
volume as a DICOM CT series) and manifest.json with the scores and spleen voxel count that the
same models give on the original PNG / NIfTI files, computed here with the API code.

    python tools/make_live_fixtures.py --nih-images <folder with the PNGs> --spleen-nifti <spleen_9.nii.gz> --out demo/live_fixtures
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.nifti_to_dicom_series import nifti_to_dicom_series  # noqa: E402
from tools.png_to_dicom_dx import png_to_dicom  # noqa: E402

DEFAULT_IMAGES = ["00028173_004.png", "00028173_005.png", "00028173_009.png"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nih-images", type=Path, required=True)
    ap.add_argument("--spleen-nifti", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--images", nargs="+", default=DEFAULT_IMAGES)
    a = ap.parse_args()
    from fastapi.testclient import TestClient
    import api.app as app_module
    client = TestClient(app_module.app)
    manifest: dict = {"dx": {}, "ct": {}}
    (a.out / "dx").mkdir(parents=True, exist_ok=True)
    for name in a.images:
        dcm = png_to_dicom(a.nih_images / name, a.out / "dx" / name.replace(".png", ".dcm"), patient_id=name.split("_")[0])
        with open(dcm, "rb") as fh:
            r = client.post("/predict", files={"file": (dcm.name, fh, "application/dicom")}).json()
        manifest["dx"][dcm.name] = {f["label"]: f["score"] for f in r["findings"]}
    ct_name = "ct_" + a.spleen_nifti.name.split(".")[0]
    n = nifti_to_dicom_series(a.spleen_nifti, a.out / ct_name)
    with open(a.spleen_nifti, "rb") as fh:
        r = client.post("/segment/spleen", files={"file": (a.spleen_nifti.name, fh, "application/gzip")}).json()
    manifest["ct"] = {"series_file": ct_name, "instances": n,
                      "expected_predicted_spleen_voxels": r["predicted_spleen_voxels"],
                      "expected_volume_ml": r["estimated_predicted_volume_ml"]}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("wrote", a.out, manifest["ct"])


if __name__ == "__main__":
    main()
