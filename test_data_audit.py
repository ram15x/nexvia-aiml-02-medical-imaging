import json

import pandas as pd
import pytest

from data_audit import build_msd_manifest, build_nih_manifest, patient_split


def test_patient_split_is_stable_and_disjoint_by_patient(tmp_path):
    metadata = pd.DataFrame([
        {"Image Index": f"{patient}_{index:03}.png", "Finding Labels": label, "Patient ID": patient}
        for patient, label in [("p1", "No Finding"), ("p2", "Mass|Effusion"), ("p3", "No Finding")]
        for index in range(2)
    ])
    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    trainval, test = tmp_path / "trainval.txt", tmp_path / "test.txt"
    trainval.write_text("\n".join(metadata["Image Index"][:4]), encoding="utf-8")
    test.write_text("\n".join(metadata["Image Index"][4:]), encoding="utf-8")
    result = build_nih_manifest(metadata_path, trainval, test, tmp_path, tmp_path / "out")
    written = pd.read_csv(tmp_path / "out" / "nih_manifest.csv")
    assert result["images"] == 6
    assert written.groupby("Patient ID").split.nunique().max() == 1
    assert result["positive_label_image_counts"]["Mass"] == 2
    assert patient_split("patient-5") == patient_split("patient-5")


def test_nih_rejects_cross_split_patient_leakage(tmp_path):
    metadata = pd.DataFrame([
        {"Image Index": "p1_000.png", "Finding Labels": "No Finding", "Patient ID": "p1"},
        {"Image Index": "p1_001.png", "Finding Labels": "No Finding", "Patient ID": "p1"},
    ])
    meta = tmp_path / "metadata.csv"
    metadata.to_csv(meta, index=False)
    trainval, test = tmp_path / "trainval.txt", tmp_path / "test.txt"
    trainval.write_text("p1_000.png", encoding="utf-8")
    test.write_text("p1_001.png", encoding="utf-8")
    with pytest.raises(ValueError, match="Patient leakage"):
        build_nih_manifest(meta, trainval, test, tmp_path, tmp_path / "out")


def test_msd_validates_pairing_and_safe_paths(tmp_path):
    dataset = {"name": "Task09_Spleen", "training": [
        {"image": "imagesTr/spleen_1.nii.gz", "label": "labelsTr/spleen_1.nii.gz"}
    ]}
    source = tmp_path / "dataset.json"
    source.write_text(json.dumps(dataset), encoding="utf-8")
    assert build_msd_manifest(source, tmp_path, tmp_path / "out")["training_cases"] == 1
    dataset["training"][0]["image"] = "../outside.nii.gz"
    source.write_text(json.dumps(dataset), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsafe"):
        build_msd_manifest(source, tmp_path, tmp_path / "out")
