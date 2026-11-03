import json

import pytest

from train_segmentation import frozen_split, make_model, read_cases


def make_dataset(tmp_path, count=10):
    root = tmp_path / "Task09_Spleen"
    images, labels = root / "imagesTr", root / "labelsTr"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    rows = []
    for index in range(count):
        name = f"spleen_{index:02}.nii.gz"
        (images / name).write_bytes(b"fixture")
        (labels / name).write_bytes(b"fixture")
        rows.append({"image": f"imagesTr/{name}", "label": f"labelsTr/{name}"})
    dataset_json = root / "dataset.json"
    dataset_json.write_text(json.dumps({"name": "Task09_Spleen", "training": rows}), encoding="utf-8")
    return root, dataset_json


def test_split_is_frozen_case_level_and_deterministic(tmp_path):
    root, dataset_json = make_dataset(tmp_path)
    cases = read_cases(dataset_json, root)
    split_path = tmp_path / "split.json"
    first = frozen_split(cases, seed=42, path=split_path)
    second = frozen_split(cases, seed=99, path=split_path)

    assert first == second
    assert set(first["train"]).isdisjoint(first["validation"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["validation"]).isdisjoint(first["test"])
    assert sum(len(first[key]) for key in ("train", "validation", "test")) == 10


def test_split_rejects_dataset_roster_change(tmp_path):
    root, dataset_json = make_dataset(tmp_path, count=8)
    cases = read_cases(dataset_json, root)
    split_path = tmp_path / "split.json"
    frozen_split(cases, seed=42, path=split_path)
    with pytest.raises(ValueError, match="does not match"):
        frozen_split(cases[:-1], seed=42, path=split_path)


def test_case_path_escape_is_rejected(tmp_path):
    root, dataset_json = make_dataset(tmp_path, count=4)
    source = json.loads(dataset_json.read_text(encoding="utf-8"))
    source["training"][0]["image"] = "../outside.nii.gz"
    dataset_json.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Unsafe or missing"):
        read_cases(dataset_json, root)


@pytest.mark.parametrize("architecture", ["monai_unet_3d", "monai_segresnet_3d"])
def test_segmentation_architectures_return_binary_logits(architecture):
    import torch

    model = make_model(architecture).eval()
    with torch.inference_mode():
        output = model(torch.zeros((1, 1, 32, 32, 32)))
    assert output.shape == (1, 2, 32, 32, 32)
    assert torch.isfinite(output).all()
