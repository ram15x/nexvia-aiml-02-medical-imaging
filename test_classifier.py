import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from api.modeling import build_classifier, gradcam_target
from data_audit import NIH_LABELS
from evaluate_classifier import classify_test_scope
from scripts.analyze_validation_threshold_sensitivity import select_f1_threshold
from train_classifier import (NIHChestXrayDataset, evaluate, labels_to_vector,
                              MaskedNegativeBCE, checkpoint_selection_score,
                              positive_class_weights)


def test_labels_to_vector_handles_no_finding_and_multilabel():
    assert labels_to_vector("No Finding").sum() == 0
    vector = labels_to_vector("Mass|Effusion")
    assert vector[NIH_LABELS.index("Mass")] == 1
    assert vector[NIH_LABELS.index("Effusion")] == 1
    assert vector.sum() == 2


def test_positive_class_weight_modes_keep_default_and_cap_full_ratio():
    targets = np.zeros((2, len(NIH_LABELS)), dtype=np.float32)
    targets[0, 0] = 1
    sqrt_weights = positive_class_weights(targets, exponent=0.5)
    full_weights = positive_class_weights(targets, exponent=1.0)
    assert sqrt_weights[0] == pytest.approx(1.0)
    assert sqrt_weights[1] == pytest.approx(np.sqrt(3.0))
    assert full_weights[1] == pytest.approx(3.0)
    assert np.all(positive_class_weights(np.zeros((101, len(NIH_LABELS))), 1.0) <= 10.0)


def test_checkpoint_selection_uses_declared_primary_metric():
    metrics = {
        "macro_auprc_defined_labels": 0.17,
        "macro_auroc_defined_labels": 0.78,
    }
    assert checkpoint_selection_score(metrics, "macro_auprc") == pytest.approx(0.17)
    assert checkpoint_selection_score(metrics, "macro_auroc") == pytest.approx(0.78)
    assert checkpoint_selection_score({"macro_auprc_defined_labels": None}, "macro_auprc") is None
    with pytest.raises(ValueError, match="Unsupported"):
        checkpoint_selection_score(metrics, "accuracy")


def test_masked_bce_uses_positive_and_no_finding_terms_only():
    targets = torch.zeros((2, len(NIH_LABELS)))
    targets[0, NIH_LABELS.index("Mass")] = 1.0
    logits = torch.zeros_like(targets, requires_grad=True)
    criterion = MaskedNegativeBCE(torch.ones(len(NIH_LABELS)))

    loss = criterion(logits, targets)
    loss.backward()

    mass = NIH_LABELS.index("Mass")
    unmentioned = NIH_LABELS.index("Edema")
    assert torch.isfinite(loss)
    assert logits.grad[0, mass] != 0
    assert logits.grad[0, unmentioned] == 0
    assert logits.grad[1, unmentioned] != 0  # No Finding rows remain negative evidence.


def test_evaluator_marks_fully_present_subset_manifest_as_subset():
    assert classify_test_scope(1492, 0) == ("official held-out test subset", False)
    assert classify_test_scope(25_596, 0) == ("official held-out test", True)
    assert classify_test_scope(25_596, 24_629) == ("official held-out test subset", False)


def test_dataset_reads_only_selected_split_and_normalizes_pixels(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.fromarray(np.full((16, 16), 128, dtype=np.uint8)).save(image_root / "p1_000.png")
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(image_root / "p2_000.png")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"Image Index": "p1_000.png", "Finding Labels": "Mass", "Patient ID": "p1", "split": "train"},
        {"Image Index": "p2_000.png", "Finding Labels": "No Finding", "Patient ID": "p2", "split": "validation"},
    ]).to_csv(manifest, index=False)

    dataset = NIHChestXrayDataset(manifest, image_root, "train", None, seed=4, image_size=32)
    image, target = dataset[0]
    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert target.shape == (14,)
    assert target[NIH_LABELS.index("Mass")] == 1


def test_dataset_supports_resolution_screen_size(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.fromarray(np.full((16, 24), 100, dtype=np.uint8)).save(image_root / "p1_000.png")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"Image Index": "p1_000.png", "Finding Labels": "Mass", "Patient ID": "p1", "split": "train"},
    ]).to_csv(manifest, index=False)

    dataset = NIHChestXrayDataset(manifest, image_root, "train", None, seed=4, image_size=320)
    image, _ = dataset[0]
    assert image.shape == (3, 320, 320)
    assert torch.isfinite(image).all()


def test_light_affine_is_train_only_and_keeps_dataset_shapes(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    pixels = np.zeros((32, 32), dtype=np.uint8)
    pixels[8:24, 12:20] = 180
    Image.fromarray(pixels).save(image_root / "p1_000.png")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"Image Index": "p1_000.png", "Finding Labels": "Mass", "Patient ID": "p1", "split": "train"},
        {"Image Index": "p1_000.png", "Finding Labels": "Mass", "Patient ID": "p1", "split": "validation"},
    ]).to_csv(manifest, index=False)

    baseline_validation = NIHChestXrayDataset(
        manifest, image_root, "validation", None, seed=4, image_size=32,
    )
    affine_validation = NIHChestXrayDataset(
        manifest, image_root, "validation", None, seed=4, image_size=32,
        augmentation_policy="light_affine",
    )
    train = NIHChestXrayDataset(
        manifest, image_root, "train", None, seed=4, image_size=32,
        augmentation_policy="light_affine",
    )

    validation_image, validation_target = affine_validation[0]
    baseline_image, _ = baseline_validation[0]
    train_image, train_target = train[0]
    assert torch.equal(validation_image, baseline_image)
    assert train_image.shape == baseline_image.shape == (3, 32, 32)
    assert torch.isfinite(train_image).all()
    assert torch.equal(train_target, validation_target)


def test_dataset_rejects_path_escape(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"Image Index": "../outside.png", "Finding Labels": "No Finding", "Patient ID": "p1", "split": "train"},
    ]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="Unsafe"):
        NIHChestXrayDataset(manifest, image_root, "train", None, seed=4)


def test_partial_dataset_requires_explicit_opt_in(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(image_root / "p1_000.png")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"Image Index": "p1_000.png", "Finding Labels": "No Finding", "Patient ID": "p1", "split": "train"},
        {"Image Index": "p2_000.png", "Finding Labels": "Mass", "Patient ID": "p2", "split": "train"},
    ]).to_csv(manifest, index=False)

    with pytest.raises(FileNotFoundError, match="Use --allow-available-subset"):
        NIHChestXrayDataset(manifest, image_root, "train", None, seed=4)
    dataset = NIHChestXrayDataset(manifest, image_root, "train", None, seed=4,
                                  allow_available_subset=True)
    assert len(dataset) == 1
    assert dataset.available_images == 1
    assert dataset.missing_images == 1


def test_evaluation_reports_auc_ap_and_confusion_counts():
    model = nn.Linear(2, len(NIH_LABELS))
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    targets = torch.zeros((2, len(NIH_LABELS)))
    targets[0, 0] = 1
    loader = DataLoader(TensorDataset(torch.zeros((2, 2)), targets), batch_size=2)
    metrics, y_true, scores = evaluate(model, loader, torch.device("cpu"))

    assert y_true.shape == scores.shape == (2, len(NIH_LABELS))
    assert metrics["per_label"][NIH_LABELS[0]]["confusion_at_0_5"] == {
        "true_positive": 1, "false_positive": 1, "false_negative": 0, "true_negative": 0,
    }
    assert metrics["per_label"][NIH_LABELS[0]]["auroc"] == 0.5


def test_validation_threshold_search_finds_perfect_cutoff_on_separable_scores():
    selected = select_f1_threshold(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.1, 0.4, 0.6, 0.9]),
    )
    assert selected["threshold"] == pytest.approx(0.6)
    assert selected["precision"] == pytest.approx(1.0)
    assert selected["recall"] == pytest.approx(1.0)
    assert selected["f1"] == pytest.approx(1.0)


def test_evidence_concentration_head_is_finite_and_gradcam_compatible():
    model = build_classifier(
        "densenet121_evidence_concentration_gate",
        len(NIH_LABELS),
        pretrained=False,
    )
    model.eval()
    logits = model(torch.zeros((1, 3, 224, 224)))

    assert logits.shape == (1, len(NIH_LABELS))
    assert torch.isfinite(logits).all()
    target_layer, channels_last = gradcam_target(
        model, "densenet121_evidence_concentration_gate"
    )
    assert target_layer is model.encoder.features
    assert channels_last is False


def test_efficientnet_b0_head_and_gradcam_target_are_configured():
    model = build_classifier("efficientnet_b0", len(NIH_LABELS), pretrained=False)
    model.eval()
    with torch.inference_mode():
        logits = model(torch.zeros((1, 3, 224, 224)))
    assert logits.shape == (1, len(NIH_LABELS))
    assert torch.isfinite(logits).all()
    target_layer, channels_last = gradcam_target(model, "efficientnet_b0")
    assert target_layer is model.features[-1]
    assert channels_last is False
