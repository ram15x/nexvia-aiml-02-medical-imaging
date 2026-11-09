"""Tests for SHAP attribution and the guarded OpenRouter question endpoint."""

from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient
from PIL import Image

from api import app as api_module
from api import qa
from api.app import CASES, CHECKPOINT, LABELS, app

ROOT = Path(__file__).resolve().parent
EXAMPLE = ROOT / "demo" / "chest_xray_example.png"
client = TestClient(app)


def _stored_case(case_ref: str = "case-test-1") -> str:
    CASES.put(case_ref, {
        "model": "DenseNet-121 (test)",
        "model_metadata": {"model_name": "DenseNet-121 (test)", "validation_metrics": None,
                           "promotion_evaluation": None, "checkpoint_sha256": "not-used"},
        "score_semantics": "Uncalibrated sigmoid scores; not diagnostic probabilities.",
        "findings": [{"label": LABELS[0], "score": 0.42}],
    })
    return case_ref


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._payload


def _fake_openrouter(captured, text="The score is 0.42 on a 0 to 1 scale."):
    def post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        return _FakeResponse({"choices": [{"message": {"content": text}}]})
    return post


@pytest.mark.skipif(not CHECKPOINT.is_file() or not EXAMPLE.is_file(),
                    reason="needs the served checkpoint and the demo image")
def test_shap_values_are_additive_and_repeatable():
    from api.shap_explain import shap_attribution

    model = api_module._load_model()
    image = Image.open(EXAMPLE).convert("RGB")
    first = shap_attribution(model, image, LABELS.index("Effusion"), nsamples=200, seed=0)
    second = shap_attribution(model, image, LABELS.index("Effusion"), nsamples=200, seed=0)

    assert first["grid"] == 8
    assert len(first["values"]) == 8 and all(len(row) == 8 for row in first["values"])
    assert first["additivity_error"] < 1e-3
    assert first["values"] == second["values"]
    assert 0.0 <= first["prediction"] <= 1.0
    assert first["top_cells"][0]["region"].startswith(("upper", "middle", "lower"))


def test_overlay_is_png_and_matches_grid_shape():
    import numpy as np
    from api.shap_explain import overlay_png

    values = [[0.0] * 8 for _ in range(8)]
    values[0][0] = 0.2
    values[7][7] = -0.1
    png = overlay_png(np.full((224, 224, 3), 120, dtype=np.uint8), values)
    assert png.startswith(b"\x89PNG")


def test_qa_without_key_is_unavailable(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    case_ref = _stored_case("case-no-key")
    response = client.post("/qa", json={"case_ref": case_ref, "question": "What does 0.42 mean?"})
    assert response.status_code == 503


def test_qa_urgent_symptom_never_reaches_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-urgent")

    def fail(*_args, **_kwargs):
        raise AssertionError("OpenRouter must not be called for urgent questions")

    monkeypatch.setattr(qa.requests, "post", fail)
    case_ref = _stored_case("case-urgent")
    response = client.post("/qa", json={"case_ref": case_ref,
                                        "question": "I have chest pain and trouble breathing"})
    body = response.json()
    assert response.status_code == 200
    assert body["urgent"] is True
    assert "emergency" in body["answer"].lower()


def test_qa_sends_guarded_prompt_and_hides_key(monkeypatch):
    key = "sk-test-do-not-leak-123"
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    monkeypatch.setenv("OPENROUTER_MODEL", "example/test-model")
    captured = {}
    monkeypatch.setattr(qa.requests, "post", _fake_openrouter(captured))
    case_ref = _stored_case("case-guarded")

    response = client.post("/qa", json={"case_ref": case_ref, "question": "What does this score mean?",
                                        "audience": "patient"})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "example/test-model"
    assert body["answer"].endswith(qa.SAFETY_FOOTER)
    assert key not in response.text

    assert captured["url"] == qa.OPENROUTER_URL
    assert captured["headers"]["Authorization"] == f"Bearer {key}"
    messages = captured["json"]["messages"]
    assert messages[0]["role"] == "system" and "Never state or imply a diagnosis" in messages[0]["content"]
    user_text = messages[1]["content"]
    assert "Audience: patient" in user_text
    assert "What does this score mean?" in user_text
    assert "image" not in user_text.lower().replace("image regions", "")
    assert captured["timeout"] == qa.TIMEOUT_SECONDS


def test_qa_upstream_failure_is_reported_without_key(monkeypatch):
    key = "sk-test-upstream-456"
    monkeypatch.setenv("OPENROUTER_API_KEY", key)

    def down(*_args, **_kwargs):
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(qa.requests, "post", down)
    case_ref = _stored_case("case-upstream")
    response = client.post("/qa", json={"case_ref": case_ref, "question": "Explain the result"})
    assert response.status_code == 502
    assert key not in response.text


def test_qa_unknown_result_and_bad_audience(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-404")
    missing = client.post("/qa", json={"case_ref": "does-not-exist", "question": "Explain it"})
    assert missing.status_code == 404
    bad_audience = client.post("/qa", json={"case_ref": _stored_case("case-aud"),
                                            "question": "Explain it", "audience": "lawyer"})
    assert bad_audience.status_code == 422


def test_qa_rejects_overlong_question(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-long")
    response = client.post("/qa", json={"case_ref": _stored_case("case-long"),
                                        "question": "x" * (qa.MAX_QUESTION_CHARS + 1)})
    assert response.status_code == 422


def test_health_reports_configuration_without_key(monkeypatch):
    key = "sk-test-health-789"
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["question_answering"]["configured"] is True
    assert key not in response.text


def test_case_store_is_bounded():
    store = qa.CaseStore(limit=2)
    for i in range(3):
        store.put(f"c{i}", {"n": i})
    assert store.get("c0") is None
    assert store.get("c2") == {"n": 2}
