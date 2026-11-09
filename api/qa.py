"""Question answering over one stored model result, through OpenRouter.

The model sees only the structured output already produced for a case: finding
scores, the threshold, the model metadata and, if requested, the top SHAP cells.
It never receives the image or any patient identifier. The OpenRouter key is read
from the OPENROUTER_API_KEY environment variable on the server and is never
returned to the browser, logged or written to the repository.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections import OrderedDict
from typing import Any

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openrouter/auto"
TIMEOUT_SECONDS = 45
MAX_QUESTION_CHARS = 500
AUDIENCES = {"patient", "doctor", "clinician"}
MAX_CASES = 256

SAFETY_FOOTER = (
    "Educational demonstration only. This answer restates model output and is not a diagnosis, "
    "treatment advice or a substitute for a qualified clinician. Discuss any concern with a "
    "doctor."
)

URGENT_PATTERN = re.compile(
    r"chest pain|can'?t breathe|cannot breathe|can not breathe|short(ness)? of breath|"
    r"coughing (up )?blood|blue lips|faint|fainting|unconscious|severe|emergency|urgent|"
    r"\bdying\b|\bsuicid|\bself[- ]harm",
    re.IGNORECASE,
)
URGENT_ANSWER = (
    "Your question mentions symptoms that can need urgent care. This demo cannot assess them. "
    "If you have chest pain, trouble breathing, fainting, or another emergency, call your local "
    "emergency number now or go to the nearest emergency department. For any other questions "
    "about this result, speak with a doctor."
)

SYSTEM_PROMPT = """You answer questions about the output of an educational chest X-ray classifier.
Rules you must follow:
- Use only the JSON case data in the user message. If the data does not answer the question, say so.
- Explain what the scores are: uncalibrated sigmoid outputs from 0 to 1 for 14 finding labels, not probabilities of disease, and not calibrated to any population.
- Never state or imply a diagnosis, a prognosis, or whether the person has or does not have a condition.
- Never recommend medicines, doses, procedures or treatment changes.
- For questions about the SHAP cells, describe them only as image regions that moved the model's score. Do not name anatomy, and do not claim a region is the cause of a finding.
- Say that the model was not validated for individual decisions and that a qualified clinician must review the image.
- Keep the answer under 250 words.
Audience style:
- patient: plain language, no jargon, short sentences, and encourage a conversation with their doctor.
- doctor or clinician: concise technical language; cite the numeric scores and the limitations from model_metadata when relevant."""


class CaseStore:
    """Small in-memory store of recent results, keyed by case_ref."""

    def __init__(self, limit: int = MAX_CASES) -> None:
        self._cases: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._limit = limit
        self._lock = threading.Lock()

    def put(self, case_ref: str, record: dict[str, Any]) -> None:
        with self._lock:
            self._cases[case_ref] = record
            self._cases.move_to_end(case_ref)
            while len(self._cases) > self._limit:
                self._cases.popitem(last=False)

    def update(self, case_ref: str, **fields: Any) -> bool:
        with self._lock:
            if case_ref not in self._cases:
                return False
            self._cases[case_ref].update(fields)
            return True

    def get(self, case_ref: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._cases.get(case_ref)
            return dict(record) if record else None


def configured_model() -> str:
    return (os.environ.get("OPENROUTER_MODEL") or DEFAULT_MODEL).strip()


def is_configured() -> bool:
    return bool((os.environ.get("OPENROUTER_API_KEY") or "").strip())


def is_urgent(question: str) -> bool:
    return bool(URGENT_PATTERN.search(question))


def build_case_context(record: dict[str, Any]) -> dict[str, Any]:
    """Return only the fields the model may use. No image, no identifiers."""
    context: dict[str, Any] = {
        "model": record.get("model"),
        "score_semantics": record.get("score_semantics"),
        "decision_threshold_for_reference": 0.5,
        "findings": record.get("findings", []),
        "model_metadata": {
            key: value for key, value in (record.get("model_metadata") or {}).items()
            if key in {"model_name", "validation_metrics", "promotion_evaluation"}
        },
    }
    attribution = record.get("attribution")
    if attribution:
        context["shap_attribution"] = {
            "label": attribution["label"],
            "method": attribution["method"],
            "baseline": attribution["baseline"],
            "prediction": attribution["prediction"],
            "top_cells": attribution["top_cells"],
            "note": "Values are contributions to this one finding's score relative to a neutral grey baseline.",
        }
    return context


def build_messages(context: dict[str, Any], question: str, audience: str) -> list[dict[str, str]]:
    user = (
        f"Audience: {audience}\n"
        f"Case data (JSON):\n{json.dumps(context, indent=2)}\n\n"
        f"Question: {question}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def ask_openrouter(messages: list[dict[str, str]], api_key: str, model: str) -> str:
    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "Nexvia AIML-02 educational demo",
        },
        json={"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 450},
        timeout=TIMEOUT_SECONDS,
    )
    if response.status_code >= 400:
        # Log the upstream status and a short, key-free error excerpt to diagnose failures.
        print(f"OpenRouter HTTP {response.status_code}: {response.text[:300]}", flush=True)
    response.raise_for_status()
    body = response.json()
    return body["choices"][0]["message"]["content"].strip()


def answer(record: dict[str, Any], question: str, audience: str, api_key: str, model: str) -> str:
    """Return a guarded answer. Urgent-symptom questions never reach the model."""
    if is_urgent(question):
        return URGENT_ANSWER
    context = build_case_context(record)
    text = ask_openrouter(build_messages(context, question, audience), api_key, model)
    return f"{text}\n\n{SAFETY_FOOTER}"
