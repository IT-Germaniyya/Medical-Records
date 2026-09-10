from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from app.ai_schemas import PatientLevelReview
from app.config import Settings
from app.services.providers import OpenRouterMedicalVisionProvider, provider_for_settings


class _FakeChatCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeChatCompletions()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()


class _ReviewClient(_FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.chat.completions.create = self._create_review

    def _create_review(self, **kwargs):
        self.chat.completions.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(PatientLevelReview().model_dump())))])


def _configuration() -> Settings:
    return Settings(
        ai_provider="openrouter",
        openrouter_api_key="test-only",
        openrouter_model="provider/test-model",
        openrouter_base_url="https://openrouter.example/v1",
        ai_max_retries=0,
    )


def test_openrouter_uses_chat_completions_for_text_and_image() -> None:
    client = _FakeClient()
    provider = OpenRouterMedicalVisionProvider(_configuration(), client=client)
    assert provider.openrouter_base_url == "https://openrouter.example/v1"

    provider.test_text_connection()
    provider.test_vision_connection()

    calls = client.chat.completions.calls
    assert len(calls) == 2
    assert all(call["model"] == "provider/test-model" for call in calls)
    assert all("messages" in call for call in calls)
    vision_parts = calls[1]["messages"][0]["content"]
    assert any(part["type"] == "image_url" for part in vision_parts)
    assert "responses" not in calls[0]


def test_openrouter_translates_structured_output_schema() -> None:
    client = _FakeClient()
    provider = OpenRouterMedicalVisionProvider(_configuration(), client=client)

    provider._create_response(
        model="provider/test-model",
        input=[{"role": "user", "content": [{"type": "input_text", "text": "Return JSON"}]}],
        text={"format": {"type": "json_schema", "name": "test_schema", "strict": True, "schema": {"type": "object"}}},
        store=False,
    )

    request = client.chat.completions.calls[0]
    assert request["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "test_schema", "strict": True, "schema": {"type": "object"}},
    }


def test_openrouter_patient_review_uses_canonical_schema() -> None:
    client = _ReviewClient()
    provider = OpenRouterMedicalVisionProvider(_configuration(), client=client)

    review = provider.review_deidentified_record({"patient": {"patient_name": "REDACTED"}, "diagnoses": []})

    assert isinstance(review, PatientLevelReview)
    assert client.chat.completions.calls[0]["model"] == "provider/test-model"
    assert client.chat.completions.calls[0]["response_format"]["json_schema"]["name"] == "patient_level_medical_review"


def test_provider_factory_selects_openrouter(monkeypatch) -> None:
    fake_client = _FakeClient()
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: fake_client))
    provider = provider_for_settings(_configuration(), diagnostics=None)
    assert isinstance(provider, OpenRouterMedicalVisionProvider)
    assert provider.configured_model == "provider/test-model"
