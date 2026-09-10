from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.config import Settings
from app.db import create_schema, make_session_factory
from app.services.ai_diagnostics import AIDiagnosticsStore, classify_error, safe_error_message
from app.services.providers import OpenAIMedicalVisionProvider


class _FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text="OK")


class _FakeClient:
    def __init__(self) -> None:
        self.responses = _FakeResponses()


def test_connection_tests_record_safe_success_and_readiness(tmp_path: Path) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'diagnostics.db'}"
    configuration = Settings(database_url=database_url, ai_provider="openai", openai_api_key="test-only", ai_max_retries=0)
    create_schema(database_url)
    store = AIDiagnosticsStore(make_session_factory(database_url))
    provider = OpenAIMedicalVisionProvider(configuration, client=_FakeClient(), diagnostics=store)

    provider.test_text_connection()
    provider.test_vision_connection()
    snapshot = store.snapshot(include_technical=True)

    assert snapshot["api_key_configured"] is True
    assert snapshot["ready_for_patient_processing"] is True
    assert snapshot["technical_details"]["request_count"] == 2
    assert snapshot["technical_details"]["last_exception_type"] is None


def test_error_classification_is_actionable_and_redacted() -> None:
    class AuthenticationError(Exception):
        status_code = 401
        message = "invalid api key sk-proj-THIS_MUST_NOT_BE_STORED"

    code, message = classify_error(AuthenticationError(), stage="text_connection_test")
    assert code == "authentication_failed"
    assert message == "OpenAI authentication failed"
    assert "sk-proj-THIS_MUST_NOT_BE_STORED" not in safe_error_message(AuthenticationError())
