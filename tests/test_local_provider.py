from __future__ import annotations

import json as jsonlib
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from app.ai_schemas import PatientBundleItem, PatientLevelReview
from app.config import Settings
from app.db import create_schema, make_session_factory
from app.pipeline import PatientPipeline
from app.repository import RecordRepository
from app.services.ai_diagnostics import AIDiagnosticsStore
from app.services.providers import LocalAIProviderConfigurationError, LocalMedicalVisionProvider, PatientLevelAIError, provider_for_settings
from tests.helpers import text_pdf
from tests.test_patient_level_review import GoldenPatientProvider


class _Response:
    def __init__(self, payload: dict):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _FakeOllama:
    def __init__(self, review: PatientLevelReview | None = None):
        self.review = review or PatientLevelReview()
        self.posts: list[dict] = []
        self.gets: list[str] = []

    def post(self, path: str, *, json: dict, timeout=None):
        self.posts.append(json)
        return _Response({"response": jsonlib.dumps(self.review.model_dump()) if json.get("format") else "OK"})

    def get(self, path: str, *, timeout=None):
        self.gets.append(path)
        if path == "/api/tags":
            return _Response({"models": [{"name": "local-test-vision:latest"}]})
        return _Response({"models": [{"name": "local-test-vision:latest", "size_vram": 123456789}]})


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite+pysqlite:///{tmp_path / 'local.db'}",
        ai_provider="local",
        local_ai_base_url="http://ollama:11434",
        local_ai_model="local-test-vision",
        local_ocr_enabled=False,
        ai_max_retries=0,
    )


def test_local_health_and_synthetic_vision_use_ollama_only(tmp_path: Path) -> None:
    fake = _FakeOllama()
    provider = LocalMedicalVisionProvider(_settings(tmp_path), http_client=fake)

    health = provider.health()
    result = provider.test_vision_connection()

    assert health["status"] == "healthy"
    assert health["gpu_status"] == "gpu"
    assert health["gpu_memory_bytes"] == 123456789
    assert result["provider"] == "local"
    assert fake.gets == ["/api/tags", "/api/ps"]
    assert fake.posts[0]["model"] == "local-test-vision"
    assert fake.posts[0]["images"]


def test_local_patient_review_validates_canonical_schema_and_multi_image_input(tmp_path: Path) -> None:
    expected = PatientLevelReview(
        patient={"sex": "Female", "source_files": ["source_document_1"], "source_pages": [1]},
        diagnoses_documented=[{"diagnosis": "Bronchial asthma", "certainty": "confirmed", "documented_vs_inferred": "documented"}],
        medications=[{"name_original": "Ventolin", "dose": None, "needs_verification": True}],
        laboratory_results=[{"test_name": "Hb", "value": 10.2, "value_text": "10.2 g/dL"}],
        growth_measurements=[{"weight_kg": 22.0, "certainty": "documented"}],
        uncertain_items=[{"field": "medications.dose", "reason": "Dose is illegible; clinician verification required."}],
        erp_summary_markdown="Bronchial asthma documented; medication dose requires verification.",
    )
    fake = _FakeOllama(expected)
    image_one = tmp_path / "one.png"
    image_two = tmp_path / "two.png"
    Image.new("RGB", (20, 20), "white").save(image_one)
    Image.new("RGB", (20, 20), "gray").save(image_two)
    bundle = [
        PatientBundleItem(source_id="s1", source_file="one.png", relative_path="one.png", page_number=1, order_index=0, media_type="image/png", local_path=str(image_one)),
        PatientBundleItem(source_id="s2", source_file="two.png", relative_path="two.png", page_number=1, order_index=1, media_type="image/png", local_path=str(image_two)),
    ]
    provider = LocalMedicalVisionProvider(_settings(tmp_path), http_client=fake)

    actual = provider.review_patient(bundle)

    assert actual.diagnoses_documented == expected.diagnoses_documented
    assert actual.medications[0]["dose"] is None
    assert actual.uncertain_items[0]["reason"].startswith("Dose is illegible")
    assert actual.erp_summary_markdown == expected.erp_summary_markdown
    assert len(fake.posts[0]["images"]) == 2


def test_local_provider_does_not_fallback_to_cloud(tmp_path: Path) -> None:
    class FailingOllama(_FakeOllama):
        def post(self, path: str, *, json: dict, timeout=None):
            raise OSError("Ollama is unavailable")

    provider = LocalMedicalVisionProvider(_settings(tmp_path), http_client=FailingOllama())

    with pytest.raises(PatientLevelAIError, match="Local AI request failed"):
        provider.test_vision_connection()


def test_local_provider_rejects_unpreparable_image_input(tmp_path: Path) -> None:
    provider = LocalMedicalVisionProvider(_settings(tmp_path), http_client=_FakeOllama())

    with pytest.raises(ValueError, match="image input could not be prepared"):
        provider._to_ollama_input(
            [{"role": "user", "content": [{"type": "input_image", "image_url": "not-a-data-url"}]}],
            None,
        )


def test_local_metrics_are_redacted_and_persisted(tmp_path: Path) -> None:
    configuration = _settings(tmp_path)
    create_schema(configuration.database_url)
    store = AIDiagnosticsStore(make_session_factory(configuration.database_url))
    provider = LocalMedicalVisionProvider(configuration, http_client=_FakeOllama(), diagnostics=store)

    store.record_request(provider="local", model=configuration.local_ai_model, stage="page_extraction", retry_count=0, latency_ms=12.5, success=True)
    store.record_request(provider="local", model=configuration.local_ai_model, stage="patient_review", retry_count=1, latency_ms=456.7, success=True)
    store.record_local_runtime(ollama_status="healthy", gpu_status="gpu", gpu_memory_bytes=9876543210)
    snapshot = store.snapshot()

    assert snapshot["page_latency_ms"] == 12.5
    assert snapshot["patient_latency_ms"] == 456.7
    assert snapshot["gpu_memory_bytes"] == 9876543210
    assert snapshot["ollama_status"] == "healthy"


def test_provider_factory_selects_local_without_an_api_key(tmp_path: Path) -> None:
    provider = provider_for_settings(_settings(tmp_path), diagnostics=None)
    assert isinstance(provider, LocalMedicalVisionProvider)


def test_local_provider_rejects_public_or_credentialed_endpoints(tmp_path: Path) -> None:
    configuration = _settings(tmp_path)
    with pytest.raises(LocalAIProviderConfigurationError, match="local/private"):
        LocalMedicalVisionProvider(replace(configuration, local_ai_base_url="https://example.com/ollama"))
    with pytest.raises(LocalAIProviderConfigurationError, match="credentials"):
        LocalMedicalVisionProvider(replace(configuration, local_ai_base_url="http://user:pass@localhost:11434"))


def test_local_provider_runs_the_cloud_golden_patient_contract(tmp_path: Path) -> None:
    """The local adapter must preserve the physician-reviewed golden case."""

    patient = tmp_path / "GOLDEN_PATIENT"
    patient.mkdir()
    text_pdf(patient / "01_visit.pdf", ["Date: 2025-01-10", "Hb: 10.2 g/dL"])
    text_pdf(patient / "02_followup.pdf", ["Date: 2026-01-12", "Hb: 10.8 g/dL"])
    golden_bundle = [
        PatientBundleItem(source_id="s1", source_file="01_visit.pdf", relative_path="01_visit.pdf", page_number=1, order_index=0, media_type="application/pdf", local_path=str(patient / "01_visit.pdf")),
        PatientBundleItem(source_id="s2", source_file="02_followup.pdf", relative_path="02_followup.pdf", page_number=1, order_index=1, media_type="application/pdf", local_path=str(patient / "02_followup.pdf")),
    ]
    expected = GoldenPatientProvider().review_patient(golden_bundle)
    config = Settings(
        database_url=f"sqlite+pysqlite:///{tmp_path / 'golden-local.db'}",
        storage_root=tmp_path / "data",
        output_root=tmp_path / "output",
        ai_provider="local",
        ai_extraction_mode="patient_level",
        local_ai_model="local-test-vision",
        local_ai_base_url="http://ollama:11434",
        ai_max_retries=0,
    )
    create_schema(config.database_url)
    repository = RecordRepository(make_session_factory(config.database_url), storage_root=config.storage_root, output_root=config.output_root)
    diagnostics = AIDiagnosticsStore(repository.session_factory)
    diagnostics.record_test_success(stage="text_connection_test", model=config.local_ai_model)
    diagnostics.record_test_success(stage="vision_connection_test", model=config.local_ai_model)
    fake = _FakeOllama(expected)
    provider = LocalMedicalVisionProvider(config, http_client=fake, diagnostics=diagnostics)

    record = PatientPipeline(repository, config, provider=provider).process(patient).record

    assert record.patient.full_name.value == "Amina Hassan"
    assert record.diagnoses[0].term_original == "Bronchial asthma"
    assert [lab.value for lab in record.laboratory_results] == [10.2, 10.8]
    assert record.medications[0].dose is None and record.medications[0].verification_required is True
    assert "not biochemically confirmed" in record.ai_clinical_review["clinical_interpretations"][0]["value"]
    assert "bronchial asthma" in record.ai_clinical_review["erp_summary_markdown"].lower()
    assert record.verification_queue
    assert len(fake.posts[0]["images"]) == 2
