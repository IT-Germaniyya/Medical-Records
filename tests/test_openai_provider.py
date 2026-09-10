from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.ai_schemas import MedicalPageExtraction
from app.config import Settings
from app.services.extraction import extract_documented_facts
from app.services.providers import OpenAIMedicalVisionProvider


def _payload(confidence: float = 0.96) -> str:
    return json.dumps({
        "transcription": "Diagnosis: Asthma",
        "document_type": "outpatient_note",
        "complexity": "simple_printed",
        "encounter_date": None,
        "patient_identity": {"patient_name": None, "patient_id": None, "hospital_file_number": None, "date_of_birth": None, "sex": None, "nationality": None, "address": None},
        "diagnoses": [{"value": "Asthma", "normalized_value": "Asthma", "confidence": confidence, "certainty": "confirmed", "documented_vs_inferred": "documented", "source_page": 1, "source_file": "note.jpg", "needs_verification": False, "status": "active"}],
        "medications": [], "laboratory_results": [], "radiology": [], "growth_measurements": [], "uncertain_items": [],
        "overall_confidence": confidence,
    })


class _FakeClient:
    def __init__(self, payloads: list[str]):
        self.payloads = payloads
        self.calls: list[dict] = []
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.payloads.pop(0))


def test_openai_provider_uses_strict_responses_and_escalates(tmp_path: Path) -> None:
    image = tmp_path / "note.jpg"
    image.write_bytes(b"not-a-real-image")
    client = _FakeClient([_payload(0.40), _payload(0.98)])
    configuration = Settings(
        ai_provider="openai", openai_api_key="test-only", openai_fast_model="fast", openai_medical_model="medical",
        fast_model_min_confidence=0.80, ai_max_retries=0, openai_prompt_root=Path("prompts/openai"),
    )
    provider = OpenAIMedicalVisionProvider(configuration, client=client)
    provider._page_input = lambda document, page_number: (None, None, "plain printed text")  # type: ignore[method-assign]
    result = provider.extract_page(image, 1, source_filename="note.jpg")
    assert result.structured is not None
    assert result.model == "medical"
    assert result.retried_with_strong_model is True
    assert [call["model"] for call in client.calls] == ["fast", "medical"]
    assert client.calls[0]["text"]["format"]["strict"] is True
    assert client.calls[0]["store"] is False


def test_ai_page_maps_only_typed_documented_facts() -> None:
    page = MedicalPageExtraction.model_validate(json.loads(_payload()))
    extracted = extract_documented_facts("P1", page.transcription, "note.jpg", 1, "outpatient_note", ai_page=page, model="medical", prompt_version="medical_page_v1")
    assert extracted.diagnoses[0].term_original == "Asthma"
    assert extracted.diagnoses[0].source_ref.extraction_method == "openai_responses"
    assert extracted.audit[0].model == "medical"
