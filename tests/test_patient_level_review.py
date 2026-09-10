"""Golden contract test for the holistic patient-level review path.

The provider is mocked so the test is deterministic and never sends PHI or
requests to OpenAI.  The expected assertions mirror a physician-reviewed
golden case: demographics, documented diagnosis, longitudinal CBC pattern,
medication uncertainty, and the ERP summary.
"""
from pathlib import Path

from app.ai_schemas import PatientLevelReview
from app.config import Settings
from app.db import create_schema, make_session_factory
from app.pipeline import PatientPipeline
from app.repository import RecordRepository
from tests.helpers import text_pdf


class GoldenPatientProvider:
    def review_patient(self, bundle):
        assert [item.relative_path for item in bundle] == ["01_visit.pdf", "02_followup.pdf"]
        return PatientLevelReview.model_validate({
            "patient": {"patient_name": "Amina Hassan", "mrn": "MRN-77", "date_of_birth": "2018-02-03", "sex": "Female", "source_files": ["01_visit.pdf"], "source_pages": [1], "confidence": 0.99},
            "record_quality": {"value": "Two dated visits; handwriting readable", "confidence": 0.95},
            "birth_history": {}, "past_medical_history": [],
            "active_problems": [{"problem": "Persistent mild anemia", "status": "active", "source_files": ["02_followup.pdf"], "source_pages": [1], "confidence": 0.94}],
            "resolved_or_historical_problems": [], "encounters": [], "symptoms": [], "clinical_findings": [],
            "diagnoses_documented": [{"diagnosis": "Bronchial asthma", "type": "documented", "status": "active", "certainty": "confirmed", "source_files": ["01_visit.pdf"], "source_pages": [1], "confidence": 0.97}],
            "clinical_interpretations": [{"value": "Persistent mild anemia with an initially microcytic pattern; iron deficiency is not biochemically confirmed.", "source_files": ["01_visit.pdf", "02_followup.pdf"], "source_pages": [1], "confidence": 0.88}],
            "medications": [{"name_original": "Ventolin", "dose": None, "verification_required": True, "source_files": ["01_visit.pdf"], "source_pages": [1], "confidence": 0.91}],
            "allergies": [], "laboratory_results": [
                {"test_name": "Hb", "value": 10.2, "value_text": "10.2 g/dL", "unit": "g/dL", "date": "2025-01-10", "source_files": ["01_visit.pdf"], "source_pages": [1], "confidence": 0.98},
                {"test_name": "Hb", "value": 10.8, "value_text": "10.8 g/dL", "unit": "g/dL", "date": "2026-01-12", "source_files": ["02_followup.pdf"], "source_pages": [1], "confidence": 0.98},
            ], "radiology": [], "procedures": [], "vaccinations": [], "growth_measurements": [],
            "growth_interpretation": "Growth appears to track within expected percentile curves.",
            "clinical_timeline": [{"date": "2025-01-10", "event": "Initial visit", "summary": "Asthma review and CBC", "source_files": ["01_visit.pdf"], "source_pages": [1], "confidence": 0.95}],
            "uncertain_items": [{"field": "medications.dose", "proposed_value": None, "reason": "Dose is illegible; clinician verification required.", "source_files": ["01_visit.pdf"], "source_pages": [1], "confidence": 0.45}],
            "conflicts": [],
            "detailed_report_markdown": "# Detailed Medical Report\nPersistent mild anemia with an initially microcytic pattern. Iron deficiency is not biochemically confirmed.",
            "erp_summary_markdown": "Amina Hassan: bronchial asthma documented. Persistent mild anemia with an initially microcytic pattern; iron deficiency is not biochemically confirmed. Ventolin dose requires verification.",
        })


def test_golden_patient_level_review_preserves_longitudinal_context(tmp_path: Path) -> None:
    patient = tmp_path / "GOLDEN_PATIENT"
    patient.mkdir()
    text_pdf(patient / "01_visit.pdf", ["Date: 2025-01-10", "Hb: 10.2 g/dL"])
    text_pdf(patient / "02_followup.pdf", ["Date: 2026-01-12", "Hb: 10.8 g/dL"])
    config = Settings(database_url=f"sqlite+pysqlite:///{tmp_path / 'golden.db'}", storage_root=tmp_path / "data", output_root=tmp_path / "output", ai_provider="openai", ai_extraction_mode="patient_level")
    create_schema(config.database_url)
    repository = RecordRepository(make_session_factory(config.database_url), storage_root=config.storage_root, output_root=config.output_root)
    record = PatientPipeline(repository, config, provider=GoldenPatientProvider()).process(patient).record

    assert record.patient.full_name.value == "Amina Hassan"
    assert record.diagnoses[0].term_original == "Bronchial asthma"
    assert [lab.value for lab in record.laboratory_results] == [10.2, 10.8]
    assert record.medications[0].dose is None and record.medications[0].verification_required is True
    assert "not biochemically confirmed" in record.ai_clinical_review["clinical_interpretations"][0]["value"]
    assert "bronchial asthma" in record.ai_clinical_review["erp_summary_markdown"].lower()
    assert record.verification_queue
