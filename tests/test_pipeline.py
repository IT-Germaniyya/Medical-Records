from pathlib import Path

import pytest

from app.config import Settings
from app.db import create_schema, make_session_factory
from app.pipeline import PatientPipeline
from app.repository import DuplicateSourceError, RecordRepository
from app.schemas import ReportType, ReviewStatus
from app.services.reports import ReportGenerator
from pypdf import PdfReader
from tests.helpers import text_pdf


def make_pipeline(tmp_path: Path) -> PatientPipeline:
    configuration = Settings(
        database_url=f"sqlite+pysqlite:///{tmp_path / 'mvp.db'}",
        storage_root=tmp_path / "data",
        output_root=tmp_path / "output",
    )
    create_schema(configuration.database_url)
    return PatientPipeline(RecordRepository(make_session_factory(configuration.database_url)), configuration)


def test_pdf_pipeline_preserves_provenance_and_outputs(tmp_path: Path) -> None:
    patient = tmp_path / "PATIENT_001"
    patient.mkdir()
    text_pdf(patient / "CBC_lab_report.pdf", [
        "Date: 2026-01-14", "Diagnosis: Bronchial asthma", "Medication: Salbutamol 100 mcg inhaler",
        "Hb: 10.2 g/dL", "MCV: 74 fL", "Weight: 18.2 kg", "Height: 108 cm",
    ])
    pipeline = make_pipeline(tmp_path)
    result = pipeline.process(patient)
    assert result.record.patient_status == "completed_with_warnings"
    assert result.record.laboratory_results[0].source_ref.source_file == "CBC_lab_report.pdf"
    assert {item.test_name_normalized for item in result.record.laboratory_results} >= {"Hemoglobin", "Mean corpuscular volume"}
    assert result.record.diagnoses[0].term_original == "Bronchial asthma"
    assert result.record.medications[0].dose is None
    assert any(item.field == "medication.dose" for item in result.record.verification_queue)
    assert (result.output_directory / "structured_record.json").exists()
    assert (result.output_directory / "physician_summary.txt").exists()
    assert (result.output_directory / "timeline.json").exists()
    assert (result.output_directory / "verification_queue.json").exists()
    assert pipeline.repository.get_record("PATIENT_001") is not None


def test_same_patient_resumes_without_reprocessing(tmp_path: Path) -> None:
    patient = tmp_path / "PATIENT_002"
    patient.mkdir()
    text_pdf(patient / "lab.pdf", ["Hb: 10.2 g/dL"])
    pipeline = make_pipeline(tmp_path)
    first = pipeline.process(patient)
    second = pipeline.process(patient)
    assert not first.resumed
    assert second.resumed


def test_existing_patient_append_and_search_by_name_or_mrn(tmp_path: Path) -> None:
    initial = tmp_path / "initial"
    additional = tmp_path / "additional"
    initial.mkdir()
    additional.mkdir()
    text_pdf(initial / "visit.pdf", ["Hb: 10.2 g/dL"])
    text_pdf(additional / "followup.pdf", ["MCV: 74 fL"])
    pipeline = make_pipeline(tmp_path)
    pipeline.process(initial, patient_id="PATIENT_APPEND", demographics={"patient_name": "Amina Hassan", "hospital_file_number": "MRN-77"})
    appended = pipeline.process(additional, patient_id="PATIENT_APPEND", resume=False)
    assert len(appended.record.source_files) == 2
    assert {item.source_ref.source_file for item in appended.record.laboratory_results} == {"visit.pdf", "followup.pdf"}
    for query in ("MRN-77", "Amina", "PATIENT_APPEND"):
        matches = pipeline.repository.search_patients(query)
        assert len(matches) == 1
        assert matches[0].patient_name == "Amina Hassan"


def test_cross_patient_duplicate_source_is_rejected(tmp_path: Path) -> None:
    patient_a, patient_b = tmp_path / "PATIENT_A", tmp_path / "PATIENT_B"
    patient_a.mkdir()
    patient_b.mkdir()
    text_pdf(patient_a / "lab.pdf", ["Hb: 10.2 g/dL"])
    (patient_b / "same.pdf").write_bytes((patient_a / "lab.pdf").read_bytes())
    pipeline = make_pipeline(tmp_path)
    pipeline.process(patient_a)
    with pytest.raises(DuplicateSourceError):
        pipeline.process(patient_b)


def test_reviewer_decision_is_audited_without_altering_raw_fact(tmp_path: Path) -> None:
    patient = tmp_path / "PATIENT_REVIEW"
    patient.mkdir()
    text_pdf(patient / "lab.pdf", ["Diagnosis: Bronchial asthma"])
    pipeline = make_pipeline(tmp_path)
    result = pipeline.process(patient)
    item = result.record.verification_queue[0]
    updated = pipeline.repository.review_item("PATIENT_REVIEW", item.item_id, ReviewStatus.APPROVED, "reviewer-1")
    assert updated is not None
    assert updated.diagnoses[0].term_original == "Bronchial asthma"
    assert next(candidate for candidate in updated.verification_queue if candidate.item_id == item.item_id).review_status == ReviewStatus.APPROVED
    assert updated.audit_events[-1].reviewed_by == "reviewer-1"


def test_reports_are_grounded_versioned_and_separate(tmp_path: Path) -> None:
    patient = tmp_path / "PATIENT_REPORT"
    patient.mkdir()
    text_pdf(patient / "lab.pdf", ["Date: 2026-01-14", "Diagnosis: Bronchial asthma", "Hb: 10.2 g/dL"])
    pipeline = make_pipeline(tmp_path)
    pipeline.process(patient)
    generator = ReportGenerator(pipeline.repository, pipeline.configuration)
    detailed = generator.generate("PATIENT_REPORT", ReportType.DETAILED)
    erp = generator.generate("PATIENT_REPORT", ReportType.ERP_SUMMARY)
    erp_v2 = generator.generate("PATIENT_REPORT", ReportType.ERP_SUMMARY)
    assert detailed.status == "needs_review"
    assert detailed.version == 1
    assert erp.version == 1
    assert erp_v2.version == 2
    assert len(PdfReader(detailed.file_path).pages) >= 2  # report body + original PDF appendix page
    erp_text = "\n".join(page.extract_text() or "" for page in PdfReader(erp.file_path).pages)
    assert "Appendix: Original Source Documents" not in erp_text
    assert "10.2" in erp_text
