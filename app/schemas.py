from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DocumentType(str, Enum):
    PATIENT_REGISTRATION = "patient_registration"
    OUTPATIENT_NOTE = "outpatient_note"
    INPATIENT_NOTE = "inpatient_note"
    EMERGENCY_NOTE = "emergency_note"
    PRESCRIPTION = "prescription"
    LAB_REPORT = "lab_report"
    RADIOLOGY_REPORT = "radiology_report"
    DISCHARGE_SUMMARY = "discharge_summary"
    OPERATIVE_NOTE = "operative_note"
    VACCINATION_RECORD = "vaccination_record"
    GROWTH_CHART = "growth_chart"
    ECG = "ECG"
    VITAL_SIGNS = "vital_signs"
    REFERRAL = "referral"
    UNKNOWN = "unknown"


class FactStatus(str, Enum):
    DOCUMENTED = "documented"
    INFERRED = "inferred"
    NEEDS_VERIFICATION = "needs_verification"
    ILLEGIBLE = "illegible"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"
    UNREADABLE = "unreadable"


class SourceRef(BaseModel):
    source_file: str
    source_page: int = Field(ge=1)
    bounding_box: list[float] | None = None
    extraction_method: str
    prompt_version: str | None = None


class ProvenancedValue(BaseModel):
    value: Any | None = None
    raw_text: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    status: FactStatus
    source_ref: SourceRef


class PatientDemographics(BaseModel):
    patient_id: str
    hospital_file_number: ProvenancedValue | None = None
    full_name: ProvenancedValue | None = None
    sex: ProvenancedValue | None = None
    date_of_birth: ProvenancedValue | None = None
    nationality: ProvenancedValue | None = None
    address: ProvenancedValue | None = None
    candidates: dict[str, list[ProvenancedValue]] = Field(default_factory=dict)


class Encounter(BaseModel):
    encounter_id: str
    date: str | None = None
    encounter_type: str | None = None
    facility: str | None = None
    department: str | None = None
    physician: str | None = None
    chief_complaint: list[ProvenancedValue] = Field(default_factory=list)
    symptoms: list[ProvenancedValue] = Field(default_factory=list)
    clinical_findings: list[ProvenancedValue] = Field(default_factory=list)
    diagnoses: list[ProvenancedValue] = Field(default_factory=list)
    medications: list[ProvenancedValue] = Field(default_factory=list)
    investigations: list[ProvenancedValue] = Field(default_factory=list)
    procedures: list[ProvenancedValue] = Field(default_factory=list)
    follow_up: list[ProvenancedValue] = Field(default_factory=list)
    source_refs: list[SourceRef] = Field(default_factory=list)


class Diagnosis(BaseModel):
    term_original: str
    term_normalized: str | None = None
    type: str = Field(pattern="^(documented|inferred|differential|historical)$")
    status: str = Field(pattern="^(active|resolved|unknown)$")
    certainty: str = Field(pattern="^(confirmed|probable|possible|uncertain)$")
    date: str | None = None
    icd10_code: str | None = None
    snomed_code: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    source_ref: SourceRef


class Medication(BaseModel):
    name_original: str | None = None
    name_normalized: str | None = None
    generic_name: str | None = None
    brand_name: str | None = None
    strength: str | None = None
    dose: str | None = None
    dose_unit: str | None = None
    route: str | None = None
    frequency: str | None = None
    duration: str | None = None
    indication: str | None = None
    status: str = "unknown"
    confidence: float = Field(ge=0.0, le=1.0)
    source_ref: SourceRef
    verification_required: bool = False


class LaboratoryResult(BaseModel):
    test_name_original: str
    test_name_normalized: str | None = None
    loinc_code: str | None = None
    value: float | None = None
    value_text: str = ""
    unit: str | None = None
    reference_range: str | None = None
    abnormal_flag: str | None = None
    date: str | None = None
    specimen: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    source_ref: SourceRef


class GrowthMeasurement(BaseModel):
    date: str | None = None
    age_months: float | None = Field(default=None, ge=0)
    weight_kg: float | None = Field(default=None, ge=0)
    height_cm: float | None = Field(default=None, ge=0)
    head_circumference_cm: float | None = Field(default=None, ge=0)
    bmi: float | None = Field(default=None, ge=0)
    source: SourceRef
    confidence: float = Field(ge=0.0, le=1.0)


class RadiologyReport(BaseModel):
    study_type: str | None = None
    body_part: str | None = None
    date: str | None = None
    report_text: str | None = None
    findings: list[str] = Field(default_factory=list)
    impression: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    source_ref: SourceRef


class TimelineEvent(BaseModel):
    date: str | None = None
    event: str
    summary: str
    source_refs: list[SourceRef] = Field(default_factory=list)


class ProblemItem(BaseModel):
    problem: str
    status: str
    source_type: str
    source_refs: list[SourceRef] = Field(default_factory=list)


class VerificationItem(BaseModel):
    item_id: str
    field: str
    proposed_value: str | None = None
    raw_text: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    severity: str = Field(pattern="^(data_entry|medical_review)$")
    source_ref: SourceRef
    review_status: ReviewStatus = ReviewStatus.PENDING


class SourceFile(BaseModel):
    file_id: str
    original_filename: str
    file_type: str
    page_count: int | None = None
    checksum: str
    processing_status: str
    original_path: str
    processed_path: str | None = None
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Archive provenance is retained on each extracted source document.
    archive_filename: str | None = None
    archive_type: str | None = None
    original_relative_path: str | None = None
    extracted_filename: str | None = None
    # Stable order for holistic review: archive/page order when available,
    # followed by the original relative path.
    source_order: int | None = None


class SourcePageInfo(BaseModel):
    page_id: str
    file_id: str
    page_number: int = Field(ge=1)
    document_type: DocumentType
    classification_confidence: float = Field(ge=0.0, le=1.0)
    extraction_status: str


class AuditEvent(BaseModel):
    fact_id: str
    patient_id: str
    field: str
    value: str | None = None
    source_file: str
    source_page: int
    model: str | None = None
    model_version: str | None = None
    prompt_version: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = Field(ge=0.0, le=1.0)
    reviewed_by: str | None = None
    review_status: ReviewStatus = ReviewStatus.PENDING


class StructuredRecord(BaseModel):
    schema_version: str = "0.1"
    patient: PatientDemographics
    source_files: list[SourceFile] = Field(default_factory=list)
    source_pages: list[SourcePageInfo] = Field(default_factory=list)
    encounters: list[Encounter] = Field(default_factory=list)
    diagnoses: list[Diagnosis] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    laboratory_results: list[LaboratoryResult] = Field(default_factory=list)
    radiology_reports: list[RadiologyReport] = Field(default_factory=list)
    growth_measurements: list[GrowthMeasurement] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    problem_list: list[ProblemItem] = Field(default_factory=list)
    verification_queue: list[VerificationItem] = Field(default_factory=list)
    audit_events: list[AuditEvent] = Field(default_factory=list)
    # Optional, reviewable output from the patient-level reasoning stage. The
    # canonical facts above remain the source of truth for reports.
    clinical_synthesis: dict[str, Any] | None = None
    # Canonical patient-level multimodal reconstruction.  The narrow typed
    # collections above remain available for exports and backward-compatible
    # integrations; this object is the authoritative AI review for reports.
    ai_clinical_review: dict[str, Any] | None = None
    ai_review_version: str | None = None
    ai_review_model: str | None = None
    ai_review_prompt_version: str | None = None
    ai_review_source_hash: str | None = None
    ai_review_created_at: datetime | None = None
    ai_review_history: list[dict[str, Any]] = Field(default_factory=list)
    processing_warnings: list[str] = Field(default_factory=list)
    patient_status: str = "completed"

    @field_validator("patient_status")
    @classmethod
    def valid_status(cls, value: str) -> str:
        if value not in {"completed", "completed_with_warnings", "failed"}:
            raise ValueError("invalid patient processing status")
        return value


class ReportType(str, Enum):
    DETAILED = "detailed_medical_report"
    ERP_SUMMARY = "erp_summary"


class ReportRecord(BaseModel):
    report_id: str
    patient_id: str
    patient_name: str | None = None
    hospital_file_number: str | None = None
    report_type: ReportType
    version: int = Field(ge=1)
    file_path: str
    text_path: str | None = None
    json_path: str | None = None
    status: str = Field(pattern="^(ready|needs_review|failed)$")
    created_at: datetime
    created_by: str
    model_name: str
    prompt_version: str
    source_record_hash: str
    review_status: str
    source_file_ids: list[str] = Field(default_factory=list)
    ai_review_version: str | None = None


class ReportGenerationRequest(BaseModel):
    report_type: ReportType
    created_by: str = "hospital-user"


class PatientSearchResult(BaseModel):
    patient_id: str
    patient_name: str | None = None
    sex: str | None = None
    date_of_birth: str | None = None
    hospital_file_number: str | None = None
    last_processed_at: datetime | None = None
    processing_status: str
    review_status: str


class JobStatus(BaseModel):
    job_id: str
    patient_id: str | None = None
    status: str
    progress: int = Field(ge=0, le=100)
    message: str
    error: str | None = None
    created_at: datetime
    updated_at: datetime
