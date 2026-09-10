"""Typed contracts for the optional OpenAI extraction provider.

The local MVP intentionally keeps these contracts separate from the canonical
record schemas.  The provider validates the model response first, then the
pipeline maps only documented values into the existing, provenance-aware
record.  This prevents model prose from becoming clinical data by accident.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Confidence = Field(default=0.0, ge=0.0, le=1.0)


class AIField(BaseModel):
    value: Any | None = None
    normalized_value: str | None = None
    confidence: float = Confidence
    certainty: Literal["confirmed", "probable", "possible", "uncertain"] = "uncertain"
    documented_vs_inferred: Literal["documented", "inferred"] = "documented"
    source_page: int = Field(default=1, ge=1)
    source_file: str = ""
    needs_verification: bool = False


class AIPatientIdentity(BaseModel):
    patient_name: AIField | None = None
    patient_id: AIField | None = None
    hospital_file_number: AIField | None = None
    date_of_birth: AIField | None = None
    sex: AIField | None = None
    nationality: AIField | None = None
    address: AIField | None = None


class AIDiagnosis(AIField):
    status: Literal["active", "resolved", "unknown"] = "unknown"


class AIMedication(AIField):
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


class AILaboratoryResult(AIField):
    test_name_original: str = ""
    test_name_normalized: str | None = None
    value: float | None = None
    value_text: str = ""
    unit: str | None = None
    reference_range: str | None = None
    abnormal_flag: str | None = None
    date: str | None = None
    specimen: str | None = None


class AIGrowthMeasurement(AIField):
    date: str | None = None
    age_months: float | None = Field(default=None, ge=0)
    weight_kg: float | None = Field(default=None, ge=0)
    height_cm: float | None = Field(default=None, ge=0)
    head_circumference_cm: float | None = Field(default=None, ge=0)
    bmi: float | None = Field(default=None, ge=0)


class AIRadiologyReport(AIField):
    study_type: str | None = None
    body_part: str | None = None
    date: str | None = None
    report_text: str | None = None
    findings: list[str] = Field(default_factory=list)
    impression: list[str] = Field(default_factory=list)


class AIUncertainItem(BaseModel):
    field: str
    proposed_value: str | None = None
    raw_text: str | None = None
    confidence: float = Confidence
    reason: str
    severity: Literal["data_entry", "medical_review"] = "data_entry"
    source_page: int = Field(default=1, ge=1)
    source_file: str = ""


class MedicalPageExtraction(BaseModel):
    """Strict page-level response returned by the Responses API."""

    transcription: str = ""
    document_type: str = "unknown"
    complexity: str = "mixed"
    encounter_date: str | None = None
    patient_identity: AIPatientIdentity = Field(default_factory=AIPatientIdentity)
    diagnoses: list[AIDiagnosis] = Field(default_factory=list)
    medications: list[AIMedication] = Field(default_factory=list)
    laboratory_results: list[AILaboratoryResult] = Field(default_factory=list)
    radiology: list[AIRadiologyReport] = Field(default_factory=list)
    growth_measurements: list[AIGrowthMeasurement] = Field(default_factory=list)
    uncertain_items: list[AIUncertainItem] = Field(default_factory=list)
    overall_confidence: float = Confidence


class ClinicalSynthesis(BaseModel):
    """Patient-level synthesis generated only from validated structured facts."""

    problem_list: list[str] = Field(default_factory=list)
    timeline: list[str] = Field(default_factory=list)
    longitudinal_trends: list[str] = Field(default_factory=list)
    physician_summary: str = ""
    erp_summary: str = ""
    overall_confidence: float = Confidence


class PatientLevelReview(BaseModel):
    """Strict, source-grounded reconstruction for one complete patient chart.

    The nested clinical collections intentionally remain extensible: medical
    documents contain heterogeneous facts (ECGs, charts, discharge notes,
    vaccination records, and handwritten annotations).  The top-level contract
    is fixed so the UI and report generators can safely rely on one canonical
    review object.
    """

    patient: dict[str, Any] = Field(default_factory=dict)
    record_quality: dict[str, Any] = Field(default_factory=dict)
    birth_history: dict[str, Any] = Field(default_factory=dict)
    past_medical_history: list[dict[str, Any]] = Field(default_factory=list)
    active_problems: list[dict[str, Any]] = Field(default_factory=list)
    resolved_or_historical_problems: list[dict[str, Any]] = Field(default_factory=list)
    encounters: list[dict[str, Any]] = Field(default_factory=list)
    symptoms: list[dict[str, Any]] = Field(default_factory=list)
    clinical_findings: list[dict[str, Any]] = Field(default_factory=list)
    diagnoses_documented: list[dict[str, Any]] = Field(default_factory=list)
    clinical_interpretations: list[dict[str, Any]] = Field(default_factory=list)
    medications: list[dict[str, Any]] = Field(default_factory=list)
    allergies: list[dict[str, Any]] = Field(default_factory=list)
    laboratory_results: list[dict[str, Any]] = Field(default_factory=list)
    radiology: list[dict[str, Any]] = Field(default_factory=list)
    procedures: list[dict[str, Any]] = Field(default_factory=list)
    vaccinations: list[dict[str, Any]] = Field(default_factory=list)
    growth_measurements: list[dict[str, Any]] = Field(default_factory=list)
    growth_interpretation: str = ""
    clinical_timeline: list[dict[str, Any]] = Field(default_factory=list)
    uncertain_items: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    detailed_report_markdown: str = ""
    erp_summary_markdown: str = ""


class PatientBundleItem(BaseModel):
    """Manifest entry passed to the multimodal patient-level provider."""

    source_id: str
    source_file: str
    relative_path: str
    page_number: int = Field(default=1, ge=1)
    order_index: int = Field(default=0, ge=0)
    media_type: str
    local_path: str
