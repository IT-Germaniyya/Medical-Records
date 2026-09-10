from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re
from uuid import uuid4

from app.ai_schemas import MedicalPageExtraction

from app.schemas import (
    AuditEvent,
    Diagnosis,
    DocumentType,
    FactStatus,
    GrowthMeasurement,
    LaboratoryResult,
    Medication,
    ProvenancedValue,
    RadiologyReport,
    SourceRef,
    VerificationItem,
)

DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
LAB_RE = re.compile(
    r"(?im)\b(?P<name>Hb|Hgb|Hemoglobin|MCV|MCH|Platelets?|WBC)\s*[:=]\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>g/dL|g/L|fL|pg|x10\^?9/L|10\^?9/L|/uL)?"
)
GROWTH_RE = re.compile(
    r"(?im)\b(?P<name>weight|height|length|head circumference)\s*[:=]\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>kg|cm)\b"
)


@dataclass
class PageExtraction:
    diagnoses: list[Diagnosis] = field(default_factory=list)
    medications: list[Medication] = field(default_factory=list)
    labs: list[LaboratoryResult] = field(default_factory=list)
    radiology: list[RadiologyReport] = field(default_factory=list)
    growth: list[GrowthMeasurement] = field(default_factory=list)
    verification: list[VerificationItem] = field(default_factory=list)
    audit: list[AuditEvent] = field(default_factory=list)


def source_ref(filename: str, page_number: int, method: str = "local_pdf_text") -> SourceRef:
    return SourceRef(source_file=filename, source_page=page_number, extraction_method=method)


def _documented_date(text: str) -> str | None:
    values = DATE_RE.findall(text)
    for value in values:
        try:
            datetime.strptime(value, "%Y-%m-%d")
            return value
        except ValueError:
            continue
    return None


def _audit(
    patient_id: str,
    field: str,
    value: str | None,
    ref: SourceRef,
    confidence: float,
    *,
    model: str = "local_safe",
    model_version: str = "1",
    prompt_version: str | None = None,
) -> AuditEvent:
    return AuditEvent(
        fact_id=str(uuid4()), patient_id=patient_id, field=field, value=value,
        source_file=ref.source_file, source_page=ref.source_page, model=model,
        model_version=model_version, prompt_version=prompt_version, confidence=confidence,
    )


def _review(field: str, value: str | None, raw_text: str | None, confidence: float, reason: str, ref: SourceRef, severity: str = "data_entry") -> VerificationItem:
    return VerificationItem(
        item_id=str(uuid4()), field=field, proposed_value=value, raw_text=raw_text,
        confidence=confidence, reason=reason, severity=severity, source_ref=ref,
    )


def extract_documented_facts(
    patient_id: str,
    text: str,
    filename: str,
    page_number: int,
    document_type: DocumentType,
    *,
    ai_page: MedicalPageExtraction | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
) -> PageExtraction:
    """Extract narrow, explicitly labeled facts. This function does not infer diagnoses or doses."""
    if ai_page is not None:
        return _extract_ai_facts(patient_id, ai_page, filename, page_number, document_type, model or "openai", prompt_version)
    ref = source_ref(filename, page_number)
    output = PageExtraction()
    collected_date = _documented_date(text)

    for match in LAB_RE.finditer(text):
        original = match.group("name")
        normalized = {
            "hb": "Hemoglobin", "hgb": "Hemoglobin", "hemoglobin": "Hemoglobin",
            "mcv": "Mean corpuscular volume", "mch": "Mean corpuscular hemoglobin",
            "platelet": "Platelets", "platelets": "Platelets", "wbc": "White blood cell count",
        }.get(original.lower(), original)
        unit = match.group("unit") or None
        result = LaboratoryResult(
            test_name_original=original, test_name_normalized=normalized,
            value=float(match.group("value")), value_text=match.group(0).strip(), unit=unit,
            date=collected_date, confidence=0.95, source_ref=ref,
        )
        output.labs.append(result)
        output.audit.append(_audit(patient_id, f"laboratory_results.{normalized}", str(result.value), ref, result.confidence))

    measures: dict[str, float] = {}
    for match in GROWTH_RE.finditer(text):
        name, value, unit = match.group("name").lower(), float(match.group("value")), match.group("unit").lower()
        if name == "weight" and unit == "kg":
            measures["weight_kg"] = value
        elif name in {"height", "length"} and unit == "cm":
            measures["height_cm"] = value
        elif name == "head circumference" and unit == "cm":
            measures["head_circumference_cm"] = value
    if measures:
        growth = GrowthMeasurement(date=collected_date, source=ref, confidence=0.93, **measures)
        output.growth.append(growth)
        output.audit.append(_audit(patient_id, "growth_measurements", str(measures), ref, growth.confidence))

    for line in re.findall(r"(?im)^(?:diagnosis|assessment)\s*:\s*(.+)$", text):
        for raw in (part.strip(" .") for part in line.split(";")):
            if not raw:
                continue
            lowered = raw.lower()
            certainty = "probable" if "probable" in lowered else "possible" if "possible" in lowered else "uncertain" if "?" in raw else "confirmed"
            diagnosis_type = "differential" if certainty in {"probable", "possible", "uncertain"} else "documented"
            diagnosis = Diagnosis(
                term_original=raw, term_normalized=raw, type=diagnosis_type,
                status="unknown", certainty=certainty, date=collected_date,
                confidence=0.92, source_ref=ref,
            )
            output.diagnoses.append(diagnosis)
            output.audit.append(_audit(patient_id, "diagnoses", raw, ref, diagnosis.confidence))

    for line in re.findall(r"(?im)^(?:medication|rx|prescription)\s*:\s*(.+)$", text):
        for raw in (part.strip(" .") for part in line.split(";")):
            if not raw:
                continue
            strength = re.search(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|mL)\b", raw, re.I)
            medication = Medication(
                name_original=raw, name_normalized=raw, strength=strength.group(0) if strength else None,
                confidence=0.90, source_ref=ref, verification_required=True,
            )
            output.medications.append(medication)
            output.audit.append(_audit(patient_id, "medications", raw, ref, medication.confidence))

    if document_type == DocumentType.UNKNOWN:
        output.verification.append(_review("document_classification", None, None, 0.0, "document type could not be classified", ref))
    return output


def _extract_ai_facts(
    patient_id: str,
    page: MedicalPageExtraction,
    filename: str,
    page_number: int,
    document_type: DocumentType,
    model: str,
    prompt_version: str | None,
) -> PageExtraction:
    """Map only validated, documented model fields into canonical MVP schemas."""
    output = PageExtraction()

    def ref_for(item: object) -> SourceRef:
        return SourceRef(
            # Source identity is supplied by the ingestion pipeline, not by
            # model output, so a hallucinated filename/page cannot corrupt
            # provenance.
            source_file=filename,
            source_page=page_number,
            extraction_method="openai_responses",
            prompt_version=prompt_version,
        )

    def confidence_for(item: object) -> float:
        return max(0.0, min(1.0, float(getattr(item, "confidence", page.overall_confidence))))

    for item in page.diagnoses:
        raw = item.value
        if raw in (None, ""):
            continue
        value = str(raw)
        ref = ref_for(item)
        certainty = item.certainty
        diagnosis_type = "documented" if item.documented_vs_inferred == "documented" and certainty == "confirmed" else "differential"
        diagnosis = Diagnosis(
            term_original=value,
            term_normalized=item.normalized_value or value,
            type=diagnosis_type,
            status=item.status,
            certainty=certainty,
            date=page.encounter_date,
            confidence=confidence_for(item),
            source_ref=ref,
        )
        output.diagnoses.append(diagnosis)
        output.audit.append(_audit(patient_id, "diagnoses", value, ref, diagnosis.confidence, model=model, model_version=model, prompt_version=prompt_version))
        if item.needs_verification or diagnosis.confidence < 0.90 or diagnosis_type != "documented":
            output.verification.append(_review("diagnosis", value, value, diagnosis.confidence, "AI extraction requires clinician verification before use", ref, "medical_review"))

    for item in page.medications:
        name = item.name_original or (str(item.value) if item.value not in (None, "") else None)
        if not name:
            continue
        ref = ref_for(item)
        medication = Medication(
            name_original=name,
            name_normalized=item.name_normalized or item.normalized_value,
            generic_name=item.generic_name,
            brand_name=item.brand_name,
            strength=item.strength,
            dose=item.dose,
            dose_unit=item.dose_unit,
            route=item.route,
            frequency=item.frequency,
            duration=item.duration,
            indication=item.indication,
            confidence=confidence_for(item),
            source_ref=ref,
            verification_required=True,
        )
        output.medications.append(medication)
        output.audit.append(_audit(patient_id, "medications", name, ref, medication.confidence, model=model, model_version=model, prompt_version=prompt_version))
        if item.needs_verification or not item.dose or not item.frequency or medication.confidence < 0.90:
            output.verification.append(_review("medication", name, name, medication.confidence, "Medication identity or dose/frequency requires reconciliation", ref, "medical_review"))

    for item in page.laboratory_results:
        if not item.test_name_original:
            continue
        ref = ref_for(item)
        numeric_value = item.value
        if numeric_value is None and item.value_text:
            try:
                numeric_value = float(item.value_text.strip())
            except (TypeError, ValueError):
                numeric_value = None
        lab = LaboratoryResult(
            test_name_original=item.test_name_original,
            test_name_normalized=item.test_name_normalized,
            value=numeric_value,
            value_text=item.value_text or (str(item.value) if item.value is not None else ""),
            unit=item.unit,
            reference_range=item.reference_range,
            abnormal_flag=item.abnormal_flag,
            date=item.date or page.encounter_date,
            specimen=item.specimen,
            confidence=confidence_for(item),
            source_ref=ref,
        )
        output.labs.append(lab)
        output.audit.append(_audit(patient_id, f"laboratory_results.{lab.test_name_original}", lab.value_text, ref, lab.confidence, model=model, model_version=model, prompt_version=prompt_version))
        if item.needs_verification or lab.confidence < 0.90:
            output.verification.append(_review("laboratory_result", lab.value_text, lab.value_text, lab.confidence, "Laboratory value requires verification against the source", ref))

    for item in page.growth_measurements:
        if not any(value is not None for value in (item.weight_kg, item.height_cm, item.head_circumference_cm, item.bmi)):
            continue
        ref = ref_for(item)
        growth = GrowthMeasurement(
            date=item.date or page.encounter_date,
            age_months=item.age_months,
            weight_kg=item.weight_kg,
            height_cm=item.height_cm,
            head_circumference_cm=item.head_circumference_cm,
            bmi=item.bmi,
            source=ref,
            confidence=confidence_for(item),
        )
        output.growth.append(growth)
        output.audit.append(_audit(patient_id, "growth_measurements", str(growth.model_dump()), ref, growth.confidence, model=model, model_version=model, prompt_version=prompt_version))
        if item.needs_verification or growth.confidence < 0.90:
            output.verification.append(_review("growth_measurement", str(growth.model_dump()), None, growth.confidence, "Chart values require verification; no percentile was inferred", ref))

    for item in page.radiology:
        ref = ref_for(item)
        report = RadiologyReport(
            study_type=item.study_type,
            body_part=item.body_part,
            date=item.date or page.encounter_date,
            report_text=item.report_text,
            findings=item.findings,
            impression=item.impression,
            confidence=confidence_for(item),
            source_ref=ref,
        )
        output.radiology.append(report)
        output.audit.append(_audit(patient_id, "radiology", report.report_text, ref, report.confidence, model=model, model_version=model, prompt_version=prompt_version))
        if item.needs_verification or report.confidence < 0.90:
            output.verification.append(_review("radiology", report.report_text, report.report_text, report.confidence, "Radiology extraction requires review against the source", ref, "medical_review"))

    for item in page.uncertain_items:
        ref = SourceRef(
            source_file=filename,
            source_page=page_number,
            extraction_method="openai_responses",
            prompt_version=prompt_version,
        )
        output.verification.append(_review(item.field, item.proposed_value, item.raw_text, item.confidence, item.reason, ref, item.severity))

    if page.document_type == "unknown" and not output.verification:
        output.verification.append(_review("document_classification", None, None, page.overall_confidence, "document type could not be classified", source_ref(filename, page_number, "openai_responses")))
    return output


def unreadable_page_item(filename: str, page_number: int, reason: str) -> VerificationItem:
    ref = source_ref(filename, page_number, "not_available")
    return _review("page_text", None, None, 0.0, reason, ref, "data_entry")


def extract_identity_facts(text: str, filename: str, page_number: int) -> dict[str, ProvenancedValue]:
    """Read only explicitly labeled identity fields for mismatch checks; never infer identity from filenames."""
    ref = source_ref(filename, page_number)
    patterns = {
        "patient_id": r"(?im)^\s*(?:patient\s*id|patient\s*number)\s*[:=]\s*([^\s]+)",
        "full_name": r"(?im)^\s*(?:patient\s*)?name\s*[:=]\s*(.+?)\s*$",
        "hospital_file_number": r"(?im)^\s*(?:mrn|medical\s*record\s*(?:number|no\.?))\s*[:=]\s*([^\s]+)",
        "sex": r"(?im)^\s*sex\s*[:=]\s*([^\s]+)",
        "date_of_birth": r"(?im)^\s*(?:dob|date\s*of\s*birth)\s*[:=]\s*([^\s]+)",
    }
    facts: dict[str, ProvenancedValue] = {}
    for field_name, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            raw = match.group(1).strip(" ,;")
            if raw:
                facts[field_name] = ProvenancedValue(value=raw, raw_text=match.group(0).strip(), confidence=0.94, status=FactStatus.DOCUMENTED, source_ref=ref)
    return facts
