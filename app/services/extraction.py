from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re
from uuid import uuid4

from app.schemas import (
    AuditEvent,
    Diagnosis,
    DocumentType,
    FactStatus,
    GrowthMeasurement,
    LaboratoryResult,
    Medication,
    ProvenancedValue,
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


def _audit(patient_id: str, field: str, value: str | None, ref: SourceRef, confidence: float) -> AuditEvent:
    return AuditEvent(
        fact_id=str(uuid4()), patient_id=patient_id, field=field, value=value,
        source_file=ref.source_file, source_page=ref.source_page, model="local_safe",
        model_version="1", confidence=confidence,
    )


def _review(field: str, value: str | None, raw_text: str | None, confidence: float, reason: str, ref: SourceRef, severity: str = "data_entry") -> VerificationItem:
    return VerificationItem(
        item_id=str(uuid4()), field=field, proposed_value=value, raw_text=raw_text,
        confidence=confidence, reason=reason, severity=severity, source_ref=ref,
    )


def extract_documented_facts(patient_id: str, text: str, filename: str, page_number: int, document_type: DocumentType) -> PageExtraction:
    """Extract narrow, explicitly labeled facts. This function does not infer diagnoses or doses."""
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
