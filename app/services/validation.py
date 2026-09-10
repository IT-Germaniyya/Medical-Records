from __future__ import annotations

from datetime import date
from uuid import uuid4

from app.schemas import SourceRef, StructuredRecord, VerificationItem


def _item(field: str, value: str | None, reason: str, ref: SourceRef, severity: str = "data_entry", confidence: float = 0.0) -> VerificationItem:
    return VerificationItem(item_id=str(uuid4()), field=field, proposed_value=value, confidence=confidence, reason=reason, severity=severity, source_ref=ref)


def apply_confidence_policy(record: StructuredRecord, high: float, medium: float, high_risk: float) -> None:
    """Configurable confidence routing; it never upgrades a fact's clinical certainty."""
    def confidence_band(confidence: float) -> str:
        return "medium" if confidence >= medium else "low"

    for diagnosis in record.diagnoses:
        if diagnosis.confidence < high_risk:
            record.verification_queue.append(_item("diagnosis", diagnosis.term_original, "high-risk clinical field requires medical review", diagnosis.source_ref, "medical_review", diagnosis.confidence))
    for medication in record.medications:
        if medication.confidence < high_risk:
            record.verification_queue.append(_item("medication", medication.name_original, f"{confidence_band(medication.confidence)}-confidence high-risk medication field requires review", medication.source_ref, "medical_review", medication.confidence))
    for lab in record.laboratory_results:
        if lab.confidence < high:
            record.verification_queue.append(_item("laboratory_result", lab.value_text, f"{confidence_band(lab.confidence)}-confidence extraction", lab.source_ref, "data_entry", lab.confidence))
    for growth in record.growth_measurements:
        if growth.confidence < high:
            record.verification_queue.append(_item("growth_measurement", None, f"{confidence_band(growth.confidence)}-confidence extraction", growth.source, "data_entry", growth.confidence))


def apply_rule_validations(record: StructuredRecord) -> None:
    """Rules only flag impossible/incomplete values. They never mutate extracted facts."""
    today = date.today().isoformat()
    for diagnosis in record.diagnoses:
        if diagnosis.date and diagnosis.date > today:
            record.verification_queue.append(_item("diagnosis.date", diagnosis.date, "future clinical date", diagnosis.source_ref, "medical_review"))
    for lab in record.laboratory_results:
        if lab.value is not None and lab.value < 0:
            record.verification_queue.append(_item("laboratory_result.value", str(lab.value), "negative lab value indicates a parsing error", lab.source_ref, "medical_review"))
        if lab.date and lab.date > today:
            record.verification_queue.append(_item("laboratory_result.date", lab.date, "future clinical date", lab.source_ref, "medical_review"))
    for medication in record.medications:
        if not medication.dose:
            record.verification_queue.append(_item("medication.dose", medication.name_original, "dose missing; do not infer", medication.source_ref, "medical_review"))
        if not medication.frequency:
            record.verification_queue.append(_item("medication.frequency", medication.name_original, "frequency missing; do not infer", medication.source_ref, "medical_review"))
    for growth in record.growth_measurements:
        if growth.date and growth.date > today:
            record.verification_queue.append(_item("growth_measurement.date", growth.date, "future clinical date", growth.source, "medical_review"))
