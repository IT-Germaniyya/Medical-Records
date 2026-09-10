from __future__ import annotations

from app.schemas import StructuredRecord


def fhir_bundle(record: StructuredRecord) -> dict:
    """Minimal FHIR-style adapter. No terminology codes are invented."""
    entries: list[dict] = [{"resource": {"resourceType": "Patient", "id": record.patient.patient_id}}]
    for diagnosis in record.diagnoses:
        entries.append({"resource": {
            "resourceType": "Condition", "subject": {"reference": f"Patient/{record.patient.patient_id}"},
            "code": {"text": diagnosis.term_original},
            "verificationStatus": {"text": diagnosis.certainty},
        }})
    for lab in record.laboratory_results:
        resource = {"resourceType": "Observation", "subject": {"reference": f"Patient/{record.patient.patient_id}"}, "code": {"text": lab.test_name_original}, "valueQuantity": {"value": lab.value}}
        if lab.unit:
            resource["valueQuantity"]["unit"] = lab.unit
        entries.append({"resource": resource})
    for medication in record.medications:
        entries.append({"resource": {"resourceType": "MedicationStatement", "subject": {"reference": f"Patient/{record.patient.patient_id}"}, "medicationCodeableConcept": {"text": medication.name_original}}})
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}
