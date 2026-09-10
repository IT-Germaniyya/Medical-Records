from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Iterable

from app.exports.fhir import fhir_bundle
from app.schemas import StructuredRecord
from app.services.clinical import physician_summary


def _write_csv(path: Path, items: Iterable[dict]) -> None:
    rows = list(items)
    fields = sorted({key for row in rows for key in row}) or ["no_records"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(record: StructuredRecord, output_root: Path) -> Path:
    target = output_root / record.patient.patient_id
    target.mkdir(parents=True, exist_ok=True)
    payload = record.model_dump(mode="json")
    (target / "structured_record.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (target / "timeline.json").write_text(json.dumps(payload["timeline"], indent=2), encoding="utf-8")
    (target / "problem_list.json").write_text(json.dumps(payload["problem_list"], indent=2), encoding="utf-8")
    (target / "verification_queue.json").write_text(json.dumps(payload["verification_queue"], indent=2), encoding="utf-8")
    (target / "audit.json").write_text(json.dumps(payload["audit_events"], indent=2), encoding="utf-8")
    summary = (record.ai_clinical_review or {}).get("erp_summary_markdown") or physician_summary(record)
    (target / "physician_summary.txt").write_text(summary, encoding="utf-8")
    (target / "physician_summary.html").write_text(f"<!doctype html><html><body><pre>{html.escape(summary)}</pre></body></html>", encoding="utf-8")
    (target / "patient.json").write_text(json.dumps(payload["patient"], indent=2), encoding="utf-8")
    if record.ai_clinical_review:
        (target / "ai_clinical_review.json").write_text(json.dumps(record.ai_clinical_review, indent=2), encoding="utf-8")
    (target / "fhir_bundle.json").write_text(json.dumps(fhir_bundle(record), indent=2), encoding="utf-8")
    _write_csv(target / "patients.csv", [{"patient_id": record.patient.patient_id}])
    _write_csv(target / "encounters.csv", [item.model_dump(mode="json") for item in record.encounters])
    _write_csv(target / "diagnoses.csv", [item.model_dump(mode="json") for item in record.diagnoses])
    _write_csv(target / "medications.csv", [item.model_dump(mode="json") for item in record.medications])
    _write_csv(target / "labs.csv", [item.model_dump(mode="json") for item in record.laboratory_results])
    _write_csv(target / "radiology.csv", [item.model_dump(mode="json") for item in record.radiology_reports])
    _write_csv(target / "vitals.csv", [])
    _write_csv(target / "growth.csv", [item.model_dump(mode="json") for item in record.growth_measurements])
    _write_csv(target / "review_queue.csv", [item.model_dump(mode="json") for item in record.verification_queue])
    return target
