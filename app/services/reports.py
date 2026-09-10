from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from uuid import uuid4

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.config import Settings, settings
from app.repository import RecordRepository
from app.schemas import ReportRecord, ReportType, StructuredRecord

PROMPT_VERSION = "physician_report_v1"
MODEL_NAME = "structured_record_renderer"


def _safe_component(value: str | None) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "unknown")).strip("._")
    return clean[:80] or "unknown"


def _value(provenanced) -> str:
    if not provenanced or provenanced.value in {None, ""}:
        return "Not documented"
    return str(provenanced.value)


def _clinical_summary(record: StructuredRecord) -> str:
    documented = [d.term_original for d in record.diagnoses if d.type == "documented"]
    probable = [d.term_original for d in record.diagnoses if d.type != "documented"]
    parts: list[str] = []
    if documented:
        parts.append("Documented diagnoses: " + "; ".join(documented) + ".")
    if probable:
        parts.append("Qualified or non-confirmed diagnoses: " + "; ".join(probable) + ".")
    if not parts:
        parts.append("No clearly documented diagnosis was extracted from the available structured record.")
    if record.timeline:
        parts.append(f"{len(record.timeline)} dated timeline event(s) are available.")
    return " ".join(parts)


def erp_summary_payload(record: StructuredRecord) -> dict:
    return {
        "patient": record.patient.model_dump(mode="json"),
        "active_problems": [item.model_dump(mode="json") for item in record.problem_list],
        "relevant_history": [{"date": item.date, "event": item.event, "summary": item.summary} for item in record.timeline],
        "key_labs": [item.model_dump(mode="json") for item in record.laboratory_results],
        "medications": [item.model_dump(mode="json") for item in record.medications],
        "allergies": None,
        "growth": [item.model_dump(mode="json") for item in record.growth_measurements],
        "follow_up": [item.model_dump(mode="json") for item in record.verification_queue],
        "verification_items": [item.model_dump(mode="json") for item in record.verification_queue],
    }


def grounding_check(record: StructuredRecord, report_type: ReportType) -> list[str]:
    """Check that the report's source record contains every fact the renderer is allowed to mention."""
    problems: list[str] = []
    if report_type == ReportType.ERP_SUMMARY:
        payload = erp_summary_payload(record)
        if payload["key_labs"] != [item.model_dump(mode="json") for item in record.laboratory_results]:
            problems.append("laboratory payload diverged from structured record")
        if payload["medications"] != [item.model_dump(mode="json") for item in record.medications]:
            problems.append("medication payload diverged from structured record")
        if payload["active_problems"] != [item.model_dump(mode="json") for item in record.problem_list]:
            problems.append("problem list diverged from structured record")
    # Source refs are mandatory on all clinical facts in the canonical schema.
    for field_name, values in (("diagnosis", record.diagnoses), ("medication", record.medications), ("laboratory_result", record.laboratory_results)):
        for value in values:
            source = getattr(value, "source_ref", None)
            if not source or not source.source_file or source.source_page < 1:
                problems.append(f"{field_name} has no valid source reference")
    return problems


class ReportGenerator:
    def __init__(self, repository: RecordRepository, configuration: Settings = settings):
        self.repository = repository
        self.configuration = configuration
        self.template_env = Environment(loader=FileSystemLoader(str(Path(__file__).parents[2] / "templates" / "reports")), autoescape=select_autoescape(["html", "xml"]))

    def generate(self, patient_id: str, report_type: ReportType, created_by: str = "hospital-user") -> ReportRecord:
        record = self.repository.get_record(patient_id)
        if not record:
            raise ValueError("patient record not found")
        grounding_errors = grounding_check(record, report_type)
        version = self.repository.next_report_version(patient_id, report_type)
        report_dir = self.configuration.output_root / patient_id / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"{_safe_component(_value(record.patient.full_name) if record.patient.full_name else patient_id)}_{_safe_component(_value(record.patient.hospital_file_number) if record.patient.hospital_file_number else patient_id)}"
        suffix = "Detailed_Medical_Report" if report_type == ReportType.DETAILED else "ERP_Summary"
        pdf_path = report_dir / f"{base_name}_{suffix}_v{version}.pdf"
        html_path = report_dir / f"{base_name}_{suffix}_v{version}.html"
        text_path = report_dir / f"{base_name}_{suffix}_v{version}.txt"
        json_path = report_dir / f"{base_name}_{suffix}_v{version}.json" if report_type == ReportType.ERP_SUMMARY else None
        created_at = datetime.now(timezone.utc)
        context = self._context(record, version, created_at)
        template_name = "detailed_medical_report.html" if report_type == ReportType.DETAILED else "erp_summary.html"
        rendered_html = self.template_env.get_template(template_name).render(**context)
        html_path.write_text(rendered_html, encoding="utf-8")
        text = self._render_pdf(record, report_type, context, pdf_path)
        text_path.write_text(text, encoding="utf-8")
        if json_path:
            json_path.write_text(json.dumps(erp_summary_payload(record), indent=2), encoding="utf-8")
        source_hash = sha256(json.dumps(record.model_dump(mode="json"), sort_keys=True).encode("utf-8")).hexdigest()
        status = "needs_review" if grounding_errors or any(item.review_status.value == "pending" for item in record.verification_queue) else "ready"
        report = ReportRecord(report_id=str(uuid4()), patient_id=patient_id, patient_name=_value(record.patient.full_name) if record.patient.full_name else None, hospital_file_number=_value(record.patient.hospital_file_number) if record.patient.hospital_file_number else None, report_type=report_type, version=version, file_path=str(pdf_path), text_path=str(text_path), json_path=str(json_path) if json_path else None, status=status, created_at=created_at, created_by=created_by, model_name=MODEL_NAME, prompt_version=PROMPT_VERSION, source_record_hash=source_hash, review_status="pending" if status == "needs_review" else "not_required", source_file_ids=[item.file_id for item in record.source_files])
        source_links = [(page.file_id, page.page_number) for page in record.source_pages]
        if not source_links:
            source_links = [(item.file_id, None) for item in record.source_files]
        return self.repository.create_report(report, source_links)

    def _context(self, record: StructuredRecord, version: int, created_at: datetime) -> dict:
        return {"patient": record.patient, "source_files": record.source_files, "timeline": record.timeline, "problem_list": record.problem_list, "diagnoses": record.diagnoses, "labs": record.laboratory_results, "medications": record.medications, "growth": record.growth_measurements, "radiology": record.radiology_reports, "verification": record.verification_queue, "clinical_summary": _clinical_summary(record), "version": version, "created_at": created_at.strftime("%Y-%m-%d %H:%M UTC"), "value": _value}

    def _render_pdf(self, record: StructuredRecord, report_type: ReportType, context: dict, destination: Path) -> str:
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#0E4F64"), fontSize=18, spaceAfter=8))
        styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], textColor=colors.HexColor("#176B80"), spaceBefore=10, spaceAfter=4))
        body = styles["BodyText"]
        story = [Paragraph("Detailed Medical Report" if report_type == ReportType.DETAILED else "ERP Physician Summary", styles["ReportTitle"]), Paragraph(f"Patient ID: {escape(record.patient.patient_id)} &nbsp; | &nbsp; Version {context['version']}", body), Spacer(1, 6)]
        sections = [("Patient Identification", f"Name: {escape(_value(record.patient.full_name))}<br/>Hospital file number: {escape(_value(record.patient.hospital_file_number))}<br/>Sex: {escape(_value(record.patient.sex))}<br/>Date of birth: {escape(_value(record.patient.date_of_birth))}"), ("Clinical Summary", escape(context["clinical_summary"])), ("Active Problems", "; ".join(escape(item.problem) for item in record.problem_list) or "No documented active problems extracted."), ("Key Investigations", "; ".join(escape(item.value_text or str(item.value)) for item in record.laboratory_results) or "No structured laboratory values extracted."), ("Medication History", "; ".join(escape(item.name_original or "Unidentified medication") for item in record.medications) or "No structured medications extracted."), ("Growth / Relevant Vitals", "; ".join(f"{item.date or 'Date not documented'}: {item.weight_kg or '-'} kg, {item.height_cm or '-'} cm" for item in record.growth_measurements) or "No structured growth measurements extracted."), ("Allergy Status", "No allergy status was extracted; this is not evidence of no allergy."), ("Follow-up / Verification Needs", f"{len(record.verification_queue)} item(s) require human review before clinical or ERP use." if record.verification_queue else "No pending verification items.")]
        if report_type == ReportType.DETAILED:
            sections.insert(2, ("Clinical Timeline", "<br/>".join(f"{escape(item.date or 'Date not documented')} — {escape(item.event)}: {escape(item.summary)}" for item in record.timeline) or "No dated clinical timeline events were extracted."))
            sections.insert(3, ("Diagnoses", "<br/>".join(f"{escape(item.term_original)} ({escape(item.type)}, {escape(item.certainty)})" for item in record.diagnoses) or "No structured diagnoses extracted."))
        for heading, content in sections:
            story.extend([Paragraph(heading, styles["Section"]), Paragraph(content, body)])
        if report_type == ReportType.DETAILED:
            story.append(PageBreak())
            story.append(Paragraph("Appendix: Original Source Documents", styles["Section"]))
            story.append(Paragraph("Original source pages are appended without cropping clinically meaningful content.", body))
            for source in record.source_files:
                if source.file_type.lower() in {"jpg", "jpeg", "png"} and Path(source.original_path).is_file():
                    story.append(Spacer(1, 6))
                    story.append(Paragraph(escape(source.original_filename), body))
                    try:
                        from PIL import Image as PILImage
                        with PILImage.open(source.original_path) as image:
                            width, height = image.size
                        max_w, max_h = 170 * mm, 230 * mm
                        scale = min(max_w / width, max_h / height)
                        story.append(Image(source.original_path, width=width * scale, height=height * scale))
                    except Exception:
                        story.append(Paragraph("Image could not be embedded; inspect the original source file.", body))
                elif source.file_type.lower() == "pdf":
                    story.append(Paragraph(escape(f"{source.original_filename} — original PDF pages appended after this report."), body))
        # Keep the intermediate file inside the mounted output volume; the
        # container runs as a non-root user and cannot write to /var/lib/tmp.
        temp_dir = self.configuration.output_root / ".tmp_pdfs"
        temp_dir.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(suffix=".pdf", dir=temp_dir, delete=False) as temp:
            temporary_pdf = Path(temp.name)
        doc = SimpleDocTemplate(str(temporary_pdf), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=18 * mm, bottomMargin=18 * mm)
        doc.build(story, onFirstPage=self._footer(record.patient.patient_id), onLaterPages=self._footer(record.patient.patient_id))
        if report_type == ReportType.DETAILED:
            writer = PdfWriter()
            for page in PdfReader(str(temporary_pdf)).pages:
                writer.add_page(page)
            for source in record.source_files:
                if source.file_type.lower() == "pdf" and Path(source.original_path).is_file():
                    for page in PdfReader(source.original_path).pages:
                        writer.add_page(page)
            with destination.open("wb") as handle:
                writer.write(handle)
        else:
            temporary_pdf.replace(destination)
        temporary_pdf.unlink(missing_ok=True)
        return self._plain_text(record, report_type)

    @staticmethod
    def _footer(patient_id: str):
        def draw(canvas, doc):
            canvas.saveState()
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(colors.HexColor("#536B78"))
            canvas.drawString(18 * mm, 10 * mm, f"Medical Records | Patient {patient_id}")
            canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
            canvas.restoreState()
        return draw

    @staticmethod
    def _plain_text(record: StructuredRecord, report_type: ReportType) -> str:
        lines = ["ERP Physician Summary" if report_type == ReportType.ERP_SUMMARY else "Detailed Medical Report", f"Patient ID: {record.patient.patient_id}", "", "Clinical Summary", _clinical_summary(record), "", "Key Investigations", "; ".join(item.value_text or str(item.value) for item in record.laboratory_results) or "No structured laboratory values extracted.", "", "Medication History", "; ".join(item.name_original or "Unidentified medication" for item in record.medications) or "No structured medications extracted.", "", "Active Problems", "; ".join(item.problem for item in record.problem_list) or "No documented active problems extracted.", "", "Follow-up / Verification Needs", f"{len(record.verification_queue)} item(s) require human review before clinical or ERP use." if record.verification_queue else "No pending verification items."]
        return "\n".join(lines)
