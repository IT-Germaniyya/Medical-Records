from __future__ import annotations

from collections.abc import Iterable
from uuid import uuid4
from datetime import datetime, timedelta
from pathlib import Path
import shutil

from sqlalchemy import delete, select
from sqlalchemy import func, or_
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.models import (
    AuditEventModel,
    ClinicalInterpretationModel,
    DiagnosisModel,
    EncounterModel,
    GrowthMeasurementModel,
    LaboratoryResultModel,
    MedicationModel,
    ObservationModel,
    PatientModel,
    PatientRecordModel,
    ReportGenerationJobModel,
    ReportModel,
    ReportSourceLinkModel,
    ProblemListModel,
    RadiologyReportModel,
    SourceFileModel,
    SourcePageModel,
    VerificationItemModel,
)
from app.schemas import AuditEvent, JobStatus, PatientSearchResult, ReportRecord, ReportType, ReviewStatus, StructuredRecord


class DuplicateSourceError(ValueError):
    """A checksum is already bound to a different patient record."""


class RecordRepository:
    def __init__(self, session_factory: sessionmaker[Session], *, storage_root: Path | None = None, output_root: Path | None = None):
        self.session_factory = session_factory
        self.storage_root = Path(storage_root) if storage_root is not None else settings.storage_root
        self.output_root = Path(output_root) if output_root is not None else settings.output_root

    def existing_checksum(self, checksum: str) -> SourceFileModel | None:
        with self.session_factory() as session:
            return session.scalar(select(SourceFileModel).where(SourceFileModel.checksum == checksum))

    def ensure_patient(self, patient_id: str) -> None:
        """Create a lightweight patient shell so queued jobs satisfy the FK constraint."""
        with self.session_factory() as session:
            if session.get(PatientModel, patient_id) is None:
                session.add(PatientModel(patient_id=patient_id, processing_status="pending", review_status="pending"))
                session.commit()

    def mark_patient_failed(self, patient_id: str) -> None:
        """Prevent a failed background upload from remaining indefinitely in pending extraction."""
        with self.session_factory() as session:
            patient = session.get(PatientModel, patient_id)
            if patient is not None:
                patient.processing_status = "failed"
                patient.review_status = "needs_retry"
                session.commit()

    def save(self, record: StructuredRecord, pages: Iterable[dict[str, object]]) -> None:
        patient_id = record.patient.patient_id
        sources = record.source_files
        with self.session_factory() as session:
            for source in sources:
                duplicate = session.scalar(
                    select(SourceFileModel).where(SourceFileModel.checksum == source.checksum)
                )
                if duplicate and duplicate.patient_id != patient_id:
                    raise DuplicateSourceError(
                        f"checksum for {source.original_filename!r} already belongs to a different patient"
                    )

            patient = session.get(PatientModel, patient_id)
            hfn = self._preferred_value(record.patient.hospital_file_number)
            full_name = self._preferred_value(record.patient.full_name)
            sex = self._preferred_value(record.patient.sex)
            date_of_birth = self._preferred_value(record.patient.date_of_birth)
            nationality = self._preferred_value(record.patient.nationality)
            address = self._preferred_value(record.patient.address)
            review_status = "pending" if record.verification_queue else "complete"
            if patient is None:
                session.add(PatientModel(patient_id=patient_id, hospital_file_number=hfn, full_name=full_name, sex=sex, date_of_birth=date_of_birth, nationality=nationality, address=address, processing_status=record.patient_status, review_status=review_status))
            else:
                patient.hospital_file_number = hfn or patient.hospital_file_number
                patient.full_name = full_name or patient.full_name
                patient.sex = sex or patient.sex
                patient.date_of_birth = date_of_birth or patient.date_of_birth
                patient.nationality = nationality or patient.nationality
                patient.address = address or patient.address
                patient.processing_status = record.patient_status
                patient.review_status = review_status

            source_ids = list(session.scalars(select(SourceFileModel.file_id).where(SourceFileModel.patient_id == patient_id)))
            if source_ids:
                session.execute(delete(SourcePageModel).where(SourcePageModel.file_id.in_(source_ids)))
            for model in (
                SourceFileModel, EncounterModel, DiagnosisModel, MedicationModel,
                LaboratoryResultModel, RadiologyReportModel, GrowthMeasurementModel,
                ProblemListModel, VerificationItemModel, AuditEventModel,
            ):
                session.execute(delete(model).where(model.patient_id == patient_id))
            # Ensure replacement rows do not collide with this patient's previous checksum rows.
            session.flush()

            for source in sources:
                session.add(SourceFileModel(
                    file_id=source.file_id, patient_id=patient_id,
                    original_filename=source.original_filename, file_type=source.file_type,
                    page_count=source.page_count, checksum=source.checksum,
                    processing_status=source.processing_status, original_path=source.original_path,
                    processed_path=source.processed_path, ingested_at=source.ingested_at,
                    archive_filename=source.archive_filename, archive_type=source.archive_type,
                    original_relative_path=source.original_relative_path,
                    extracted_filename=source.extracted_filename,
                    source_order=source.source_order,
                ))
            for page in pages:
                session.add(SourcePageModel(
                    page_id=str(page["page_id"]), file_id=str(page["file_id"]),
                    page_number=int(page["page_number"]), document_type=str(page["document_type"]),
                    classification_confidence=float(page["classification_confidence"]),
                    extraction_status=str(page["extraction_status"]),
                ))
            for encounter in record.encounters:
                session.add(EncounterModel(encounter_id=encounter.encounter_id, patient_id=patient_id, encounter_date=encounter.date, payload=encounter.model_dump(mode="json")))
            for diagnosis in record.diagnoses:
                session.add(DiagnosisModel(diagnosis_id=str(uuid4()), patient_id=patient_id, term_original=diagnosis.term_original, certainty=diagnosis.certainty, confidence=diagnosis.confidence, payload=diagnosis.model_dump(mode="json")))
            for medication in record.medications:
                session.add(MedicationModel(medication_id=str(uuid4()), patient_id=patient_id, name_normalized=medication.name_normalized, confidence=medication.confidence, payload=medication.model_dump(mode="json")))
            for lab in record.laboratory_results:
                session.add(LaboratoryResultModel(result_id=str(uuid4()), patient_id=patient_id, test_name=lab.test_name_original, value=lab.value, unit=lab.unit, confidence=lab.confidence, payload=lab.model_dump(mode="json")))
            for report in record.radiology_reports:
                session.add(RadiologyReportModel(report_id=str(uuid4()), patient_id=patient_id, study_type=report.study_type, confidence=report.confidence, payload=report.model_dump(mode="json")))
            for growth in record.growth_measurements:
                session.add(GrowthMeasurementModel(measurement_id=str(uuid4()), patient_id=patient_id, measurement_date=growth.date, confidence=growth.confidence, payload=growth.model_dump(mode="json")))
            for problem in record.problem_list:
                session.add(ProblemListModel(problem_id=str(uuid4()), patient_id=patient_id, problem=problem.problem, status=problem.status, source_type=problem.source_type, payload=problem.model_dump(mode="json")))
            for item in record.verification_queue:
                session.add(VerificationItemModel(item_id=item.item_id, patient_id=patient_id, field=item.field, confidence=item.confidence, reason=item.reason, review_status=item.review_status.value, payload=item.model_dump(mode="json")))
            for event in record.audit_events:
                session.add(AuditEventModel(fact_id=event.fact_id, patient_id=patient_id, field=event.field, source_file=event.source_file, source_page=event.source_page, confidence=event.confidence, review_status=event.review_status.value, payload=event.model_dump(mode="json")))

            serialized = record.model_dump(mode="json")
            existing = session.get(PatientRecordModel, patient_id)
            if existing:
                existing.payload = serialized
                existing.processing_status = record.patient_status
                existing.schema_version = record.schema_version
            else:
                session.add(PatientRecordModel(patient_id=patient_id, schema_version=record.schema_version, processing_status=record.patient_status, payload=serialized))
            session.commit()

    def get_record(self, patient_id: str) -> StructuredRecord | None:
        with self.session_factory() as session:
            persisted = session.get(PatientRecordModel, patient_id)
            return StructuredRecord.model_validate(persisted.payload) if persisted else None

    def delete_patient(self, patient_id: str) -> bool:
        """Delete one patient and every persisted record that belongs to them.

        Foreign keys are intentionally deleted in dependency order because the
        existing schema predates database-level ON DELETE CASCADE constraints.
        Source and generated report files are removed only from the configured
        per-patient storage/output directories after the transaction commits.
        """
        with self.session_factory() as session:
            patient = session.get(PatientModel, patient_id)
            if patient is None:
                return False
            self._delete_patient_rows(session, patient_id)
            session.delete(patient)
            session.commit()
        self._remove_patient_directories(patient_id)
        return True

    def delete_all_patients(self) -> int:
        """Delete all patients and their records, returning the count removed."""
        with self.session_factory() as session:
            patient_ids = list(session.scalars(select(PatientModel.patient_id)))
            if not patient_ids:
                return 0
            for patient_id in patient_ids:
                self._delete_patient_rows(session, patient_id)
            session.execute(delete(PatientModel).where(PatientModel.patient_id.in_(patient_ids)))
            session.commit()
        for patient_id in patient_ids:
            self._remove_patient_directories(patient_id)
        return len(patient_ids)

    @staticmethod
    def _delete_patient_rows(session: Session, patient_id: str) -> None:
        """Remove all child rows for a patient in foreign-key-safe order."""
        source_ids = list(session.scalars(select(SourceFileModel.file_id).where(SourceFileModel.patient_id == patient_id)))
        report_ids = list(session.scalars(select(ReportModel.report_id).where(ReportModel.patient_id == patient_id)))
        if report_ids:
            session.execute(delete(ReportSourceLinkModel).where(ReportSourceLinkModel.report_id.in_(report_ids)))
        if source_ids:
            session.execute(delete(SourcePageModel).where(SourcePageModel.file_id.in_(source_ids)))

        for model in (
            ReportGenerationJobModel,
            ReportModel,
            SourceFileModel,
            PatientRecordModel,
            EncounterModel,
            DiagnosisModel,
            MedicationModel,
            LaboratoryResultModel,
            RadiologyReportModel,
            ObservationModel,
            GrowthMeasurementModel,
            ClinicalInterpretationModel,
            ProblemListModel,
            VerificationItemModel,
            AuditEventModel,
        ):
            session.execute(delete(model).where(model.patient_id == patient_id))

    def _remove_patient_directories(self, patient_id: str) -> None:
        """Remove only the exact per-patient directories under configured roots."""
        if not patient_id or Path(patient_id).name != patient_id or patient_id in {".", ".."}:
            return
        for root in (self.storage_root, self.output_root):
            root_resolved = root.resolve()
            patient_dir = (root / patient_id).resolve()
            if patient_dir.parent == root_resolved and patient_dir.name == patient_id and patient_dir.is_dir():
                shutil.rmtree(patient_dir, ignore_errors=True)

    def source_path(self, patient_id: str, file_id: str) -> str | None:
        with self.session_factory() as session:
            source = session.get(SourceFileModel, file_id)
            if source and source.patient_id == patient_id:
                return source.original_path
            return None

    def source_pages(self, patient_id: str) -> list[dict]:
        with self.session_factory() as session:
            rows = session.execute(
                select(SourcePageModel).join(SourceFileModel, SourcePageModel.file_id == SourceFileModel.file_id).where(SourceFileModel.patient_id == patient_id).order_by(SourceFileModel.source_order, SourceFileModel.original_filename, SourcePageModel.page_number)
            ).scalars()
            return [{"page_id": row.page_id, "file_id": row.file_id, "page_number": row.page_number, "document_type": row.document_type, "classification_confidence": row.classification_confidence, "extraction_status": row.extraction_status} for row in rows]

    def search_patients(self, query: str = "", limit: int = 50) -> list[PatientSearchResult]:
        with self.session_factory() as session:
            statement = select(PatientModel).order_by(PatientModel.last_processed_at.desc()).limit(min(max(limit, 1), 100))
            query = query.strip()
            if query:
                token = f"%{query}%"
                statement = statement.where(or_(PatientModel.patient_id.ilike(token), PatientModel.full_name.ilike(token), PatientModel.hospital_file_number.ilike(token)))
            rows = session.scalars(statement).all()
            return [PatientSearchResult(patient_id=row.patient_id, patient_name=row.full_name, sex=row.sex, date_of_birth=row.date_of_birth, hospital_file_number=row.hospital_file_number, last_processed_at=row.last_processed_at, processing_status=row.processing_status, review_status=row.review_status) for row in rows]

    def dashboard_metrics(self) -> dict[str, object]:
        with self.session_factory() as session:
            total = session.scalar(select(func.count(PatientModel.patient_id))) or 0
            pending_review = session.scalar(select(func.count(PatientModel.patient_id)).where(PatientModel.review_status == "pending")) or 0
            pending_extraction = session.scalar(select(func.count(PatientModel.patient_id)).where(PatientModel.processing_status.in_(["pending", "processing"]))) or 0
            processed = session.scalar(select(func.count(PatientModel.patient_id)).where(PatientModel.processing_status.in_(["completed", "completed_with_warnings"]))) or 0
            reports = session.scalar(select(func.count(ReportModel.report_id))) or 0
            failed = session.scalar(select(func.count(PatientModel.patient_id)).where(PatientModel.processing_status == "failed")) or 0
            recent = self.search_patients("", 8)
            recent_reports = session.scalars(select(ReportModel).order_by(ReportModel.created_at.desc()).limit(8)).all()
            return {"total_patients": total, "processed_patients": processed, "pending_extraction": pending_extraction, "pending_review": pending_review, "reports_generated": reports, "failed_processing": failed, "recent_patients": [item.model_dump(mode="json") for item in recent], "recent_reports": [self._report_schema(session, item).model_dump(mode="json") for item in recent_reports]}

    def review_item(self, patient_id: str, item_id: str, status: ReviewStatus, reviewer_id: str | None, edited_value: str | None = None) -> StructuredRecord | None:
        """Record a reviewer decision without changing an unverified clinical fact automatically."""
        with self.session_factory() as session:
            persisted = session.get(PatientRecordModel, patient_id)
            item_model = session.get(VerificationItemModel, item_id)
            if not persisted or not item_model or item_model.patient_id != patient_id:
                return None
            record = StructuredRecord.model_validate(persisted.payload)
            item = next((candidate for candidate in record.verification_queue if candidate.item_id == item_id), None)
            if item is None:
                return None
            item.review_status = status
            if edited_value is not None:
                item.proposed_value = edited_value
            item_model.review_status = status.value
            payload = item_model.payload
            payload["review_status"] = status.value
            if edited_value is not None:
                payload["proposed_value"] = edited_value
            item_model.payload = payload
            record.audit_events.append(AuditEvent(
                fact_id=str(uuid4()), patient_id=patient_id, field=f"review.{item.field}",
                value=item.proposed_value, source_file=item.source_ref.source_file,
                source_page=item.source_ref.source_page, confidence=item.confidence,
                reviewed_by=reviewer_id, review_status=status,
            ))
            persisted.payload = record.model_dump(mode="json")
            session.add(AuditEventModel(
                fact_id=record.audit_events[-1].fact_id, patient_id=patient_id,
                field=record.audit_events[-1].field, source_file=item.source_ref.source_file,
                source_page=item.source_ref.source_page, confidence=item.confidence,
                review_status=status.value, payload=record.audit_events[-1].model_dump(mode="json"),
            ))
            session.commit()
            return record

    def create_report(self, report: ReportRecord, source_links: list[tuple[str, int | None]]) -> ReportRecord:
        with self.session_factory() as session:
            session.add(ReportModel(report_id=report.report_id, patient_id=report.patient_id, report_type=report.report_type.value, version=report.version, file_path=report.file_path, text_path=report.text_path, json_path=report.json_path, status=report.status, created_at=report.created_at, created_by=report.created_by, model_name=report.model_name, prompt_version=report.prompt_version, source_record_hash=report.source_record_hash, review_status=report.review_status, ai_review_version=report.ai_review_version))
            for file_id, page_number in source_links:
                session.add(ReportSourceLinkModel(link_id=str(uuid4()), report_id=report.report_id, file_id=file_id, page_number=page_number))
            session.commit()
            return report

    def next_report_version(self, patient_id: str, report_type: ReportType) -> int:
        with self.session_factory() as session:
            current = session.scalar(select(func.max(ReportModel.version)).where(ReportModel.patient_id == patient_id, ReportModel.report_type == report_type.value))
            return int(current or 0) + 1

    def list_reports(self, query: str = "", report_type: ReportType | None = None, limit: int = 100, patient_id: str | None = None, date: str | None = None) -> list[ReportRecord]:
        with self.session_factory() as session:
            statement = select(ReportModel).join(PatientModel, ReportModel.patient_id == PatientModel.patient_id).order_by(ReportModel.created_at.desc()).limit(min(max(limit, 1), 200))
            if report_type:
                statement = statement.where(ReportModel.report_type == report_type.value)
            if patient_id:
                statement = statement.where(ReportModel.patient_id == patient_id)
            if date:
                try:
                    start = datetime.strptime(date, "%Y-%m-%d")
                    statement = statement.where(ReportModel.created_at >= start, ReportModel.created_at < start + timedelta(days=1))
                except ValueError:
                    pass
            if query.strip():
                token = f"%{query.strip()}%"
                statement = statement.where(or_(ReportModel.patient_id.ilike(token), PatientModel.full_name.ilike(token), PatientModel.hospital_file_number.ilike(token)))
            rows = session.scalars(statement).all()
            return [self._report_schema(session, row) for row in rows]

    def get_report(self, report_id: str) -> ReportRecord | None:
        with self.session_factory() as session:
            row = session.get(ReportModel, report_id)
            return self._report_schema(session, row) if row else None

    def create_job(self, job: JobStatus, report_type: str = "") -> JobStatus:
        with self.session_factory() as session:
            session.add(ReportGenerationJobModel(job_id=job.job_id, patient_id=job.patient_id, report_type=report_type, status=job.status, progress=job.progress, message=job.message, error=job.error, created_at=job.created_at, updated_at=job.updated_at))
            session.commit()
        return job

    def update_job(self, job_id: str, *, status: str, progress: int, message: str, error: str | None = None) -> JobStatus | None:
        with self.session_factory() as session:
            row = session.get(ReportGenerationJobModel, job_id)
            if not row:
                return None
            row.status, row.progress, row.message, row.error = status, progress, message, error
            session.commit()
            return JobStatus(job_id=row.job_id, patient_id=row.patient_id, status=row.status, progress=row.progress, message=row.message, error=row.error, created_at=row.created_at, updated_at=row.updated_at)

    def get_job(self, job_id: str) -> JobStatus | None:
        with self.session_factory() as session:
            row = session.get(ReportGenerationJobModel, job_id)
            return JobStatus(job_id=row.job_id, patient_id=row.patient_id, status=row.status, progress=row.progress, message=row.message, error=row.error, created_at=row.created_at, updated_at=row.updated_at) if row else None

    @staticmethod
    def _report_schema(session: Session, row: ReportModel) -> ReportRecord:
        links = session.scalars(select(ReportSourceLinkModel).where(ReportSourceLinkModel.report_id == row.report_id)).all()
        patient = session.get(PatientModel, row.patient_id)
        return ReportRecord(report_id=row.report_id, patient_id=row.patient_id, patient_name=patient.full_name if patient else None, hospital_file_number=patient.hospital_file_number if patient else None, report_type=ReportType(row.report_type), version=row.version, file_path=row.file_path, text_path=row.text_path, json_path=row.json_path, status=row.status, created_at=row.created_at, created_by=row.created_by, model_name=row.model_name, prompt_version=row.prompt_version, source_record_hash=row.source_record_hash, review_status=row.review_status, source_file_ids=[link.file_id for link in links], ai_review_version=row.ai_review_version)

    @staticmethod
    def _preferred_value(value):
        return str(value.value) if value and value.value is not None else None
