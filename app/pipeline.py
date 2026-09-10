from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import mimetypes
from pathlib import Path
import shutil
from uuid import uuid4
from typing import Any

from app.config import Settings, settings
from app.exports.writer import write_outputs
from app.repository import RecordRepository
from app.schemas import DocumentType, FactStatus, PatientDemographics, ProblemItem, ProvenancedValue, SourcePageInfo, SourceRef, StructuredRecord
from app.services.classification import classify
from app.services.clinical import build_timeline_and_problems
from app.services.extraction import extract_documented_facts, extract_identity_facts, source_ref, unreadable_page_item
from app.services.ingestion import ArchiveLimits, IngestionError, extract_archive, detect_archive_type, ingest_folder
from app.services.preprocessing import preprocess
from app.ai_schemas import PatientBundleItem, PatientLevelReview
from app.services.providers import MedicalAIProvider, PageAnalysis, PatientLevelAIError, provider_for_settings
from app.services.ai_diagnostics import AIDiagnosticsStore
from app.services.validation import apply_confidence_policy, apply_rule_validations


@dataclass
class ProcessingResult:
    record: StructuredRecord
    output_directory: Path
    resumed: bool = False
    warnings: list[str] = field(default_factory=list)


class PatientPipeline:
    def __init__(self, repository: RecordRepository, configuration: Settings = settings, provider: MedicalAIProvider | None = None):
        self.repository = repository
        self.configuration = configuration
        self.diagnostics = AIDiagnosticsStore(repository.session_factory)
        self.provider = provider or provider_for_settings(configuration, diagnostics=self.diagnostics)

    def process(self, source: Path, patient_id: str | None = None, resume: bool = True, demographics: dict[str, str | None] | None = None, source_mime_type: str | None = None) -> ProcessingResult:
        archive_result = None
        archive_root = None
        warnings: list[str] = []
        try:
            if self._use_patient_level_review() and hasattr(self.provider, "test_text_connection") and not self.diagnostics.is_ready(provider=getattr(self.provider, "name", "openai"), model=getattr(self.provider, "configured_model", self.configuration.openai_medical_model)):
                raise PatientLevelAIError("AI diagnostics are not ready: run Test AI Connection and Test Vision Extraction before processing patient records")
            if source.is_file():
                archive_kind = detect_archive_type(source, source_mime_type)
                if archive_kind:
                    limits = ArchiveLimits(
                        max_files=self.configuration.archive_max_files,
                        max_uncompressed_bytes=self.configuration.archive_max_uncompressed_bytes,
                        max_single_file_bytes=self.configuration.archive_max_single_file_bytes,
                        max_compression_ratio=self.configuration.archive_max_compression_ratio,
                        extraction_timeout_seconds=self.configuration.archive_extraction_timeout_seconds,
                        allow_nested_archives=self.configuration.allow_nested_archive_extraction,
                    )
                    archive_result = extract_archive(source, self.configuration.storage_root / "incoming", source_mime_type, limits)
                    archive_root = archive_result.root
                    source = archive_root
                    warnings.extend(archive_result.warnings)
            if not source.is_dir():
                raise IngestionError("input must be a patient folder or a ZIP/RAR archive")
            resolved_patient_id = patient_id or source.name
            if not resolved_patient_id:
                raise IngestionError("patient identifier is required")
            existing = self.repository.get_record(resolved_patient_id)
            if resume and existing and existing.patient_status in {"completed", "completed_with_warnings"}:
                return ProcessingResult(existing, self.configuration.output_root / resolved_patient_id, resumed=True, warnings=list(existing.processing_warnings))

            existing_checksums = {source_file.checksum for source_file in existing.source_files} if existing else set()
            ingested = ingest_folder(
                resolved_patient_id,
                source,
                self.configuration.storage_root,
                existing_checksums,
                archive_filename=archive_result.archive_filename if archive_result else None,
                archive_type=archive_result.archive_type if archive_result else None,
            )
            if archive_result and archive_result.unsupported_count:
                warnings.insert(0, f"{len(ingested)} files were processed. {archive_result.unsupported_count} unsupported files were skipped.")
            if archive_result and archive_result.extracted_count > len(ingested):
                warnings.append(f"{archive_result.extracted_count - len(ingested)} duplicate files were skipped.")
            record = existing.model_copy(deep=True) if existing else StructuredRecord(patient=PatientDemographics(patient_id=resolved_patient_id))
            if existing and not record.source_pages:
                record.source_pages = [SourcePageInfo.model_validate(item) for item in self.repository.source_pages(resolved_patient_id)]
            source_pages: list[dict[str, object]] = [item.model_dump(mode="json") for item in record.source_pages]
            self._apply_upload_demographics(record, demographics)
            processed_files = list(record.source_files)
            if self._use_patient_level_review():
                prepared = self._prepare_patient_files(resolved_patient_id, ingested, processed_files)
                existing_ids = {source_file.file_id for source_file in processed_files}
                processed_files.extend(source_file for _, _, source_file in prepared if source_file.file_id not in existing_ids)
                record.source_files = processed_files
                source_pages = self._ensure_source_pages(record, source_pages)
                bundle = self._patient_bundle(record, prepared)
                review = self.provider.review_patient(bundle)  # type: ignore[attr-defined]
                if review is None:
                    raise PatientLevelAIError("AI medical review could not be completed")
                self._apply_patient_review(record, review, resolved_patient_id)
                # The patient-level review has already inspected every source;
                # do not run the legacy per-page extraction path afterwards.
                ingested = []
            for item in ingested:
                result = preprocess(item, self.configuration.storage_root / resolved_patient_id / "processed")
                source_file = item.source_file.model_copy(update={"processed_path": str(result.processed_path), "processing_status": "processed"})
                processed_files.append(source_file)
                total_pages = source_file.page_count or 1
                for page_number in range(1, total_pages + 1):
                    filename_hint, _ = classify(source_file.original_filename, None)
                    page_analysis = self.provider.extract_page(
                        result.processed_path,
                        page_number,
                        source_filename=source_file.original_filename,
                        document_type_hint=filename_hint.value,
                    )
                    text = page_analysis.text
                    document_type, classification_confidence = classify(source_file.original_filename, text)
                    if page_analysis.structured:
                        try:
                            structured_type = DocumentType(page_analysis.structured.document_type.casefold())
                        except (ValueError, AttributeError):
                            structured_type = DocumentType.UNKNOWN
                        if structured_type != DocumentType.UNKNOWN:
                            document_type = structured_type
                            classification_confidence = page_analysis.confidence
                    status = "extracted" if text or page_analysis.structured else "needs_verification"
                    page_info = SourcePageInfo(
                        page_id=str(uuid4()), file_id=source_file.file_id, page_number=page_number,
                        document_type=document_type, classification_confidence=classification_confidence,
                        extraction_status=status,
                    )
                    record.source_pages.append(page_info)
                    source_pages.append(page_info.model_dump(mode="json"))
                    if page_analysis.structured:
                        identity = self._identity_from_ai(page_analysis, source_file.original_filename, page_number)
                        self._merge_document_identity(record, identity, resolved_patient_id)
                        if identity:
                            from app.services.extraction import _review
                            for field_name, candidate in identity.items():
                                if candidate.status != FactStatus.DOCUMENTED:
                                    record.verification_queue.append(_review(
                                        field_name,
                                        str(candidate.value),
                                        candidate.raw_text,
                                        candidate.confidence,
                                        "AI identity field is inferred or marked uncertain",
                                        candidate.source_ref,
                                        "medical_review",
                                    ))
                        extracted = extract_documented_facts(
                            resolved_patient_id,
                            text or "",
                            source_file.original_filename,
                            page_number,
                            document_type,
                            ai_page=page_analysis.structured,
                            model=page_analysis.model,
                            prompt_version=page_analysis.prompt_version,
                        )
                        record.diagnoses.extend(extracted.diagnoses)
                        record.medications.extend(extracted.medications)
                        record.laboratory_results.extend(extracted.labs)
                        record.radiology_reports.extend(extracted.radiology)
                        record.growth_measurements.extend(extracted.growth)
                        record.verification_queue.extend(extracted.verification)
                        record.audit_events.extend(extracted.audit)
                    elif text:
                        identity = extract_identity_facts(text, source_file.original_filename, page_number)
                        self._merge_document_identity(record, identity, resolved_patient_id)
                        extracted = extract_documented_facts(resolved_patient_id, text, source_file.original_filename, page_number, document_type)
                        record.diagnoses.extend(extracted.diagnoses)
                        record.medications.extend(extracted.medications)
                        record.laboratory_results.extend(extracted.labs)
                        record.radiology_reports.extend(extracted.radiology)
                        record.growth_measurements.extend(extracted.growth)
                        record.verification_queue.extend(extracted.verification)
                        record.audit_events.extend(extracted.audit)
                    else:
                        reason = page_analysis.error or "no machine-readable text extracted; configure validated OCR/vision provider or review source"
                        record.verification_queue.append(unreadable_page_item(source_file.original_filename, page_number, reason))
                for warning in [*item.warnings, *result.warnings]:
                    record.verification_queue.append(unreadable_page_item(source_file.original_filename, 1, warning))

            record.source_files = processed_files
            record.processing_warnings = list(dict.fromkeys([*record.processing_warnings, *warnings]))
            build_timeline_and_problems(record)
            apply_confidence_policy(record, self.configuration.confidence_high, self.configuration.confidence_medium, self.configuration.high_risk_confidence)
            apply_rule_validations(record)
            synthesize = getattr(self.provider, "synthesize", None)
            if callable(synthesize):
                # Stage C receives canonical structured facts only. Raw images,
                # OCR text, and unvalidated model output are deliberately not
                # sent to the reasoning model.
                synthesis_input = record.model_dump(
                    mode="json",
                    include={"encounters", "diagnoses", "medications", "laboratory_results", "radiology_reports", "growth_measurements", "timeline", "problem_list", "verification_queue"},
                )
                synthesis = synthesize(synthesis_input)
                if synthesis is not None:
                    record.clinical_synthesis = synthesis.model_dump(mode="json")
            if record.verification_queue or record.processing_warnings:
                record.patient_status = "completed_with_warnings"
            self.repository.save(record, source_pages)
            output_directory = write_outputs(record, self.configuration.output_root)
            return ProcessingResult(record, output_directory, warnings=list(warnings))
        finally:
            if archive_root is not None:
                shutil.rmtree(archive_root, ignore_errors=True)

    def _use_patient_level_review(self) -> bool:
        return (
            self.configuration.ai_extraction_mode.casefold() == "patient_level"
            and callable(getattr(self.provider, "review_patient", None))
        )

    def _prepare_patient_files(self, patient_id: str, ingested: list, existing: list) -> list[tuple[Any, Any, Any]]:
        prepared: list[tuple[Any, Any, Any]] = []
        processed_root = self.configuration.storage_root / patient_id / "processed"
        existing_offset = max((item.source_order or 0 for item in existing), default=-1) + 1
        for item in ingested:
            result = preprocess(item, processed_root)
            source_file = item.source_file.model_copy(update={"processed_path": str(result.processed_path), "processing_status": "processed", "source_order": existing_offset + (item.source_file.source_order or 0)})
            prepared.append((item, result, source_file))
        # Reprocessing uses the forensic originals already stored for this
        # patient.  It must not create duplicate source rows or lose order.
        if not prepared:
            for source_file in existing:
                path = Path(source_file.processed_path or source_file.original_path)
                if path.is_file():
                    prepared.append((None, None, source_file))
        return prepared

    @staticmethod
    def _ensure_source_pages(record: StructuredRecord, source_pages: list[dict[str, object]]) -> list[dict[str, object]]:
        known = {(page.file_id, page.page_number) for page in record.source_pages}
        for source_file in record.source_files:
            for page_number in range(1, (source_file.page_count or 1) + 1):
                if (source_file.file_id, page_number) in known:
                    continue
                document_type, confidence = classify(source_file.original_filename, None)
                page = SourcePageInfo(
                    page_id=str(uuid4()), file_id=source_file.file_id, page_number=page_number,
                    document_type=document_type, classification_confidence=confidence,
                    extraction_status="pending_ai_review",
                )
                record.source_pages.append(page)
                source_pages.append(page.model_dump(mode="json"))
        return source_pages

    @staticmethod
    def _patient_bundle(record: StructuredRecord, prepared: list[tuple[Any, Any, Any]]) -> list[PatientBundleItem]:
        ordered = sorted(
            (source_file for _, _, source_file in prepared),
            key=lambda item: (item.source_order if item.source_order is not None else 10**9, item.original_relative_path or item.original_filename),
        )
        items: list[PatientBundleItem] = []
        for index, source_file in enumerate(ordered):
            # The multimodal review receives the forensic original.  Derived
            # preprocessing is retained for local/page-level fallback only.
            path = Path(source_file.original_path)
            media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            for page_number in range(1, (source_file.page_count or 1) + 1):
                # PDF pages share one native input_file; image files naturally
                # have one page.  The manifest still retains every page ID.
                items.append(PatientBundleItem(
                    source_id=source_file.file_id,
                    source_file=source_file.original_filename,
                    relative_path=source_file.original_relative_path or source_file.original_filename,
                    page_number=page_number,
                    order_index=index * 100000 + page_number,
                    media_type=media_type,
                    local_path=str(path),
                ))
        return items

    def _apply_patient_review(self, record: StructuredRecord, review: PatientLevelReview, patient_id: str) -> None:
        """Map the strict patient-level object into canonical typed views."""
        payload = review.model_dump(mode="json")
        if record.ai_clinical_review:
            record.ai_review_history.append({
                "version": record.ai_review_version,
                "model": record.ai_review_model,
                "prompt_version": record.ai_review_prompt_version,
                "source_hash": record.ai_review_source_hash,
                "created_at": record.ai_review_created_at.isoformat() if record.ai_review_created_at else None,
                "review": record.ai_clinical_review,
            })
        source_hash = sha256("".join(item.checksum for item in sorted(record.source_files, key=lambda value: value.source_order if value.source_order is not None else 10**9)).encode()).hexdigest()
        record.ai_clinical_review = payload
        record.ai_review_version = f"v{len(record.ai_review_history) + 1}"
        record.ai_review_model = self.configuration.openai_medical_model
        record.ai_review_prompt_version = "patient_level_review_v1"
        record.ai_review_source_hash = source_hash
        record.ai_review_created_at = datetime.now(timezone.utc)

        patient_data = payload.get("patient") or {}
        if isinstance(patient_data, dict):
            mapping = {"patient_name": "full_name", "full_name": "full_name", "mrn": "hospital_file_number", "hospital_file_number": "hospital_file_number", "dob": "date_of_birth", "date_of_birth": "date_of_birth", "sex": "sex", "nationality": "nationality", "address": "address"}
            for source_name, target_name in mapping.items():
                if source_name not in patient_data:
                    continue
                value = self._fact_value(patient_data[source_name])
                if value in (None, ""):
                    continue
                field_item = patient_data[source_name] if isinstance(patient_data[source_name], dict) else patient_data
                reference = self._fact_ref(field_item, record, "patient_level_review")
                confidence = self._fact_confidence(field_item) or 0.90
                setattr(record.patient, target_name, ProvenancedValue(value=value, raw_text=str(value), confidence=confidence, status=FactStatus.DOCUMENTED, source_ref=reference))

        # A fresh AI review is canonical; never append stale page-level facts.
        record.encounters = []
        record.diagnoses = [self._diagnosis_from_fact(item, record) for item in payload.get("diagnoses_documented", []) if self._diagnosis_from_fact(item, record)]
        record.medications = [self._medication_from_fact(item, record) for item in payload.get("medications", []) if self._medication_from_fact(item, record)]
        record.laboratory_results = [self._lab_from_fact(item, record) for item in payload.get("laboratory_results", []) if self._lab_from_fact(item, record)]
        record.radiology_reports = [self._radiology_from_fact(item, record) for item in payload.get("radiology", []) if self._radiology_from_fact(item, record)]
        record.growth_measurements = [self._growth_from_fact(item, record) for item in payload.get("growth_measurements", []) if self._growth_from_fact(item, record)]
        record.timeline = [self._timeline_from_fact(item, record) for item in payload.get("clinical_timeline", []) if self._timeline_from_fact(item, record)]
        record.problem_list = []
        for item in [*payload.get("active_problems", []), *payload.get("resolved_or_historical_problems", [])]:
            value = self._fact_value(item)
            if value:
                status = str(item.get("status", "active")) if isinstance(item, dict) else "active"
                record.problem_list.append(ProblemItem(problem=str(value), status=status, source_type="ai_clinical_review", source_refs=[self._fact_ref(item, record, "patient_level_review")]))
        record.verification_queue = []
        for item in payload.get("uncertain_items", []):
            if not isinstance(item, dict):
                continue
            record.verification_queue.append(unreadable_page_item(
                self._fact_ref(item, record, "patient_level_review").source_file,
                self._fact_ref(item, record, "patient_level_review").source_page,
                str(item.get("reason") or item.get("verification_required") or "Uncertain or illegible clinical information requires clinician verification"),
            ))

        record.clinical_synthesis = {
            "record_quality": payload.get("record_quality", {}),
            "clinical_interpretations": payload.get("clinical_interpretations", []),
            "growth_interpretation": payload.get("growth_interpretation", ""),
            "conflicts": payload.get("conflicts", []),
        }

    @staticmethod
    def _fact_value(item: Any) -> Any:
        if isinstance(item, dict):
            for key in ("value", "term", "diagnosis", "problem", "name", "test_name", "text", "summary"):
                if item.get(key) not in (None, ""):
                    return item[key]
        return item if item not in (None, "") else None

    @staticmethod
    def _fact_confidence(item: Any) -> float:
        try:
            return min(1.0, max(0.0, float(item.get("confidence", 0.0)))) if isinstance(item, dict) else 0.0
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _fact_ref(item: Any, record: StructuredRecord, method: str) -> SourceRef:
        value = item if isinstance(item, dict) else {}
        files = value.get("source_files") or value.get("source_file") or "unknown"
        if isinstance(files, list):
            files = files[0] if files else "unknown"
        page_values = value.get("source_pages") or value.get("source_page") or 1
        if isinstance(page_values, list):
            page_values = page_values[0] if page_values else 1
        try:
            page = max(1, int(page_values))
        except (TypeError, ValueError):
            page = 1
        source_file = str(files)
        if source_file not in {source.original_filename for source in record.source_files}:
            source = next((source for source in record.source_files if source.file_id == source_file), None)
            source_file = source.original_filename if source else source_file
        return SourceRef(source_file=source_file, source_page=page, extraction_method=method, prompt_version="patient_level_review_v1")

    @classmethod
    def _diagnosis_from_fact(cls, item: Any, record: StructuredRecord):
        from app.schemas import Diagnosis
        value = cls._fact_value(item)
        if not value:
            return None
        ref = cls._fact_ref(item, record, "patient_level_review")
        kind = str(item.get("type", "documented")) if isinstance(item, dict) else "documented"
        if kind not in {"documented", "inferred", "differential", "historical"}:
            kind = "documented"
        certainty = str(item.get("certainty", "confirmed")) if isinstance(item, dict) else "confirmed"
        if certainty not in {"confirmed", "probable", "possible", "uncertain"}:
            certainty = "uncertain"
        status = str(item.get("status", "unknown")) if isinstance(item, dict) else "unknown"
        if status not in {"active", "resolved", "unknown"}:
            status = "unknown"
        return Diagnosis(term_original=str(value), term_normalized=item.get("normalized") if isinstance(item, dict) else None, type=kind, status=status, certainty=certainty, date=item.get("date") if isinstance(item, dict) else None, confidence=cls._fact_confidence(item), source_ref=ref)

    @classmethod
    def _medication_from_fact(cls, item: Any, record: StructuredRecord):
        from app.schemas import Medication
        if not isinstance(item, dict):
            return None
        value = cls._fact_value(item)
        if not value:
            return None
        return Medication(name_original=str(item.get("name_original") or value), name_normalized=item.get("name_normalized"), generic_name=item.get("generic_name"), brand_name=item.get("brand_name"), strength=item.get("strength"), dose=item.get("dose"), dose_unit=item.get("dose_unit"), route=item.get("route"), frequency=item.get("frequency"), duration=item.get("duration"), indication=item.get("indication"), status=str(item.get("status", "unknown")), confidence=cls._fact_confidence(item), source_ref=cls._fact_ref(item, record, "patient_level_review"), verification_required=bool(item.get("verification_required", item.get("dose") in (None, ""))))

    @classmethod
    def _lab_from_fact(cls, item: Any, record: StructuredRecord):
        from app.schemas import LaboratoryResult
        if not isinstance(item, dict):
            return None
        value = cls._fact_value(item)
        name = item.get("test_name_original") or item.get("test_name") or item.get("name")
        if not name:
            return None
        numeric = item.get("value") if isinstance(item.get("value"), (int, float)) else None
        return LaboratoryResult(test_name_original=str(name), test_name_normalized=item.get("test_name_normalized"), value=numeric, value_text=str(item.get("value_text") or value or ""), unit=item.get("unit"), reference_range=item.get("reference_range"), abnormal_flag=item.get("abnormal_flag"), date=item.get("date"), specimen=item.get("specimen"), confidence=cls._fact_confidence(item), source_ref=cls._fact_ref(item, record, "patient_level_review"))

    @classmethod
    def _radiology_from_fact(cls, item: Any, record: StructuredRecord):
        from app.schemas import RadiologyReport
        if not isinstance(item, dict):
            return None
        if not cls._fact_value(item) and not item.get("study_type"):
            return None
        return RadiologyReport(study_type=item.get("study_type"), body_part=item.get("body_part"), date=item.get("date"), report_text=item.get("report_text") or item.get("text"), findings=list(item.get("findings") or []), impression=list(item.get("impression") or []), confidence=cls._fact_confidence(item), source_ref=cls._fact_ref(item, record, "patient_level_review"))

    @classmethod
    def _growth_from_fact(cls, item: Any, record: StructuredRecord):
        from app.schemas import GrowthMeasurement
        if not isinstance(item, dict):
            return None
        if not any(item.get(key) is not None for key in ("weight_kg", "height_cm", "head_circumference_cm", "bmi")):
            return None
        return GrowthMeasurement(date=item.get("date"), age_months=item.get("age_months"), weight_kg=item.get("weight_kg"), height_cm=item.get("height_cm"), head_circumference_cm=item.get("head_circumference_cm"), bmi=item.get("bmi"), source=cls._fact_ref(item, record, "patient_level_review"), confidence=cls._fact_confidence(item))

    @classmethod
    def _timeline_from_fact(cls, item: Any, record: StructuredRecord):
        from app.schemas import TimelineEvent
        if not isinstance(item, dict):
            return None
        summary = str(item.get("summary") or item.get("event") or item.get("text") or "")
        if not summary:
            return None
        return TimelineEvent(date=item.get("date"), event=str(item.get("event") or "Clinical encounter"), summary=summary, source_refs=[cls._fact_ref(item, record, "patient_level_review")])

    @staticmethod
    def _identity_from_ai(page_analysis: PageAnalysis, filename: str, page_number: int) -> dict[str, ProvenancedValue]:
        if not page_analysis.structured:
            return {}
        identity: dict[str, ProvenancedValue] = {}
        ref_method = "openai_responses"
        for field_name, field in page_analysis.structured.patient_identity.model_dump().items():
            if not field or field.get("value") in (None, ""):
                continue
            value = field.get("value")
            field_status = (
                FactStatus.NEEDS_VERIFICATION
                if field.get("needs_verification")
                else FactStatus.INFERRED
                if field.get("documented_vs_inferred") == "inferred"
                else FactStatus.DOCUMENTED
            )
            identity[field_name] = ProvenancedValue(
                value=value,
                raw_text=str(value),
                confidence=float(field.get("confidence", page_analysis.confidence)),
                status=field_status,
                source_ref=SourceRef(
                    source_file=filename,
                    source_page=page_number,
                    extraction_method=ref_method,
                    prompt_version=page_analysis.prompt_version,
                ),
            )
        return identity

    @staticmethod
    def _apply_upload_demographics(record: StructuredRecord, demographics: dict[str, str | None] | None) -> None:
        if not demographics:
            return
        ref = SourceRef(source_file="web_upload_metadata", source_page=1, extraction_method="user_entered")
        mapping = {"patient_name": "full_name", "hospital_file_number": "hospital_file_number", "sex": "sex", "date_of_birth": "date_of_birth", "nationality": "nationality", "address": "address"}
        for incoming_key, field_name in mapping.items():
            incoming = (demographics.get(incoming_key) or "").strip()
            if not incoming:
                continue
            current = getattr(record.patient, field_name)
            if current and current.value and str(current.value).casefold() != incoming.casefold():
                from app.services.extraction import _review
                record.verification_queue.append(_review(field_name, incoming, incoming, 1.0, "possible patient identity or demographic mismatch; do not merge automatically", ref, "medical_review"))
                record.patient.candidates.setdefault(field_name, []).append(ProvenancedValue(value=incoming, raw_text=incoming, confidence=1.0, status=FactStatus.NEEDS_VERIFICATION, source_ref=ref))
                continue
            setattr(record.patient, field_name, ProvenancedValue(value=incoming, raw_text=incoming, confidence=1.0, status=FactStatus.DOCUMENTED, source_ref=ref))

    @staticmethod
    def _merge_document_identity(record: StructuredRecord, identity: dict[str, ProvenancedValue], resolved_patient_id: str) -> None:
        from app.services.extraction import _review
        source_id = identity.get("patient_id")
        if source_id and str(source_id.value).casefold() != resolved_patient_id.casefold():
            record.verification_queue.append(_review("patient_id", str(source_id.value), source_id.raw_text, source_id.confidence, "possible patient identity mismatch; document must be reviewed before merge", source_id.source_ref, "medical_review"))
        for field_name in ("full_name", "hospital_file_number", "sex", "date_of_birth"):
            incoming = identity.get(field_name)
            if not incoming:
                continue
            current = getattr(record.patient, field_name)
            if current and current.value and str(current.value).casefold() != str(incoming.value).casefold():
                record.patient.candidates.setdefault(field_name, []).append(incoming)
                record.verification_queue.append(_review(field_name, str(incoming.value), incoming.raw_text, incoming.confidence, "conflicting demographic candidate; do not overwrite preferred value automatically", incoming.source_ref, "medical_review"))
            elif current is None:
                setattr(record.patient, field_name, incoming)
