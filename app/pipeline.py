from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
from uuid import uuid4

from app.config import Settings, settings
from app.exports.writer import write_outputs
from app.repository import RecordRepository
from app.schemas import DocumentType, FactStatus, PatientDemographics, ProvenancedValue, SourcePageInfo, SourceRef, StructuredRecord
from app.services.classification import classify
from app.services.clinical import build_timeline_and_problems
from app.services.extraction import extract_documented_facts, extract_identity_facts, source_ref, unreadable_page_item
from app.services.ingestion import ArchiveLimits, IngestionError, extract_archive, detect_archive_type, ingest_folder
from app.services.preprocessing import preprocess
from app.services.providers import MedicalAIProvider, PageAnalysis, provider_for_settings
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
        self.provider = provider or provider_for_settings(configuration)

    def process(self, source: Path, patient_id: str | None = None, resume: bool = True, demographics: dict[str, str | None] | None = None, source_mime_type: str | None = None) -> ProcessingResult:
        archive_result = None
        archive_root = None
        warnings: list[str] = []
        try:
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
