from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import shutil

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import settings
from app.db import make_session_factory
from app.pipeline import PatientPipeline
from app.repository import DuplicateSourceError, RecordRepository
from app.schemas import JobStatus, ReportGenerationRequest, ReportRecord, ReportType, ReviewStatus, StructuredRecord
from app.services.ingestion import ArchiveCorruptError, ArchivePasswordError, IngestionError, detect_archive_type
from app.services.reports import ReportGenerator
from app.services.providers import PatientLevelAIError
from app.services.providers import OpenAIMedicalVisionProvider, OpenRouterMedicalVisionProvider
from app.services.ai_diagnostics import AIDiagnosticsStore, safe_error_message

app = FastAPI(title="Medical Records Digitization MVP", version="0.1.0")
repository = RecordRepository(make_session_factory())
pipeline = PatientPipeline(repository, settings)
report_generator = ReportGenerator(repository, settings)
ai_diagnostic_store = AIDiagnosticsStore(repository.session_factory)
_configured_model = settings.openrouter_model if settings.ai_provider.casefold().startswith("openrouter") else settings.openai_medical_model
_configured_key = settings.openrouter_api_key if settings.ai_provider.casefold().startswith("openrouter") else settings.openai_api_key
ai_diagnostic_store.mark_runtime(role="api", provider=settings.ai_provider, model=_configured_model, api_key_configured=bool(_configured_key))


class IngestRequest(BaseModel):
    patient_folder: str
    patient_id: str | None = None
    resume: bool = True


class ReviewRequest(BaseModel):
    review_status: ReviewStatus
    reviewer_id: str | None = None
    edited_value: str | None = None


class AIComparisonRequest(BaseModel):
    patient_id: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _technical_details_enabled() -> bool:
    return bool(settings.ai_debug or settings.admin_mode or settings.app_environment.casefold() in {"development", "dev", "test"})


@app.get("/admin/ai")
def get_ai_diagnostics() -> dict[str, object]:
    """Return redacted provider diagnostics; no request payloads or secrets."""

    snapshot = ai_diagnostic_store.snapshot(include_technical=_technical_details_enabled())
    snapshot.update({
        "configured_provider": settings.ai_provider,
        "configured_model": settings.openrouter_model if settings.ai_provider.casefold().startswith("openrouter") else settings.openai_medical_model,
        "provider_options": ["openai", "openrouter"],
        "provider_switching": "environment_configuration",
    })
    return snapshot


def _configured_openai_provider() -> OpenAIMedicalVisionProvider:
    if settings.ai_provider.casefold() not in {"openai", "openai_vision", "openai_medical"}:
        raise HTTPException(status_code=503, detail="OpenAI provider is not enabled")
    if not settings.openai_api_key:
        ai_diagnostic_store.mark_runtime(role="api", provider=settings.ai_provider, model=settings.openai_medical_model, api_key_configured=False)
        raise HTTPException(status_code=503, detail="OpenAI API key is not configured")
    if isinstance(pipeline.provider, OpenAIMedicalVisionProvider):
        return pipeline.provider
    return OpenAIMedicalVisionProvider(settings, diagnostics=ai_diagnostic_store)


def _configured_current_provider() -> OpenAIMedicalVisionProvider | OpenRouterMedicalVisionProvider:
    """Build the currently configured provider for the admin smoke tests."""

    if settings.ai_provider.casefold().startswith("openrouter"):
        if not settings.openrouter_api_key:
            ai_diagnostic_store.mark_runtime(role="api", provider=settings.ai_provider, model=settings.openrouter_model, api_key_configured=False)
            raise HTTPException(status_code=503, detail="OpenRouter API key is not configured")
        if isinstance(pipeline.provider, OpenRouterMedicalVisionProvider):
            return pipeline.provider
        return OpenRouterMedicalVisionProvider(settings, diagnostics=ai_diagnostic_store)
    return _configured_openai_provider()


def _provider_for_comparison(name: str):
    provider = name.casefold()
    if provider == "openai":
        if not settings.openai_api_key:
            raise PatientLevelAIError("OpenAI API key is not configured")
        return OpenAIMedicalVisionProvider(settings)
    if provider == "openrouter":
        if not settings.openrouter_api_key:
            raise PatientLevelAIError("OpenRouter API key is not configured")
        return OpenRouterMedicalVisionProvider(settings)
    raise PatientLevelAIError("Unsupported comparison provider")


def _deidentify_for_comparison(record: StructuredRecord) -> dict[str, object]:
    """Remove direct identifiers while retaining comparable clinical facts."""

    source = record.ai_clinical_review or record.model_dump(mode="json", include={
        "patient", "diagnoses", "medications", "laboratory_results", "growth_measurements", "verification_queue",
    })
    redacted_keys = {"patient_id", "full_name", "patient_name", "hospital_file_number", "mrn", "address", "date_of_birth", "source_file", "original_filename", "filename"}
    omitted_text_keys = {"detailed_report_markdown", "erp_summary_markdown", "raw_text", "transcription", "report_text", "summary", "event", "reason"}

    def redact(value: object, key: str = "") -> object:
        if isinstance(value, dict):
            output: dict[str, object] = {}
            for name, item in value.items():
                normalized = name.casefold()
                if normalized in omitted_text_keys:
                    continue
                if normalized in redacted_keys:
                    output[name] = "REDACTED"
                elif normalized == "source_files" and isinstance(item, list):
                    output[name] = [f"source_document_{index}" for index, _ in enumerate(item, start=1)]
                else:
                    output[name] = redact(item, name)
            return output
        if isinstance(value, list):
            return [redact(item, key) for item in value]
        return value

    return redact(source)  # type: ignore[return-value]


@app.post("/admin/ai/compare")
def compare_ai_providers(request: AIComparisonRequest) -> dict[str, object]:
    record = repository.get_record(request.patient_id)
    if not record:
        raise HTTPException(status_code=404, detail="patient record not found")
    redacted = _deidentify_for_comparison(record)
    results: dict[str, object] = {}
    for provider_name in ("openai", "openrouter"):
        try:
            provider = _provider_for_comparison(provider_name)
            review = provider.review_deidentified_record(redacted)
            results[provider_name] = {"status": "success", "model": provider.configuration.openai_medical_model, "review": review.model_dump(mode="json")}
        except Exception as exc:
            results[provider_name] = {"status": "failed", "message": safe_error_message(exc)}
    return {
        "patient_id": request.patient_id,
        "deidentified": True,
        "automatic_selection": False,
        "comparison_fields": ["demographics", "diagnoses", "medications", "labs", "growth", "uncertain_items", "erp_summary"],
        "results": results,
    }


def _run_ai_connection_test(stage: str) -> dict[str, object]:
    provider = _configured_current_provider()
    try:
        result = provider.test_text_connection() if stage == "text_connection_test" else provider.test_vision_connection()
        return {"ok": True, "message": "AI connection test succeeded", "result": result, "diagnostics": ai_diagnostic_store.snapshot(include_technical=_technical_details_enabled())}
    except PatientLevelAIError as exc:
        snapshot = ai_diagnostic_store.snapshot(include_technical=_technical_details_enabled())
        raise HTTPException(status_code=503, detail={"message": str(exc), "diagnostics": snapshot}) from exc


@app.post("/admin/ai/test-connection")
def test_ai_connection() -> dict[str, object]:
    return _run_ai_connection_test("text_connection_test")


@app.post("/admin/ai/test-vision")
def test_ai_vision() -> dict[str, object]:
    return _run_ai_connection_test("vision_connection_test")


@app.post("/ingestions", response_model=StructuredRecord)
def ingest(request: IngestRequest) -> StructuredRecord:
    try:
        return pipeline.process(Path(request.patient_folder), request.patient_id, request.resume).record
    except PatientLevelAIError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, DuplicateSourceError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/patients/{patient_id}", response_model=StructuredRecord)
def patient_record(patient_id: str) -> StructuredRecord:
    record = repository.get_record(patient_id)
    if not record:
        raise HTTPException(status_code=404, detail="patient record not found")
    return record


@app.get("/patients", response_model=list)
def search_patients(q: str = "", limit: int = 50):
    return [item.model_dump(mode="json") for item in repository.search_patients(q, limit)]


@app.delete("/patients/{patient_id}")
def delete_patient(patient_id: str) -> dict[str, object]:
    if not repository.delete_patient(patient_id):
        raise HTTPException(status_code=404, detail="patient record not found")
    return {"deleted": True, "patient_id": patient_id}


@app.delete("/patients")
def delete_all_patients(confirm: str = Query(default="")) -> dict[str, object]:
    if confirm != "DELETE ALL":
        raise HTTPException(status_code=400, detail="confirmation required: pass confirm=DELETE ALL")
    deleted_count = repository.delete_all_patients()
    return {"deleted": True, "deleted_count": deleted_count}


@app.get("/dashboard")
def dashboard():
    return repository.dashboard_metrics()


@app.get("/patients/{patient_id}/source-files/{file_id}")
def source_file(patient_id: str, file_id: str) -> FileResponse:
    path = repository.source_path(patient_id, file_id)
    if not path or not Path(path).is_file():
        raise HTTPException(status_code=404, detail="source file not found")
    # TODO: Require RBAC/record-level authorization before exposing outside trusted local deployments.
    return FileResponse(path, filename=Path(path).name)


@app.post("/patients/{patient_id}/review-items/{item_id}", response_model=StructuredRecord)
def review_item(patient_id: str, item_id: str, request: ReviewRequest) -> StructuredRecord:
    updated = repository.review_item(patient_id, item_id, request.review_status, request.reviewer_id, request.edited_value)
    if not updated:
        raise HTTPException(status_code=404, detail="review item not found")
    return updated


def _run_upload_job(job_id: str, source: Path, patient_id: str, metadata: dict[str, str | None]) -> None:
    is_archive = source.suffix.lower() in {".zip", ".rar"}
    repository.update_job(job_id, status="processing", progress=15, message="Preparing medical record..." if is_archive else "Preparing medical record...")
    if settings.ai_extraction_mode.casefold() == "patient_level" and settings.ai_provider.casefold() in {"openai", "openai_vision", "openai_medical"}:
        repository.update_job(job_id, status="processing", progress=25, message="AI is reviewing the complete patient record...")
    try:
        result = pipeline.process(source, patient_id=patient_id, resume=False, demographics=metadata)
        if is_archive and result.warnings:
            repository.update_job(job_id, status="complete", progress=100, message="Archive processed with warnings")
        elif is_archive:
            repository.update_job(job_id, status="complete", progress=100, message="Clinical extraction complete")
        else:
            repository.update_job(job_id, status="complete", progress=100, message="Clinical extraction complete")
    except PatientLevelAIError as exc:
        repository.mark_patient_failed(patient_id)
        message = safe_error_message(exc)
        repository.update_job(job_id, status="failed", progress=100, message=message, error=message)
    except ArchivePasswordError:
        repository.mark_patient_failed(patient_id)
        repository.update_job(job_id, status="failed", progress=100, message="This archive is password protected. Please upload an unencrypted archive.", error="Password-protected archive")
    except ArchiveCorruptError:
        repository.mark_patient_failed(patient_id)
        repository.update_job(job_id, status="failed", progress=100, message="The archive could not be extracted. It may be damaged or incomplete.", error="Unable to read archive")
    except IngestionError as exc:
        repository.mark_patient_failed(patient_id)
        repository.update_job(job_id, status="failed", progress=100, message=str(exc), error="Unable to extract archive" if is_archive else "Processing failed")
    except Exception as exc:
        repository.mark_patient_failed(patient_id)
        repository.update_job(job_id, status="failed", progress=100, message="Processing failed; source files were preserved", error=type(exc).__name__)
    finally:
        # The multipart staging directory is temporary. Originals are copied into the patient store first.
        staging_root = source.parent if source.is_file() else source
        if "incoming" in staging_root.parts and staging_root.name:
            shutil.rmtree(staging_root, ignore_errors=True)


def _run_ai_reprocess_job(job_id: str, patient_id: str) -> None:
    repository.update_job(job_id, status="processing", progress=20, message="AI is reviewing the complete patient record...")
    try:
        source = settings.storage_root / patient_id / "original"
        result = pipeline.process(source, patient_id=patient_id, resume=False)
        repository.update_job(job_id, status="complete", progress=100, message="Clinical extraction complete")
    except PatientLevelAIError as exc:
        repository.mark_patient_failed(patient_id)
        message = safe_error_message(exc)
        repository.update_job(job_id, status="failed", progress=100, message=message, error=message)
    except Exception as exc:
        repository.update_job(job_id, status="failed", progress=100, message="AI medical review could not be completed.", error=type(exc).__name__)


@app.post("/patients/{patient_id}/ai-review/reprocess", response_model=JobStatus)
def reprocess_ai_review(patient_id: str, background_tasks: BackgroundTasks) -> JobStatus:
    if not repository.get_record(patient_id):
        raise HTTPException(status_code=404, detail="patient record not found")
    job = JobStatus(job_id=str(uuid4()), patient_id=patient_id, status="queued", progress=5, message="Preparing medical record...", created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
    repository.create_job(job, report_type="ai_review_reprocess")
    background_tasks.add_task(_run_ai_reprocess_job, job.job_id, patient_id)
    return job


@app.post("/uploads", response_model=JobStatus)
async def upload_patient(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    patient_id: str | None = Form(default=None),
    patient_name: str | None = Form(default=None),
    hospital_file_number: str | None = Form(default=None),
    sex: str | None = Form(default=None),
    date_of_birth: str | None = Form(default=None),
    nationality: str | None = Form(default=None),
    address: str | None = Form(default=None),
):
    if not files:
        raise HTTPException(status_code=422, detail="at least one JPG, PNG, PDF, ZIP, or RAR file is required")
    resolved_patient_id = (patient_id or f"P{uuid4().hex[:12].upper()}").strip()
    upload_root = settings.storage_root / "incoming" / "web" / uuid4().hex
    upload_root.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []
    allowed = {".jpg", ".jpeg", ".png", ".pdf", ".zip", ".rar"}
    archive_files: list[tuple[Path, str | None]] = []
    for upload in files:
        safe_name = Path(upload.filename or "upload").name
        if Path(safe_name).suffix.lower() not in allowed:
            raise HTTPException(status_code=422, detail=f"unsupported file type: {Path(safe_name).suffix}")
        destination = upload_root / safe_name
        with destination.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)
        saved_files.append(destination)
        if destination.suffix.lower() in {".zip", ".rar"}:
            try:
                detect_archive_type(destination, upload.content_type)
            except IngestionError as exc:
                shutil.rmtree(upload_root, ignore_errors=True)
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            archive_files.append((destination, upload.content_type))
    if len(archive_files) > 1 or (archive_files and len(saved_files) > len(archive_files)):
        shutil.rmtree(upload_root, ignore_errors=True)
        raise HTTPException(status_code=422, detail="Upload one ZIP or RAR archive at a time, or upload source documents without an archive.")
    repository.ensure_patient(resolved_patient_id)
    source = archive_files[0][0] if archive_files else upload_root
    job = JobStatus(job_id=str(uuid4()), patient_id=resolved_patient_id, status="queued", progress=5, message="Upload received", created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
    repository.create_job(job, report_type="ingestion")
    metadata = {"patient_name": patient_name, "hospital_file_number": hospital_file_number, "sex": sex, "date_of_birth": date_of_birth, "nationality": nationality, "address": address}
    background_tasks.add_task(_run_upload_job, job.job_id, source, resolved_patient_id, metadata)
    return job


@app.get("/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str) -> JobStatus:
    job = repository.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


def _run_report_job(job_id: str, patient_id: str, report_type: ReportType, created_by: str) -> None:
    repository.update_job(job_id, status="processing", progress=20, message="Rendering grounded report")
    try:
        report_generator.generate(patient_id, report_type, created_by)
        repository.update_job(job_id, status="complete", progress=100, message="Report ready")
    except Exception as exc:
        repository.update_job(job_id, status="failed", progress=100, message="Report generation failed", error=type(exc).__name__)


@app.post("/patients/{patient_id}/reports/generate", response_model=JobStatus)
def generate_report(patient_id: str, request: ReportGenerationRequest, background_tasks: BackgroundTasks) -> JobStatus:
    if not repository.get_record(patient_id):
        raise HTTPException(status_code=404, detail="patient record not found")
    job = JobStatus(job_id=str(uuid4()), patient_id=patient_id, status="queued", progress=5, message="Report queued", created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
    repository.create_job(job, report_type=request.report_type.value)
    background_tasks.add_task(_run_report_job, job.job_id, patient_id, request.report_type, request.created_by)
    return job


@app.get("/patients/{patient_id}/reports", response_model=list[ReportRecord])
def patient_reports(patient_id: str) -> list[ReportRecord]:
    return repository.list_reports(patient_id=patient_id)


@app.get("/reports", response_model=list[ReportRecord])
def search_reports(q: str = "", report_type: ReportType | None = None, limit: int = 100, date: str | None = None) -> list[ReportRecord]:
    return repository.list_reports(q, report_type, limit, date=date)


@app.get("/reports/{report_id}", response_model=ReportRecord)
def report(report_id: str) -> ReportRecord:
    result = repository.get_report(report_id)
    if not result:
        raise HTTPException(status_code=404, detail="report not found")
    return result


@app.get("/reports/{report_id}/download")
def download_report(report_id: str) -> FileResponse:
    result = repository.get_report(report_id)
    if not result or not Path(result.file_path).is_file():
        raise HTTPException(status_code=404, detail="report file not found")
    return FileResponse(result.file_path, filename=Path(result.file_path).name, media_type="application/pdf")


@app.get("/reports/{report_id}/text")
def report_text(report_id: str) -> FileResponse:
    result = repository.get_report(report_id)
    if not result or not result.text_path or not Path(result.text_path).is_file():
        raise HTTPException(status_code=404, detail="report text not found")
    return FileResponse(result.text_path, filename=Path(result.text_path).name, media_type="text/plain")


@app.get("/reports/{report_id}/json")
def report_json(report_id: str) -> FileResponse:
    result = repository.get_report(report_id)
    if not result or not result.json_path or not Path(result.json_path).is_file():
        raise HTTPException(status_code=404, detail="report JSON not found")
    return FileResponse(result.json_path, filename=Path(result.json_path).name, media_type="application/json")
