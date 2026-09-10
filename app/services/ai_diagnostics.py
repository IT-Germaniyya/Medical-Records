"""Safe, shared diagnostics for configured multimodal AI requests.

Only redacted metadata is persisted here.  Request bodies, source filenames,
authorization headers, and model output are deliberately never stored.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

from sqlalchemy import select

from app.models import AIDiagnosticModel


_SECRET_RE = re.compile(r"(?:sk-[A-Za-z0-9_-]{12,}|Bearer\s+[A-Za-z0-9._-]+)", re.IGNORECASE)
_DATA_URL_RE = re.compile(r"data:[^,\s]+;base64,[A-Za-z0-9+/=]+", re.IGNORECASE)
_LONG_TOKEN_RE = re.compile(r"\b[A-Za-z0-9+/=_-]{160,}\b")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def safe_error_message(exc: BaseException) -> str:
    """Return a bounded error message with secrets and payloads removed."""

    message = getattr(exc, "message", None) or str(exc) or "The provider returned an error without a message."
    message = _DATA_URL_RE.sub("[redacted data]", str(message))
    message = _SECRET_RE.sub("[redacted secret]", message)
    message = _LONG_TOKEN_RE.sub("[redacted token]", message)
    message = " ".join(message.replace("\x00", " ").split())
    return message[:1000]


def exception_type(exc: BaseException) -> str:
    return f"{type(exc).__module__}.{type(exc).__name__}"


def http_status(exc: BaseException) -> int | None:
    value = getattr(exc, "status_code", None)
    if value is None:
        response = getattr(exc, "response", None)
        value = getattr(response, "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def classify_error(exc: BaseException, *, stage: str, provider: str = "openai") -> tuple[str, str]:
    """Map SDK/network/schema failures to safe, actionable product language."""

    provider_key = provider.casefold()
    label = "OpenRouter" if provider_key.startswith("openrouter") else "Local AI" if provider_key.startswith("local") else "OpenAI"
    name = type(exc).__name__.casefold()
    message = safe_error_message(exc).casefold()
    status = http_status(exc)
    combined = f"{name} {message}"
    if status == 401 or "authentication" in combined or "invalid_api_key" in combined or "api key" in combined:
        return "authentication_failed", f"{label} authentication failed"
    if "insufficient_quota" in combined or "credit_balance_exhausted" in combined or "no credits remaining" in combined:
        return "quota_exhausted", f"{label} quota or credits are exhausted"
    if status == 404 or "model" in combined and any(token in combined for token in ("not found", "does not exist", "unavailable")):
        return "model_unavailable", f"Configured {label} model is unavailable"
    if status == 413 or any(token in combined for token in ("too large", "context length", "payload", "maximum context", "input too long")):
        return "request_too_large", f"{label} request exceeded size limits"
    if any(token in combined for token in ("schema", "structured output", "json_schema", "response_format")) or (stage in {"patient_review", "patient_synthesis"} and "validation" in combined):
        return "structured_output_schema_rejected", f"{label} structured output schema was rejected"
    if stage in {"vision_connection_test", "patient_review", "patient_synthesis", "page_extraction"} and any(token in combined for token in ("image", "file", "mime", "media", "input")):
        return "input_preparation_failed", f"{label} image/file input could not be prepared"
    if "timeout" in combined or "timed out" in combined:
        return "request_timeout", f"{label} request timed out"
    if "rate" in combined and "limit" in combined:
        return "rate_limit", f"{label} rate limit reached"
    if "connection" in combined or "dns" in combined or "network" in combined:
        return "connection_failed", f"Could not connect to {label}"
    return "ai_request_failed", f"{label} request failed"


class AIDiagnosticsStore:
    """A single redacted row shared by API and worker through PostgreSQL."""

    def __init__(self, session_factory: Any):
        self.session_factory = session_factory

    def _row(self, session: Any) -> AIDiagnosticModel:
        row = session.get(AIDiagnosticModel, 1)
        if row is None:
            row = AIDiagnosticModel(diagnostic_id=1, provider="unknown", model="unknown", updated_at=utc_now())
            session.add(row)
            session.flush()
        return row

    def mark_runtime(self, *, role: str, provider: str, model: str, api_key_configured: bool) -> None:
        with self.session_factory() as session:
            row = self._row(session)
            row.provider = provider
            row.model = model
            if role == "worker":
                row.worker_api_key_configured = bool(api_key_configured)
            else:
                row.api_key_configured = bool(api_key_configured)
            row.updated_at = utc_now()
            session.commit()

    def record_request(
        self,
        *,
        provider: str,
        model: str,
        stage: str,
        retry_count: int,
        latency_ms: float,
        success: bool,
        exc: BaseException | None = None,
    ) -> None:
        now = utc_now()
        with self.session_factory() as session:
            row = self._row(session)
            row.provider = provider
            row.model = model
            row.last_request_status = "success" if success else "failed"
            row.last_request_at = now
            row.last_request_stage = stage
            row.last_retry_count = max(0, int(retry_count))
            row.request_count = int(row.request_count or 0) + 1
            previous_count = row.request_count - 1
            row.average_latency_ms = ((float(row.average_latency_ms or 0.0) * previous_count) + max(0.0, latency_ms)) / row.request_count
            if stage == "page_extraction":
                row.last_page_latency_ms = max(0.0, latency_ms)
            elif stage in {"patient_review", "patient_synthesis"}:
                row.last_patient_latency_ms = max(0.0, latency_ms)
            if success:
                row.success_count = int(row.success_count or 0) + 1
                row.last_successful_request = now
            elif exc is not None:
                code, safe_message = classify_error(exc, stage=stage, provider=provider)
                row.last_error_code = code
                row.last_error_message = safe_message
                row.last_exception_type = exception_type(exc)
                row.last_http_status = http_status(exc)
            row.updated_at = now
            session.commit()

    def record_local_runtime(
        self,
        *,
        ollama_status: str,
        gpu_status: str,
        gpu_memory_bytes: int | None = None,
    ) -> None:
        with self.session_factory() as session:
            row = self._row(session)
            row.ollama_status = ollama_status[:32]
            row.gpu_status = gpu_status[:64]
            row.gpu_memory_bytes = gpu_memory_bytes
            row.updated_at = utc_now()
            session.commit()

    def record_test_success(self, *, stage: str, model: str, at: datetime | None = None) -> None:
        now = at or utc_now()
        with self.session_factory() as session:
            row = self._row(session)
            if stage == "text_connection_test":
                row.last_text_success_at = now
                row.last_text_success_model = model
            elif stage == "vision_connection_test":
                row.last_vision_success_at = now
                row.last_vision_success_model = model
            row.updated_at = now
            session.commit()

    def is_ready(self, *, provider: str, model: str) -> bool:
        with self.session_factory() as session:
            row = session.get(AIDiagnosticModel, 1)
            return bool(
                row
                and row.provider == provider
                and row.last_text_success_at
                and row.last_vision_success_at
                and row.last_text_success_model == model
                and row.last_vision_success_model == model
            )

    def snapshot(self, *, include_technical: bool = False) -> dict[str, Any]:
        with self.session_factory() as session:
            row = session.get(AIDiagnosticModel, 1)
            if row is None:
                return {
                    "provider": "unknown", "model": "unknown", "api_key_configured": False,
                    "worker_api_key_configured": False, "last_request_status": "never",
                    "last_error_code": None, "last_error_message": None, "last_successful_request": None,
                    "average_latency_ms": 0.0, "page_latency_ms": None, "patient_latency_ms": None,
                    "gpu_memory_bytes": None, "ollama_status": None, "gpu_status": None,
                    "ready_for_patient_processing": False,
                }
            result = {
                "provider": row.provider,
                "model": row.model,
                "api_key_configured": bool(row.api_key_configured),
                "worker_api_key_configured": bool(row.worker_api_key_configured),
                "last_request_status": row.last_request_status or "never",
                "last_error_code": row.last_error_code,
                "last_error_message": row.last_error_message,
                "last_successful_request": iso(row.last_successful_request),
                "average_latency_ms": round(float(row.average_latency_ms or 0.0), 1),
                "page_latency_ms": round(float(row.last_page_latency_ms), 1) if row.last_page_latency_ms is not None else None,
                "patient_latency_ms": round(float(row.last_patient_latency_ms), 1) if row.last_patient_latency_ms is not None else None,
                "gpu_memory_bytes": row.gpu_memory_bytes,
                "ollama_status": row.ollama_status,
                "gpu_status": row.gpu_status,
                "ready_for_patient_processing": bool(row.last_text_success_at and row.last_vision_success_at and row.last_text_success_model == row.model and row.last_vision_success_model == row.model),
            }
            if include_technical:
                result["technical_details"] = {
                    "last_exception_type": row.last_exception_type,
                    "http_status": row.last_http_status,
                    "request_stage": row.last_request_stage,
                    "retry_count": row.last_retry_count,
                    "timestamp": iso(row.last_request_at),
                    "text_test_success": iso(row.last_text_success_at),
                    "vision_test_success": iso(row.last_vision_success_at),
                    "request_count": row.request_count,
                    "success_count": row.success_count,
                }
            return result
