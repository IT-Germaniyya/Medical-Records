from __future__ import annotations

from abc import ABC, abstractmethod
from base64 import b64decode, b64encode
from dataclasses import dataclass, replace
from io import BytesIO
import json
import logging
import mimetypes
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import urlsplit
import ipaddress

from pypdf import PdfReader

from app.ai_schemas import ClinicalSynthesis, MedicalPageExtraction, PatientBundleItem, PatientLevelReview
from app.config import Settings, settings
from app.services.ai_diagnostics import AIDiagnosticsStore, classify_error, safe_error_message


logger = logging.getLogger("medicaldata.ai")


_PATIENT_FACT_PROPERTIES: dict[str, Any] = {
    "value": {"type": ["string", "number", "boolean", "null"]},
    "term": {"type": ["string", "null"]},
    "diagnosis": {"type": ["string", "null"]},
    "problem": {"type": ["string", "null"]},
    "name": {"type": ["string", "null"]},
    "patient_name": {"type": ["string", "null"]},
    "full_name": {"type": ["string", "null"]},
    "mrn": {"type": ["string", "null"]},
    "hospital_file_number": {"type": ["string", "null"]},
    "dob": {"type": ["string", "null"]},
    "date_of_birth": {"type": ["string", "null"]},
    "sex": {"type": ["string", "null"]},
    "nationality": {"type": ["string", "null"]},
    "address": {"type": ["string", "null"]},
    "normalized": {"type": ["string", "null"]},
    "normalized_value": {"type": ["string", "null"]},
    "test_name": {"type": ["string", "null"]},
    "test_name_original": {"type": ["string", "null"]},
    "test_name_normalized": {"type": ["string", "null"]},
    "value_text": {"type": ["string", "null"]},
    "unit": {"type": ["string", "null"]},
    "reference_range": {"type": ["string", "null"]},
    "abnormal_flag": {"type": ["string", "null"]},
    "date": {"type": ["string", "null"]},
    "specimen": {"type": ["string", "null"]},
    "name_original": {"type": ["string", "null"]},
    "name_normalized": {"type": ["string", "null"]},
    "generic_name": {"type": ["string", "null"]},
    "brand_name": {"type": ["string", "null"]},
    "strength": {"type": ["string", "null"]},
    "dose": {"type": ["string", "null"]},
    "dose_unit": {"type": ["string", "null"]},
    "route": {"type": ["string", "null"]},
    "frequency": {"type": ["string", "null"]},
    "duration": {"type": ["string", "null"]},
    "indication": {"type": ["string", "null"]},
    "status": {"type": ["string", "null"]},
    "type": {"type": ["string", "null"]},
    "certainty": {"type": ["string", "null"]},
    "confidence": {"type": ["number", "null"]},
    "study_type": {"type": ["string", "null"]},
    "body_part": {"type": ["string", "null"]},
    "report_text": {"type": ["string", "null"]},
    "text": {"type": ["string", "null"]},
    "event": {"type": ["string", "null"]},
    "summary": {"type": ["string", "null"]},
    "reason": {"type": ["string", "null"]},
    "field": {"type": ["string", "null"]},
    "proposed_value": {"type": ["string", "null"]},
    "raw_text": {"type": ["string", "null"]},
    "verification_required": {"type": ["boolean", "null"]},
    "age_months": {"type": ["number", "null"]},
    "weight_kg": {"type": ["number", "null"]},
    "height_cm": {"type": ["number", "null"]},
    "head_circumference_cm": {"type": ["number", "null"]},
    "bmi": {"type": ["number", "null"]},
    "facility": {"type": ["string", "null"]},
    "department": {"type": ["string", "null"]},
    "physician": {"type": ["string", "null"]},
    "source_files": {"type": "array", "items": {"type": "string"}},
    "source_pages": {"type": "array", "items": {"type": "integer"}},
    "findings": {"type": "array", "items": {"type": "string"}},
    "impression": {"type": "array", "items": {"type": "string"}},
}
_PATIENT_FACT_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "properties": _PATIENT_FACT_PROPERTIES,
    "required": list(_PATIENT_FACT_PROPERTIES),
}


@dataclass
class PageAnalysis:
    """Provider output kept separate from the canonical clinical record."""

    text: str | None
    structured: MedicalPageExtraction | None = None
    confidence: float = 0.0
    model: str | None = None
    prompt_version: str | None = None
    error: str | None = None
    retried_with_strong_model: bool = False


class DocumentVisionProvider(ABC):
    """Provider boundary. Production providers must pass privacy/security review."""

    name = "abstract"
    version = ""

    @abstractmethod
    def extract_text(self, document: Path, page_number: int) -> str | None:
        raise NotImplementedError

    def extract_page(
        self,
        document: Path,
        page_number: int,
        *,
        source_filename: str | None = None,
        document_type_hint: str | None = None,
    ) -> PageAnalysis:
        return PageAnalysis(text=self.extract_text(document, page_number), model=self.name, prompt_version=self.version)


class LocalSafeProvider(DocumentVisionProvider):
    """No-network MVP provider: PDF text extraction only, images are queued for review."""

    name = "local_safe"
    version = "1"

    def extract_text(self, document: Path, page_number: int) -> str | None:
        if document.suffix.lower() != ".pdf":
            return None
        try:
            pages = PdfReader(str(document)).pages
            if 1 <= page_number <= len(pages):
                value = pages[page_number - 1].extract_text()
                return value.strip() if value and value.strip() else None
        except Exception:
            return None
        return None


class OpenAIProviderConfigurationError(RuntimeError):
    """Raised when OpenAI is selected without a usable backend configuration."""


class OpenRouterProviderConfigurationError(RuntimeError):
    """Raised when OpenRouter is selected without a usable backend configuration."""


class LocalAIProviderConfigurationError(RuntimeError):
    """Raised when the local Ollama provider cannot be configured safely."""


class PatientLevelAIError(RuntimeError):
    """Raised when the required holistic patient review cannot be completed."""


class OpenAIMedicalVisionProvider(DocumentVisionProvider):
    """Multimodal extraction through the official OpenAI Responses API.

    The provider returns a validated ``MedicalPageExtraction`` object.  It never
    exposes the API key to the frontend and never logs request contents.  The
    canonical pipeline remains conservative: only documented values from this
    response are mapped into the existing record, while uncertain values enter
    the human-review queue.
    """

    name = "openai"
    version = "responses-v1"

    _PATIENT_REVIEW_SCHEMA: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "patient": _PATIENT_FACT_SCHEMA,
            "record_quality": _PATIENT_FACT_SCHEMA,
            "birth_history": _PATIENT_FACT_SCHEMA,
            **{name: {"type": "array", "items": _PATIENT_FACT_SCHEMA} for name in (
                "past_medical_history", "active_problems", "resolved_or_historical_problems", "encounters",
                "symptoms", "clinical_findings", "diagnoses_documented", "clinical_interpretations",
                "medications", "allergies", "laboratory_results", "radiology", "procedures", "vaccinations",
                "growth_measurements", "clinical_timeline", "uncertain_items", "conflicts",
            )},
            "growth_interpretation": {"type": "string"},
            "detailed_report_markdown": {"type": "string"},
            "erp_summary_markdown": {"type": "string"},
        },
        "required": [
            "patient", "record_quality", "birth_history", "past_medical_history", "active_problems",
            "resolved_or_historical_problems", "encounters", "symptoms", "clinical_findings",
            "diagnoses_documented", "clinical_interpretations", "medications", "allergies",
            "laboratory_results", "radiology", "procedures", "vaccinations", "growth_measurements",
            "growth_interpretation", "clinical_timeline", "uncertain_items", "conflicts",
            "detailed_report_markdown", "erp_summary_markdown",
        ],
    }

    _PAGE_SCHEMA: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "transcription": {"type": "string"},
            "document_type": {"type": "string"},
            "complexity": {"type": "string"},
            "encounter_date": {"type": ["string", "null"]},
            "patient_identity": {"$ref": "#/$defs/patient_identity"},
            "diagnoses": {"type": "array", "items": {"$ref": "#/$defs/diagnosis"}},
            "medications": {"type": "array", "items": {"$ref": "#/$defs/medication"}},
            "laboratory_results": {"type": "array", "items": {"$ref": "#/$defs/lab"}},
            "radiology": {"type": "array", "items": {"$ref": "#/$defs/radiology"}},
            "growth_measurements": {"type": "array", "items": {"$ref": "#/$defs/growth"}},
            "uncertain_items": {"type": "array", "items": {"$ref": "#/$defs/uncertain"}},
            "overall_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": [
            "transcription", "document_type", "complexity", "encounter_date", "patient_identity",
            "diagnoses", "medications", "laboratory_results", "radiology", "growth_measurements",
            "uncertain_items", "overall_confidence",
        ],
        "$defs": {
            "field": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "value": {"type": ["string", "number", "null"]}, "normalized_value": {"type": ["string", "null"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "certainty": {"type": "string"}, "documented_vs_inferred": {"type": "string"},
                    "source_page": {"type": "integer", "minimum": 1}, "source_file": {"type": "string"},
                    "needs_verification": {"type": "boolean"},
                },
                "required": ["value", "normalized_value", "confidence", "certainty", "documented_vs_inferred", "source_page", "source_file", "needs_verification"],
            },
            "patient_identity": {
                "type": "object", "additionalProperties": False,
                "properties": {name: {"anyOf": [{"$ref": "#/$defs/field"}, {"type": "null"}]} for name in (
                    "patient_name", "patient_id", "hospital_file_number", "date_of_birth", "sex", "nationality", "address"
                )},
                "required": ["patient_name", "patient_id", "hospital_file_number", "date_of_birth", "sex", "nationality", "address"],
            },
            "diagnosis": {
                "type": "object", "additionalProperties": False,
                "properties": {**{"value": {"type": ["string", "number", "null"]}, "normalized_value": {"type": ["string", "null"]}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "certainty": {"type": "string"}, "documented_vs_inferred": {"type": "string"}, "source_page": {"type": "integer", "minimum": 1}, "source_file": {"type": "string"}, "needs_verification": {"type": "boolean"}}, "status": {"type": "string"}},
                "required": ["value", "normalized_value", "confidence", "certainty", "documented_vs_inferred", "source_page", "source_file", "needs_verification", "status"],
            },
            "medication": {
                "type": "object", "additionalProperties": False,
                "properties": {**{name: {"type": ["string", "null"]} for name in ("value", "normalized_value", "name_original", "name_normalized", "generic_name", "brand_name", "strength", "dose", "dose_unit", "route", "frequency", "duration", "indication")}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "certainty": {"type": "string"}, "documented_vs_inferred": {"type": "string"}, "source_page": {"type": "integer", "minimum": 1}, "source_file": {"type": "string"}, "needs_verification": {"type": "boolean"}},
                "required": ["value", "normalized_value", "name_original", "name_normalized", "generic_name", "brand_name", "strength", "dose", "dose_unit", "route", "frequency", "duration", "indication", "confidence", "certainty", "documented_vs_inferred", "source_page", "source_file", "needs_verification"],
            },
            "lab": {
                "type": "object", "additionalProperties": False,
                "properties": {**{name: {"type": ["string", "null"]} for name in ("normalized_value", "test_name_original", "test_name_normalized", "value_text", "unit", "reference_range", "abnormal_flag", "date", "specimen")}, "value": {"type": ["number", "string", "null"]}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "certainty": {"type": "string"}, "documented_vs_inferred": {"type": "string"}, "source_page": {"type": "integer", "minimum": 1}, "source_file": {"type": "string"}, "needs_verification": {"type": "boolean"}},
                "required": ["value", "normalized_value", "test_name_original", "test_name_normalized", "value_text", "unit", "reference_range", "abnormal_flag", "date", "specimen", "confidence", "certainty", "documented_vs_inferred", "source_page", "source_file", "needs_verification"],
            },
            "growth": {
                "type": "object", "additionalProperties": False,
                "properties": {**{name: {"type": ["string", "null"]} for name in ("value", "normalized_value", "date")}, **{name: {"type": ["number", "null"]} for name in ("age_months", "weight_kg", "height_cm", "head_circumference_cm", "bmi")}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "certainty": {"type": "string"}, "documented_vs_inferred": {"type": "string"}, "source_page": {"type": "integer", "minimum": 1}, "source_file": {"type": "string"}, "needs_verification": {"type": "boolean"}},
                "required": ["value", "normalized_value", "date", "age_months", "weight_kg", "height_cm", "head_circumference_cm", "bmi", "confidence", "certainty", "documented_vs_inferred", "source_page", "source_file", "needs_verification"],
            },
            "radiology": {
                "type": "object", "additionalProperties": False,
                "properties": {**{name: {"type": ["string", "null"]} for name in ("value", "normalized_value", "study_type", "body_part", "date", "report_text")}, "findings": {"type": "array", "items": {"type": "string"}}, "impression": {"type": "array", "items": {"type": "string"}}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "certainty": {"type": "string"}, "documented_vs_inferred": {"type": "string"}, "source_page": {"type": "integer", "minimum": 1}, "source_file": {"type": "string"}, "needs_verification": {"type": "boolean"}},
                "required": ["value", "normalized_value", "study_type", "body_part", "date", "report_text", "findings", "impression", "confidence", "certainty", "documented_vs_inferred", "source_page", "source_file", "needs_verification"],
            },
            "uncertain": {
                "type": "object", "additionalProperties": False,
                "properties": {"field": {"type": "string"}, "proposed_value": {"type": ["string", "null"]}, "raw_text": {"type": ["string", "null"]}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "reason": {"type": "string"}, "severity": {"type": "string"}, "source_page": {"type": "integer", "minimum": 1}, "source_file": {"type": "string"}},
                "required": ["field", "proposed_value", "raw_text", "confidence", "reason", "severity", "source_page", "source_file"],
            },
        },
    }
    _SYNTHESIS_SCHEMA: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "problem_list": {"type": "array", "items": {"type": "string"}},
            "timeline": {"type": "array", "items": {"type": "string"}},
            "longitudinal_trends": {"type": "array", "items": {"type": "string"}},
            "physician_summary": {"type": "string"},
            "erp_summary": {"type": "string"},
            "overall_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["problem_list", "timeline", "longitudinal_trends", "physician_summary", "erp_summary", "overall_confidence"],
    }

    def __init__(self, configuration: Settings = settings, *, client: Any | None = None, api_key: str | None = None, base_url: str | None = None, diagnostics: AIDiagnosticsStore | None = None) -> None:
        if api_key and not configuration.openai_api_key:
            # Explicit injection is intended for tests/secret managers; normal
            # deployments should rely on OPENAI_API_KEY in the environment.
            configuration = replace(configuration, openai_api_key=api_key)
        self.configuration = configuration
        self.diagnostics = diagnostics
        self.base_url = base_url
        self.configured_model = configuration.openai_medical_model
        self._request_slots = threading.BoundedSemaphore(max(1, configuration.ai_max_concurrent_requests))
        if not configuration.openai_api_key and client is None:
            raise OpenAIProviderConfigurationError("OPENAI_API_KEY is required when AI_EXTRACTION_PROVIDER=openai")
        if client is not None:
            self.client = client
            self._mark_runtime()
            return
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised in deployment image
            raise OpenAIProviderConfigurationError("install the optional openai dependency") from exc
        client_kwargs: dict[str, Any] = {"api_key": configuration.openai_api_key, "timeout": configuration.ai_request_timeout}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self._mark_runtime()

    def _create_response(self, **kwargs: Any) -> Any:
        """Create a provider response; OpenRouter overrides this adapter."""

        return self.client.responses.create(**kwargs)

    @staticmethod
    def _response_text(response: Any) -> str | None:
        return getattr(response, "output_text", None)

    def _mark_runtime(self) -> None:
        if self.diagnostics is not None:
            self.diagnostics.mark_runtime(
                role="api_or_worker",
                provider=self.name,
                model=self.configuration.openai_medical_model,
                api_key_configured=bool(self.configuration.openai_api_key),
            )

    def _record_request(self, *, model: str, stage: str, retry_count: int, started: float, success: bool, exc: BaseException | None = None) -> None:
        if self.diagnostics is not None:
            self.diagnostics.record_request(
                provider=self.name,
                model=model,
                stage=stage,
                retry_count=retry_count,
                latency_ms=(time.monotonic() - started) * 1000,
                success=success,
                exc=exc,
            )

    def _log_failure(self, exc: BaseException, *, stage: str, model: str, retry_count: int) -> None:
        code, safe_message = classify_error(exc, stage=stage, provider=self.name)
        provider_label = "OpenRouter" if self.name == "openrouter" else "Local AI" if self.name == "local" else "OpenAI"
        if self.configuration.ai_debug:
            logger.exception(
                "%s request failed stage=%s model=%s code=%s retry_count=%s message=%s",
                provider_label,
                stage,
                model,
                code,
                retry_count,
                safe_message,
            )
        else:
            logger.warning(
                "%s request failed stage=%s model=%s code=%s retry_count=%s message=%s",
                provider_label,
                stage,
                model,
                code,
                retry_count,
                safe_message,
            )

    def test_text_connection(self) -> dict[str, Any]:
        """Run a minimal text-only Responses API request with no patient data."""

        stage = "text_connection_test"
        model = self.configuration.openai_medical_model
        started = time.monotonic()
        last_attempt = 0
        for attempt in range(self.configuration.ai_max_retries + 1):
            last_attempt = attempt
            try:
                with self._request_slots:
                    response = self._create_response(
                        model=model,
                        input="Reply with exactly OK.",
                        store=False,
                        timeout=self.configuration.ai_request_timeout,
                    )
                if not self._response_text(response):
                    raise ValueError("empty text test response")
                self._record_request(model=model, stage=stage, retry_count=attempt, started=started, success=True)
                if self.diagnostics is not None:
                    self.diagnostics.record_test_success(stage=stage, model=model)
                return {"success": True, "stage": stage, "provider": self.name, "model": model}
            except Exception as exc:
                if attempt >= self.configuration.ai_max_retries or not self._retryable(exc):
                    self._record_request(model=model, stage=stage, retry_count=attempt, started=started, success=False, exc=exc)
                    self._log_failure(exc, stage=stage, model=model, retry_count=attempt)
                    code, safe_message = classify_error(exc, stage=stage, provider=self.name)
                    raise PatientLevelAIError(safe_message) from exc
                time.sleep(min(2**attempt, 8))
        raise PatientLevelAIError("OpenAI request failed")

    @staticmethod
    def _synthetic_medical_image() -> bytes:
        """Create a deterministic, PHI-free medical-style test image in memory."""

        from PIL import Image, ImageDraw

        image = Image.new("RGB", (640, 360), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((18, 18, 622, 342), outline="#335c67", width=3)
        draw.text((32, 30), "SYNTHETIC MEDICAL VISION TEST", fill="#102a43")
        draw.text((32, 58), "No patient data / test fixture", fill="#617887")
        draw.line((32, 126, 608, 126), fill="#d7e3e8", width=1)
        points = [(32, 196), (70, 196), (88, 150), (106, 238), (126, 182), (150, 196), (188, 196), (206, 164), (224, 224), (244, 190), (270, 196), (310, 196), (332, 142), (350, 246), (370, 176), (394, 196), (440, 196), (460, 158), (480, 230), (500, 184), (530, 196), (608, 196)]
        draw.line(points, fill="#a13b46", width=4, joint="curve")
        draw.text((32, 286), "Synthetic ECG-like waveform", fill="#617887")
        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()

    def test_vision_connection(self) -> dict[str, Any]:
        """Run a minimal vision request against a bundled synthetic fixture."""

        stage = "vision_connection_test"
        model = self.configuration.openai_medical_model
        started = time.monotonic()
        image_data = self._synthetic_medical_image()
        payload = [{"role": "user", "content": [
            {"type": "input_text", "text": "Describe this synthetic test image in one short sentence. Do not infer a real diagnosis."},
            {"type": "input_image", "image_url": f"data:image/png;base64,{b64encode(image_data).decode('ascii')}", "detail": self.configuration.openai_image_detail},
        ]}]
        for attempt in range(self.configuration.ai_max_retries + 1):
            try:
                with self._request_slots:
                    response = self._create_response(model=model, input=payload, store=False, timeout=self.configuration.ai_request_timeout)
                if not self._response_text(response):
                    raise ValueError("empty vision test response")
                self._record_request(model=model, stage=stage, retry_count=attempt, started=started, success=True)
                if self.diagnostics is not None:
                    self.diagnostics.record_test_success(stage=stage, model=model)
                return {"success": True, "stage": stage, "provider": self.name, "model": model}
            except Exception as exc:
                if attempt >= self.configuration.ai_max_retries or not self._retryable(exc):
                    self._record_request(model=model, stage=stage, retry_count=attempt, started=started, success=False, exc=exc)
                    self._log_failure(exc, stage=stage, model=model, retry_count=attempt)
                    code, safe_message = classify_error(exc, stage=stage, provider=self.name)
                    raise PatientLevelAIError(safe_message) from exc
                time.sleep(min(2**attempt, 8))
        raise PatientLevelAIError("OpenAI request failed")

    def extract_text(self, document: Path, page_number: int) -> str | None:
        return self.extract_page(document, page_number).text

    def extract_page(
        self,
        document: Path,
        page_number: int,
        *,
        source_filename: str | None = None,
        document_type_hint: str | None = None,
    ) -> PageAnalysis:
        source_filename = source_filename or document.name
        visual_data, media_type, embedded_text = self._page_input(document, page_number)
        complexity = self._complexity(source_filename, embedded_text)
        first_model = self._model_for(complexity, strong=False)
        first = self._request(
            model=first_model,
            source_filename=source_filename,
            page_number=page_number,
            document_type_hint=document_type_hint,
            complexity=complexity,
            embedded_text=embedded_text,
            visual_data=visual_data,
            media_type=media_type,
        )
        if first.error or first.structured is None:
            return first
        if (
            first_model != self.configuration.openai_medical_model
            and first.confidence < self.configuration.fast_model_min_confidence
        ):
            if self.configuration.ai_debug:
                logger.info("Fast extraction confidence %.2f is below threshold; running second pass for page %s", first.confidence, page_number)
            second = self._request(
                model=self.configuration.openai_medical_model,
                source_filename=source_filename,
                page_number=page_number,
                document_type_hint=document_type_hint,
                complexity=complexity,
                embedded_text=embedded_text,
                visual_data=visual_data,
                media_type=media_type,
            )
            second.retried_with_strong_model = True
            return second
        if self.configuration.ai_debug:
            logger.info("Second pass not required for page %s", page_number)
        return first

    def review_patient(self, bundle: list[PatientBundleItem | dict[str, Any]]) -> PatientLevelReview:
        """Review the complete patient bundle in one multimodal call.

        A size-triggered hierarchical pass is the only fallback.  It still
        sends the actual source visuals to the model and performs a final
        patient-level synthesis; no OCR-first or page-by-page downgrade is
        performed.
        """
        items = [item if isinstance(item, PatientBundleItem) else PatientBundleItem.model_validate(item) for item in bundle]
        if not items:
            raise PatientLevelAIError("AI medical review could not be completed: no source documents were available")
        total_bytes = sum(Path(item.local_path).stat().st_size for item in items if Path(item.local_path).is_file())
        if total_bytes <= self.configuration.ai_max_patient_bytes:
            return self._request_patient_group(items)
        if not self.configuration.ai_allow_hierarchical_fallback:
            raise PatientLevelAIError("AI medical review could not be completed: patient record exceeds configured multimodal size limits")
        group_size = max(1, self.configuration.ai_pages_per_request)
        groups = [items[index:index + group_size] for index in range(0, len(items), group_size)]
        logger.info("Patient record is large; using hierarchical multimodal review with %s source groups", len(groups))
        reviews = [self._request_patient_group(group) for group in groups]
        return self._synthesize_patient_reviews(reviews)

    def _request_patient_group(self, items: list[PatientBundleItem]) -> PatientLevelReview:
        instructions = (
            "You are reviewing the complete medical record of a single patient.\n"
            "Read ALL attached pages and images before creating the final clinical reconstruction.\n"
            "Correlate information across visits and reports. Identify repeated findings, longitudinal changes, "
            "resolved problems, persistent problems, and conflicting data. Do not treat each page as an isolated document.\n\n"
            "Distinguish explicitly documented diagnoses, examination findings, laboratory findings, historical diagnoses, "
            "AI clinical interpretation, suspected/probable diagnoses, and unreadable or uncertain information. "
            "Never invent a date, medication dose, diagnosis, percentile, or allergy. Use null or an empty array when absent. "
            "Every major fact must include source_files and source_pages and a confidence. Handwriting that is not reliable must be marked "
            "'Illegible / requires clinician verification'. Return only the strict patient-level JSON schema.\n\n"
            "Source manifest (preserve these IDs and order in all source references):\n"
            + json.dumps([item.model_dump(mode="json", exclude={"local_path"}) for item in items], ensure_ascii=False)
        )
        content: list[dict[str, Any]] = [{"type": "input_text", "text": instructions}]
        seen_files: set[str] = set()
        for item in sorted(items, key=lambda value: value.order_index):
            path = Path(item.local_path)
            if not path.is_file():
                continue
            data = path.read_bytes()
            media_type = item.media_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if media_type == "application/pdf":
                if str(path) in seen_files:
                    continue
                content.append({"type": "input_file", "filename": item.relative_path or path.name, "file_data": f"data:application/pdf;base64,{b64encode(data).decode('ascii')}"})
                seen_files.add(str(path))
            elif media_type.startswith("image/"):
                content.append({"type": "input_image", "image_url": f"data:{media_type};base64,{b64encode(self._resize_image(data, media_type)).decode('ascii')}", "detail": self.configuration.openai_image_detail})
        payload = [{"role": "user", "content": content}]
        started = time.monotonic()
        for attempt in range(self.configuration.ai_max_retries + 1):
            try:
                with self._request_slots:
                    response = self._create_response(
                        model=self.configuration.openai_medical_model,
                        instructions=instructions,
                        input=payload,
                        text={"format": {"type": "json_schema", "name": "patient_level_medical_review", "strict": True, "schema": self._PATIENT_REVIEW_SCHEMA}},
                        store=False,
                        timeout=self.configuration.ai_request_timeout,
                    )
                raw = self._response_text(response)
                if not raw:
                    raise ValueError("empty patient-level response")
                result = PatientLevelReview.model_validate(json.loads(raw))
                self._record_request(model=self.configuration.openai_medical_model, stage="patient_review", retry_count=attempt, started=started, success=True)
                if self.configuration.ai_debug:
                    logger.info("Patient-level multimodal review received in %.1f seconds", time.monotonic() - started)
                return result
            except Exception as exc:
                if attempt >= self.configuration.ai_max_retries or not self._retryable(exc):
                    self._record_request(model=self.configuration.openai_medical_model, stage="patient_review", retry_count=attempt, started=started, success=False, exc=exc)
                    self._log_failure(exc, stage="patient_review", model=self.configuration.openai_medical_model, retry_count=attempt)
                    _, safe_message = classify_error(exc, stage="patient_review", provider=self.name)
                    raise PatientLevelAIError(safe_message) from exc
                time.sleep(min(2**attempt, 8))
        raise PatientLevelAIError("AI medical review could not be completed")

    def review_deidentified_record(self, payload: dict[str, Any]) -> PatientLevelReview:
        """Run the same canonical review schema over a de-identified record.

        Comparison mode deliberately receives only redacted structured facts;
        it never sends the patient's original files or identifiers.
        """

        stage = "comparison_review"
        instructions = (
            "Review this de-identified structured medical record conservatively. Use only documented facts in the payload. "
            "Never invent dates, doses, diagnoses, percentiles, allergies, or identifiers. Preserve uncertainty, conflicts, and source references. "
            "Return the exact patient-level schema, including a detailed report and ERP summary."
        )
        payload_input = [{"role": "user", "content": [{"type": "input_text", "text": instructions + "\nREDACTED RECORD:\n" + json.dumps(payload, ensure_ascii=False)}]}]
        started = time.monotonic()
        try:
            with self._request_slots:
                response = self._create_response(
                    model=self.configuration.openai_medical_model,
                    instructions=instructions,
                    input=payload_input,
                    text={"format": {"type": "json_schema", "name": "patient_level_medical_review", "strict": True, "schema": self._PATIENT_REVIEW_SCHEMA}},
                    store=False,
                    timeout=self.configuration.ai_request_timeout,
                )
            raw = self._response_text(response)
            if not raw:
                raise ValueError("empty comparison response")
            result = PatientLevelReview.model_validate(json.loads(raw))
            self._record_request(model=self.configuration.openai_medical_model, stage=stage, retry_count=0, started=started, success=True)
            return result
        except Exception as exc:
            self._record_request(model=self.configuration.openai_medical_model, stage=stage, retry_count=0, started=started, success=False, exc=exc)
            self._log_failure(exc, stage=stage, model=self.configuration.openai_medical_model, retry_count=0)
            _, safe_message = classify_error(exc, stage=stage, provider=self.name)
            raise PatientLevelAIError(safe_message) from exc

    def _synthesize_patient_reviews(self, reviews: list[PatientLevelReview]) -> PatientLevelReview:
        """Synthesize multimodal group reviews into the same canonical schema."""
        instructions = (
            "You are synthesizing multimodal reviews of one patient's complete medical record. "
            "Reconcile duplicates and conflicts, preserve source references and uncertainty, and produce the exact same "
            "patient-level schema. Do not add facts absent from the group reviews. Write a coherent detailed physician report "
            "and a concise approximately one-page ERP physician summary."
        )
        payload = [{"role": "user", "content": [{"type": "input_text", "text": instructions + "\nGROUP REVIEWS:\n" + json.dumps([item.model_dump(mode="json") for item in reviews], ensure_ascii=False)}]}]
        started = time.monotonic()
        try:
            with self._request_slots:
                response = self._create_response(
                    model=self.configuration.openai_summary_model,
                    instructions=instructions,
                    input=payload,
                    text={"format": {"type": "json_schema", "name": "patient_level_medical_review", "strict": True, "schema": self._PATIENT_REVIEW_SCHEMA}},
                    store=False,
                    timeout=self.configuration.ai_request_timeout,
                )
            raw = self._response_text(response)
            if not raw:
                raise ValueError("empty patient-level synthesis response")
            result = PatientLevelReview.model_validate(json.loads(raw))
            self._record_request(model=self.configuration.openai_summary_model, stage="patient_synthesis", retry_count=0, started=started, success=True)
            return result
        except Exception as exc:
            self._record_request(model=self.configuration.openai_summary_model, stage="patient_synthesis", retry_count=0, started=started, success=False, exc=exc)
            self._log_failure(exc, stage="patient_synthesis", model=self.configuration.openai_summary_model, retry_count=0)
            _, safe_message = classify_error(exc, stage="patient_synthesis", provider=self.name)
            raise PatientLevelAIError(safe_message) from exc

    def synthesize(self, structured_facts: dict[str, Any]) -> ClinicalSynthesis | None:
        """Run stage C using only the already validated structured record."""
        prompt_path = Path(self.configuration.openai_prompt_root) / "clinical_synthesis_v1.txt"
        instructions = prompt_path.read_text(encoding="utf-8") if prompt_path.is_file() else (
            "Use only validated structured facts. Do not invent diagnoses, doses, dates, allergies, or trends. "
            "Separate documented facts from uncertainty and return only the requested JSON schema."
        )
        payload = [{"role": "user", "content": [{"type": "input_text", "text": json.dumps(structured_facts, ensure_ascii=False, default=str)}]}]
        started = time.monotonic()
        if self.configuration.ai_debug:
            logger.info("Sending validated structured facts to OpenAI for longitudinal synthesis")
        for attempt in range(self.configuration.ai_max_retries + 1):
            try:
                with self._request_slots:
                    response = self._create_response(
                        model=self.configuration.openai_reasoning_model,
                        instructions=instructions,
                        input=payload,
                        text={"format": {"type": "json_schema", "name": "clinical_synthesis", "strict": True, "schema": self._SYNTHESIS_SCHEMA}},
                        store=False,
                        timeout=self.configuration.ai_request_timeout,
                    )
                raw = self._response_text(response)
                if not raw:
                    raise ValueError("empty synthesis response")
                synthesis = ClinicalSynthesis.model_validate(json.loads(raw))
                if self.configuration.ai_debug:
                    logger.info("Longitudinal synthesis received in %.1f seconds", time.monotonic() - started)
                return synthesis
            except Exception as exc:
                if attempt >= self.configuration.ai_max_retries or not self._retryable(exc):
                    return None
                time.sleep(min(2**attempt, 8))
        return None

    def _model_for(self, complexity: str, *, strong: bool) -> str:
        if strong or complexity in {"handwritten", "prescription", "growth_chart", "complex_form", "poor_quality"}:
            return self.configuration.openai_medical_model
        return self.configuration.openai_fast_model

    @staticmethod
    def _complexity(filename: str, text: str | None) -> str:
        haystack = f"{filename} {text or ''}".casefold()
        if any(word in haystack for word in ("prescription", "rx", "medication")):
            return "prescription"
        if any(word in haystack for word in ("growth", "height", "weight", "percentile")):
            return "growth_chart"
        if any(word in haystack for word in ("handwritten", "manuscript")):
            return "handwritten"
        if not text:
            return "poor_quality"
        if any(word in haystack for word in ("form", "registration", "table", "lab", "cbc")):
            return "table"
        return "simple_printed"

    def _request(
        self,
        *,
        model: str,
        source_filename: str,
        page_number: int,
        document_type_hint: str | None,
        complexity: str,
        embedded_text: str | None,
        visual_data: bytes | None,
        media_type: str | None,
    ) -> PageAnalysis:
        prompt_version = self._prompt_version(document_type_hint)
        instructions = self._prompt(document_type_hint)
        content: list[dict[str, Any]] = [{"type": "input_text", "text": self._user_instruction(source_filename, page_number, complexity, embedded_text)}]
        if visual_data is not None and media_type:
            content.append({"type": "input_image", "image_url": f"data:{media_type};base64,{b64encode(visual_data).decode('ascii')}"})
        payload = [{"role": "user", "content": content}]
        last_error = "request_failed"
        started = time.monotonic()
        if self.configuration.ai_debug:
            logger.info("Sending page %s to OpenAI using model %s", page_number, model)
        for attempt in range(self.configuration.ai_max_retries + 1):
            try:
                with self._request_slots:
                    response = self._create_response(
                        model=model,
                        instructions=instructions,
                        input=payload,
                        text={"format": {"type": "json_schema", "name": "medical_page_extraction", "strict": True, "schema": self._PAGE_SCHEMA}},
                        store=False,
                        timeout=self.configuration.ai_request_timeout,
                    )
                raw = self._response_text(response)
                if not raw:
                    raise ValueError("empty structured response")
                parsed = MedicalPageExtraction.model_validate(json.loads(raw))
                self._record_request(model=model, stage="page_extraction", retry_count=attempt, started=started, success=True)
                if self.configuration.ai_debug:
                    logger.info("OpenAI response received in %.1f seconds", time.monotonic() - started)
                    logger.info("Extraction stored for page %s", page_number)
                return PageAnalysis(text=parsed.transcription or None, structured=parsed, confidence=parsed.overall_confidence, model=model, prompt_version=prompt_version)
            except Exception as exc:  # SDK exception classes vary between releases.
                last_error = self._error_code(exc)
                if attempt >= self.configuration.ai_max_retries or not self._retryable(exc):
                    self._record_request(model=model, stage="page_extraction", retry_count=attempt, started=started, success=False, exc=exc)
                    self._log_failure(exc, stage="page_extraction", model=model, retry_count=attempt)
                    break
                time.sleep(min(2**attempt, 8))
        return PageAnalysis(text=None, confidence=0.0, model=model, prompt_version=prompt_version, error=last_error)

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        name = type(exc).__name__.casefold()
        message = str(exc).casefold()
        if "insufficient_quota" in message or "credit_balance_exhausted" in message or "no credits remaining" in message:
            return False
        return any(token in name for token in ("timeout", "ratelimit", "connection", "internalserver", "serviceunavailable", "apierror"))

    @staticmethod
    def _error_code(exc: Exception) -> str:
        name = type(exc).__name__.casefold()
        if "rate" in name:
            return "rate_limit"
        if "timeout" in name:
            return "timeout"
        if "json" in name or "validation" in name or "valueerror" in name:
            return "invalid_structured_response"
        return "api_error"

    @staticmethod
    def _prompt_version(document_type_hint: str | None) -> str:
        slug = (document_type_hint or "medical_page").casefold().replace(" ", "_")
        return f"{slug}_v1"

    def _prompt(self, document_type_hint: str | None) -> str:
        prompt_name = {
            "lab_report": "lab_extraction_v1.txt",
            "prescription": "prescription_extraction_v1.txt",
            "growth_chart": "growth_chart_v1.txt",
            "clinical_note": "medical_page_extraction_v1.txt",
        }.get((document_type_hint or "").casefold(), "medical_page_extraction_v1.txt")
        base = Path(self.configuration.openai_prompt_root) / prompt_name
        if base.is_file():
            return base.read_text(encoding="utf-8")
        return (
            "You are a medical records transcription and extraction service. Transcribe exactly before interpreting. "
            "Never guess unreadable handwriting, medication dose, dates, laboratory values, or diagnoses. "
            "Separate documented source facts from inference; mark uncertainty and conflicts for human review. "
            f"Use the specialized document mode: {document_type_hint or 'unknown'}. Return only the requested JSON schema."
        )

    @staticmethod
    def _user_instruction(filename: str, page_number: int, complexity: str, embedded_text: str | None) -> str:
        text = embedded_text or "(no reliable embedded text; inspect the supplied page image)"
        return (
            f"Analyze source file {filename!r}, page {page_number}. Complexity hint: {complexity}. "
            "Return structured facts found on this page only. Preserve exact dates and terminology. "
            "Do not infer facts from filename or page order. Embedded text, if present, is only a reference:\n" + text[:20000]
        )

    def _page_input(self, document: Path, page_number: int) -> tuple[bytes | None, str | None, str | None]:
        if document.suffix.casefold() != ".pdf":
            try:
                data = document.read_bytes()
                media_type = mimetypes.guess_type(document.name)[0] or "image/jpeg"
                return self._resize_image(data, media_type), media_type, None
            except OSError:
                return None, None, None
        embedded_text: str | None = None
        try:
            pages = PdfReader(str(document)).pages
            if 1 <= page_number <= len(pages):
                value = pages[page_number - 1].extract_text()
                embedded_text = value.strip() if value and value.strip() else None
        except Exception:
            pass
        # PyMuPDF is an optional runtime dependency used only for scanned/visual PDFs.
        try:
            try:
                import pymupdf as fitz  # type: ignore
            except ImportError:  # pragma: no cover - older PyMuPDF releases
                import fitz  # type: ignore

            pdf = fitz.open(str(document))
            if 1 <= page_number <= len(pdf):
                pixmap = pdf[page_number - 1].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                return self._resize_image(bytes(pixmap.tobytes("png")), "image/png"), "image/png", embedded_text
        except Exception:
            pass
        return None, None, embedded_text

    def _resize_image(self, data: bytes, media_type: str) -> bytes:
        """Bound very large uploads while retaining the original on disk."""
        limit = max(512, self.configuration.ai_max_image_dimension)
        try:
            from PIL import Image

            with Image.open(BytesIO(data)) as image:
                if max(image.size) <= limit:
                    return data
                ratio = limit / max(image.size)
                resized = image.resize((max(1, int(image.width * ratio)), max(1, int(image.height * ratio))), Image.Resampling.LANCZOS)
                output = BytesIO()
                format_name = "PNG" if media_type == "image/png" else "JPEG"
                if format_name == "JPEG":
                    resized.convert("RGB").save(output, format=format_name, quality=92)
                else:
                    resized.save(output, format=format_name)
                return output.getvalue()
        except Exception:
            return data


class OpenRouterMedicalVisionProvider(OpenAIMedicalVisionProvider):
    """OpenRouter's OpenAI-compatible chat-completions multimodal provider.

    The canonical patient schema and safety instructions remain inherited from
    the OpenAI provider.  Only transport adaptation differs: Responses input
    items are translated to chat message content, and PDF file inputs are
    rendered to page images because OpenRouter chat models accept image parts
    but do not expose the Responses ``input_file`` contract uniformly.
    """

    name = "openrouter"
    version = "openrouter-chat-v1"

    def __init__(self, configuration: Settings = settings, *, client: Any | None = None, diagnostics: AIDiagnosticsStore | None = None) -> None:
        if not configuration.openrouter_api_key and client is None:
            raise OpenRouterProviderConfigurationError("OPENROUTER_API_KEY is required when AI_PROVIDER=openrouter")
        model = configuration.openrouter_model
        router_configuration = replace(
            configuration,
            openai_api_key=configuration.openrouter_api_key,
            openai_medical_model=model,
            openai_fast_model=model,
            openai_reasoning_model=model,
            openai_summary_model=model,
        )
        super().__init__(router_configuration, client=client, base_url=configuration.openrouter_base_url, diagnostics=diagnostics)
        self.openrouter_model = model
        self.openrouter_base_url = configuration.openrouter_base_url
        self.configured_model = model

    def _create_response(self, **kwargs: Any) -> Any:
        model = kwargs.pop("model")
        instructions = kwargs.pop("instructions", None)
        input_value = kwargs.pop("input", "")
        text = kwargs.pop("text", None)
        timeout = kwargs.pop("timeout", None)
        kwargs.pop("store", None)
        messages = self._to_chat_messages(input_value, instructions)
        request: dict[str, Any] = {"model": model, "messages": messages}
        if text:
            response_format = self._chat_response_format(text)
            if response_format:
                request["response_format"] = response_format
        if timeout is not None:
            request["timeout"] = timeout
        return self.client.chat.completions.create(**request)

    @staticmethod
    def _response_text(response: Any) -> str | None:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return None
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if isinstance(content, list):
            return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict)) or None
        return content if isinstance(content, str) else None

    @staticmethod
    def _chat_response_format(text: dict[str, Any]) -> dict[str, Any] | None:
        fmt = text.get("format") if isinstance(text, dict) else None
        if not isinstance(fmt, dict) or fmt.get("type") != "json_schema":
            return None
        return {
            "type": "json_schema",
            "json_schema": {
                "name": fmt.get("name", "medical_record_output"),
                "strict": bool(fmt.get("strict", True)),
                "schema": fmt.get("schema", {}),
            },
        }

    def _to_chat_messages(self, input_value: Any, instructions: str | None) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        if isinstance(input_value, str):
            messages.append({"role": "user", "content": input_value})
            return messages
        for message in input_value or []:
            if not isinstance(message, dict):
                continue
            role = message.get("role", "user")
            content = message.get("content", "")
            if isinstance(content, str):
                messages.append({"role": role, "content": content})
                continue
            parts: list[dict[str, Any]] = []
            for part in content or []:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "input_text":
                    parts.append({"type": "text", "text": part.get("text", "")})
                elif part_type == "input_image":
                    image_url: dict[str, Any] = {"url": part.get("image_url", "")}
                    if part.get("detail"):
                        image_url["detail"] = part["detail"]
                    parts.append({"type": "image_url", "image_url": image_url})
                elif part_type == "input_file":
                    parts.extend(self._pdf_file_parts(part))
            messages.append({"role": role, "content": parts})
        return messages

    def _pdf_file_parts(self, part: dict[str, Any]) -> list[dict[str, Any]]:
        file_data = part.get("file_data") or ""
        if not isinstance(file_data, str) or "," not in file_data:
            return [{"type": "text", "text": "[Source file could not be prepared for this provider; verify it manually.]"}]
        try:
            try:
                import pymupdf as fitz  # type: ignore
            except ImportError:
                import fitz  # type: ignore

            raw = __import__("base64").b64decode(file_data.split(",", 1)[1])
            document = fitz.open(stream=raw, filetype="pdf")
            output: list[dict[str, Any]] = []
            for page in document:
                pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                encoded = b64encode(pixmap.tobytes("png")).decode("ascii")
                output.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}", "detail": self.configuration.openai_image_detail}})
            return output or [{"type": "text", "text": "[Empty PDF source; verify it manually.]"}]
        except Exception:
            return [{"type": "text", "text": "[PDF source could not be prepared for this provider; verify it manually.]"}]


class LocalMedicalVisionProvider(OpenAIMedicalVisionProvider):
    """Patient-level multimodal extraction through a local Ollama server.

    This deliberately reuses the canonical patient schema, prompts, validation,
    provenance, and safety policy from the OpenAI provider.  Transport is a
    local-only Ollama ``/api/generate`` request with images encoded in memory;
    there is no cloud fallback and no API key.
    """

    name = "local"
    version = "ollama-generate-v1"

    def __init__(
        self,
        configuration: Settings = settings,
        *,
        http_client: Any | None = None,
        diagnostics: AIDiagnosticsStore | None = None,
    ) -> None:
        model = configuration.local_ai_model.strip()
        if not model:
            raise LocalAIProviderConfigurationError("LOCAL_AI_MODEL is required when AI_PROVIDER=local")
        base_url = configuration.local_ai_base_url.strip().rstrip("/")
        if not base_url or not base_url.startswith(("http://", "https://")):
            raise LocalAIProviderConfigurationError("LOCAL_AI_BASE_URL must be an http(s) URL")
        parsed_base_url = urlsplit(base_url)
        if parsed_base_url.username or parsed_base_url.password:
            raise LocalAIProviderConfigurationError("LOCAL_AI_BASE_URL must not contain URL credentials")
        hostname = (parsed_base_url.hostname or "").casefold().rstrip(".")
        try:
            local_host = ipaddress.ip_address(hostname).is_private or ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            local_host = hostname in {"localhost", "host.docker.internal", "ollama"} or hostname.endswith(".local") or "." not in hostname
        if not local_host:
            raise LocalAIProviderConfigurationError("LOCAL_AI_BASE_URL must point to a local/private Ollama host")
        local_configuration = replace(
            configuration,
            openai_api_key=None,
            openai_medical_model=model,
            openai_fast_model=model,
            openai_reasoning_model=model,
            openai_summary_model=model,
        )
        # The parent constructor is used only for shared schemas, request
        # accounting, and safety logic. It never contacts OpenAI when a client
        # object is supplied; all network traffic goes through http_client.
        super().__init__(local_configuration, client=object(), diagnostics=diagnostics)
        self.local_ai_base_url = base_url
        self.local_ai_model = model
        if http_client is not None:
            self.http_client = http_client
        else:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - dependency is part of the service image
                raise LocalAIProviderConfigurationError("install the httpx dependency for local AI") from exc
            self.http_client = httpx.Client(
                base_url=base_url,
                timeout=configuration.ai_request_timeout,
                trust_env=False,
            )
        self._ocr_engine: Any | None = None
        self._ocr_unavailable_logged = False

    def _create_response(self, **kwargs: Any) -> Any:
        model = kwargs.pop("model")
        instructions = kwargs.pop("instructions", None)
        input_value = kwargs.pop("input", "")
        text = kwargs.pop("text", None)
        timeout = kwargs.pop("timeout", None)
        kwargs.pop("store", None)
        prompt, images = self._to_ollama_input(input_value, instructions)
        request: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "images": images,
            "stream": False,
        }
        output_format = self._ollama_response_format(text)
        if output_format is not None:
            request["format"] = output_format
        response = self.http_client.post("/api/generate", json=request, timeout=timeout)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _response_text(response: Any) -> str | None:
        if isinstance(response, dict):
            value = response.get("response")
            return value if isinstance(value, str) else None
        value = getattr(response, "response", None)
        return value if isinstance(value, str) else None

    @staticmethod
    def _ollama_response_format(text: Any) -> dict[str, Any] | str | None:
        if not isinstance(text, dict):
            return None
        fmt = text.get("format")
        if not isinstance(fmt, dict):
            return "json" if fmt == "json" else None
        if fmt.get("type") == "json_schema":
            # Ollama accepts a JSON schema directly in `format`.
            return fmt.get("schema") or None
        return fmt if fmt.get("type") == "object" else None

    def _to_ollama_input(self, input_value: Any, instructions: str | None) -> tuple[str, list[str]]:
        prompt_parts: list[str] = []
        if instructions:
            prompt_parts.append(instructions)
        images: list[str] = []
        if isinstance(input_value, str):
            prompt_parts.append(input_value)
            return "\n\n".join(prompt_parts), images
        for message in input_value or []:
            if not isinstance(message, dict):
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                prompt_parts.append(content)
                continue
            for part in content or []:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "input_text":
                    prompt_parts.append(str(part.get("text", "")))
                elif part_type == "input_image":
                    raw = self._decode_data_url(part.get("image_url"))
                    if raw is None:
                        raise ValueError("image input could not be prepared locally")
                    images.append(b64encode(raw).decode("ascii"))
                    ocr_text = self._optional_ocr(raw)
                    if ocr_text:
                        prompt_parts.append("[Optional local PaddleOCR-VL text for the following image]\n" + ocr_text)
                elif part_type == "input_file":
                    rendered_pages = self._pdf_file_images(part)
                    if not rendered_pages:
                        raise ValueError("PDF input could not be prepared locally")
                    for raw, page_number in rendered_pages:
                        images.append(b64encode(raw).decode("ascii"))
                        ocr_text = self._optional_ocr(raw)
                        if ocr_text:
                            prompt_parts.append(f"[Optional local PaddleOCR-VL text for PDF page {page_number}]\n{ocr_text}")
        return "\n\n".join(part for part in prompt_parts if part), images

    @staticmethod
    def _decode_data_url(value: Any) -> bytes | None:
        if not isinstance(value, str) or "," not in value or not value.startswith("data:"):
            return None
        try:
            return b64decode(value.split(",", 1)[1], validate=True)
        except Exception:
            return None

    def _pdf_file_images(self, part: dict[str, Any]) -> list[tuple[bytes, int]]:
        raw_pdf = self._decode_data_url(part.get("file_data"))
        if raw_pdf is None:
            raise ValueError("PDF input could not be prepared locally")
        try:
            try:
                import pymupdf as fitz  # type: ignore
            except ImportError:  # pragma: no cover - older PyMuPDF releases
                import fitz  # type: ignore
            document = fitz.open(stream=raw_pdf, filetype="pdf")
            pages: list[tuple[bytes, int]] = []
            for index, page in enumerate(document, start=1):
                pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                pages.append((self._resize_image(bytes(pixmap.tobytes("png")), "image/png"), index))
            return pages
        except Exception as exc:
            if self.configuration.ai_debug:
                logger.exception("Local PDF page rendering failed")
            raise ValueError("PDF input could not be prepared locally") from exc

    def _optional_ocr(self, image_data: bytes) -> str | None:
        if not self.configuration.local_ocr_enabled:
            return None
        try:
            if self._ocr_engine is None:
                from paddleocr import PaddleOCRVL  # type: ignore
                self._ocr_engine = PaddleOCRVL()
            try:
                result = self._ocr_engine.predict(image_data)
            except TypeError:
                result = self._ocr_engine.predict(input=image_data)
            return self._extract_ocr_text(result)
        except ImportError:
            if not self._ocr_unavailable_logged:
                logger.warning("LOCAL_OCR_ENABLED=true but PaddleOCR-VL is not installed; continuing with local VLM images only")
                self._ocr_unavailable_logged = True
            return None
        except Exception:
            # OCR is an optional local augmentation. Never replace the image
            # path or fall back to a cloud service when it fails.
            if self.configuration.ai_debug:
                logger.exception("Local PaddleOCR-VL augmentation failed")
            return None

    @staticmethod
    def _extract_ocr_text(result: Any) -> str | None:
        values: list[str] = []

        def visit(value: Any) -> None:
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
            elif isinstance(value, dict):
                for key in ("text", "rec_texts", "texts", "content"):
                    if key in value:
                        visit(value[key])
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)
            elif hasattr(value, "json"):
                try:
                    payload = value.json() if callable(value.json) else value.json
                    visit(payload)
                except Exception:
                    pass

        visit(result)
        deduped = list(dict.fromkeys(values))
        return "\n".join(deduped)[:20000] if deduped else None

    def health(self) -> dict[str, Any]:
        """Check only the configured local Ollama endpoint and GPU metadata."""

        try:
            tags_response = self.http_client.get("/api/tags", timeout=min(self.configuration.ai_request_timeout, 10))
            tags_response.raise_for_status()
            tags_payload = tags_response.json() or {}
            models = tags_payload.get("models", []) if isinstance(tags_payload, dict) else []
            names = [str(item.get("name", "")) for item in models if isinstance(item, dict)]
            configured_model_base = self.local_ai_model.split(":", 1)[0]
            model_available = any(name == self.local_ai_model or name.split(":", 1)[0] == configured_model_base for name in names)
            gpu_status = "unknown"
            gpu_memory_bytes: int | None = None
            try:
                ps_response = self.http_client.get("/api/ps", timeout=min(self.configuration.ai_request_timeout, 10))
                ps_response.raise_for_status()
                ps_payload = ps_response.json() or {}
                running_models = ps_payload.get("models", []) if isinstance(ps_payload, dict) else []
                vram_values = [int(item.get("size_vram", 0)) for item in running_models if isinstance(item, dict) and item.get("size_vram") is not None]
                gpu_memory_bytes = sum(vram_values) if vram_values else 0
                gpu_status = "gpu" if gpu_memory_bytes > 0 else "cpu_or_idle"
            except Exception:
                gpu_status = "unknown"
            ollama_status = "healthy" if model_available else "model_missing"
            if self.diagnostics is not None:
                self.diagnostics.record_local_runtime(ollama_status=ollama_status, gpu_status=gpu_status, gpu_memory_bytes=gpu_memory_bytes)
            return {
                "status": ollama_status,
                "base_url": self.local_ai_base_url,
                "model": self.local_ai_model,
                "model_available": model_available,
                "available_models": names[:50],
                "gpu_status": gpu_status,
                "gpu_memory_bytes": gpu_memory_bytes,
                "ocr_enabled": bool(self.configuration.local_ocr_enabled),
            }
        except Exception as exc:
            if self.diagnostics is not None:
                self.diagnostics.record_local_runtime(ollama_status="unavailable", gpu_status="unknown", gpu_memory_bytes=None)
            return {
                "status": "unavailable",
                "base_url": self.local_ai_base_url,
                "model": self.local_ai_model,
                "model_available": False,
                "available_models": [],
                "gpu_status": "unknown",
                "gpu_memory_bytes": None,
                "ocr_enabled": bool(self.configuration.local_ocr_enabled),
                "error": safe_error_message(exc),
            }


MedicalAIProvider = DocumentVisionProvider


def provider_for_settings(configuration: Settings = settings, *, diagnostics: AIDiagnosticsStore | None = None) -> DocumentVisionProvider:
    """Build the configured provider without ever passing secrets to the UI."""

    provider = configuration.ai_provider.casefold()
    if provider in {"openai", "openai_vision", "openai_medical"}:
        return OpenAIMedicalVisionProvider(configuration, diagnostics=diagnostics)
    if provider in {"openrouter", "openrouter_vision", "openrouter_medical"}:
        return OpenRouterMedicalVisionProvider(configuration, diagnostics=diagnostics)
    if provider in {"local", "local_ai", "ollama", "ollama_local"}:
        return LocalMedicalVisionProvider(configuration, diagnostics=diagnostics)
    return LocalSafeProvider()
