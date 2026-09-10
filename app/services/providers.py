from __future__ import annotations

from abc import ABC, abstractmethod
from base64 import b64encode
from dataclasses import dataclass, replace
from io import BytesIO
import json
import logging
import mimetypes
from pathlib import Path
import threading
import time
from typing import Any

from pypdf import PdfReader

from app.ai_schemas import ClinicalSynthesis, MedicalPageExtraction
from app.config import Settings, settings


logger = logging.getLogger("medicaldata.ai")


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

    def __init__(self, configuration: Settings = settings, *, client: Any | None = None, api_key: str | None = None) -> None:
        if api_key and not configuration.openai_api_key:
            # Explicit injection is intended for tests/secret managers; normal
            # deployments should rely on OPENAI_API_KEY in the environment.
            configuration = replace(configuration, openai_api_key=api_key)
        self.configuration = configuration
        self._request_slots = threading.BoundedSemaphore(max(1, configuration.ai_max_concurrent_requests))
        if not configuration.openai_api_key and client is None:
            raise OpenAIProviderConfigurationError("OPENAI_API_KEY is required when AI_EXTRACTION_PROVIDER=openai")
        if client is not None:
            self.client = client
            return
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised in deployment image
            raise OpenAIProviderConfigurationError("install the optional openai dependency") from exc
        self.client = OpenAI(api_key=configuration.openai_api_key, timeout=configuration.ai_request_timeout)

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
                    response = self.client.responses.create(
                        model=self.configuration.openai_reasoning_model,
                        instructions=instructions,
                        input=payload,
                        text={"format": {"type": "json_schema", "name": "clinical_synthesis", "strict": True, "schema": self._SYNTHESIS_SCHEMA}},
                        store=False,
                        timeout=self.configuration.ai_request_timeout,
                    )
                raw = getattr(response, "output_text", None)
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
                    response = self.client.responses.create(
                        model=model,
                        instructions=instructions,
                        input=payload,
                        text={"format": {"type": "json_schema", "name": "medical_page_extraction", "strict": True, "schema": self._PAGE_SCHEMA}},
                        store=False,
                        timeout=self.configuration.ai_request_timeout,
                    )
                raw = getattr(response, "output_text", None)
                if not raw:
                    raise ValueError("empty structured response")
                parsed = MedicalPageExtraction.model_validate(json.loads(raw))
                if self.configuration.ai_debug:
                    logger.info("OpenAI response received in %.1f seconds", time.monotonic() - started)
                    logger.info("Extraction stored for page %s", page_number)
                return PageAnalysis(text=parsed.transcription or None, structured=parsed, confidence=parsed.overall_confidence, model=model, prompt_version=prompt_version)
            except Exception as exc:  # SDK exception classes vary between releases.
                last_error = self._error_code(exc)
                if attempt >= self.configuration.ai_max_retries or not self._retryable(exc):
                    break
                time.sleep(min(2**attempt, 8))
        return PageAnalysis(text=None, confidence=0.0, model=model, prompt_version=prompt_version, error=last_error)

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        name = type(exc).__name__.casefold()
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


MedicalAIProvider = DocumentVisionProvider


def provider_for_settings(configuration: Settings = settings) -> DocumentVisionProvider:
    """Build the configured provider without ever passing secrets to the UI."""

    if configuration.ai_provider.casefold() in {"openai", "openai_vision", "openai_medical"}:
        return OpenAIMedicalVisionProvider(configuration)
    return LocalSafeProvider()
