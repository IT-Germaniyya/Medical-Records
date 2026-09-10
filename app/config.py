from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "sqlite+pysqlite:///./medical_mvp.db")
    storage_root: Path = Path(os.getenv("STORAGE_ROOT", "./data"))
    output_root: Path = Path(os.getenv("OUTPUT_ROOT", "./output"))
    confidence_high: float = float(os.getenv("CONFIDENCE_HIGH", "0.90"))
    confidence_medium: float = float(os.getenv("CONFIDENCE_MEDIUM", "0.70"))
    high_risk_confidence: float = float(os.getenv("HIGH_RISK_CONFIDENCE", "0.98"))
    archive_max_files: int = int(os.getenv("ARCHIVE_MAX_FILES", "1000"))
    archive_max_uncompressed_bytes: int = int(os.getenv("ARCHIVE_MAX_UNCOMPRESSED_BYTES", str(500 * 1024 * 1024)))
    archive_max_single_file_bytes: int = int(os.getenv("ARCHIVE_MAX_SINGLE_FILE_BYTES", str(100 * 1024 * 1024)))
    archive_max_compression_ratio: float = float(os.getenv("ARCHIVE_MAX_COMPRESSION_RATIO", "1000"))
    archive_extraction_timeout_seconds: float = float(os.getenv("ARCHIVE_EXTRACTION_TIMEOUT_SECONDS", "120"))
    allow_nested_archive_extraction: bool = os.getenv("ALLOW_NESTED_ARCHIVE_EXTRACTION", "false").casefold() in {"1", "true", "yes", "on"}
    # AI is opt-in.  The default remains the no-network local provider so an
    # installation cannot accidentally send PHI to an external service.
    ai_provider: str = os.getenv("AI_PROVIDER") or os.getenv("AI_EXTRACTION_PROVIDER", "local_safe")
    # Patient-level review is the primary multimodal workflow.  The value is
    # deliberately configurable so a deployment can stage the migration, but
    # the default is the holistic patient review requested by the product.
    ai_extraction_mode: str = os.getenv("AI_EXTRACTION_MODE", "patient_level")
    legacy_ocr_enabled: bool = os.getenv("LEGACY_OCR_ENABLED", "false").casefold() in {"1", "true", "yes", "on"}
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_medical_model: str = os.getenv("OPENAI_MEDICAL_MODEL", "gpt-5.6-sol")
    openai_fast_model: str = os.getenv("OPENAI_FAST_MODEL", os.getenv("OPENAI_MEDICAL_MODEL", "gpt-5.6-sol"))
    openai_reasoning_model: str = os.getenv("OPENAI_REASONING_MODEL", "gpt-5.6-sol")
    openai_summary_model: str = os.getenv("OPENAI_SUMMARY_MODEL", os.getenv("OPENAI_REASONING_MODEL", "gpt-5.6-sol"))
    openai_image_detail: str = os.getenv("OPENAI_IMAGE_DETAIL", "high")
    ai_allow_hierarchical_fallback: bool = os.getenv("AI_ALLOW_HIERARCHICAL_FALLBACK", "true").casefold() in {"1", "true", "yes", "on"}
    ai_max_patient_bytes: int = int(os.getenv("AI_MAX_PATIENT_BYTES", str(40 * 1024 * 1024)))
    fast_model_min_confidence: float = float(os.getenv("FAST_MODEL_MIN_CONFIDENCE", "0.80"))
    auto_accept_confidence: float = float(os.getenv("AUTO_ACCEPT_CONFIDENCE", "0.95"))
    physician_review_confidence: float = float(os.getenv("PHYSICIAN_REVIEW_CONFIDENCE", "0.70"))
    ai_max_concurrent_requests: int = int(os.getenv("AI_MAX_CONCURRENT_REQUESTS", "4"))
    ai_request_timeout: float = float(os.getenv("AI_REQUEST_TIMEOUT_SECONDS", os.getenv("AI_REQUEST_TIMEOUT", "90")))
    ai_max_retries: int = int(os.getenv("AI_MAX_RETRIES", "2"))
    ai_max_image_dimension: int = int(os.getenv("AI_MAX_IMAGE_DIMENSION", "4096"))
    ai_pages_per_request: int = int(os.getenv("AI_PAGES_PER_REQUEST", "3"))
    ai_debug: bool = os.getenv("AI_DEBUG", "false").casefold() in {"1", "true", "yes", "on"}
    openai_prompt_root: Path = Path(os.getenv("OPENAI_PROMPT_ROOT", "./prompts/openai"))
    app_environment: str = os.getenv("APP_ENV", "development")
    admin_mode: bool = os.getenv("ADMIN_MODE", "false").casefold() in {"1", "true", "yes", "on"}
    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY") or None
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "openrouter/free")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


settings = Settings()
