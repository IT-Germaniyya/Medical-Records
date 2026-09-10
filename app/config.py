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
    ai_provider: str = os.getenv("AI_PROVIDER", "local_safe")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_medical_model: str = os.getenv("OPENAI_MEDICAL_MODEL", "gpt-5.2")
    openai_fast_model: str = os.getenv("OPENAI_FAST_MODEL", "gpt-5.2")
    openai_reasoning_model: str = os.getenv("OPENAI_REASONING_MODEL", "gpt-5.2")
    fast_model_min_confidence: float = float(os.getenv("FAST_MODEL_MIN_CONFIDENCE", "0.80"))
    auto_accept_confidence: float = float(os.getenv("AUTO_ACCEPT_CONFIDENCE", "0.95"))
    physician_review_confidence: float = float(os.getenv("PHYSICIAN_REVIEW_CONFIDENCE", "0.70"))
    ai_max_concurrent_requests: int = int(os.getenv("AI_MAX_CONCURRENT_REQUESTS", "4"))
    ai_request_timeout: float = float(os.getenv("AI_REQUEST_TIMEOUT", "60"))
    ai_max_retries: int = int(os.getenv("AI_MAX_RETRIES", "2"))
    ai_max_image_dimension: int = int(os.getenv("AI_MAX_IMAGE_DIMENSION", "4096"))
    openai_prompt_root: Path = Path(os.getenv("OPENAI_PROMPT_ROOT", "./prompts/openai"))


settings = Settings()
