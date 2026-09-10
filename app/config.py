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


settings = Settings()
