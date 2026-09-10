from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pypdf import PdfReader


class MedicalAIProvider(ABC):
    """Provider boundary. Production providers must pass privacy/security review."""

    name = "abstract"
    version = ""

    @abstractmethod
    def extract_text(self, document: Path, page_number: int) -> str | None:
        raise NotImplementedError


class LocalSafeProvider(MedicalAIProvider):
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
