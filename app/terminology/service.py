from __future__ import annotations

from abc import ABC, abstractmethod


class TerminologyService(ABC):
    """Implementations may use locally licensed terminology services; null is always safe."""

    @abstractmethod
    def normalize_diagnosis(self, term: str) -> str | None: ...

    @abstractmethod
    def map_icd10(self, term: str) -> str | None: ...

    @abstractmethod
    def map_snomed(self, term: str) -> str | None: ...

    @abstractmethod
    def map_loinc(self, test_name: str) -> str | None: ...


class NullTerminologyService(TerminologyService):
    """Never fabricate codes when a licensed/validated terminology source is absent."""

    def normalize_diagnosis(self, term: str) -> str | None:
        return None

    def map_icd10(self, term: str) -> str | None:
        return None

    def map_snomed(self, term: str) -> str | None:
        return None

    def map_loinc(self, test_name: str) -> str | None:
        return None

