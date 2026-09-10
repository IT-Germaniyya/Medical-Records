"""Opt-in smoke test for the configured OpenAI medical page provider.

This command intentionally requires ``AI_PROVIDER=openai`` and
``OPENAI_API_KEY`` in the backend environment. It never prints the key or raw
request payload; use it only with an approved, de-identified test document.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import settings
from app.services.providers import OpenAIMedicalVisionProvider


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one opt-in OpenAI medical extraction smoke test")
    parser.add_argument("path", type=Path)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--document-type", default=None)
    args = parser.parse_args()
    if settings.ai_provider.casefold() not in {"openai", "openai_vision", "openai_medical"}:
        parser.error("set AI_PROVIDER=openai to enable the real API test")
    provider = OpenAIMedicalVisionProvider(settings)
    result = provider.extract_page(args.path, args.page, document_type_hint=args.document_type)
    print({
        "document_classification": result.structured.document_type if result.structured else None,
        "demographics": result.structured.patient_identity.model_dump(mode="json") if result.structured else None,
        "structured_medical_extraction": result.structured.model_dump(mode="json") if result.structured else None,
        "confidence": result.confidence,
        "verification_items": result.structured.uncertain_items if result.structured else [],
        "error": result.error,
        "model": result.model,
    })
    return 0 if result.structured and not result.error else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

