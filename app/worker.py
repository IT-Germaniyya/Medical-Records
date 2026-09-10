"""Conservative polling worker for the MVP; replace with Celery/RQ after queue infrastructure validation."""
from __future__ import annotations

import os
from pathlib import Path
from time import sleep

from app.config import settings
from app.db import make_session_factory
from app.pipeline import PatientPipeline
from app.repository import RecordRepository
from app.services.ai_diagnostics import AIDiagnosticsStore


def main() -> None:
    root = Path(os.getenv("BATCH_INPUT", "/var/lib/medical-data/incoming/patients"))
    poll_seconds = max(5, int(os.getenv("WORKER_POLL_SECONDS", "30")))
    repository = RecordRepository(make_session_factory())
    diagnostics = AIDiagnosticsStore(repository.session_factory)
    configured_model = settings.openrouter_model if settings.ai_provider.casefold().startswith("openrouter") else settings.openai_medical_model
    configured_key = settings.openrouter_api_key if settings.ai_provider.casefold().startswith("openrouter") else settings.openai_api_key
    for _ in range(30):
        try:
            diagnostics.mark_runtime(role="worker", provider=settings.ai_provider, model=configured_model, api_key_configured=bool(configured_key))
            break
        except Exception:
            # The API container applies migrations on startup; wait briefly if
            # the worker wins the initial container race.
            sleep(2)
    while True:
        if root.is_dir():
            for folder in (item for item in root.iterdir() if item.is_dir()):
                try:
                    PatientPipeline(repository, settings).process(folder, resume=True)
                except Exception:
                    # Do not emit source filenames/content: they may contain PHI.
                    pass
        sleep(poll_seconds)


if __name__ == "__main__":
    main()
