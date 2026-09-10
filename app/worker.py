"""Conservative polling worker for the MVP; replace with Celery/RQ after queue infrastructure validation."""
from __future__ import annotations

import os
from pathlib import Path
from time import sleep

from app.config import settings
from app.db import make_session_factory
from app.pipeline import PatientPipeline
from app.repository import RecordRepository


def main() -> None:
    root = Path(os.getenv("BATCH_INPUT", "/var/lib/medical-data/incoming/patients"))
    poll_seconds = max(5, int(os.getenv("WORKER_POLL_SECONDS", "30")))
    while True:
        if root.is_dir():
            for folder in (item for item in root.iterdir() if item.is_dir()):
                try:
                    PatientPipeline(RecordRepository(make_session_factory()), settings).process(folder, resume=True)
                except Exception:
                    # Do not emit source filenames/content: they may contain PHI.
                    pass
        sleep(poll_seconds)


if __name__ == "__main__":
    main()
