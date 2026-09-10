from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from app.config import settings
from app.db import create_schema, make_session_factory
from app.pipeline import PatientPipeline
from app.repository import RecordRepository


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumable batch processor for patient folders")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--create-schema", action="store_true")
    args = parser.parse_args()
    if args.create_schema:
        create_schema()
    folders = [path for path in args.input.iterdir() if path.is_dir()]

    def run(folder: Path):
        repo = RecordRepository(make_session_factory())
        return PatientPipeline(repo, settings).process(folder)

    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run, folder): folder for folder in folders}
        for future in as_completed(futures):
            folder = futures[future]
            try:
                result = future.result()
                print(f"{folder.name}: {result.record.patient_status}{' (resumed)' if result.resumed else ''}")
            except Exception as exc:
                # Do not print file contents or source text (potential PHI).
                failures += 1
                print(f"{folder.name}: failed ({type(exc).__name__})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
