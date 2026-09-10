from __future__ import annotations

import argparse
from pathlib import Path

from app.config import settings
from app.db import create_schema, make_session_factory
from app.pipeline import PatientPipeline
from app.repository import RecordRepository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Conservative patient-folder medical record digitizer")
    parser.add_argument("patient_folder", type=Path, help="Folder, ZIP, or RAR archive for one patient")
    parser.add_argument("--patient-id", help="Override patient ID; defaults to folder name")
    parser.add_argument("--no-resume", action="store_true", help="Reprocess a completed patient")
    parser.add_argument("--create-schema", action="store_true", help="Local SQLite convenience; use Alembic in production")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.create_schema:
        create_schema()
    repository = RecordRepository(make_session_factory())
    result = PatientPipeline(repository, settings).process(args.patient_folder, args.patient_id, resume=not args.no_resume)
    print(f"patient={result.record.patient.patient_id} status={result.record.patient_status} output={result.output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
