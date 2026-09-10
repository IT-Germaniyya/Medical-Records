from __future__ import annotations

import argparse

from app.config import settings
from app.db import create_schema, make_session_factory
from app.repository import RecordRepository
from app.schemas import ReportType
from app.services.reports import ReportGenerator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate grounded patient report(s)")
    parser.add_argument("patient_id")
    parser.add_argument("--type", choices=["detailed_medical_report", "erp_summary", "both"], default="both")
    parser.add_argument("--create-schema", action="store_true")
    args = parser.parse_args(argv)
    if args.create_schema:
        create_schema()
    generator = ReportGenerator(RecordRepository(make_session_factory()), settings)
    report_types = [ReportType.DETAILED, ReportType.ERP_SUMMARY] if args.type == "both" else [ReportType(args.type)]
    for report_type in report_types:
        report = generator.generate(args.patient_id, report_type)
        print(f"{report.report_type.value} v{report.version}: {report.status} ({report.file_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
