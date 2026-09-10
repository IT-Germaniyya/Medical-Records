from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation import evaluate_gold_standard


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare exported records against physician-labeled gold-standard JSON")
    parser.add_argument("--predictions", type=Path, required=True, help="Directory of canonical prediction JSON files")
    parser.add_argument("--gold-standard", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("pilot_evaluation_report.json"))
    args = parser.parse_args()
    report = evaluate_gold_standard(args.predictions, args.gold_standard)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"compared={report['patients_compared']} report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
