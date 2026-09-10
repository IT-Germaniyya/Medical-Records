"""Pilot/gold-standard evaluation that reports field-level metrics, not a misleading single accuracy."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class Metric:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def as_dict(self) -> dict[str, float | int]:
        precision = self.true_positive / (self.true_positive + self.false_positive) if self.true_positive + self.false_positive else 0.0
        recall = self.true_positive / (self.true_positive + self.false_negative) if self.true_positive + self.false_negative else 0.0
        return {"true_positive": self.true_positive, "false_positive": self.false_positive, "false_negative": self.false_negative, "precision": round(precision, 4), "recall": round(recall, 4)}


def _values(record: dict, key: str, field: str) -> set[str]:
    return {str(item.get(field)).strip().casefold() for item in record.get(key, []) if item.get(field) not in {None, ""}}


def _update(metric: Metric, predicted: set[str], gold: set[str]) -> None:
    metric.true_positive += len(predicted & gold)
    metric.false_positive += len(predicted - gold)
    metric.false_negative += len(gold - predicted)


def evaluate_gold_standard(prediction_directory: Path, gold_directory: Path) -> dict:
    metrics = {name: Metric() for name in ("diagnosis", "medication_name", "medication_dose", "lab_value", "date", "growth_measurement")}
    compared = 0
    for gold_file in sorted(gold_directory.glob("*.json")):
        predicted_file = prediction_directory / gold_file.name
        if not predicted_file.exists():
            continue
        gold, predicted = json.loads(gold_file.read_text(encoding="utf-8")), json.loads(predicted_file.read_text(encoding="utf-8"))
        compared += 1
        _update(metrics["diagnosis"], _values(predicted, "diagnoses", "term_original"), _values(gold, "diagnoses", "term_original"))
        _update(metrics["medication_name"], _values(predicted, "medications", "name_normalized"), _values(gold, "medications", "name_normalized"))
        _update(metrics["medication_dose"], _values(predicted, "medications", "dose"), _values(gold, "medications", "dose"))
        _update(metrics["lab_value"], _values(predicted, "laboratory_results", "value"), _values(gold, "laboratory_results", "value"))
        _update(metrics["date"], _values(predicted, "timeline", "date"), _values(gold, "timeline", "date"))
        _update(metrics["growth_measurement"], _values(predicted, "growth_measurements", "weight_kg"), _values(gold, "growth_measurements", "weight_kg"))
    return {"patients_compared": compared, "metrics": {name: metric.as_dict() for name, metric in metrics.items()}}
