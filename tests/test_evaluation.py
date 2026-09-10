import json

from app.evaluation import evaluate_gold_standard


def test_evaluation_reports_field_level_precision_and_recall(tmp_path) -> None:
    predictions, gold = tmp_path / "predictions", tmp_path / "gold"
    predictions.mkdir()
    gold.mkdir()
    predicted = {"diagnoses": [{"term_original": "Asthma"}], "medications": [{"name_normalized": "Salbutamol", "dose": None}], "laboratory_results": [{"value": 10.2}], "timeline": [{"date": "2026-01-01"}], "growth_measurements": [{"weight_kg": 12.0}]}
    labeled = {"diagnoses": [{"term_original": "Asthma"}], "medications": [{"name_normalized": "Salbutamol", "dose": "2 puffs"}], "laboratory_results": [{"value": 10.2}], "timeline": [{"date": "2026-01-01"}], "growth_measurements": [{"weight_kg": 12.0}]}
    (predictions / "PATIENT_001.json").write_text(json.dumps(predicted), encoding="utf-8")
    (gold / "PATIENT_001.json").write_text(json.dumps(labeled), encoding="utf-8")
    report = evaluate_gold_standard(predictions, gold)
    assert report["patients_compared"] == 1
    assert report["metrics"]["diagnosis"]["precision"] == 1.0
    assert report["metrics"]["medication_dose"]["recall"] == 0.0
