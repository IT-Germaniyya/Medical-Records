from __future__ import annotations

from app.schemas import DocumentType


_KEYWORDS: list[tuple[DocumentType, tuple[str, ...]]] = [
    (DocumentType.LAB_REPORT, ("lab", "cbc", "haemat", "hemat", "blood")),
    (DocumentType.PRESCRIPTION, ("prescrip", "rx", "medication")),
    (DocumentType.RADIOLOGY_REPORT, ("radiology", "xray", "x-ray", "ultrasound", "mri", "ct")),
    (DocumentType.GROWTH_CHART, ("growth", "weight", "height")),
    (DocumentType.PATIENT_REGISTRATION, ("registration", "demographic", "patient card")),
    (DocumentType.DISCHARGE_SUMMARY, ("discharge",)),
    (DocumentType.OUTPATIENT_NOTE, ("outpatient", "clinic", "visit")),
    (DocumentType.EMERGENCY_NOTE, ("emergency", "er_", "ed_")),
    (DocumentType.ECG, ("ecg", "ekg")),
    (DocumentType.VITAL_SIGNS, ("vital",)),
]


def classify(filename: str, text: str | None) -> tuple[DocumentType, float]:
    haystack = f"{filename} {text or ''}".lower()
    for document_type, keywords in _KEYWORDS:
        if any(keyword in haystack for keyword in keywords):
            return document_type, 0.92 if text else 0.74
    return DocumentType.UNKNOWN, 0.0
