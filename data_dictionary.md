# Data dictionary

| Entity | Purpose | Integrity rule |
| --- | --- | --- |
| `patients` | Patient identifier and preferred hospital file number | Conflicting demographics are retained as candidates in canonical JSON. |
| `source_files` | Original file metadata, checksum, paths, status | `checksum` is globally unique to prevent duplicate ingestion. |
| `source_pages` | Page-level class, confidence, extraction status | PDF pages are treated independently. |
| `encounters` | Independent clinical visits | Never merge without documented date/context. |
| `diagnoses` | Explicit or qualified diagnosis facts | Codes stay null unless reliable terminology mapping exists. |
| `medications` | Source medication details | Missing dose/frequency is a warning, not an inferred value. |
| `laboratory_results` | Exact laboratory text/value/unit | Source value is never silently corrected. |
| `radiology_reports` | Extracted written report fields | No image diagnosis is generated. |
| `observations` / `growth_measurements` | Vitals and pediatric measurements | No percentile is inferred from an unreadable plotted chart. |
| `clinical_interpretations` | Derived clinical reasoning | Must remain separate from documented diagnoses. |
| `problem_list` | Concise active/historical problems | Sources identify documented vs derived items. |
| `verification_items` | Reviewer work queue | Contains proposed value, confidence, reason, source, review status. |
| `audit_events` | Field-level provenance/audit record | Every extracted fact links to source file/page/confidence. |
