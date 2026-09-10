# Architecture

```text
patient folder / ZIP
  -> ingestion (checksum + immutable original copy)
  -> derived preprocessing (processed copy only)
  -> per-page classification
  -> provider abstraction (local embedded-PDF text in MVP)
  -> narrow structured extraction + source provenance
  -> independent rule validation
  -> human verification queue
  -> timeline/problem view + one-page summary
  -> PostgreSQL + JSON/CSV/FHIR-style outputs
```

The core pipeline is intentionally separate from integrations. `MedicalAIProvider` is the only extraction-provider boundary; `TerminologyService` returns null until a reliable licensed mapping service is installed; and the FHIR adapter is isolated under `app/exports/` so core schemas do not depend on FHIR.

`PatientPipeline` treats each patient as a failure-isolation boundary. One unreadable page yields a verification item while the rest of the patient output is retained. The repository stores source files/pages and relational projections alongside the canonical serialized record, preserving faithful API retrieval and auditability.

The current local provider reads embedded PDF text only. Implementing OCR/multimodal extraction is a deployment integration, not a silent fallback: its model/provider version and prompt version must be written into each audit event, and it must operate only after the hospital has approved its PHI/data-processing arrangement.
