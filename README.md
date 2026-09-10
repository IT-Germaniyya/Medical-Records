# Medical Records Digitization MVP

A conservative, traceable prototype for digitizing one legacy patient folder into a structured longitudinal record, physician-facing summary, review queue, ERP exports, and a FHIR-style bundle. It is designed around one non-negotiable rule: **a missing or uncertain fact remains uncertain**.

## What works now

- Ingests one folder, ZIP, or RAR archive of JPG, JPEG, PNG, and PDF documents. ZIP/RAR archives use the same upload workflow, are detected from magic bytes, and may contain root files or nested folders.
- RAR extraction uses `rarfile` with Unar/libarchive inside Docker; archive limits, traversal/symlink checks, duplicate checks, and nested-archive blocking are applied before medical ingestion.
- Copies originals to immutable-style `original/` storage, produces separate `processed/` derivatives (EXIF orientation correction, cautious uniform-border crop, contrast enhancement, low-resolution and possible-glare flags), SHA-256 deduplicates documents, and preserves file/page provenance.
- Uses `AI_EXTRACTION_MODE=patient_level` by default when a remote provider is enabled: the complete ZIP/RAR/PDF/image bundle is reviewed as one clinical case with high-detail image inputs, strict structured output, source grounding, longitudinal reasoning, and a single canonical review feeding both reports. OpenAI uses the Responses API; OpenRouter uses its OpenAI-compatible multimodal chat endpoint with the same clinical schema and safety prompts. Large bundles use a hierarchical multimodal fallback; the system never silently falls back to OCR when the required AI review fails. API keys remain backend/worker-only environment secrets.
- Classifies each page conservatively, extracts only narrow explicitly labeled fields (laboratory values, diagnoses, medication lines, and numeric growth measures), and stores field-level audit events.
- Persists canonical JSON plus relational PostgreSQL rows, with Alembic migrations for the base schema and the patient-workspace/report/job tables.
- Produces the MVP files requested per patient: `structured_record.json`, `physician_summary.txt`, `timeline.json`, and `verification_queue.json`, plus CSV, HTML, audit, patient JSON, and FHIR-style exports.
- Supports resumable batch processing: a completed patient is not processed again unless `--no-resume` is passed.
- The workspace exposes an **AI Clinical Review** tab and **Reprocess with AI** action. Each review stores its version, model, prompt version, source hash, and creation time; generated reports reference the review version used.
- The **AI diagnostics** page (`/admin/ai`) shows redacted provider health, API/worker configuration visibility, last request metadata, latency, and safe error categories. **Test AI Connection** sends a minimal text-only request and **Test Vision Extraction** uses a deterministic synthetic image; patient processing is blocked until both tests succeed.
- Provider selection is configuration-based: set `AI_PROVIDER=openai` or `AI_PROVIDER=openrouter` and restart both the API and worker. Configure OpenRouter with `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` (a fixed model slug or the default `openrouter/free`), and `OPENROUTER_BASE_URL`. The admin page shows the active provider/model and warns that free routing can vary between requests.
- A fully local provider is available with `AI_PROVIDER=local`: it sends patient-level multimodal prompts only to Ollama at `LOCAL_AI_BASE_URL` using the configurable `LOCAL_AI_MODEL` (for example, a locally installed vision model). It requires no API key, never falls back to cloud providers, supports multi-image/PDF review, and can optionally augment the prompt with PaddleOCR-VL text via `LOCAL_OCR_ENABLED=true`.
- The admin page can compare the same de-identified structured patient record through OpenAI and OpenRouter across demographics, diagnoses, medications, labs, growth, uncertain items, and ERP summary. Results are shown side by side and are never automatically selected by cost or score.
- Local performance diagnostics include the latest page latency, patient-review latency, Ollama reachability/model status, GPU/CPU status, and reported GPU memory when Ollama exposes it.

## Deliberate safety boundaries

- The local provider can inspect document images, handwriting, tables, prescriptions, laboratory pages, and growth charts, but it does **not** claim clinical validation: it must not infer a diagnosis, calculate growth percentiles, correct abnormal labs, assign a dose, or assign ICD-10/SNOMED/LOINC codes. Optional PaddleOCR-VL output is only an additional local transcription aid.
- OpenAI and OpenRouter are optional extraction aids, not clinical validation. Both transcribe before interpreting, never fill unreadable medication doses, preserve source references, and queue low-confidence or high-risk values for human review. Validate privacy, security, model behavior, and local regulatory requirements before enabling either provider for PHI.
- Local AI remains an extraction aid, not clinical validation. A local model must still label illegible content, `needs_verification`, probable findings, and documented facts conservatively; no cloud fallback is attempted when Ollama is unavailable.
- A diagnosis or medication extracted from text is still queued for medical review. Missing dose/frequency becomes a review item, never a guessed dose.
- "No allergy extracted" is not reported as "no allergy." Undated information is not added to the chronological timeline.
- This is a prototype, not a clinical decision-support system or production deployment. Perform security validation, clinical validation, and local regulatory review before using real patient data.

## Quick start

Create a virtual environment with Python 3.11+ and install dependencies:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install .[dev]  # Windows PowerShell
```

For a local SQLite demonstration, create the sample and run the pipeline:

```bash
.venv/Scripts/python scripts/create_sample_patient.py
.venv/Scripts/python ingest.py sample_patients/PATIENT_DEMO_0001 --create-schema
```

The resulting files are placed in `output/PATIENT_DEMO_0001/`. Use a PostgreSQL `DATABASE_URL` and run migrations for normal service operation:

```bash
alembic upgrade head
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`POST /ingestions` accepts `{"patient_folder":"/path/to/PATIENT_000001"}` in this MVP. It is intentionally a local, trusted-network API: do not expose arbitrary filesystem paths in a production API.

## Docker

Copy `.env.example` to `.env`, change all non-demo secrets, then run:

```bash
docker compose up --build
```

The API applies the Alembic migration before starting. The image includes native Unar and `bsdtar` backends for RAR4/RAR5. The worker watches `data/incoming/patients/`; Redis is included as a future queue dependency. In production, replace the MVP polling worker with an authenticated task queue such as Celery/RQ and use a managed encrypted PostgreSQL service.

### Local Ollama on Windows

1. Install [Ollama for Windows](https://ollama.com/download/windows).
2. Pull the configured vision model, for example `ollama pull qwen3-vl` (or use the slug configured in `LOCAL_AI_MODEL`).
3. Verify it is available with `ollama list` and that Ollama is listening on `http://localhost:11434`.
4. Set `AI_PROVIDER=local`, `LOCAL_AI_BASE_URL=http://host.docker.internal:11434`, and `LOCAL_AI_MODEL=<your-installed-vision-model>` in the untracked `.env`.
5. Restart with `docker compose up -d --build`. Docker maps `host.docker.internal` for both API and worker containers; the Admin > AI page reports Ollama and GPU status.

If Ollama or the selected model is unavailable, patient-level processing stops with a local diagnostic error. The application does not switch to OpenAI or OpenRouter automatically. PaddleOCR-VL is optional and must be installed separately in the API/worker image or runtime before setting `LOCAL_OCR_ENABLED=true`.

## Batch processing

```bash
python batch_process.py --input /data/patients --workers 10
```

The batch command isolates exceptions by patient and avoids writing source content to console output. Completed patient records are resumed from the database. Cross-patient duplicates are rejected by SHA-256 checksum.

## Outputs

Each `output/<PATIENT_ID>/` directory contains:

- `structured_record.json`, `patient.json`, `timeline.json`, `problem_list.json`, `audit.json`
- `physician_summary.txt` and `physician_summary.html`
- `verification_queue.json`, `review_queue.csv`
- `patients.csv`, `encounters.csv`, `diagnoses.csv`, `medications.csv`, `labs.csv`, `radiology.csv`, `vitals.csv`, `growth.csv`
- `fhir_bundle.json` (FHIR-style adapter, not a claim of certification)

Adapt ERP column names in `erp_mapping.example.yaml`; no hospital ERP format is hardcoded.

## Tests

```bash
python -m pytest
```

Tests create synthetic embedded-text PDFs and cover provenance/output generation, resumable processing, and cross-patient checksum rejection. No real patient data is included.

## Pilot evaluation

Place physician-labeled canonical JSON files in `gold_standard/` and matching system outputs (renamed to the same patient filename) in a prediction directory. Run:

```bash
python evaluate_pilot.py --predictions pilot_predictions --gold-standard gold_standard
```

The report provides precision/recall by diagnosis, medication name/dose, lab value, date, and growth measurement, rather than only an overall accuracy score. Add document-classification and physician-summary-acceptance metrics with the reviewer dashboard/labeling workflow in Phase 7.

## Architecture and data dictionary

See [architecture.md](architecture.md), [data_dictionary.md](data_dictionary.md), and the [hospital user guide](docs/USER_GUIDE.md). Versioned, provider-facing prompt contracts reside in `prompts/`; prompts are never scattered through pipeline code.

The web workspace is the normal hospital workflow: search or add a patient, upload records, review only uncertain fields, view originals, and generate either report from structured data. The API also remains available for integrations and controlled automation.

## Production-hardening roadmap

1. Add locally hosted or contractually approved OCR and multimodal adapters, benchmarked against physician-labeled data before enabling auto-accept.
2. Add RBAC/OIDC, access audit logs, encryption/key management, TLS termination, retention/deletion policies, PHI-safe monitoring, and signed document storage.
3. Replace polling with Celery/RQ, Redis-backed idempotent tasks, dead-letter handling, metrics, per-page retries, and backpressure.
4. Build the Phase 4 reviewer dashboard with source-image/page pane, evidence-highlight overlay, approve/edit/reject actions, and role-specific queues.
5. Integrate licensed terminology services, WHO/CDC growth standards, validated FHIR profiles, ERP mappings, and a 200–500 patient pilot evaluation/gold-standard comparison.
