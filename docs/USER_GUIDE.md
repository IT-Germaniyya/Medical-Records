# Hospital User Guide

## Start the application

1. Install Docker Desktop.
2. Open a terminal in the MedicalData folder and run `docker compose up --build`.
3. Wait until the services are healthy.

## Open the web interface

Open [http://localhost:8080](http://localhost:8080) in the hospital workstation browser.

## Upload a patient

1. Select **Add patient**.
2. Enter the patient ID, name, hospital file number/MRN, and any known demographics.
3. Select one or more JPG, PNG, or PDF records, or one ZIP/RAR archive. The picker shows one **Upload archive** workflow; the system detects ZIP versus RAR automatically. Archives may contain files at the root or nested folders.
4. Select **Process patient**.
5. Continue working while the progress card runs. The patient workspace opens when processing is complete.

To add documents to an existing patient, open that patient and select **Add documents**. Earlier source files and extracted facts are retained. Duplicate files are skipped by checksum. Unsupported files inside an archive are skipped with a warning; nested archives are skipped for safety.

## Patient-level AI review

The default `AI_EXTRACTION_MODE=patient_level` sends the complete patient bundle to the configured multimodal model as one clinical case. Set `AI_PROVIDER=openai` to use OpenAI, or `AI_PROVIDER=openrouter` to use OpenRouter without changing the workflow. OpenAI uses `OPENAI_API_KEY` and `OPENAI_MEDICAL_MODEL`; OpenRouter uses `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, and `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1`. A fixed OpenRouter model slug can be configured instead of `openrouter/free`; free routing may use different models between requests and extraction quality can vary. The model reads all pages/images together, preserves source references, reconstructs the longitudinal history, and produces the canonical review used by both reports. Configuration is passed to both API and worker containers, and keys are never exposed to frontend JavaScript. Large records use a hierarchical multimodal fallback; if AI review cannot be completed, the patient is marked failed and the original files remain preserved.

## Search for a patient

Select **Patients**. Search using a full or partial name, patient ID, or hospital file number/MRN. Select a result to open its Patient Workspace.

The workspace tabs are:

- **Overview**: demographics, summary, alerts, and pending work.
- **Documents**: original PDFs/images, document filters, page navigation, image zoom/rotation, and page-level extracted data.
- **Timeline**, **Problems**, **Labs**, **Medications**, and **Growth / Vitals**: structured clinical views.
- **Review Items**: only uncertain or high-risk fields requiring a human decision.
- **AI Clinical Review**: the holistic clinical summary, documented diagnoses, active problems, key investigations, medication history, pediatric/growth assessment, and uncertain/illegible items before report generation.
- **AI diagnostics** (`/admin/ai`): administrators can select the configured provider, run text and synthetic-vision connection tests, and compare OpenAI/OpenRouter on a de-identified patient. The page exposes only redacted error metadata (provider, model, HTTP status, request stage, retry count, and timestamp) in development/admin mode; secrets and patient payloads are never shown. Patient-level processing remains blocked until both checks pass.
- **Reports**: generate and retrieve both report types.

Use **Reprocess with AI** to invalidate the current reconstruction and create a new review version without deleting earlier report versions.

## Review uncertain data

Open **Review Items**. Read the proposed value, confidence, reason, and source filename/page. Select:

- **Approve** when the source supports the proposed value.
- **Edit** to enter a verified value.
- **Reject** when the proposed extraction is not supported.
- **Mark unreadable** when the source cannot be safely transcribed.

Every decision is added to the audit trail. Unreadable information is never guessed, and a possible or inferred diagnosis is never shown as a confirmed diagnosis.

## Generate the two reports

Open **Reports** inside a patient workspace.

- **Detailed Medical Report**: a multi-page physician reference report. It includes structured history, timeline, labs, medications, problems, verification needs, and an appendix containing original source documents where available.
- **ERP Physician Summary**: a compact, image-free PDF plus TXT and JSON exports for ERP entry/import.

Select **Generate Detailed Report**, **Generate ERP Summary**, or **Generate Both**. Reports are versioned (`v1`, `v2`, `v3`) and are never silently overwritten. A report marked **needs review** contains pending verification items and must be checked before clinical or ERP use.

Use the central **Reports** page to find earlier report versions by patient name, ID, or MRN and open/download them later.
