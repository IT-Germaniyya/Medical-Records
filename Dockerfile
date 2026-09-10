FROM python:3.12-slim

# rarfile uses a native backend for RAR4/RAR5 extraction. Keep both Unar and
# libarchive's bsdtar available so the image works across common RAR variants.
RUN apt-get update \
    && apt-get install -y --no-install-recommends unar libarchive-tools \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /service
COPY pyproject.toml README.md ./
COPY app ./app
COPY templates ./templates
COPY alembic ./alembic
COPY alembic.ini batch_process.py ingest.py evaluate_pilot.py generate_reports.py ./
RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 appuser
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
