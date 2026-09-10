"""Add local AI health and performance metrics.

Revision ID: b18_local_ai_diagnostics
Revises: a17d
"""

from alembic import op
import sqlalchemy as sa


revision = "b18_local_ai_diagnostics"
down_revision = "a17d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("ai_diagnostics", sa.Column("last_page_latency_ms", sa.Float(), nullable=True))
    op.add_column("ai_diagnostics", sa.Column("last_patient_latency_ms", sa.Float(), nullable=True))
    op.add_column("ai_diagnostics", sa.Column("gpu_memory_bytes", sa.BigInteger(), nullable=True))
    op.add_column("ai_diagnostics", sa.Column("ollama_status", sa.String(length=32), nullable=True))
    op.add_column("ai_diagnostics", sa.Column("gpu_status", sa.String(length=64), nullable=True))


def downgrade() -> None:
    op.drop_column("ai_diagnostics", "gpu_status")
    op.drop_column("ai_diagnostics", "ollama_status")
    op.drop_column("ai_diagnostics", "gpu_memory_bytes")
    op.drop_column("ai_diagnostics", "last_patient_latency_ms")
    op.drop_column("ai_diagnostics", "last_page_latency_ms")
