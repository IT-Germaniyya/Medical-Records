"""redacted shared AI diagnostics

Revision ID: a17d
Revises: 9c4a
"""

from alembic import op
import sqlalchemy as sa


revision = "a17d"
down_revision = "9c4a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ai_diagnostics",
        sa.Column("diagnostic_id", sa.Integer(), primary_key=True),
        sa.Column("provider", sa.String(length=64), nullable=False, server_default="unknown"),
        sa.Column("model", sa.String(length=128), nullable=False, server_default="unknown"),
        sa.Column("api_key_configured", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("worker_api_key_configured", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("last_request_status", sa.String(length=32), nullable=True),
        sa.Column("last_error_code", sa.String(length=64), nullable=True),
        sa.Column("last_error_message", sa.Text(), nullable=True),
        sa.Column("last_exception_type", sa.String(length=255), nullable=True),
        sa.Column("last_http_status", sa.Integer(), nullable=True),
        sa.Column("last_request_stage", sa.String(length=64), nullable=True),
        sa.Column("last_retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_request_at", sa.DateTime(), nullable=True),
        sa.Column("last_successful_request", sa.DateTime(), nullable=True),
        sa.Column("last_text_success_at", sa.DateTime(), nullable=True),
        sa.Column("last_text_success_model", sa.String(length=128), nullable=True),
        sa.Column("last_vision_success_at", sa.DateTime(), nullable=True),
        sa.Column("last_vision_success_model", sa.String(length=128), nullable=True),
        sa.Column("average_latency_ms", sa.Float(), nullable=False, server_default="0"),
        sa.Column("request_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("success_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.execute(sa.text("INSERT INTO ai_diagnostics (diagnostic_id, provider, model, updated_at) VALUES (1, 'unknown', 'unknown', CURRENT_TIMESTAMP)"))


def downgrade() -> None:
    op.drop_table("ai_diagnostics")
