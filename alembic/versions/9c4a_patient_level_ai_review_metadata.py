"""patient-level AI review and source ordering metadata

Revision ID: 9c4a
Revises: 8a60bdf38177
"""
from alembic import op
import sqlalchemy as sa


revision = "9c4a"
down_revision = "4f2e7b8c9d10"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("source_files", sa.Column("source_order", sa.Integer(), nullable=True))
    op.add_column("reports", sa.Column("ai_review_version", sa.String(length=64), nullable=True))


def downgrade() -> None:
    op.drop_column("reports", "ai_review_version")
    op.drop_column("source_files", "source_order")
