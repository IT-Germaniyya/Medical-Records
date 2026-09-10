"""store ZIP/RAR archive provenance on extracted source files

Revision ID: 4f2e7b8c9d10
Revises: 8a60bdf38177
Create Date: 2026-09-10
"""

from alembic import op
import sqlalchemy as sa


revision = "4f2e7b8c9d10"
down_revision = "8a60bdf38177"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("source_files", sa.Column("archive_filename", sa.String(length=512), nullable=True))
    op.add_column("source_files", sa.Column("archive_type", sa.String(length=16), nullable=True))
    op.add_column("source_files", sa.Column("original_relative_path", sa.String(length=1024), nullable=True))
    op.add_column("source_files", sa.Column("extracted_filename", sa.String(length=1024), nullable=True))


def downgrade() -> None:
    op.drop_column("source_files", "extracted_filename")
    op.drop_column("source_files", "original_relative_path")
    op.drop_column("source_files", "archive_type")
    op.drop_column("source_files", "archive_filename")
