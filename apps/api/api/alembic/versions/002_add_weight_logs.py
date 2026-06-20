"""add weight_logs table

Revision ID: 002
Revises: 001
Create Date: 2026-05-13

"""
from alembic import op
import sqlalchemy as sa

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "weight_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("experiment_id", sa.String(), nullable=False),
        sa.Column("epoch", sa.Integer(), nullable=False),
        sa.Column("file_path", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"]),
    )
    op.create_index("ix_weight_logs_exp_epoch", "weight_logs", ["experiment_id", "epoch"])


def downgrade() -> None:
    op.drop_index("ix_weight_logs_exp_epoch", table_name="weight_logs")
    op.drop_table("weight_logs")
