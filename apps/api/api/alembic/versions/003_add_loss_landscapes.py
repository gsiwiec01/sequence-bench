"""add loss_landscapes table

Revision ID: 003
Revises: 002
Create Date: 2026-05-14

"""
from alembic import op
import sqlalchemy as sa

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = inspector.get_table_names()

    if "loss_landscapes" not in existing_tables:
        op.create_table(
            "loss_landscapes",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("experiment_id", sa.String(), nullable=False),
            sa.Column("param_x", sa.String(), nullable=False),
            sa.Column("param_y", sa.String(), nullable=False),
            sa.Column("grid_size", sa.Integer(), nullable=False),
            sa.Column("x_range", sa.JSON(), nullable=True),
            sa.Column("y_range", sa.JSON(), nullable=True),
            sa.Column("file_path", sa.String(), nullable=True),
            sa.Column("celery_task_id", sa.String(), nullable=True),
            sa.Column("status", sa.String(), nullable=False, server_default="pending"),
            sa.Column("error_message", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"]),
            sa.PrimaryKeyConstraint("id"),
        )

    existing_indexes = {idx["name"] for idx in inspector.get_indexes("loss_landscapes")} if "loss_landscapes" in existing_tables else set()
    if "ix_loss_landscapes_experiment_id" not in existing_indexes:
        op.create_index("ix_loss_landscapes_experiment_id", "loss_landscapes", ["experiment_id"])


def downgrade() -> None:
    op.drop_index("ix_loss_landscapes_experiment_id", table_name="loss_landscapes")
    op.drop_table("loss_landscapes")
