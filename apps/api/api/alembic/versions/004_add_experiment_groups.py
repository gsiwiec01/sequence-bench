"""add experiment_groups table and group_id to experiments

Revision ID: 004
Revises: 003
Create Date: 2026-05-21

"""
from alembic import op
import sqlalchemy as sa

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = inspector.get_table_names()

    if "experiment_groups" not in existing_tables:
        op.create_table(
            "experiment_groups",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("description", sa.String(), nullable=True),
            sa.Column("tags", sa.JSON(), nullable=False, server_default="[]"),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("dataset", sa.String(), nullable=True),
            sa.Column("created_from_matrix", sa.Boolean(), nullable=False, server_default="false"),
            sa.PrimaryKeyConstraint("id"),
        )

    existing_indexes_groups = (
        {idx["name"] for idx in inspector.get_indexes("experiment_groups")}
        if "experiment_groups" in existing_tables
        else set()
    )
    if "ix_experiment_groups_created_at" not in existing_indexes_groups:
        op.create_index("ix_experiment_groups_created_at", "experiment_groups", ["created_at"])

    # Add group_id column to experiments
    existing_cols = {col["name"] for col in inspector.get_columns("experiments")}
    if "group_id" not in existing_cols:
        op.add_column(
            "experiments",
            sa.Column(
                "group_id",
                sa.String(),
                sa.ForeignKey("experiment_groups.id", ondelete="SET NULL"),
                nullable=True,
            ),
        )
        op.create_index("ix_experiments_group_id", "experiments", ["group_id"])


def downgrade() -> None:
    op.drop_index("ix_experiments_group_id", table_name="experiments")
    op.drop_column("experiments", "group_id")
    op.drop_index("ix_experiment_groups_created_at", table_name="experiment_groups")
    op.drop_table("experiment_groups")
