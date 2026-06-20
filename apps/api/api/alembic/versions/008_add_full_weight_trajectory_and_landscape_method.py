"""add full weight trajectory columns + loss landscape method

Revision ID: 008
Revises: 007
Create Date: 2026-06-14

"""
from alembic import op
import sqlalchemy as sa

revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def _columns(inspector, table: str) -> set[str]:
    return {c["name"] for c in inspector.get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    exp_cols = _columns(inspector, "experiments")
    if "full_weight_trajectory" not in exp_cols:
        op.add_column(
            "experiments",
            sa.Column(
                "full_weight_trajectory",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
        )
    if "full_weight_path" not in exp_cols:
        op.add_column(
            "experiments",
            sa.Column("full_weight_path", sa.String(), nullable=True),
        )

    ls_cols = _columns(inspector, "loss_landscapes")
    if "method" not in ls_cols:
        op.add_column(
            "loss_landscapes",
            sa.Column(
                "method",
                sa.String(),
                nullable=False,
                server_default="scalar",
            ),
        )


def downgrade() -> None:
    op.drop_column("loss_landscapes", "method")
    op.drop_column("experiments", "full_weight_path")
    op.drop_column("experiments", "full_weight_trajectory")
