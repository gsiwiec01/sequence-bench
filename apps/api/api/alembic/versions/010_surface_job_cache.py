"""turn loss_landscapes into a cached async surface job

Dodaje cache_key (unikalny), params, skalary (explained_variance, anchor_loss)
i updated_at, by powierzchnia PCA była trwałym, cache'owanym zadaniem async
analogicznym do eksperymentów.

Revision ID: 010
Revises: 009
Create Date: 2026-06-14

"""
from alembic import op
import sqlalchemy as sa

revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def _columns(inspector, table: str) -> set[str]:
    if not inspector.has_table(table):
        return set()
    return {c["name"] for c in inspector.get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    cols = _columns(inspector, "loss_landscapes")
    if not cols:
        return

    if "cache_key" not in cols:
        op.add_column("loss_landscapes", sa.Column("cache_key", sa.String(), nullable=True))
        op.create_index(
            "ix_loss_landscapes_cache_key", "loss_landscapes", ["cache_key"], unique=True
        )
    if "params" not in cols:
        op.add_column("loss_landscapes", sa.Column("params", sa.JSON(), nullable=True))
    if "explained_variance" not in cols:
        op.add_column("loss_landscapes", sa.Column("explained_variance", sa.Float(), nullable=True))
    if "anchor_loss" not in cols:
        op.add_column("loss_landscapes", sa.Column("anchor_loss", sa.Float(), nullable=True))
    if "updated_at" not in cols:
        op.add_column("loss_landscapes", sa.Column("updated_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column("loss_landscapes", "updated_at")
    op.drop_column("loss_landscapes", "anchor_loss")
    op.drop_column("loss_landscapes", "explained_variance")
    op.drop_column("loss_landscapes", "params")
    op.drop_index("ix_loss_landscapes_cache_key", table_name="loss_landscapes")
    op.drop_column("loss_landscapes", "cache_key")
