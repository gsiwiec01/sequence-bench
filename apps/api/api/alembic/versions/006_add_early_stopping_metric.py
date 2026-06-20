"""add early_stopping_metric column to experiments

Revision ID: 006
Revises: 005
Create Date: 2026-06-10

"""
from alembic import op
import sqlalchemy as sa

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    cols = {col["name"] for col in inspector.get_columns("experiments")}
    if "early_stopping_metric" not in cols:
        op.add_column("experiments", sa.Column("early_stopping_metric", sa.String(), nullable=True))
        op.execute("UPDATE experiments SET early_stopping_metric = 'val_loss' WHERE early_stopping_metric IS NULL")
        op.alter_column("experiments", "early_stopping_metric", nullable=False)


def downgrade() -> None:
    op.drop_column("experiments", "early_stopping_metric")
