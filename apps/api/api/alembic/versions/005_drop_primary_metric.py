"""drop primary_metric and metric_value columns

Revision ID: 005
Revises: 004
Create Date: 2026-06-10

"""
from alembic import op
import sqlalchemy as sa

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    exp_cols = {col["name"] for col in inspector.get_columns("experiments")}
    if "primary_metric" in exp_cols:
        op.drop_column("experiments", "primary_metric")

    em_cols = {col["name"] for col in inspector.get_columns("epoch_metrics")}
    if "metric_value" in em_cols:
        op.drop_column("epoch_metrics", "metric_value")
    if "train_metric_value" in em_cols:
        op.drop_column("epoch_metrics", "train_metric_value")


def downgrade() -> None:
    op.add_column("epoch_metrics", sa.Column("train_metric_value", sa.Float(), nullable=True))
    op.add_column("epoch_metrics", sa.Column("metric_value", sa.Float(), nullable=True))
    op.add_column("experiments", sa.Column("primary_metric", sa.String(), nullable=False, server_default="accuracy"))
