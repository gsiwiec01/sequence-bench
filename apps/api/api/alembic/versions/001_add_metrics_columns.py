"""add metrics columns

Revision ID: 001
Revises:
Create Date: 2026-05-07

"""
from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def _existing_columns(table: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    return {c["name"] for c in inspector.get_columns(table)}


def _table_exists(table: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table in inspector.get_table_names()


def _add_col(table: str, col: sa.Column) -> None:
    if col.name not in _existing_columns(table):
        op.add_column(table, col)


def upgrade() -> None:
    # epoch_metrics
    _add_col("epoch_metrics", sa.Column("train_metric_value", sa.Float(), nullable=True))
    _add_col("epoch_metrics", sa.Column("grad_norm_mean", sa.Float(), nullable=True))
    _add_col("epoch_metrics", sa.Column("grad_norm_max", sa.Float(), nullable=True))
    _add_col("epoch_metrics", sa.Column("learning_rate", sa.Float(), nullable=True))

    # gradient_logs
    _add_col("gradient_logs", sa.Column("grad_norm_mean", sa.Float(), nullable=True))
    _add_col("gradient_logs", sa.Column("n_layers", sa.Integer(), nullable=True))
    _add_col("gradient_logs", sa.Column("n_steps", sa.Integer(), nullable=True))

    # experiments
    _add_col("experiments", sa.Column("additional_metrics", sa.JSON(), nullable=True))
    _add_col("experiments", sa.Column("n_parameters", sa.Integer(), nullable=True))
    _add_col("experiments", sa.Column("total_training_time_s", sa.Float(), nullable=True))
    _add_col("experiments", sa.Column("convergence_epoch", sa.Integer(), nullable=True))
    _add_col("experiments", sa.Column("final_train_loss", sa.Float(), nullable=True))
    _add_col("experiments", sa.Column("final_val_loss", sa.Float(), nullable=True))

    # additional_metrics table
    if not _table_exists("additional_metrics"):
        op.create_table(
            "additional_metrics",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("experiment_id", sa.String(), nullable=False),
            sa.Column("epoch", sa.Integer(), nullable=False),
            sa.Column("metric_name", sa.String(64), nullable=False),
            sa.Column("metric_value", sa.Float(), nullable=False),
            sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ix_additional_metrics_exp_epoch_name",
            "additional_metrics",
            ["experiment_id", "epoch", "metric_name"],
        )


def downgrade() -> None:
    if _table_exists("additional_metrics"):
        op.drop_index("ix_additional_metrics_exp_epoch_name", table_name="additional_metrics")
        op.drop_table("additional_metrics")

    for col in ["final_val_loss", "final_train_loss", "convergence_epoch",
                "total_training_time_s", "n_parameters", "additional_metrics"]:
        if col in _existing_columns("experiments"):
            op.drop_column("experiments", col)

    for col in ["n_steps", "n_layers", "grad_norm_mean"]:
        if col in _existing_columns("gradient_logs"):
            op.drop_column("gradient_logs", col)

    for col in ["learning_rate", "grad_norm_max", "grad_norm_mean", "train_metric_value"]:
        if col in _existing_columns("epoch_metrics"):
            op.drop_column("epoch_metrics", col)
