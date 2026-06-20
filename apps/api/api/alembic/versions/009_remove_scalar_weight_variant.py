"""remove 2-scalar weight variant: weight_logs table, landscape method, full flag

PCA jest jedyną metodą powierzchni błędu. Usuwamy artefakty wariantu
2-skalarowego: tabelę weight_logs (lekki log 2 skalarów), kolumnę
loss_landscapes.method oraz experiments.full_weight_trajectory (logowanie pełnej
trajektorii jest teraz domyślne, sterowane hyperparametrem).

Revision ID: 009
Revises: 008
Create Date: 2026-06-14

"""
from alembic import op
import sqlalchemy as sa

revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def _columns(inspector, table: str) -> set[str]:
    if not inspector.has_table(table):
        return set()
    return {c["name"] for c in inspector.get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if inspector.has_table("weight_logs"):
        existing_indexes = {idx["name"] for idx in inspector.get_indexes("weight_logs")}
        if "ix_weight_logs_exp_epoch" in existing_indexes:
            op.drop_index("ix_weight_logs_exp_epoch", table_name="weight_logs")
        op.drop_table("weight_logs")

    if "method" in _columns(inspector, "loss_landscapes"):
        op.drop_column("loss_landscapes", "method")

    if "full_weight_trajectory" in _columns(inspector, "experiments"):
        op.drop_column("experiments", "full_weight_trajectory")


def downgrade() -> None:
    op.add_column(
        "experiments",
        sa.Column("full_weight_trajectory", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "loss_landscapes",
        sa.Column("method", sa.String(), nullable=False, server_default="scalar"),
    )
    op.create_table(
        "weight_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("experiment_id", sa.String(), nullable=False),
        sa.Column("epoch", sa.Integer(), nullable=False),
        sa.Column("file_path", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_weight_logs_exp_epoch", "weight_logs", ["experiment_id", "epoch"])
