"""drop param_x/param_y/grid_size from loss_landscapes

Trajektoria wag liczy stały rzut 2D (rnn.weight_hh_l0[0,0] vs [0,1]) na
stałej siatce N×N, więc te kolumny przestały przechowywać konfigurowalne dane.

Revision ID: 007
Revises: 006
Create Date: 2026-06-13

"""
from alembic import op
import sqlalchemy as sa

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None

_COLUMNS = ("param_x", "param_y", "grid_size")


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "loss_landscapes" not in inspector.get_table_names():
        return
    existing = {col["name"] for col in inspector.get_columns("loss_landscapes")}
    for name in _COLUMNS:
        if name in existing:
            op.drop_column("loss_landscapes", name)


def downgrade() -> None:
    op.add_column("loss_landscapes", sa.Column("param_x", sa.String(), nullable=False, server_default=""))
    op.add_column("loss_landscapes", sa.Column("param_y", sa.String(), nullable=False, server_default=""))
    op.add_column("loss_landscapes", sa.Column("grid_size", sa.Integer(), nullable=False, server_default="25"))
