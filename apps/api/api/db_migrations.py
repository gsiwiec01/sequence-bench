from __future__ import annotations

from pathlib import Path

import sqlalchemy as sa
from sqlalchemy import create_engine, text

from api.config import settings

_ALEMBIC_DIR = Path(__file__).parent / "alembic"
_LOCK_KEY = 42
_SENTINEL_TABLE = "epoch_metrics"
_SENTINEL_COLUMN = "grad_norm_mean"

def run_pending_migrations() -> None:
    sync_url = settings.database_connection_string.replace("+asyncpg", "+psycopg2")
    engine = create_engine(sync_url, pool_pre_ping=True)

    with engine.connect() as conn:
        conn.execute(text(f"SELECT pg_advisory_lock({_LOCK_KEY})"))
        try:
            _apply(engine)
        finally:
            conn.execute(text(f"SELECT pg_advisory_unlock({_LOCK_KEY})"))
            conn.commit()

    engine.dispose()

def _apply(engine) -> None:
    from alembic.config import Config
    from alembic import command

    cfg = Config()
    cfg.set_main_option("script_location", str(_ALEMBIC_DIR))

    inspector = sa.inspect(engine)

    if inspector.has_table("alembic_version"):
        command.upgrade(cfg, "head")
        return

    already_migrated = (
        inspector.has_table(_SENTINEL_TABLE)
        and any(
            c["name"] == _SENTINEL_COLUMN
            for c in inspector.get_columns(_SENTINEL_TABLE)
        )
    )

    if already_migrated:
        command.stamp(cfg, "head")
    else:
        command.upgrade(cfg, "head")
