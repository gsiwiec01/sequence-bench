import asyncio
import logging
from contextlib import asynccontextmanager

import torch
import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from api.config import settings
from api.json_utils import NaNSafeJSONResponse
from api.models.database import engine, Base, AsyncSessionLocal
from api.routes.datasets import seed_builtin_datasets

logger = logging.getLogger(__name__)

_DB_RETRIES = 10
_DB_RETRY_DELAY = 3.0


async def _wait_for_db() -> None:
    for attempt in range(1, _DB_RETRIES + 1):
        try:
            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1"))

            return
        except Exception as exc:
            if attempt == _DB_RETRIES:
                raise
            logger.warning("DB not ready (attempt %d/%d): %s -retrying in %.0fs", attempt, _DB_RETRIES, exc, _DB_RETRY_DELAY)
            await asyncio.sleep(_DB_RETRY_DELAY)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await _wait_for_db()

    async with engine.begin() as conn:
        await conn.execute(text("SELECT pg_advisory_xact_lock(7472650)"))
        await conn.run_sync(lambda c: Base.metadata.create_all(c, checkfirst=True))

    from api.db_migrations import run_pending_migrations
    await asyncio.to_thread(run_pending_migrations)

    async with engine.begin() as conn:
        await conn.execute(text(
            "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS "
            "task_type VARCHAR NOT NULL DEFAULT 'classification'"
        ))
        await conn.execute(text("""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'datasets' AND column_name = 'default_task_type'
                ) THEN
                    UPDATE datasets SET task_type = default_task_type
                    WHERE default_task_type IS NOT NULL AND task_type = 'classification';
                    ALTER TABLE datasets DROP COLUMN IF EXISTS default_task_type;
                END IF;
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'datasets' AND column_name = 'compatible_task_types'
                ) THEN
                    ALTER TABLE datasets DROP COLUMN IF EXISTS compatible_task_types;
                END IF;
            END
            $$;
        """))

    async with AsyncSessionLocal() as db:
        await seed_builtin_datasets(db)

    yield
    await engine.dispose()


app = FastAPI(
    title="sequence-bench API",
    version="0.1.0",
    lifespan=lifespan,
    default_response_class=NaNSafeJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.routes import datasets, experiments, gradients, groups, results, monitoring
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["experiments"])
app.include_router(gradients.router, prefix="/api/gradients", tags=["gradients"])
app.include_router(groups.router, prefix="/api/groups", tags=["groups"])
app.include_router(results.router, prefix="/api/results", tags=["results"])
app.include_router(monitoring.router, prefix="/api", tags=["monitoring"])


@app.get("/health")
async def health() -> dict:
    checks: dict[str, str] = {}

    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        checks["db"] = "ok"
    except Exception as e:
        checks["db"] = f"error: {e}"

    try:
        r = aioredis.from_url(settings.redis_url)
        await r.ping()
        await r.aclose()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"

    try:
        checks["gpu"] = "ok" if torch.cuda.is_available() else "none"
    except Exception:
        checks["gpu"] = "unavailable"

    overall = "ok" if all(v == "ok" for k, v in checks.items() if k != "gpu") else "degraded"
    return {"status": overall, **checks}
