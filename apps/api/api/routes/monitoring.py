import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
import redis.asyncio as aioredis
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings
from api.models.database import get_db
from api.models.experiment import Experiment, ExperimentStatus
from api.models.metrics import EpochMetric, LossLandscape

router = APIRouter()

@router.get("/experiments/{experiment_id}/stream")
async def stream_metrics(experiment_id: str, db: AsyncSession = Depends(get_db)):
    r = aioredis.from_url(settings.redis_url)
    channel = f"experiment:{experiment_id}:metrics"
    pubsub = r.pubsub()
    await pubsub.subscribe(channel)

    result = await db.execute(
        select(EpochMetric)
        .where(EpochMetric.experiment_id == experiment_id)
        .order_by(EpochMetric.epoch)
    )
    historical = result.scalars().all()

    async def event_generator():
        last_sent_epoch = -1
        try:
            for m in historical:
                data = json.dumps({
                    "epoch": m.epoch,
                    "train_loss": m.train_loss,
                    "val_loss": m.val_loss,
                    "epoch_time_s": m.epoch_time_s,
                    "gpu_memory_mb": m.gpu_memory_mb,
                    "grad_norm_mean": m.grad_norm_mean,
                    "grad_norm_max": m.grad_norm_max,
                    "learning_rate": m.learning_rate,
                })
                yield f"event: metric\ndata: {data}\n\n"
                last_sent_epoch = m.epoch

            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode()
                    parsed = json.loads(data)

                    if "event" in parsed and parsed["event"] in ("completed", "failed"):
                        yield f"event: {parsed['event']}\ndata: {data}\n\n"
                        break
                    else:
                        if parsed.get("epoch", -1) > last_sent_epoch:
                            last_sent_epoch = parsed["epoch"]
                            yield f"event: metric\ndata: {data}\n\n"
        finally:
            await pubsub.unsubscribe(channel)
            await r.aclose()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Nginx: wyłącz buforowanie SSE
        },
    )


@router.get("/experiments/{experiment_id}/loss_landscape/{job_id}/stream")
async def stream_surface_status(
    experiment_id: str, job_id: str, db: AsyncSession = Depends(get_db)
):
    r = aioredis.from_url(settings.redis_url)
    channel = f"surface:{job_id}:status"
    pubsub = r.pubsub()
    await pubsub.subscribe(channel)

    job = (await db.execute(
        select(LossLandscape).where(
            LossLandscape.id == job_id, LossLandscape.experiment_id == experiment_id
        )
    )).scalar_one_or_none()
    initial_status = job.status if job else "none"

    async def event_generator():
        try:
            if initial_status in ("completed", "failed"):
                payload = json.dumps({"status": initial_status})
                yield f"event: {initial_status}\ndata: {payload}\n\n"
                return

            yield f"event: status\ndata: {json.dumps({'status': initial_status})}\n\n"

            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                data = message["data"]
                if isinstance(data, bytes):
                    data = data.decode()
                parsed = json.loads(data)
                event = parsed.get("event")
                if event in ("completed", "failed"):
                    yield f"event: {event}\ndata: {data}\n\n"
                    break

                yield f"event: status\ndata: {data}\n\n"
        finally:
            await pubsub.unsubscribe(channel)
            await r.aclose()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
