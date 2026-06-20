import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from api.models.database import get_db
from api.models.metrics import GradientLog
from api.plots import render_gradient_trend_png, render_gradient_heatmap_png
from ml_engine.gradient_monitor import load_gradient_file

router = APIRouter()

@router.get("/{experiment_id}/epochs")
async def list_gradient_epochs(experiment_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(GradientLog.epoch)
        .where(GradientLog.experiment_id == experiment_id)
        .order_by(GradientLog.epoch)
    )
    return {"epochs": [row[0] for row in result.all()]}

@router.get("/{experiment_id}/param-trends")
async def get_param_trends(experiment_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(GradientLog)
        .where(GradientLog.experiment_id == experiment_id)
        .order_by(GradientLog.epoch)
    )
    logs = result.scalars().all()

    epochs: list[int] = []
    params: dict[str, list[float | None]] = {}

    for log in logs:
        epochs.append(log.epoch)
        try:
            norms = load_gradient_file(log.file_path)
        except (FileNotFoundError, Exception):
            norms = {}

        all_keys = set(params.keys()) | set(norms.keys())
        for key in all_keys:
            series = params.setdefault(key, [None] * (len(epochs) - 1))
            arr = norms.get(key)
            nonzero = [v for v in (arr or []) if v > 0]
            series.append(float(np.mean(nonzero)) if nonzero else None)

    return {"epochs": epochs, "params": params}

@router.get("/{experiment_id}/param-trends.png")
async def get_param_trends_png(experiment_id: str, db: AsyncSession = Depends(get_db)):
    data = await get_param_trends(experiment_id, db)
    png = await run_in_threadpool(render_gradient_trend_png, data)

    return Response(
        content=png,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{experiment_id}_gradient_trend.png"'},
    )


@router.get("/{experiment_id}/{epoch:int}")
async def get_gradient_norms(
    experiment_id: str,
    epoch: int,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(GradientLog).where(
            GradientLog.experiment_id == experiment_id,
            GradientLog.epoch == epoch,
        )
    )
    log = result.scalar_one_or_none()
    if not log:
        raise HTTPException(404, f"Brak logów gradientów dla epoch={epoch}")

    try:
        data = load_gradient_file(log.file_path)
        return data
    except FileNotFoundError:
        raise HTTPException(404, f"Plik .npz nie znaleziony: {log.file_path}")


@router.get("/{experiment_id}/{epoch:int}/heatmap.png")
async def get_gradient_heatmap_png(
    experiment_id: str,
    epoch: int,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(GradientLog).where(
            GradientLog.experiment_id == experiment_id,
            GradientLog.epoch == epoch,
        )
    )
    
    log = result.scalar_one_or_none()
    if not log:
        raise HTTPException(404, f"Brak logów gradientów dla epoch={epoch}")

    try:
        data = load_gradient_file(log.file_path)
    except FileNotFoundError:
        raise HTTPException(404, f"Plik .npz nie znaleziony: {log.file_path}")

    png = await run_in_threadpool(render_gradient_heatmap_png, data, epoch)
    return Response(
        content=png,
        media_type="image/png",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{experiment_id}_gradient_heatmap_epoch{epoch}.png"'
            )
        },
    )
