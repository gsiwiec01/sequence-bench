import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from api.models.database import get_db
from api.models.metrics import GradientLog

router = APIRouter()


@router.get("/{experiment_id}/epochs")
async def list_gradient_epochs(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(GradientLog.epoch)
        .where(GradientLog.experiment_id == experiment_id)
        .order_by(GradientLog.epoch)
    )
    return {"epochs": [row[0] for row in result.all()]}


@router.get("/{experiment_id}/{epoch}")
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
        data = np.load(log.file_path, allow_pickle=False)
        return {key: data[key].tolist() for key in data.files}
    except FileNotFoundError:
        raise HTTPException(404, f"Plik .npz nie znaleziony: {log.file_path}")
