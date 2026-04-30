from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.models.database import get_db
from api.models.dataset import Dataset
from api.models.experiment import Experiment, ExperimentStatus

router = APIRouter()


class HyperParams(BaseModel):
    hidden_size: int = 256
    num_layers: int = 1
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 10
    gradient_clip: float = 1.0


class ExperimentCreate(BaseModel):
    dataset_id: str
    architecture: str
    k1: int = Field(ge=1)
    k2: int = Field(ge=1)
    seed: int = 42
    hyperparams: HyperParams = HyperParams()


class MatrixCreate(BaseModel):
    dataset_id: str
    architectures: list[str] = ["lstm", "gru"]
    k2_ratios: list[float] = [0.05, 0.1, 0.25, 0.5, 1.0]
    k1_modes: list[str] = ["k1_eq_1", "k1_eq_k2"]
    seeds: list[int] = [42, 43, 44]
    hyperparams: HyperParams = HyperParams()


class ExperimentResponse(BaseModel):
    id: str
    dataset_id: str
    architecture: str
    k1: int
    k2: int
    seed: int
    hyperparams: dict
    status: str
    best_metric: Optional[float]
    created_at: datetime
    finished_at: Optional[datetime]

    model_config = {"from_attributes": True}


@router.post("/", response_model=ExperimentResponse, status_code=201)
async def create_experiment(body: ExperimentCreate, db: AsyncSession = Depends(get_db)):
    ds = await db.get(Dataset, body.dataset_id)
    if not ds:
        raise HTTPException(404, f"Dataset {body.dataset_id} nie znaleziony")

    if body.k1 > body.k2:
        raise HTTPException(422, f"k1={body.k1} > k2={body.k2}")

    exp = Experiment(
        dataset_id=body.dataset_id,
        architecture=body.architecture,
        k1=body.k1, k2=body.k2, seed=body.seed,
        hyperparams=body.hyperparams.model_dump(),
    )
    db.add(exp)
    await db.commit()
    await db.refresh(exp)

    from api.tasks.training import train_experiment
    task = train_experiment.delay(exp.id)
    exp.celery_task_id = task.id
    exp.status = ExperimentStatus.PENDING
    await db.commit()

    return exp


@router.post("/matrix", response_model=list[ExperimentResponse], status_code=201)
async def create_matrix(body: MatrixCreate, db: AsyncSession = Depends(get_db)):
    ds = await db.get(Dataset, body.dataset_id)
    if not ds:
        raise HTTPException(404, f"Dataset {body.dataset_id} nie znaleziony")

    T = ds.T
    experiments = []
    for arch in body.architectures:
        for ratio in body.k2_ratios:
            k2 = max(1, int(ratio * T))
            for k1_mode in body.k1_modes:
                k1 = 1 if k1_mode == "k1_eq_1" else k2
                for seed in body.seeds:
                    exp = Experiment(
                        dataset_id=body.dataset_id,
                        architecture=arch, k1=k1, k2=k2, seed=seed,
                        hyperparams=body.hyperparams.model_dump(),
                    )
                    db.add(exp)
                    experiments.append(exp)

    await db.commit()

    from api.tasks.training import train_experiment
    for exp in experiments:
        await db.refresh(exp)
        task = train_experiment.delay(exp.id)
        exp.celery_task_id = task.id
        exp.status = ExperimentStatus.PENDING

    await db.commit()
    return experiments


@router.get("/", response_model=list[ExperimentResponse])
async def list_experiments(
    dataset_id: str | None = None,
    architecture: str | None = None,
    status: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    q = select(Experiment)

    if dataset_id:
        q = q.where(Experiment.dataset_id == dataset_id)

    if architecture:
        q = q.where(Experiment.architecture == architecture)

    if status:
        q = q.where(Experiment.status == status)

    result = await db.execute(q)
    return result.scalars().all()


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: str, db: AsyncSession = Depends(get_db)):
    exp = await db.get(Experiment, experiment_id)
    if not exp:
        raise HTTPException(404)

    return exp


@router.delete("/{experiment_id}", status_code=204)
async def delete_experiment(experiment_id: str, db: AsyncSession = Depends(get_db)):
    exp = await db.get(Experiment, experiment_id)
    if not exp:
        raise HTTPException(404)

    await db.delete(exp)
    await db.commit()


@router.post("/{experiment_id}/cancel", response_model=ExperimentResponse)
async def cancel_experiment(experiment_id: str, db: AsyncSession = Depends(get_db)):
    exp = await db.get(Experiment, experiment_id)
    if not exp:
        raise HTTPException(404)

    if exp.celery_task_id:
        from api.tasks.training import celery_app
        celery_app.control.revoke(exp.celery_task_id, terminate=True)

    exp.status = ExperimentStatus.CANCELLED
    await db.commit()
    return exp
