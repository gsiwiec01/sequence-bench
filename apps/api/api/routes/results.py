from collections import defaultdict

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from api.models.database import get_db
from api.models.dataset import Dataset
from api.models.experiment import Experiment, ExperimentStatus
from api.models.metrics import EpochMetric

router = APIRouter()


@router.get("/compare")
async def compare_experiments(
    experiment_ids: list[str] = Query(..., description="Lista ID eksperymentów do porównania"),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Experiment).where(Experiment.id.in_(experiment_ids))
    )
    exps = result.scalars().all()

    return [
        {
            "id": e.id,
            "architecture": e.architecture,
            "dataset_id": e.dataset_id,
            "k1": e.k1,
            "k2": e.k2,
            "seed": e.seed,
            "status": e.status,
            "best_metric": e.best_metric,
            "hyperparams": e.hyperparams,
        }
        for e in exps
    ]


@router.get("/degradation")
async def degradation_curves(
    dataset_id: str,
    metric: str = "best_metric",
    group_by: str = "architecture",
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Experiment).where(
            Experiment.dataset_id == dataset_id,
            Experiment.status == ExperimentStatus.COMPLETED,
        )
    )
    exps = result.scalars().all()

    if not exps:
        return {}

    ds = await db.get(Dataset, dataset_id)
    T = ds.T if ds else 1

    groups: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for e in exps:
        key = getattr(e, group_by, group_by)
        ratio = round(e.k2 / T, 4)
        val = e.best_metric
        if val is not None:
            groups[key][ratio].append(val)

    output = {}
    for group_name, by_ratio in groups.items():
        ratios = sorted(by_ratio.keys())
        baseline_ratio = max(ratios)
        baseline = float(sum(by_ratio[baseline_ratio]) / len(by_ratio[baseline_ratio]))
        if baseline == 0:
            continue
        output[group_name] = {
            "k2_ratios": ratios,
            "delta_mean": [
                float(sum(by_ratio[r]) / len(by_ratio[r])) / baseline for r in ratios
            ],
            "delta_std": [
                float(
                    (
                        sum(
                            (x - sum(by_ratio[r]) / len(by_ratio[r])) ** 2
                            for x in by_ratio[r]
                        )
                        / max(len(by_ratio[r]) - 1, 1)
                    )
                    ** 0.5
                )
                / baseline
                for r in ratios
            ],
            "n_per_ratio": [len(by_ratio[r]) for r in ratios],
        }
    return output


@router.get("/{experiment_id}/epochs")
async def get_epoch_metrics(experiment_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(EpochMetric)
        .where(EpochMetric.experiment_id == experiment_id)
        .order_by(EpochMetric.epoch)
    )
    metrics = result.scalars().all()
    return [
        {
            "epoch": m.epoch,
            "train_loss": m.train_loss,
            "val_loss": m.val_loss,
            "metric_value": m.metric_value,
            "epoch_time_s": m.epoch_time_s,
            "gpu_memory_mb": m.gpu_memory_mb,
        }
        for m in metrics
    ]
