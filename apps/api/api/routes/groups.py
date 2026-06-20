from collections import defaultdict
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.models.database import get_db
from api.models.dataset import Dataset
from api.models.experiment import Experiment, ExperimentStatus
from api.models.group import ExperimentGroup
from api.plots import render_degradation_png
from api.routes.results import _fetch_metric_values

router = APIRouter()

class GroupCreate(BaseModel):
    name: str
    description: Optional[str] = None
    tags: list[str] = []
    dataset: Optional[str] = None


class GroupResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    tags: list[str]
    created_at: datetime
    dataset: Optional[str]
    created_from_matrix: bool
    n_total: int
    n_completed: int
    n_running: int
    n_pending: int
    n_failed: int
    n_cancelled: int
    best_metric_by_arch: dict[str, Optional[float]]

class GroupDetailResponse(GroupResponse):
    experiment_ids: list[str]

class GroupUpdate(BaseModel):
    name: str

class AddExperimentsBody(BaseModel):
    experiment_ids: list[str]

async def _load_experiments(group_id: str, db: AsyncSession) -> list[Experiment]:
    result = await db.execute(
        select(Experiment).where(Experiment.group_id == group_id)
    )
    return list(result.scalars().all())

def _build_response(g: ExperimentGroup, exps: list[Experiment]) -> dict:
    n_completed = sum(1 for e in exps if e.status == ExperimentStatus.COMPLETED)
    n_running = sum(1 for e in exps if e.status == ExperimentStatus.RUNNING)
    n_pending = sum(1 for e in exps if e.status == ExperimentStatus.PENDING)
    n_failed = sum(1 for e in exps if e.status == ExperimentStatus.FAILED)
    n_cancelled = sum(1 for e in exps if e.status == ExperimentStatus.CANCELLED)

    best_by_arch: dict[str, Optional[float]] = {}
    for e in exps:
        if e.best_metric is not None:
            cur = best_by_arch.get(e.architecture)
            if cur is None or e.best_metric > cur:
                best_by_arch[e.architecture] = e.best_metric

    return dict(
        id=g.id,
        name=g.name,
        description=g.description,
        tags=g.tags or [],
        created_at=g.created_at,
        dataset=g.dataset,
        created_from_matrix=g.created_from_matrix,
        n_total=len(exps),
        n_completed=n_completed,
        n_running=n_running,
        n_pending=n_pending,
        n_failed=n_failed,
        n_cancelled=n_cancelled,
        best_metric_by_arch=best_by_arch,
        experiment_ids=[e.id for e in exps],
    )

@router.get("/", response_model=list[GroupResponse])
async def list_groups(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ExperimentGroup).order_by(ExperimentGroup.created_at.desc())
    )
    groups = result.scalars().all()

    out = []
    for g in groups:
        exps = await _load_experiments(g.id, db)
        out.append(GroupResponse(**_build_response(g, exps)))

    return out


@router.get("/{group_id}", response_model=GroupDetailResponse)
async def get_group(group_id: str, db: AsyncSession = Depends(get_db)):
    g = await db.get(ExperimentGroup, group_id)
    if not g:
        raise HTTPException(404, "Grupa nie znaleziona")

    exps = await _load_experiments(group_id, db)
    return GroupDetailResponse(**_build_response(g, exps))


@router.post("/", response_model=GroupDetailResponse, status_code=201)
async def create_group(body: GroupCreate, db: AsyncSession = Depends(get_db)):
    g = ExperimentGroup(
        name=body.name,
        description=body.description,
        tags=body.tags,
        dataset=body.dataset,
        created_from_matrix=False,
    )

    db.add(g)

    await db.commit()
    await db.refresh(g)

    return GroupDetailResponse(**_build_response(g, []))


@router.patch("/{group_id}", response_model=GroupResponse)
async def rename_group(group_id: str, body: GroupUpdate, db: AsyncSession = Depends(get_db)):
    g = await db.get(ExperimentGroup, group_id)
    if not g:
        raise HTTPException(404, "Grupa nie znaleziona")

    g.name = body.name

    await db.commit()
    await db.refresh(g)

    exps = await _load_experiments(group_id, db)
    return GroupResponse(**_build_response(g, exps))


@router.post("/{group_id}/experiments", status_code=200)
async def add_experiments_to_group(
    group_id: str,
    body: AddExperimentsBody,
    db: AsyncSession = Depends(get_db),
):
    g = await db.get(ExperimentGroup, group_id)
    if not g:
        raise HTTPException(404, "Grupa nie znaleziona")

    result = await db.execute(
        select(Experiment).where(Experiment.id.in_(body.experiment_ids))
    )
    exps = result.scalars().all()
    for exp in exps:
        exp.group_id = group_id

    await db.commit()
    return {"added": len(exps)}


@router.delete("/{group_id}/experiments/{experiment_id}", status_code=204)
async def remove_experiment_from_group(
    group_id: str,
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
):
    exp = await db.get(Experiment, experiment_id)
    if not exp or exp.group_id != group_id:
        raise HTTPException(404, "Eksperyment nie należy do tej grupy")

    exp.group_id = None
    await db.commit()


@router.delete("/{group_id}", status_code=204)
async def delete_group(
    group_id: str,
    delete_experiments: bool = Query(False, description="Usuń również eksperymenty należące do grupy"),
    db: AsyncSession = Depends(get_db),
):
    g = await db.get(ExperimentGroup, group_id)
    if not g:
        raise HTTPException(404, "Grupa nie znaleziona")

    if delete_experiments:
        result = await db.execute(
            select(Experiment).where(Experiment.group_id == group_id)
        )
        for exp in result.scalars().all():
            await db.delete(exp)

    await db.delete(g)
    await db.commit()


@router.post("/{group_id}/cancel_all")
async def cancel_all_running(group_id: str, db: AsyncSession = Depends(get_db)):
    g = await db.get(ExperimentGroup, group_id)
    if not g:
        raise HTTPException(404, "Grupa nie znaleziona")

    result = await db.execute(
        select(Experiment).where(
            Experiment.group_id == group_id,
            Experiment.status.in_([ExperimentStatus.RUNNING, ExperimentStatus.PENDING]),
        )
    )
    active = result.scalars().all()

    from api.celery_client import celery_app
    for exp in active:
        if exp.celery_task_id:
            celery_app.control.revoke(exp.celery_task_id, terminate=True)
        exp.status = ExperimentStatus.CANCELLED

    await db.commit()
    return {"cancelled": len(active)}


@router.post("/{group_id}/retry_failed")
async def retry_failed(group_id: str, db: AsyncSession = Depends(get_db)):
    g = await db.get(ExperimentGroup, group_id)
    if not g:
        raise HTTPException(404, "Grupa nie znaleziona")

    result = await db.execute(
        select(Experiment).where(
            Experiment.group_id == group_id,
            Experiment.status == ExperimentStatus.FAILED,
        )
    )
    failed = result.scalars().all()

    from api.celery_client import celery_app
    for exp in failed:
        task = celery_app.send_task("tasks.train_experiment", args=[exp.id])
        exp.celery_task_id = task.id
        exp.status = ExperimentStatus.PENDING

    await db.commit()
    return {"retried": len(failed)}


@router.get("/{group_id}/degradation")
async def group_degradation(
    group_id: str,
    metric: str | None = Query(None, description="Metryka do δ(k₂); None = best_metric eksperymentu"),
    group_by: str = "architecture",
    baseline_k2: int | None = None,
    baseline_k1: int | None = None,
    only_converged: bool = False,
    db: AsyncSession = Depends(get_db),
):
    g = await db.get(ExperimentGroup, group_id)
    if not g:
        raise HTTPException(404, "Grupa nie znaleziona")

    result = await db.execute(
        select(Experiment).where(
            Experiment.group_id == group_id,
            Experiment.status == ExperimentStatus.COMPLETED,
        )
    )
    exps = list(result.scalars().all())

    if not exps:
        return {"groups": {}, "meta": {"T": 0, "total_fetched": 0, "with_metric": 0, "null_metric_ids": []}}

    ds = await db.get(Dataset, exps[0].dataset_id)
    T = ds.T if ds else 1

    exp_ids = [e.id for e in exps]
    metric_values = await _fetch_metric_values(db, exp_ids, metric)

    SAFE_ATTRS = {"architecture", "k1", "k2", "seed"}
    groups: dict[str, dict[float, list[tuple[float, str, int, int]]]] = defaultdict(lambda: defaultdict(list))
    null_ids: list[str] = []

    for e in exps:
        if group_by not in SAFE_ATTRS:
            continue

        if only_converged and e.convergence_epoch is None:
            null_ids.append(e.id)
            continue

        key = str(getattr(e, group_by))
        ratio = round(e.k2 / T, 4)
        val = metric_values.get(e.id) if metric_values else e.best_metric

        if val is not None:
            groups[key][ratio].append((val, e.id, e.k2, e.k1))
        else:
            null_ids.append(e.id)

    output: dict[str, dict] = {}
    for group_name, by_ratio in groups.items():
        ratios = sorted(by_ratio.keys())

        if baseline_k2 is not None:
            target = round(baseline_k2 / T, 4)
            baseline_ratio = min(ratios, key=lambda r: abs(r - target))
        else:
            baseline_ratio = max(ratios)

        if baseline_k1 is not None:
            baseline_vals = [v for v, _, _, k1 in by_ratio[baseline_ratio] if k1 == baseline_k1]
        else:
            baseline_vals = [v for v, _, _, _ in by_ratio[baseline_ratio]]

        if not baseline_vals:
            baseline_vals = [v for v, _, _, _ in by_ratio[baseline_ratio]]

        baseline = float(sum(baseline_vals) / len(baseline_vals))
        if baseline == 0:
            continue

        output[group_name] = {
            "k2_ratios": ratios,
            "k2_values": [by_ratio[r][0][2] for r in ratios],
            "baseline_k2_ratio": baseline_ratio,
            "delta_mean": [
                float(sum(v for v, _, _, _ in by_ratio[r]) / len(by_ratio[r])) / baseline
                for r in ratios
            ],
            "delta_std": [
                float(
                    (
                        sum(
                            (v - sum(v2 for v2, _, _, _ in by_ratio[r]) / len(by_ratio[r])) ** 2
                            for v, _, _, _ in by_ratio[r]
                        )
                        / max(len(by_ratio[r]) - 1, 1)
                    )
                    ** 0.5
                )
                / baseline
                for r in ratios
            ],
            "n_per_ratio": [len(by_ratio[r]) for r in ratios],
            "experiment_ids": [[eid for _, eid, _, _ in by_ratio[r]] for r in ratios],
        }

    return {
        "groups": output,
        "meta": {
            "T": T,
            "total_fetched": len(exps),
            "with_metric": sum(1 for e in exps if e.best_metric is not None),
            "null_metric_ids": null_ids,
        },
    }


@router.get("/{group_id}/degradation.png")
async def group_degradation_png(
    group_id: str,
    metric: str | None = Query(None, description="Metryka do δ(k₂); None = best_metric eksperymentu"),
    group_by: str = "architecture",
    baseline_k2: int | None = None,
    baseline_k1: int | None = None,
    only_converged: bool = False,
    db: AsyncSession = Depends(get_db),
):
    data = await group_degradation(group_id, metric, group_by, baseline_k2, baseline_k1, only_converged, db)
    png = await run_in_threadpool(render_degradation_png, data)

    return Response(
        content=png,
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="degradation.png"'},
    )
