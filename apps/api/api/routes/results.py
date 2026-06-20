import csv
import io
from collections import defaultdict

try:
    import openpyxl
    _OPENPYXL_AVAILABLE = True
except ImportError:
    _OPENPYXL_AVAILABLE = False

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.models.database import get_db
from api.models.dataset import Dataset
from api.models.experiment import Experiment, ExperimentStatus
from api.models.metrics import AdditionalMetric, EpochMetric, GradientLog
from api.plots import render_convergence_png, render_degradation_png, render_loss_png

router = APIRouter()

@router.get("/export")
async def export_results(
    dataset_id: str | None = Query(None),
    group_id: str | None = Query(None),
    architecture: str | None = Query(None),
    format: str = Query("csv"),
    db: AsyncSession = Depends(get_db),
):
    q = select(Experiment)
    if dataset_id:
        q = q.where(Experiment.dataset_id == dataset_id)

    if group_id:
        q = q.where(Experiment.group_id == group_id)

    if architecture:
        q = q.where(Experiment.architecture == architecture)

    result = await db.execute(q)
    exps = result.scalars().all()

    ds_ids = list({e.dataset_id for e in exps})
    ds_map: dict[str, str] = {}
    for ds_id in ds_ids:
        ds = await db.get(Dataset, ds_id)
        if ds:
            ds_map[ds_id] = ds.name

    T_map: dict[str, int] = {}
    for ds_id in ds_ids:
        ds = await db.get(Dataset, ds_id)
        if ds:
            T_map[ds_id] = ds.T

    COLUMNS = [
        "experiment_id", "dataset", "architecture", "k1", "k2", "k2_ratio",
        "seed", "best_metric", "convergence_epoch", "total_training_time_s",
        "n_parameters", "status",
    ]

    def _row(e: Experiment) -> list:
        T = T_map.get(e.dataset_id, 1)
        return [
            e.id,
            ds_map.get(e.dataset_id, e.dataset_id),
            e.architecture,
            e.k1,
            e.k2,
            round(e.k2 / T, 4),
            e.seed,
            e.best_metric,
            e.convergence_epoch,
            e.total_training_time_s,
            e.n_parameters,
            e.status,
        ]

    if format == "xlsx":
        if not _OPENPYXL_AVAILABLE:
            raise HTTPException(422, "openpyxl nie jest zainstalowany -użyj format=csv")

        exp_ids = [e.id for e in exps]
        em_result = await db.execute(
            select(EpochMetric)
            .where(EpochMetric.experiment_id.in_(exp_ids))
            .order_by(EpochMetric.experiment_id, EpochMetric.epoch)
        )
        epoch_metrics = em_result.scalars().all()

        wb = openpyxl.Workbook()
        ws1 = wb.active
        ws1.title = "Wyniki"
        ws1.append(COLUMNS)
        for e in exps:
            ws1.append(_row(e))

        ws2 = wb.create_sheet("Epoch Metrics")
        ws2.append(["experiment_id", "epoch", "train_loss", "val_loss"])
        for m in epoch_metrics:
            ws2.append([m.experiment_id, m.epoch, m.train_loss, m.val_loss])

        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=results.xlsx"},
        )

    # Default: CSV
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(COLUMNS)

    for e in exps:
        writer.writerow(_row(e))

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=results.csv"},
    )


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


_MIN_METRICS_DEGRAD = frozenset({"mse", "mae", "mape", "perplexity", "cross_entropy", "loss"})

async def _fetch_metric_values(
    db: AsyncSession,
    exp_ids: list[str],
    metric: str | None,
) -> dict[str, float]:
    if not metric or not exp_ids:
        return {}

    use_min = metric in _MIN_METRICS_DEGRAD
    agg_fn = func.min if use_min else func.max

    q = (
        select(
            AdditionalMetric.experiment_id.label("exp_id"),
            agg_fn(AdditionalMetric.metric_value).label("best_val"),
        )
        .where(
            AdditionalMetric.experiment_id.in_(exp_ids),
            AdditionalMetric.metric_name == metric,
        )
        .group_by(AdditionalMetric.experiment_id)
    )

    rows = await db.execute(q)
    return {row.exp_id: row.best_val for row in rows}


async def compute_degradation_curves(
    dataset_id: str,
    metric: str | None,
    group_by: str,
    baseline_k2: int | None,
    db: AsyncSession,
    baseline_k1: int | None = None,
    only_converged: bool = False,
) -> dict:
    result = await db.execute(
        select(Experiment).where(
            Experiment.dataset_id == dataset_id,
            Experiment.status == ExperimentStatus.COMPLETED,
        )
    )
    exps = result.scalars().all()

    if not exps:
        return {"groups": {}, "meta": {"T": 0, "total_fetched": 0, "with_metric": 0, "null_metric_ids": []}}

    ds = await db.get(Dataset, dataset_id)
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


@router.get("/degradation")
async def degradation_curves(
    dataset_id: str,
    metric: str | None = Query(None, description="Metryka do δ(k₂); None = best_metric eksperymentu"),
    group_by: str = "architecture",
    baseline_k2: int | None = None,
    baseline_k1: int | None = None,
    only_converged: bool = False,
    db: AsyncSession = Depends(get_db),
):
    return await compute_degradation_curves(dataset_id, metric, group_by, baseline_k2, db, baseline_k1, only_converged)


@router.get("/degradation.png")
async def degradation_curves_png(
    dataset_id: str,
    metric: str | None = Query(None, description="Metryka do δ(k₂); None = best_metric eksperymentu"),
    group_by: str = "architecture",
    baseline_k2: int | None = None,
    baseline_k1: int | None = None,
    only_converged: bool = False,
    db: AsyncSession = Depends(get_db),
):
    data = await compute_degradation_curves(dataset_id, metric, group_by, baseline_k2, db, baseline_k1, only_converged)
    png = await run_in_threadpool(render_degradation_png, data)

    return Response(
        content=png,
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="degradation.png"'},
    )

_CONV_DEFAULT_METRIC: dict[str, str] = {
    "regression": "mse",
    "classification": "accuracy",
    "seq2seq": "accuracy",
}

async def compute_convergence_epochs(
    dataset_id: str,
    group_by: str,
    db: AsyncSession,
    group_id: str | None = None,
    metric: str | None = None,
    metric_threshold: float | None = None,
) -> dict:
    filters = [
        Experiment.dataset_id == dataset_id,
        Experiment.status == ExperimentStatus.COMPLETED,
    ]
    if group_id is not None:
        filters.append(Experiment.group_id == group_id)

    result = await db.execute(select(Experiment).where(*filters))
    exps = result.scalars().all()

    if not exps:
        return {
            "dataset_id": dataset_id, "group_by": group_by, "points": [],
            "metric": metric, "threshold": metric_threshold, "threshold_mode": None,
        }

    ds = await db.get(Dataset, dataset_id)
    T = ds.T if ds else 1

    eff_metric = metric or _CONV_DEFAULT_METRIC.get(exps[0].task_type, "accuracy")
    mode = "min" if eff_metric in _MIN_METRICS_DEGRAD else "max"
    threshold = metric_threshold if metric_threshold is not None else (0.01 if mode == "min" else 0.5)

    metric_values = await _fetch_metric_values(db, [e.id for e in exps], eff_metric)

    def _passes(e: Experiment) -> bool:
        if e.convergence_epoch is None:
            return False

        val = metric_values.get(e.id)

        if val is None:
            return False

        return (val < threshold) if mode == "min" else (val > threshold)

    SAFE_ATTRS = {"architecture", "k1", "k2", "seed"}
    groups: dict[str, dict[float, list]] = defaultdict(lambda: defaultdict(list))

    for e in exps:
        if group_by not in SAFE_ATTRS:
            continue
        key = str(getattr(e, group_by))
        ratio = round(e.k2 / T, 4)
        ce = e.convergence_epoch if _passes(e) else None
        groups[key][ratio].append((ce, e.k2))

    points = []
    for group_name, by_ratio in groups.items():
        for ratio, data_list in sorted(by_ratio.items()):
            n_seeds = len(data_list)
            k2 = data_list[0][1]
            converged = [ce for ce, _ in data_list if ce is not None]
            n_converged = len(converged)

            if n_converged == 0:
                mean = std = min_val = max_val = None
            else:
                mean = sum(converged) / n_converged
                variance = sum((v - mean) ** 2 for v in converged) / max(n_converged - 1, 1)
                std = variance ** 0.5
                min_val = min(converged)
                max_val = max(converged)

            points.append({
                "k2": k2,
                "k2_ratio": ratio,
                group_by: group_name,
                "convergence_epoch_mean": mean,
                "convergence_epoch_std": std,
                "convergence_epoch_min": min_val,
                "convergence_epoch_max": max_val,
                "n_seeds": n_seeds,
                "n_converged": n_converged,
            })

    points.sort(key=lambda p: (str(p[group_by]), p["k2_ratio"]))
    return {
        "dataset_id": dataset_id, "group_by": group_by, "points": points,
        "metric": eff_metric, "threshold": threshold, "threshold_mode": mode,
    }


@router.get("/convergence")
async def convergence_epochs(
    dataset_id: str,
    group_by: str = "architecture",
    group_id: str | None = None,
    metric: str | None = None,
    metric_threshold: float | None = None,
    db: AsyncSession = Depends(get_db),
):
    return await compute_convergence_epochs(
        dataset_id, group_by, db, group_id, metric, metric_threshold
    )


@router.get("/convergence.png")
async def convergence_epochs_png(
    dataset_id: str,
    group_by: str = "architecture",
    group_id: str | None = None,
    metric: str | None = None,
    metric_threshold: float | None = None,
    db: AsyncSession = Depends(get_db),
):
    data = await compute_convergence_epochs(
        dataset_id, group_by, db, group_id, metric, metric_threshold
    )

    png = await run_in_threadpool(render_convergence_png, data)
    return Response(
        content=png,
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="convergence.png"'},
    )


async def _fetch_epoch_metrics(experiment_id: str, db: AsyncSession) -> list[dict]:
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
            "epoch_time_s": m.epoch_time_s,
            "gpu_memory_mb": m.gpu_memory_mb,
            "grad_norm_mean": m.grad_norm_mean,
            "grad_norm_max": m.grad_norm_max,
        }
        for m in metrics
    ]


@router.get("/{experiment_id}/epochs")
async def get_epoch_metrics(experiment_id: str, db: AsyncSession = Depends(get_db)):
    return await _fetch_epoch_metrics(experiment_id, db)


@router.get("/{experiment_id}/loss.png")
async def loss_png(experiment_id: str, db: AsyncSession = Depends(get_db)):
    metrics = await _fetch_epoch_metrics(experiment_id, db)
    png = await run_in_threadpool(render_loss_png, metrics)
    return Response(
        content=png,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{experiment_id}_loss.png"'},
    )
