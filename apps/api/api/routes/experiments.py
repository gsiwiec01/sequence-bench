import hashlib
import io
import json
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from api.celery_client import celery_app
from api.config import settings
from api.metrics_registry import AVAILABLE_METRICS, ES_MODE
from api.models.database import get_db
from api.models.dataset import Dataset, DatasetType
from api.models.experiment import Experiment, ExperimentStatus
from api.models.group import ExperimentGroup
from api.models.metrics import AdditionalMetric, EpochMetric, GradientLog, LossLandscape
from api.plots import render_metric_png, render_surface_png
from ml_engine.pca import pca_directions, project_trajectory
from ml_engine.weight_tracker import load_trajectory

router = APIRouter()

class HyperParams(BaseModel):
    hidden_size: int = 256
    num_layers: int = 1
    dropout: float = 0.0
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 10
    gradient_clip: float = 1.0
    grad_clip_mode: Literal["norm", "value", "none"] = "norm"
    max_grad_value: float | None = None
    gradient_log_interval: int = 1
    weight_track_enabled: bool = True
    weight_log_interval: int = 1

    @model_validator(mode="after")
    def validate_clip_config(self) -> "HyperParams":
        if self.grad_clip_mode == "value" and self.max_grad_value is None:
            raise ValueError("max_grad_value required when grad_clip_mode='value'")

        if self.grad_clip_mode in ("norm", "none"):
            self.grad_clip_mode = "none" if self.gradient_clip == 0 else "norm"

        return self

def _valid_es_metrics(task_type: str) -> list[str]:
    return ["val_loss"] + AVAILABLE_METRICS.get(task_type, [])

class ExperimentCreate(BaseModel):
    dataset_id: str
    architecture: str
    k1: int = Field(ge=1)
    k2: int = Field(ge=1)
    seed: int = 42
    task_type: str
    early_stopping_metric: str = "val_loss"
    hyperparams: HyperParams = HyperParams()

    @model_validator(mode="after")
    def validate_task_type(self) -> "ExperimentCreate":
        if self.task_type not in AVAILABLE_METRICS:
            raise ValueError(f"Nieznany task_type: {self.task_type!r}. Dozwolone: {list(AVAILABLE_METRICS)}")

        valid = _valid_es_metrics(self.task_type)
        if self.early_stopping_metric not in valid:
            raise ValueError(f"early_stopping_metric musi być jedną z: {valid}")

        return self


class MatrixCreate(BaseModel):
    dataset_id: str
    task_type: str
    architectures: list[str] = ["lstm", "gru"]
    k2_values: list[int] | None = None
    k2_ratios: list[float] | None = None
    k1_values: list[int] | None = None
    k1_ratios: list[float] | None = None
    seeds: list[int] = [42, 43, 44]
    early_stopping_metric: str = "val_loss"
    hyperparams: HyperParams = HyperParams()
    clip_norms: list[float | None] | None = None

    @model_validator(mode="after")
    def validate_params(self) -> "MatrixCreate":
        if (self.k2_values is None) == (self.k2_ratios is None):
            raise ValueError("Podaj jedno k2_values lub k2_ratios")

        if (self.k1_values is None) == (self.k1_ratios is None):
            raise ValueError("Podaj jedno k1_values lub k1_ratios")

        if self.task_type not in AVAILABLE_METRICS:
            raise ValueError(f"Nieznany typ zadania: {self.task_type!r}")

        valid = _valid_es_metrics(self.task_type)
        if self.early_stopping_metric not in valid:
            raise ValueError(f"early_stopping_metric musi być jedną z: {valid}")

        return self


class ExperimentResponse(BaseModel):
    id: str
    dataset_id: str
    architecture: str
    k1: int
    k2: int
    seed: int
    task_type: str
    early_stopping_metric: str
    early_stopping_mode: str
    hyperparams: dict
    additional_metrics: list[str]
    status: str
    best_metric: Optional[float]
    created_at: datetime
    finished_at: Optional[datetime]
    group_id: Optional[str]
    n_parameters: Optional[int]
    total_training_time_s: Optional[float]
    convergence_epoch: Optional[int]
    final_train_loss: Optional[float]
    final_val_loss: Optional[float]
    has_weight_trajectory: bool = False

    model_config = {"from_attributes": True}


class MatrixResponse(BaseModel):
    experiments: list[ExperimentResponse]
    group_id: str


async def _build_experiment_json(exp: "Experiment", db) -> dict:
    def _load_grad(npz_path: str) -> dict[str, list[float]]:
        path = Path(npz_path)
        manifest = path.with_suffix(".json")
        key_map: dict[str, str] = json.loads(manifest.read_text()) if manifest.exists() else {}
        data = np.load(str(path), allow_pickle=False)
        return {key_map.get(k, k): data[k].tolist() for k in data.files}

    epoch_rows = (await db.execute(
        select(EpochMetric)
        .where(EpochMetric.experiment_id == exp.id)
        .order_by(EpochMetric.epoch)
    )).scalars().all()

    grad_logs = (await db.execute(
        select(GradientLog)
        .where(GradientLog.experiment_id == exp.id)
        .order_by(GradientLog.epoch)
    )).scalars().all()

    accumulated: dict[str, list] = {}
    for log in grad_logs:
        try:
            for param, norms in _load_grad(log.file_path).items():
                accumulated.setdefault(param, []).append(norms)
        except Exception:
            pass

    weights_trajectory = None
    traj_path = _full_traj_path(exp)
    if traj_path:
        try:
            W, _names, _shapes, epochs = load_trajectory(traj_path)
            if W.shape[0] >= 3:
                d1, d2, w_end, explained = pca_directions(W)
                a_t, b_t = project_trajectory(W, w_end, d1, d2)
                weights_trajectory = {
                    "pairs": ["PC1", "PC2"],
                    "trajectory": [[float(a), float(b)] for a, b in zip(a_t, b_t)],
                    "epochs": list(epochs),
                    "explained_variance": explained,
                }
        except Exception:
            pass

    result: dict = {
        "experiment": {
            "id": exp.id,
            "architecture": exp.architecture,
            "k1": exp.k1,
            "k2": exp.k2,
            "seed": exp.seed,
            "status": exp.status.value if hasattr(exp.status, "value") else str(exp.status),
            "best_metric": exp.best_metric,
            "convergence_epoch": exp.convergence_epoch,
            "total_training_time_s": exp.total_training_time_s,
            "final_train_loss": exp.final_train_loss,
            "final_val_loss": exp.final_val_loss,
            "created_at": exp.created_at.isoformat() if exp.created_at else None,
            "finished_at": exp.finished_at.isoformat() if exp.finished_at else None,
            "hyperparams": exp.hyperparams or {},
        },
        "epochs": [
            {
                "epoch": r.epoch,
                "train_loss": r.train_loss,
                "val_loss": r.val_loss,
                "grad_norm_mean": r.grad_norm_mean,
                "grad_norm_max": r.grad_norm_max,
                "learning_rate": r.learning_rate,
            }
            for r in epoch_rows
        ],
    }

    if accumulated:
        result["gradients"] = accumulated

    if weights_trajectory:
        result["weights_trajectory"] = weights_trajectory

    return result


class BulkExportBody(BaseModel):
    experiment_ids: list[str]

@router.post("/export-bulk")
async def export_bulk(body: BulkExportBody, db: AsyncSession = Depends(get_db)):
    if not body.experiment_ids:
        raise HTTPException(400, detail="Brak identyfikatorów eksperymentów")

    buf = io.BytesIO()
    exported = 0

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for exp_id in body.experiment_ids:
            exp = await db.get(Experiment, exp_id)
            if not exp or exp.status != ExperimentStatus.COMPLETED:
                continue

            data = await _build_experiment_json(exp, db)

            filename = f"exp_{exp_id[:8]}_{exp.architecture}_k{exp.k2}.json"
            zf.writestr(filename, json.dumps(data, ensure_ascii=False))
            exported += 1

    if exported == 0:
        raise HTTPException(400, detail="Żaden z wybranych eksperymentów nie jest ukończony")

    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="experiments_export.zip"'},
    )

@router.post("/", response_model=ExperimentResponse, status_code=201)
async def create_experiment(body: ExperimentCreate, db: AsyncSession = Depends(get_db)):
    ds = await db.get(Dataset, body.dataset_id)
    if not ds:
        raise HTTPException(404, f"Dataset {body.dataset_id} nie znaleziony")

    if body.k1 > body.k2:
        raise HTTPException(422, f"k1={body.k1} > k2={body.k2}")

    effective_task_type = ds.task_type

    exp = Experiment(
        dataset_id=body.dataset_id,
        architecture=body.architecture,
        k1=body.k1, k2=body.k2, seed=body.seed,
        task_type=effective_task_type,
        early_stopping_metric=body.early_stopping_metric,
        early_stopping_mode=ES_MODE.get(body.early_stopping_metric, "min"),
        hyperparams=body.hyperparams.model_dump(),
    )

    db.add(exp)
    await db.commit()
    await db.refresh(exp)

    task = celery_app.send_task("tasks.train_experiment", args=[exp.id])
    exp.celery_task_id = task.id
    exp.status = ExperimentStatus.PENDING

    await db.commit()
    return exp


@router.post("/matrix", response_model=MatrixResponse, status_code=201)
async def create_matrix(body: MatrixCreate, db: AsyncSession = Depends(get_db)):
    ds = await db.get(Dataset, body.dataset_id)
    if not ds:
        raise HTTPException(404, f"Dataset {body.dataset_id} nie znaleziony")

    effective_task_type = ds.task_type

    T = ds.T
    k2_list = body.k2_values if body.k2_values is not None else [max(1, int(r * T)) for r in body.k2_ratios]
    k1_list = body.k1_values if body.k1_values is not None else [max(1, int(r * T)) for r in body.k1_ratios]

    clip_list: list[float | None | object] = body.clip_norms if body.clip_norms is not None else [object()]

    experiments = []
    for arch in body.architectures:
        for k2 in k2_list:
            for k1 in k1_list:
                if k1 > k2:
                    continue

                for seed in body.seeds:
                    for clip in clip_list:
                        hp_dict = body.hyperparams.model_dump()
                        if body.clip_norms is not None:
                            if clip is None:
                                hp_dict["grad_clip_mode"] = "none"
                            else:
                                hp_dict["gradient_clip"] = float(clip)  # type: ignore[arg-type]
                                hp_dict["grad_clip_mode"] = "norm"

                        exp = Experiment(
                            dataset_id=body.dataset_id,
                            architecture=arch, k1=k1, k2=k2, seed=seed,
                            task_type=effective_task_type,
                            early_stopping_metric=body.early_stopping_metric,
                            early_stopping_mode=ES_MODE.get(body.early_stopping_metric, "min"),
                            hyperparams=hp_dict,
                        )

                        db.add(exp)
                        experiments.append(exp)

    if not experiments:
        raise HTTPException(422, "Żadna para k1/k2 nie jest poprawna (k1 > k2 dla wszystkich kombinacji)")

    archs_str = ", ".join(sorted({e.architecture for e in experiments}))
    k2s_str = ", ".join(str(k) for k in sorted({e.k2 for e in experiments}))
    ts = datetime.utcnow().strftime("%m-%d %H:%M")
    group_name = f"{ds.name} -{archs_str} -k2=[{k2s_str}] -{ts}"
    tags = [ds.name] + sorted({e.architecture for e in experiments})

    group = ExperimentGroup(
        name=group_name,
        dataset=ds.name,
        tags=tags,
        created_from_matrix=True,
    )

    db.add(group)
    await db.flush()

    for exp in experiments:
        exp.group_id = group.id

    await db.commit()

    for exp in experiments:
        await db.refresh(exp)
        task = celery_app.send_task("tasks.train_experiment", args=[exp.id])
        exp.celery_task_id = task.id
        exp.status = ExperimentStatus.PENDING

    await db.commit()
    await db.refresh(group)

    return MatrixResponse(experiments=experiments, group_id=group.id)


@router.get("/", response_model=list[ExperimentResponse])
async def list_experiments(
    dataset_id: str | None = None,
    group_id: str | None = None,
    architecture: str | None = None,
    status: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    q = select(Experiment)
    if dataset_id:
        q = q.where(Experiment.dataset_id == dataset_id)

    if group_id:
        q = q.where(Experiment.group_id == group_id)

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


@router.get("/{experiment_id}/clone-config")
async def clone_config(experiment_id: str, db: AsyncSession = Depends(get_db)):
    exp = await db.get(Experiment, experiment_id)
    if not exp:
        raise HTTPException(404)

    return {
        "dataset_id": exp.dataset_id,
        "architecture": exp.architecture,
        "k1": exp.k1,
        "k2": exp.k2,
        "seed": exp.seed,
        "task_type": exp.task_type,
        "early_stopping_metric": exp.early_stopping_metric,
        "hyperparams": exp.hyperparams,
    }


@router.get("/{experiment_id}/metrics")
async def get_additional_metrics(
    experiment_id: str,
    names: str | None = Query(None, description="Nazwy metryk oddzielone przecinkami"),
    epoch_from: int = Query(0, ge=0),
    epoch_to: int | None = Query(None, ge=0),
    db: AsyncSession = Depends(get_db),
):
    exp = await db.get(Experiment, experiment_id)
    if not exp:
        raise HTTPException(404)

    requested = [n.strip() for n in names.split(",")] if names else None

    q = select(AdditionalMetric).where(
        AdditionalMetric.experiment_id == experiment_id,
        AdditionalMetric.epoch >= epoch_from,
    )

    if epoch_to is not None:
        q = q.where(AdditionalMetric.epoch <= epoch_to)

    if requested:
        q = q.where(AdditionalMetric.metric_name.in_(requested))

    q = q.order_by(AdditionalMetric.epoch)

    result = await db.execute(q)
    rows = result.scalars().all()

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row.metric_name].append({"epoch": row.epoch, "value": row.metric_value})

    return grouped


@router.get("/{experiment_id}/metric.png")
async def metric_png(
    experiment_id: str,
    name: str = Query(..., description="Nazwa metryki dodatkowej do narysowania"),
    color: str = Query("#7c3aed"),
    db: AsyncSession = Depends(get_db),
):
    exp = await db.get(Experiment, experiment_id)
    if not exp:
        raise HTTPException(404)

    result = await db.execute(
        select(AdditionalMetric)
        .where(
            AdditionalMetric.experiment_id == experiment_id,
            AdditionalMetric.metric_name == name,
        )
        .order_by(AdditionalMetric.epoch)
    )

    series = [{"epoch": r.epoch, "value": r.metric_value} for r in result.scalars().all()]
    png = await run_in_threadpool(render_metric_png, series, name, color)

    return Response(
        content=png,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{experiment_id}_{name}.png"'},
    )


@router.get("/{experiment_id}/weight_trajectory")
async def get_weight_trajectory(experiment_id: str, db: AsyncSession = Depends(get_db)):
    exp = await db.get(Experiment, experiment_id)
    if not exp:
        raise HTTPException(404)

    path = _full_traj_path(exp)
    if not path:
        raise HTTPException(404, "Nieznaleziono danych trajektorii wag dla tego eksperymentu")

    W, names, shapes, epochs = load_trajectory(path)
    if W.shape[0] < 3:
        return {"pairs": ["PC1", "PC2"], "trajectory": [], "epochs": [], "explained_variance": None}

    d1, d2, w_end, explained = pca_directions(W)
    a_t, b_t = project_trajectory(W, w_end, d1, d2)
    trajectory = [[float(a), float(b)] for a, b in zip(a_t, b_t)]
    return {
        "pairs": ["PC1", "PC2"],
        "trajectory": trajectory,
        "epochs": list(epochs),
        "explained_variance": explained,
    }


def _full_traj_path(exp) -> str | None:
    path = exp.full_weight_path
    if path:
        return path
    safe_id = exp.id.replace("-", "_")
    candidate = Path(settings.gradient_storage_dir) / f"weights_full_{safe_id}.npz"
    return str(candidate) if candidate.exists() else None

SURFACE_METHOD_VERSION = "pca-v1"

def _surface_params(resolution: int, margin: float) -> dict:
    return {"resolution": int(resolution), "margin": float(margin), "method_version": SURFACE_METHOD_VERSION}

def _surface_cache_key(experiment_id: str, params: dict) -> str:
    canonical = json.dumps({"experiment_id": experiment_id, **params}, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def _load_landscape_npz(file_path: str) -> dict:
    with np.load(file_path) as data:
        result: dict = {
            "x_values": data["x_values"].tolist(),
            "y_values": data["y_values"].tolist(),
            "loss_grid": data["loss_grid"].tolist(),
        }

        if "a_traj" in data and "b_traj" in data:
            result["a_traj"] = data["a_traj"].tolist()
            result["b_traj"] = data["b_traj"].tolist()

        if "explained_variance" in data:
            result["explained_variance"] = float(data["explained_variance"])

        if "anchor_loss" in data:
            result["anchor_loss"] = float(data["anchor_loss"])

    return result


def _landscape_base(landscape: "LossLandscape") -> dict:
    return {
        "id": landscape.id,
        "status": landscape.status,
        "params": landscape.params,
        "x_range": landscape.x_range,
        "y_range": landscape.y_range,
        "explained_variance": landscape.explained_variance,
        "anchor_loss": landscape.anchor_loss,
        "error_message": landscape.error_message,
        "created_at": landscape.created_at.isoformat() if landscape.created_at else None,
    }


async def _landscape_response(landscape: "LossLandscape") -> dict:
    response = _landscape_base(landscape)
    if landscape.status == "completed" and landscape.file_path:
        try:
            file_data = await run_in_threadpool(_load_landscape_npz, landscape.file_path)
            response.update(file_data)
        except Exception as exc:
            response["error_message"] = f"Błąd odczytu danych: {exc}"

    return response

class LandscapeCreate(BaseModel):
    resolution: int = Field(default=25, ge=5, le=101)
    margin: float = Field(default=0.15, gt=0.0, le=1.0)
    force: bool = False

@router.post("/{experiment_id}/loss_landscape", status_code=202)
async def create_loss_landscape(
    experiment_id: str,
    body: LandscapeCreate,
    db: AsyncSession = Depends(get_db),
):
    exp = await db.get(Experiment, experiment_id)
    if not exp:
        raise HTTPException(404)

    if exp.status != ExperimentStatus.COMPLETED:
        raise HTTPException(422, "Eksperyment musi być ukończony")

    if not exp.has_weight_trajectory:
        raise HTTPException(422, "Brak trajektorii wag dla tego przebiegu")

    params = _surface_params(body.resolution, body.margin)
    cache_key = _surface_cache_key(experiment_id, params)

    existing = (await db.execute(
        select(LossLandscape).where(LossLandscape.cache_key == cache_key)
    )).scalar_one_or_none()

    if existing and not body.force:
        if existing.status == "completed":
            return {"id": existing.id, "status": "completed", "cached": True}

        if existing.status in ("queued", "running"):
            return {"id": existing.id, "status": existing.status, "cached": False}

    if existing:
        existing.cache_key = None
        await db.flush()

    landscape = LossLandscape(
        experiment_id=experiment_id,
        cache_key=cache_key,
        params=params,
        status="queued",
    )

    db.add(landscape)
    await db.commit()
    await db.refresh(landscape)

    task = celery_app.send_task("tasks.compute_loss_landscape", args=[landscape.id])
    landscape.celery_task_id = task.id
    await db.commit()

    return {"id": landscape.id, "status": "queued", "cached": False}


@router.get("/{experiment_id}/loss_landscape")
async def lookup_loss_landscape(
    experiment_id: str,
    resolution: int = 25,
    margin: float = 0.15,
    db: AsyncSession = Depends(get_db),
):
    params = _surface_params(resolution, margin)
    cache_key = _surface_cache_key(experiment_id, params)

    job = (await db.execute(
        select(LossLandscape).where(LossLandscape.cache_key == cache_key)
    )).scalar_one_or_none()
    if not job:
        return {"status": "none"}

    return await _landscape_response(job)


@router.get("/{experiment_id}/loss_landscape/{landscape_id}")
async def get_loss_landscape(
    experiment_id: str,
    landscape_id: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(LossLandscape).where(
            LossLandscape.id == landscape_id,
            LossLandscape.experiment_id == experiment_id,
        )
    )

    landscape = result.scalar_one_or_none()
    if not landscape:
        raise HTTPException(404)

    return await _landscape_response(landscape)


@router.get("/{experiment_id}/loss_landscape/{landscape_id}/surface.png")
async def surface_png(
    experiment_id: str,
    landscape_id: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(LossLandscape).where(
            LossLandscape.id == landscape_id,
            LossLandscape.experiment_id == experiment_id,
        )
    )

    landscape = result.scalar_one_or_none()
    if not landscape:
        raise HTTPException(404)

    if landscape.status != "completed" or not landscape.file_path:
        raise HTTPException(409, "Powierzchnia nie jest jeszcze policzona")

    try:
        with np.load(landscape.file_path) as d:
            data = {
                "x_values": d["x_values"].tolist(),
                "y_values": d["y_values"].tolist(),
                "loss_grid": d["loss_grid"].tolist(),
                "a_traj": d["a_traj"].tolist() if "a_traj" in d else [],
                "b_traj": d["b_traj"].tolist() if "b_traj" in d else [],
                "explained_variance": float(d["explained_variance"]) if "explained_variance" in d else None,
                "anchor_loss": float(d["anchor_loss"]) if "anchor_loss" in d else None,
            }

    except Exception as exc:
        raise HTTPException(500, f"Błąd odczytu danych powierzchni: {exc}")

    png = await run_in_threadpool(render_surface_png, data)
    return Response(
        content=png,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="surface_pca_{experiment_id[:8]}.png"'},
    )

@router.get("/{experiment_id}/export")
async def export_experiment(experiment_id: str, db: AsyncSession = Depends(get_db)):
    exp = await db.get(Experiment, experiment_id)
    if not exp:
        raise HTTPException(404)

    if exp.status != ExperimentStatus.COMPLETED:
        raise HTTPException(400, detail="Eksport dostępny po zakończeniu eksperymentu")

    result = await _build_experiment_json(exp, db)
    filename = f"exp_{experiment_id[:8]}_{exp.architecture}_k{exp.k2}.json"
    return Response(
        content=json.dumps(result, ensure_ascii=False),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

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
        celery_app.control.revoke(exp.celery_task_id, terminate=True)

    exp.status = ExperimentStatus.CANCELLED
    await db.commit()
    return exp
