import math
import time
import json
from datetime import datetime, timezone

import numpy as np
import redis
import torch
import torch.nn as nn
from celery import Task
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from worker.celery_app import celery_app
from worker.dataset_registry import DATASET_REGISTRY
from api.config import settings
from api.metrics_registry import AVAILABLE_METRICS, ES_MODE, compute_additional_metrics
from api.models.dataset import Dataset
from api.models.experiment import Experiment, ExperimentStatus
from api.models.metrics import AdditionalMetric, EpochMetric, GradientLog
from ml_engine.datasets.generic import GenericSequenceDataModule
from ml_engine.datasets.upload import UploadConfig
from ml_engine.engine import MonitorConfig, TBPTTEngine
from ml_engine.gradient_monitor import GradientMonitor
from ml_engine.model_registry import create_model
from ml_engine.utils import get_device, set_seed
from ml_engine.weight_tracker import WeightTrackerr

def get_sync_session():
    sync_url = settings.database_connection_string.replace("+asyncpg", "+psycopg2")
    engine = create_engine(sync_url)
    return Session(engine)

@celery_app.task(bind=True, max_retries=2, name="tasks.train_experiment")
def train_experiment(self: Task, experiment_id: str) -> dict:
    r = redis.from_url(settings.redis_url)
    channel = f"experiment:{experiment_id}:metrics"

    with get_sync_session() as db:
        exp = db.get(Experiment, experiment_id)
        if not exp:
            return {"error": "Experiment not found"}

        _TERMINAL = {ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED}
        if exp.status in _TERMINAL:
            return {"experiment_id": experiment_id, "best_metric": exp.best_metric, "skipped": True}

        exp.status = ExperimentStatus.RUNNING
        db.commit()

        try:
            ds_record = db.get(Dataset, exp.dataset_id)
            hp = exp.hyperparams
            extra_metric_names: list[str] = AVAILABLE_METRICS.get(exp.task_type, [])

            data_module = _build_data_module(ds_record)

            set_seed(exp.seed)
            device = get_device()

            model = create_model(
                rnn_type=exp.architecture,
                input_size=ds_record.input_size,
                hidden_size=hp.get("hidden_size", 256),
                num_layers=hp.get("num_layers", 1),
                dropout=hp.get("dropout", 0.2),
                output_size=ds_record.output_size,
                task_type=exp.task_type,
            ).to(device)

            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            exp.n_parameters = n_params
            db.commit()

            criterion = _build_criterion(exp.task_type)
            optimizer = torch.optim.Adam(model.parameters(), lr=hp.get("learning_rate", 1e-3))

            monitor = GradientMonitor(model, log_dir=settings.gradient_storage_dir)

            def gradient_save_callback(epoch: int, path) -> None:
                stats = getattr(monitor, "last_stats", {})
                gl = GradientLog(
                    experiment_id=experiment_id,
                    epoch=epoch,
                    file_path=str(path),
                    grad_norm_mean=stats.get("grad_norm_mean"),
                    n_layers=stats.get("n_layers"),
                    n_steps=stats.get("n_steps"),
                )
                db.add(gl)
                db.commit()

            weight_tracker = None
            weight_save_cb = None
            weight_log_interval = max(1, int(hp.get("weight_log_interval", 1)))
            full_traj_tracker = None
            if hp.get("weight_track_enabled", True):
                weight_tracker = WeightTrackerr(
                    model=model,
                    log_dir=settings.gradient_storage_dir,
                    experiment_id=experiment_id,
                    subsample=weight_log_interval,
                )
                full_traj_tracker = weight_tracker

            need_logits = any(m in extra_metric_names for m in ("auc_macro", "cross_entropy", "perplexity"))
            task_type = exp.task_type

            compute_extra_fn = None
            if extra_metric_names:
                def compute_extra_fn(preds: np.ndarray, targets: np.ndarray, logits) -> dict[str, float]:
                    return compute_additional_metrics(
                        preds, targets, extra_metric_names, task_type, logits
                    )

            es_metric = exp.early_stopping_metric or "val_loss"
            es_mode = exp.early_stopping_mode or ES_MODE.get(es_metric, "min")

            engine = TBPTTEngine(
                model=model, optimizer=optimizer, criterion=criterion,
                k1=exp.k1, k2=exp.k2, device=device,
                max_grad_norm=hp.get("gradient_clip", 1.0),
                grad_clip_mode=hp.get("grad_clip_mode", "norm"),
                max_grad_value=hp.get("max_grad_value"),
                checkpoint_dir=f"{settings.checkpoint_dir}/{experiment_id}",
                early_stopping_metric=es_metric,
                early_stopping_mode=es_mode,
                compute_eval_extra=compute_extra_fn,
                collect_logits=need_logits,
                monitor=MonitorConfig(
                    gradient_monitor=monitor,
                    gradient_log_interval=int(hp.get("gradient_log_interval", 1)),
                    gradient_experiment_id=experiment_id,
                    gradient_save_callback=gradient_save_callback,
                    weight_tracker=weight_tracker,
                    weight_log_interval=weight_log_interval,
                    weight_save_callback=weight_save_cb,
                ),
            )

            batch_size = hp.get("batch_size", 64)
            train_loader = data_module.get_train_loader(batch_size, exp.seed)
            val_loader = data_module.get_val_loader(batch_size)

            training_start = time.perf_counter()

            def metric_callback(epoch: int, metrics: dict) -> None:
                payload = json.dumps({
                    "epoch": epoch,
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                })
                r.publish(channel, payload)

                em = EpochMetric(
                    experiment_id=experiment_id,
                    epoch=epoch,
                    train_loss=metrics.get("train_loss"),
                    val_loss=metrics.get("val_loss"),
                    epoch_time_s=metrics.get("epoch_time_s"),
                    gpu_memory_mb=metrics.get("gpu_memory_mb"),
                    grad_norm_mean=metrics.get("grad_norm_mean"),
                    grad_norm_max=metrics.get("grad_norm_max"),
                    learning_rate=metrics.get("learning_rate"),
                )
                db.add(em)

                for m_name in extra_metric_names:
                    val = metrics.get(m_name)
                    if val is not None:
                        db.add(AdditionalMetric(
                            experiment_id=experiment_id,
                            epoch=epoch,
                            metric_name=m_name,
                            metric_value=float(val),
                        ))

                db.commit()

            history = engine.train(
                train_loader=train_loader,
                val_loader=val_loader,
                max_epochs=hp.get("max_epochs", 100),
                early_stopping_patience=hp.get("early_stopping_patience", 10),
                callback=metric_callback,
            )

            total_time = time.perf_counter() - training_start
            best = _best_metric(history, es_metric, es_mode)

            if full_traj_tracker is not None and full_traj_tracker.n_steps > 0:
                saved_full = full_traj_tracker.flush()
                if hasattr(exp, "full_weight_path"):
                    exp.full_weight_path = str(saved_full)

            exp.best_metric = best
            exp.status = ExperimentStatus.COMPLETED
            exp.finished_at = datetime.now(timezone.utc)
            exp.total_training_time_s = total_time
            exp.convergence_epoch = _find_convergence_epoch(history, best, es_metric, es_mode)
            if history:
                last = history[-1]
                exp.final_train_loss = last.get("train_loss")
                exp.final_val_loss = last.get("val_loss")
            db.commit()

            r.publish(channel, json.dumps({"event": "completed", "best_metric": best}))
            return {"experiment_id": experiment_id, "best_metric": best}

        except torch.cuda.OutOfMemoryError as oom:
            torch.cuda.empty_cache()
            hp["batch_size"] = max(1, hp.get("batch_size", 64) // 2)
            exp.hyperparams = hp
            db.commit()
            raise self.retry(exc=oom, countdown=10)

        except Exception as e:
            exp.status = ExperimentStatus.FAILED
            exp.error_message = str(e)[:500]
            db.commit()
            r.publish(channel, json.dumps({"event": "failed", "error": str(e)}))
            raise


def _build_data_module(ds_record):
    if ds_record.type == "custom":
        config = UploadConfig(**ds_record.config_json)
        return GenericSequenceDataModule(ds_record.file_path, config, name=ds_record.name)

    factory = DATASET_REGISTRY.get(ds_record.name)
    if factory is None:
        known = list(DATASET_REGISTRY)
        raise ValueError(f"Nieznany dataset: {ds_record.name!r}. Znane: {known}")

    return factory()


def _build_criterion(task_type: str):
    mapping = {
        "classification": nn.CrossEntropyLoss(),
        "language_model": nn.CrossEntropyLoss(),
        "seq2seq": nn.CrossEntropyLoss(ignore_index=0),
        "regression": nn.MSELoss(),
        "forecasting": nn.MSELoss(),
    }
    return mapping.get(task_type, nn.CrossEntropyLoss())


def _best_metric(history: list[dict], metric: str, mode: str) -> float | None:
    values = [h.get(metric) for h in history if h.get(metric) is not None]
    if not values:
        return None

    return min(values) if mode == "min" else max(values)


def _find_convergence_epoch(
    history: list[dict], best_metric: float | None, metric: str, mode: str
) -> int | None:
    if best_metric is None or not history:
        return None

    for entry in history:
        val = entry.get(metric)
        if val is None:
            continue

        if mode == "min" and val <= best_metric * 1.05:
            return int(entry.get("epoch", 0))

        if mode == "max" and val >= best_metric * 0.95:
            return int(entry.get("epoch", 0))

    return None
