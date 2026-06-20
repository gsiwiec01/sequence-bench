from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import redis
import torch
from celery import Task

from worker.celery_app import celery_app
from worker.training import get_sync_session, _build_data_module, _build_criterion
from worker.pca_landscape import compute_pca_landscape_arrays, set_flat_weights
from api.config import settings
from api.models.metrics import LossLandscape
from api.models.experiment import Experiment
from api.models.dataset import Dataset
from ml_engine.model_registry import create_model
from ml_engine.utils import get_device
from ml_engine.weight_tracker import load_trajectory

GRID_SIZE = 25
def _eval_loss(model, val_loader, criterion, device, max_batches: int | None = 16) -> float:
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            if max_batches is not None and n_batches >= max_batches:
                break

            batch_x = batch_x.to(device)
            if isinstance(batch_y, torch.Tensor):
                batch_y = batch_y.to(device)

            h = model.init_hidden(batch_x.size(0), device)
            output, _ = model(batch_x, h)
            task_type = getattr(model, "task_type", "classification")
            if task_type in ("seq2seq", "language_model") and output.dim() == 3:
                output = output.reshape(-1, output.size(-1))
                if isinstance(batch_y, torch.Tensor) and batch_y.dim() == 2:
                    batch_y = batch_y.reshape(-1)

            loss = criterion(output, batch_y)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def _full_traj_path(exp) -> str | None:
    path = getattr(exp, "full_weight_path", None)
    if path:
        return path

    safe_id = exp.id.replace("-", "_")
    candidate = Path(settings.gradient_storage_dir) / f"weights_full_{safe_id}.npz"

    return str(candidate) if candidate.exists() else None

def _surface_channel(job_id: str) -> str:
    return f"surface:{job_id}:status"

@celery_app.task(bind=True, max_retries=0, name="tasks.compute_loss_landscape")
def compute_loss_landscape(self: Task, landscape_id: str) -> dict:
    r = redis.from_url(settings.redis_url)
    channel = _surface_channel(landscape_id)

    def publish(payload: dict) -> None:
        try:
            r.publish(channel, json.dumps(payload))
        except Exception:
            pass

    with get_sync_session() as db:
        landscape = db.get(LossLandscape, landscape_id)
        if not landscape:
            return {"error": "Landscape not found"}

        landscape.status = "running"
        db.commit()
        publish({"status": "running"})

        def fail(message: str) -> dict:
            landscape.status = "failed"
            landscape.error_message = message
            db.commit()
            publish({"event": "failed", "error": message})
            return {"error": message}

        try:
            exp = db.get(Experiment, landscape.experiment_id)
            ds = db.get(Dataset, exp.dataset_id)
            hp = exp.hyperparams
            params = landscape.params or {}
            resolution = int(params.get("resolution", GRID_SIZE))
            margin = float(params.get("margin", 0.15))

            full_path = _full_traj_path(exp)
            if not full_path or not Path(full_path).exists():
                return fail( "Brak pełnej trajektorii wag dla tego przebiegu")

            W, names, shapes, epochs = load_trajectory(full_path)
            if W.shape[0] < 3:
                return fail(
                    f"Pełna trajektoria ma tylko {W.shape[0]} epok; PCA-2D wymaga >= 3."
                )

            device = get_device()
            model = create_model(
                rnn_type=exp.architecture,
                input_size=ds.input_size,
                hidden_size=hp.get("hidden_size", 256),
                num_layers=hp.get("num_layers", 1),
                dropout=hp.get("dropout", 0.2),
                output_size=ds.output_size,
                task_type=exp.task_type,
            ).to(device)
            model.eval()
            set_flat_weights(model, W[-1], names, shapes)

            data_module = _build_data_module(ds)
            val_loader = data_module.get_val_loader(hp.get("batch_size", 64))
            criterion = _build_criterion(exp.task_type).to(device)

            result = compute_pca_landscape_arrays(
                model, val_loader, criterion, device, W, names, shapes,
                resolution=resolution, margin=margin, max_batches=16,
            )

            a_axis = result["a_axis"]
            b_axis = result["b_axis"]
            safe_id = landscape_id.replace("-", "_")
            out_path = Path(settings.gradient_storage_dir) / f"landscape_{safe_id}.npz"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                str(out_path),
                x_values=a_axis,
                y_values=b_axis,
                loss_grid=result["loss_grid"],
                a_traj=result["a_traj"],
                b_traj=result["b_traj"],
                explained_variance=np.array(result["explained_variance"]),
                anchor_loss=np.array(result["anchor_loss"]),
                epochs=np.asarray(epochs, dtype=np.int64),
            )

            landscape.file_path = str(out_path)
            landscape.x_range = [float(a_axis[0]), float(a_axis[-1])]
            landscape.y_range = [float(b_axis[0]), float(b_axis[-1])]
            landscape.explained_variance = float(result["explained_variance"])
            landscape.anchor_loss = float(result["anchor_loss"])
            landscape.status = "completed"
            db.commit()

            publish({
                "event": "completed",
                "explained_variance": float(result["explained_variance"]),
                "anchor_loss": float(result["anchor_loss"]),
            })
            return {"landscape_id": landscape_id, "status": "completed"}

        except Exception as exc:
            fail(str(exc)[:500])
            raise
        finally:
            try:
                r.close()
            except Exception:
                pass
