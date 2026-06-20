from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ml_engine.pca import grid_axes, pca_directions, project_trajectory
from worker.landscape import _eval_loss

__all__ = [
    "pca_directions",
    "project_trajectory",
    "grid_axes",
    "set_flat_weights",
    "get_flat_weights",
    "sweep_pca_grid",
    "compute_pca_landscape_arrays",
]

def _split_sizes(shapes: list[list[int]]) -> list[int]:
    return [int(np.prod(s)) if s else 1 for s in shapes]

def set_flat_weights(
    model: Any, vec: np.ndarray, names: list[str], shapes: list[list[int]]
) -> None:
    sizes = _split_sizes(shapes)
    params = dict(model.named_parameters())
    off = 0
    with torch.no_grad():
        for name, shape, size in zip(names, shapes, sizes):
            chunk = vec[off : off + size]
            off += size
            p = params[name]
            t = torch.as_tensor(chunk, dtype=p.dtype, device=p.device)
            p.data.copy_(t.reshape(tuple(shape) if shape else ()))

    if off != len(vec):
        raise ValueError(f"Niespójność długości wektora wag: użyto {off}, długość {len(vec)}.")


def get_flat_weights(model: Any, names: list[str]) -> np.ndarray:
    params = dict(model.named_parameters())
    with torch.no_grad():
        vec = torch.cat([params[n].detach().reshape(-1).float().cpu() for n in names])

    return vec.numpy().astype(np.float64)

def sweep_pca_grid(
    model: Any,
    val_loader: Any,
    criterion: Any,
    device: Any,
    w_end: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    names: list[str],
    shapes: list[list[int]],
    a_axis: np.ndarray,
    b_axis: np.ndarray,
    max_batches: int | None = 16,
) -> np.ndarray:
    loss_grid = np.zeros((len(b_axis), len(a_axis)), dtype=np.float32)
    for bi, b in enumerate(b_axis):
        for ai, a in enumerate(a_axis):
            vec = w_end + a * d1 + b * d2
            set_flat_weights(model, vec, names, shapes)
            loss_grid[bi, ai] = _eval_loss(model, val_loader, criterion, device, max_batches)

    set_flat_weights(model, w_end, names, shapes)
    return loss_grid


def compute_pca_landscape_arrays(
    model: Any,
    val_loader: Any,
    criterion: Any,
    device: Any,
    W: np.ndarray,
    names: list[str],
    shapes: list[list[int]],
    resolution: int = 25,
    margin: float = 0.15,
    max_batches: int | None = 16,
) -> dict[str, Any]:
    d1, d2, w_end, explained = pca_directions(W)
    a_t, b_t = project_trajectory(W, w_end, d1, d2)
    a_axis, b_axis = grid_axes(a_t, b_t, resolution, margin)

    set_flat_weights(model, w_end, names, shapes)
    anchor_loss = float(_eval_loss(model, val_loader, criterion, device, max_batches))

    loss_grid = sweep_pca_grid(
        model, val_loader, criterion, device,
        w_end, d1, d2, names, shapes, a_axis, b_axis, max_batches,
    )

    return {
        "a_axis": a_axis,
        "b_axis": b_axis,
        "loss_grid": loss_grid,
        "a_traj": a_t,
        "b_traj": b_t,
        "anchor_loss": anchor_loss,
        "explained_variance": explained,
        "d1": d1,
        "d2": d2,
        "w_end": w_end,
    }
