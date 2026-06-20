from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ml_engine.model_registry import create_model
from ml_engine.weight_tracker import WeightTrackerr
from worker.landscape import _eval_loss
from worker.pca_landscape import (
    compute_pca_landscape_arrays,
    get_flat_weights,
    grid_axes,
    pca_directions,
    project_trajectory,
    sweep_pca_grid,
)

DEVICE = torch.device("cpu")
INPUT, HIDDEN, OUTPUT, T, B = 4, 6, 3, 5, 8

def _build_fixture():
    torch.manual_seed(0)
    model = create_model("gru", input_size=INPUT, hidden_size=HIDDEN, output_size=OUTPUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    train_batches = [
        (torch.randn(B, T, INPUT), torch.randint(0, OUTPUT, (B,))) for _ in range(3)
    ]
    val_loader = [
        (torch.randn(B, T, INPUT), torch.randint(0, OUTPUT, (B,))) for _ in range(2)
    ]

    tracker = WeightTrackerr(model, ".", "test-run")
    opt = torch.optim.Adam(model.parameters(), lr=0.05)

    tracker.step()
    for _ in range(5):
        model.train()
        for xb, yb in train_batches:
            opt.zero_grad()
            h = model.init_hidden(xb.size(0), DEVICE)
            out, _ = model(xb, h)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
        tracker.step()

    model.eval()
    W = np.stack(tracker.rows).astype(np.float64)
    return model, val_loader, criterion, W, tracker.names, tracker.shapes


def test_pca_directions_orthonormal() -> None:
    _, _, _, W, _, _ = _build_fixture()
    d1, d2, w_end, explained = pca_directions(W)
    assert np.allclose(np.linalg.norm(d1), 1.0)
    assert np.allclose(np.linalg.norm(d2), 1.0)
    assert abs(float(d1 @ d2)) < 1e-10
    np.testing.assert_array_equal(w_end, W[-1])

    assert 0.0 < explained <= 1.0 + 1e-9


def test_explained_variance_reported() -> None:
    model, val_loader, criterion, W, names, shapes = _build_fixture()
    result = compute_pca_landscape_arrays(
        model, val_loader, criterion, DEVICE, W, names, shapes,
        resolution=9, margin=0.15, max_batches=None,
    )
    assert 0.0 < result["explained_variance"] <= 1.0 + 1e-9


def test_anchor_matches_recorded_final_loss() -> None:
    model, val_loader, criterion, W, names, shapes = _build_fixture()
    recorded_final = _eval_loss(model, val_loader, criterion, DEVICE, max_batches=None)

    result = compute_pca_landscape_arrays(
        model, val_loader, criterion, DEVICE, W, names, shapes,
        resolution=11, margin=0.15, max_batches=None,
    )
    assert abs(result["anchor_loss"] - recorded_final) < 1e-6


def test_trajectory_fully_inside_grid() -> None:
    model, val_loader, criterion, W, names, shapes = _build_fixture()
    result = compute_pca_landscape_arrays(
        model, val_loader, criterion, DEVICE, W, names, shapes,
        resolution=11, margin=0.15, max_batches=None,
    )
    a_axis, b_axis = result["a_axis"], result["b_axis"]
    a_t, b_t = result["a_traj"], result["b_traj"]
    assert a_t.min() >= a_axis.min() and a_t.max() <= a_axis.max()
    assert b_t.min() >= b_axis.min() and b_t.max() <= b_axis.max()


def test_grid_is_deterministic() -> None:
    model, val_loader, criterion, W, names, shapes = _build_fixture()
    d1, d2, w_end, _ = pca_directions(W)
    a_t, b_t = project_trajectory(W, w_end, d1, d2)
    a_axis, b_axis = grid_axes(a_t, b_t, resolution=9, margin=0.15)

    g1 = sweep_pca_grid(
        model, val_loader, criterion, DEVICE, w_end, d1, d2, names, shapes,
        a_axis, b_axis, max_batches=None,
    )
    g2 = sweep_pca_grid(
        model, val_loader, criterion, DEVICE, w_end, d1, d2, names, shapes,
        a_axis, b_axis, max_batches=None,
    )
    assert float(np.max(np.abs(g1 - g2))) == 0.0


def test_weights_restored_after_sweep() -> None:
    model, val_loader, criterion, W, names, shapes = _build_fixture()
    checksum_before = get_flat_weights(model, names)

    compute_pca_landscape_arrays(
        model, val_loader, criterion, DEVICE, W, names, shapes,
        resolution=9, margin=0.15, max_batches=None,
    )
    checksum_after = get_flat_weights(model, names)
    assert np.array_equal(checksum_before, checksum_after)


def test_grid_contains_origin_and_min_equals_anchor() -> None:
    model, val_loader, criterion, W, names, shapes = _build_fixture()
    result = compute_pca_landscape_arrays(
        model, val_loader, criterion, DEVICE, W, names, shapes,
        resolution=11, margin=0.15, max_batches=None,
    )
    a_axis, b_axis = result["a_axis"], result["b_axis"]
    assert float(np.min(np.abs(a_axis))) < 1e-12
    assert float(np.min(np.abs(b_axis))) < 1e-12
    assert float(result["loss_grid"].min()) <= result["anchor_loss"] + 1e-6


def test_too_short_trajectory_raises() -> None:
    W = np.random.RandomState(0).randn(2, 10)
    try:
        pca_directions(W)
        raised = False
    except ValueError:
        raised = True
    assert raised
