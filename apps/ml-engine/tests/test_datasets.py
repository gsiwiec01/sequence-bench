from __future__ import annotations

import torch

from ml_engine.datasets.adding_problem import AddingProblemDataModule
from ml_engine.datasets.copy_task import CopyTaskDataModule


def test_copy_task_shapes() -> None:
    dm = CopyTaskDataModule(T=50, n_train=100, n_val=20, n_test=20)
    loader = dm.get_train_loader(batch_size=8, seed=0)
    x, y = next(iter(loader))
    assert x.ndim == 3
    assert y.ndim == 2
    assert x.size(1) == y.size(1), "x and y must share sequence length"


def test_copy_task_input_size() -> None:
    dm = CopyTaskDataModule(T=50, n_classes=8)
    loader = dm.get_train_loader(batch_size=4, seed=0)
    x, _ = next(iter(loader))
    assert x.size(2) == dm.n_classes + 2


def test_copy_task_marker_channel() -> None:
    dm = CopyTaskDataModule(T=20, copy_len=5, n_train=32, n_val=8, n_test=8)
    loader = dm.get_train_loader(batch_size=32, seed=0)
    x, _ = next(iter(loader))
    marker_channel = x[:, :, dm.n_classes]
    assert (marker_channel.sum(dim=1) == 1).all(), "Marker must fire exactly once"
    assert (marker_channel[:, dm.T + dm.copy_len] == 1).all(), "Marker at wrong position"


def test_adding_problem_shapes() -> None:
    dm = AddingProblemDataModule(T=100, n_train=100, n_val=20, n_test=20)
    loader = dm.get_train_loader(batch_size=8, seed=0)
    x, y = next(iter(loader))
    assert x.shape[-1] == 2, "Adding Problem: d=2"
    assert y.shape[-1] == 1, "Adding Problem: scalar output"


def test_adding_problem_mask_has_two_ones() -> None:
    dm = AddingProblemDataModule(T=50, n_train=200, n_val=20, n_test=20)
    loader = dm.get_train_loader(batch_size=200, seed=0)
    x, _ = next(iter(loader))
    mask = x[:, :, 1]
    assert (mask.sum(dim=1) == 2).all(), "Each sequence must have exactly 2 mask positions"


def test_adding_problem_baseline_mse() -> None:
    dm = AddingProblemDataModule(T=200, n_train=5000, n_val=500, n_test=500)
    loader = dm.get_test_loader(batch_size=500)
    _, y = next(iter(loader))
    pred = torch.full_like(y, y.mean().item())
    mse = ((pred - y) ** 2).mean().item()
    assert 0.1 < mse < 0.25, f"Baseline MSE out of range: {mse:.4f}"


def test_metadata_consistency() -> None:
    for dm in [CopyTaskDataModule(T=50), AddingProblemDataModule(T=100)]:
        meta = dm.metadata
        assert meta.T > 1
        assert meta.input_size >= 1
        assert meta.output_size >= 1
        assert meta.n_train > 0


def test_deterministic_generation() -> None:
    dm = CopyTaskDataModule(T=30, n_train=50, n_val=10, n_test=10)
    x1, y1 = next(iter(dm.get_train_loader(batch_size=50, seed=7)))
    x2, y2 = next(iter(dm.get_train_loader(batch_size=50, seed=7)))
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
