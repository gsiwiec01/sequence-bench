from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from ml_engine.datasets.adding_problem import AddingProblemDataModule
from ml_engine.datasets.copy_task import CopyTaskDataModule
from ml_engine.datasets.generic import GenericSequenceDataModule
from ml_engine.datasets.upload import UploadConfig, load_and_validate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_npz(tmp_dir: str, N: int = 100, T: int = 50, d: int = 3, with_y: bool = True) -> str:
    path = Path(tmp_dir) / "data.npz"
    X = np.random.randn(N, T, d).astype(np.float32)
    if with_y:
        y = np.random.randint(0, 3, N).astype(np.float32)
        np.savez(path, X=X, y=y)
    else:
        np.savez(path, X=X)
    return str(path)


def _default_config() -> UploadConfig:
    return UploadConfig(task_type="classification", n_classes=3,
                        normalize="zscore", nan_strategy="error")


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


def test_copy_task_input_is_per_sample_one_hot() -> None:
    dm = CopyTaskDataModule(T=20, n_classes=8, copy_len=5, n_train=64, n_val=8, n_test=8)
    loader = dm.get_train_loader(batch_size=64, seed=0)
    x, y = next(iter(loader))
    token_region = x[:, : dm.copy_len, : dm.n_classes]
    assert (token_region.sum(dim=-1) == 1).all(), "Each token position must be one-hot per sample"
    encoded_tokens = token_region.argmax(dim=-1)
    expected_tokens = y[:, dm.T - dm.copy_len :] - 1
    assert torch.equal(encoded_tokens, expected_tokens), "Encoded tokens must match target tokens"


def test_copy_task_marker_channel() -> None:
    dm = CopyTaskDataModule(T=20, copy_len=5, n_train=32, n_val=8, n_test=8)
    loader = dm.get_train_loader(batch_size=32, seed=0)
    x, _ = next(iter(loader))
    marker_channel = x[:, :, dm.n_classes]
    assert (marker_channel.sum(dim=1) == 1).all(), "Marker must fire exactly once"
    assert (marker_channel[:, dm.T - dm.copy_len - 1] == 1).all(), "Marker at wrong position"


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
        assert meta.sequence_length > 1
        assert meta.input_size >= 1
        assert meta.output_size >= 1
        assert meta.n_train > 0
        assert meta.task_type


def test_deterministic_generation() -> None:
    dm = CopyTaskDataModule(T=30, n_train=50, n_val=10, n_test=10)
    x1, y1 = next(iter(dm.get_train_loader(batch_size=50, seed=7)))
    x2, y2 = next(iter(dm.get_train_loader(batch_size=50, seed=7)))
    assert torch.equal(x1, x2) and torch.equal(y1, y2)


# ---------------------------------------------------------------------------
# upload.py -load_and_validate
# ---------------------------------------------------------------------------

def test_valid_npz_loads() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_npz(tmp)
        result = load_and_validate(path, _default_config())
        assert "X_train" in result
        assert result["X_train"].shape[2] == 3


def test_rejects_inf() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        X = np.random.randn(50, 20, 2).astype(np.float32)
        X[0, 0, 0] = np.inf
        path = Path(tmp) / "bad.npz"
        np.savez(path, X=X)
        with pytest.raises(ValueError, match="Inf"):
            load_and_validate(str(path), _default_config())


def test_split_before_sliding_window() -> None:
    """Split nie dopuszcza data leakage -sliding window aplikowany po podziale."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_npz(tmp, N=100, T=50)
        config = UploadConfig(task_type="classification", n_classes=3,
                              normalize="none", nan_strategy="error",
                              sliding_window=10, train_split=0.7, val_split=0.15, test_split=0.15)
        result = load_and_validate(path, config)
        n_total = result["metadata"]["n_train"] + result["metadata"]["n_val"] + result["metadata"]["n_test"]
        assert n_total > 0


def test_normalize_uses_train_stats() -> None:
    """Val/test muszą być znormalizowane statystykami z train."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_npz(tmp, N=200, T=30, d=1)
        result = load_and_validate(path, _default_config())
        mu = result["X_train"].mean()
        assert abs(mu) < 0.5, f"Train mean after z-score: {mu}"


def test_npy_2d_promoted_to_3d() -> None:
    """Plik .npy z kształtem (N, T) musi zostać rozszerzony do (N, T, 1)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.npy"
        X = np.random.randn(50, 30).astype(np.float32)
        np.save(path, X)
        config = UploadConfig(task_type="regression", n_classes=None,
                              normalize="none", nan_strategy="error")
        result = load_and_validate(str(path), config)
        assert result["X_train"].shape[2] == 1


def test_rejects_unsupported_extension() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.h5"
        path.write_bytes(b"dummy")
        with pytest.raises(ValueError, match="Nieobsługiwany format"):
            load_and_validate(str(path), _default_config())


def test_nan_strategy_drop() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        X = np.random.randn(50, 20, 2).astype(np.float32)
        X[0, 5, 0] = np.nan
        path = Path(tmp) / "nan.npz"
        np.savez(path, X=X)
        config = UploadConfig(task_type="regression", n_classes=None,
                              normalize="none", nan_strategy="drop")
        result = load_and_validate(str(path), config)
        total = result["metadata"]["n_train"] + result["metadata"]["n_val"] + result["metadata"]["n_test"]
        assert total < 50, "Próbka z NaN powinna zostać usunięta"


def test_nan_strategy_error_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        X = np.random.randn(50, 20, 2).astype(np.float32)
        X[3, 3, 1] = np.nan
        path = Path(tmp) / "nan.npz"
        np.savez(path, X=X)
        with pytest.raises(ValueError, match="NaN"):
            load_and_validate(str(path), _default_config())


# ---------------------------------------------------------------------------
# generic.py -GenericSequenceDataModule
# ---------------------------------------------------------------------------

def test_generic_datamodule_loaders() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_npz(tmp, N=120, T=40, d=5)
        dm = GenericSequenceDataModule(path, _default_config(), name="test_ds")
        x, y = next(iter(dm.get_train_loader(batch_size=16)))
        assert x.ndim == 3
        assert x.shape[2] == 5


def test_generic_datamodule_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_npz(tmp, N=120, T=40, d=5)
        dm = GenericSequenceDataModule(path, _default_config(), name="my_ds")
        meta = dm.metadata
        assert meta.name == "my_ds"
        assert meta.input_size == 5
        assert meta.output_size == 3
        assert meta.n_classes == 3
        assert meta.task_type == "classification"
        assert meta.n_train > 0 and meta.n_val > 0 and meta.n_test > 0


def test_generic_datamodule_without_labels() -> None:
    """Plik bez y -loadery zwracają tylko X."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_npz(tmp, N=80, T=20, d=2, with_y=False)
        config = UploadConfig(task_type="regression", n_classes=None,
                              normalize="zscore", nan_strategy="error")
        dm = GenericSequenceDataModule(path, config)
        batch = next(iter(dm.get_val_loader(batch_size=8)))
        assert len(batch) == 1, "Bez y loader zwraca tylko X"
