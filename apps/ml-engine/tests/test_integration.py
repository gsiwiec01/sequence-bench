from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from ml_engine.datasets.adding_problem import AddingProblemDataModule
from ml_engine.datasets.copy_task import CopyTaskDataModule
from ml_engine.engine import MonitorConfig, TBPTTEngine
from ml_engine.gradient_monitor import GradientMonitor
from ml_engine.model_registry import create_model


def _make_engine(model, criterion, k1, k2, monitor=None, checkpoint_dir=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return TBPTTEngine(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        k1=k1,
        k2=k2,
        device=torch.device("cpu"),
        checkpoint_interval=100,
        checkpoint_dir=checkpoint_dir or "./checkpoints_test",
        monitor=MonitorConfig(gradient_monitor=monitor, gradient_log_interval=1) if monitor else None,
    )


def test_adding_problem_gru_trains():
    dm = AddingProblemDataModule(T=50, n_train=200, n_val=50, n_test=50)
    meta = dm.metadata
    model = create_model("gru", meta.input_size, hidden_size=32, output_size=meta.output_size, task_type="regression")

    engine = _make_engine(model, nn.MSELoss(), k1=10, k2=25)
    history = engine.train(
        dm.get_train_loader(batch_size=32, seed=0),
        dm.get_val_loader(batch_size=32),
        max_epochs=3,
        early_stopping_patience=100,
    )

    assert len(history) == 3
    for row in history:
        assert math.isfinite(row["train_loss"])
        assert math.isfinite(row["val_loss"])
        assert math.isfinite(row["epoch_time_s"])
        assert "val_accuracy" not in row, "regression must not report accuracy"


def test_copy_task_lstm_trains():
    dm = CopyTaskDataModule(T=20, n_classes=4, copy_len=5, n_train=200, n_val=50, n_test=50)
    meta = dm.metadata
    model = create_model("lstm", meta.input_size, hidden_size=32, output_size=meta.output_size, task_type="seq2seq")

    engine = _make_engine(model, nn.CrossEntropyLoss(), k1=5, k2=10)
    history = engine.train(
        dm.get_train_loader(batch_size=32, seed=0),
        dm.get_val_loader(batch_size=32),
        max_epochs=3,
        early_stopping_patience=100,
    )

    assert len(history) == 3
    for row in history:
        assert math.isfinite(row["train_loss"])
        assert math.isfinite(row["val_loss"])
        assert math.isfinite(row["epoch_time_s"])
        assert "val_accuracy" in row and 0.0 <= row["val_accuracy"] <= 1.0
        assert "train_accuracy" in row


@pytest.mark.filterwarnings("ignore:Full backward hook is firing:UserWarning")
def test_gradient_monitor_saves_data():
    dm = AddingProblemDataModule(T=50, n_train=100, n_val=50, n_test=50)
    meta = dm.metadata
    model = create_model("lstm", meta.input_size, hidden_size=32, output_size=meta.output_size, task_type="regression")

    with tempfile.TemporaryDirectory() as tmp:
        monitor = GradientMonitor(model, log_dir=tmp)
        engine = _make_engine(model, nn.MSELoss(), k1=10, k2=25, monitor=monitor, checkpoint_dir=tmp)
        engine.train(
            dm.get_train_loader(batch_size=32, seed=0),
            dm.get_val_loader(batch_size=32),
            max_epochs=2,
            early_stopping_patience=100,
        )

        saved = list(Path(tmp).glob("gradients_*.npz"))
        assert len(saved) > 0, "GradientMonitor saved no files"

        import numpy as np
        with np.load(saved[0]) as data:
            assert len(data.files) > 0, "gradient file is empty"
