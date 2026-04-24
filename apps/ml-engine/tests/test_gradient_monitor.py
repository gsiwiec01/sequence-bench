from __future__ import annotations

import tempfile

import numpy as np
import pytest
import torch

from ml_engine.gradient_monitor import GradientMonitor
from ml_engine.model_registry import SequenceModel


def test_hook_captures_gradients() -> None:
    model = SequenceModel("lstm", 10, 16, output_size=5)
    with tempfile.TemporaryDirectory() as tmp:
        monitor = GradientMonitor(model, log_dir=tmp)
        monitor.attach()

        x = torch.randn(2, 5, 10)
        h = model.init_hidden(2, torch.device("cpu"))
        out, _ = model(x, h)
        out.mean().backward()
        monitor.step()

        assert len(monitor._buffer) > 0, "Buffer empty — hooks not firing"
        monitor.detach()


def test_save_creates_npz() -> None:
    model = SequenceModel("gru", 10, 16, output_size=5)
    with tempfile.TemporaryDirectory() as tmp:
        monitor = GradientMonitor(model, log_dir=tmp)
        monitor.attach()

        for _ in range(3):
            x = torch.randn(2, 5, 10)
            h = model.init_hidden(2, torch.device("cpu"))
            out, _ = model(x, h)
            out.mean().backward()
            monitor.step()

        path = monitor.save(epoch=1)
        assert path.exists()
        with np.load(path) as data:
            assert len(data.files) > 0
        monitor.detach()


def test_detach_removes_hooks() -> None:
    model = SequenceModel("lstm", 10, 16, output_size=5)
    with tempfile.TemporaryDirectory() as tmp:
        monitor = GradientMonitor(model, log_dir=tmp)
        monitor.attach()
        assert len(monitor._hooks) > 0
        monitor.detach()
        assert len(monitor._hooks) == 0


def test_save_clears_buffer() -> None:
    model = SequenceModel("gru", 10, 16, output_size=5)
    with tempfile.TemporaryDirectory() as tmp:
        monitor = GradientMonitor(model, log_dir=tmp)
        monitor.attach()

        x = torch.randn(2, 5, 10)
        h = model.init_hidden(2, torch.device("cpu"))
        model(x, h)[0].mean().backward()
        monitor.step()

        monitor.save(epoch=0)
        assert monitor._buffer == {}, "Buffer should be cleared after save"
        monitor.detach()


def test_experiment_id_suffix() -> None:
    model = SequenceModel("lstm", 10, 16, output_size=5)
    with tempfile.TemporaryDirectory() as tmp:
        monitor = GradientMonitor(model, log_dir=tmp)
        monitor.attach()

        x = torch.randn(2, 5, 10)
        h = model.init_hidden(2, torch.device("cpu"))
        model(x, h)[0].mean().backward()
        monitor.step()

        path = monitor.save(epoch=2, experiment_id="exp42")
        assert "exp42" in path.name
        monitor.detach()


def test_large_file_warning() -> None:
    model = SequenceModel("lstm", 10, 16, output_size=5)
    with tempfile.TemporaryDirectory() as tmp:
        monitor = GradientMonitor(model, log_dir=tmp, max_file_size_gb=0.0)
        monitor.attach()

        x = torch.randn(2, 5, 10)
        h = model.init_hidden(2, torch.device("cpu"))
        model(x, h)[0].mean().backward()
        monitor.step()

        with pytest.warns(UserWarning, match="GradientMonitor"):
            monitor.save(epoch=0)
        monitor.detach()


def test_double_attach_does_not_duplicate_hooks() -> None:
    model = SequenceModel("gru", 10, 16, output_size=5)
    with tempfile.TemporaryDirectory() as tmp:
        monitor = GradientMonitor(model, log_dir=tmp)
        monitor.attach()
        count_after_first = len(monitor._hooks)
        monitor.attach()
        assert len(monitor._hooks) == count_after_first
        monitor.detach()
