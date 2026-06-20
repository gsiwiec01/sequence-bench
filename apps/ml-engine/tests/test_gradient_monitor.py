from __future__ import annotations

import tempfile
import threading

import numpy as np
import pytest
import torch

from ml_engine.gradient_monitor import GradientMonitor
from ml_engine.model_registry import create_model


def test_hook_captures_gradients() -> None:
    model = create_model("lstm", 10, 16, output_size=5)
    with tempfile.TemporaryDirectory() as tmp:
        monitor = GradientMonitor(model, log_dir=tmp)
        monitor.attach()

        x = torch.randn(2, 5, 10, requires_grad=True)
        h = model.init_hidden(2, torch.device("cpu"))
        out, _ = model(x, h)
        out.mean().backward()
        monitor.step()

        assert len(monitor._buffer) > 0, "Buffer empty - hooks not firing"
        monitor.detach()


def test_save_creates_npz() -> None:
    model = create_model("gru", 10, 16, output_size=5)
    with tempfile.TemporaryDirectory() as tmp:
        monitor = GradientMonitor(model, log_dir=tmp)
        monitor.attach()

        for _ in range(3):
            x = torch.randn(2, 5, 10, requires_grad=True)
            h = model.init_hidden(2, torch.device("cpu"))
            out, _ = model(x, h)
            out.mean().backward()
            monitor.step()

        path = monitor.save(epoch=1)
        monitor._save_thread.join()  # wait for async write
        assert path.exists()
        with np.load(path) as data:
            assert len(data.files) > 0
        monitor.detach()


def test_detach_removes_hooks() -> None:
    model = create_model("lstm", 10, 16, output_size=5)
    with tempfile.TemporaryDirectory() as tmp:
        monitor = GradientMonitor(model, log_dir=tmp)
        monitor.attach()
        assert len(monitor._hooks) > 0
        monitor.detach()
        assert len(monitor._hooks) == 0


def test_save_clears_buffer() -> None:
    model = create_model("gru", 10, 16, output_size=5)
    with tempfile.TemporaryDirectory() as tmp:
        monitor = GradientMonitor(model, log_dir=tmp)
        monitor.attach()

        x = torch.randn(2, 5, 10, requires_grad=True)
        h = model.init_hidden(2, torch.device("cpu"))
        model(x, h)[0].mean().backward()
        monitor.step()

        monitor.save(epoch=0)
        assert monitor._buffer == {}, "Buffer should be cleared after save"
        monitor._save_thread.join()
        monitor.detach()


def test_experiment_id_suffix() -> None:
    model = create_model("lstm", 10, 16, output_size=5)
    with tempfile.TemporaryDirectory() as tmp:
        monitor = GradientMonitor(model, log_dir=tmp)
        monitor.attach()

        x = torch.randn(2, 5, 10, requires_grad=True)
        h = model.init_hidden(2, torch.device("cpu"))
        model(x, h)[0].mean().backward()
        monitor.step()

        path = monitor.save(epoch=2, experiment_id="exp42")
        assert "exp42" in path.name
        monitor._save_thread.join()
        monitor.detach()


def test_large_file_warning() -> None:
    model = create_model("lstm", 10, 16, output_size=5)
    with tempfile.TemporaryDirectory() as tmp:
        monitor = GradientMonitor(model, log_dir=tmp, max_file_size_gb=0.0)
        monitor.attach()

        x = torch.randn(2, 5, 10, requires_grad=True)
        h = model.init_hidden(2, torch.device("cpu"))
        model(x, h)[0].mean().backward()
        monitor.step()

        with pytest.warns(UserWarning, match="GradientMonitor"):
            monitor.save(epoch=0)
        monitor._save_thread.join()
        monitor.detach()


def test_double_attach_does_not_duplicate_hooks() -> None:
    model = create_model("gru", 10, 16, output_size=5)
    with tempfile.TemporaryDirectory() as tmp:
        monitor = GradientMonitor(model, log_dir=tmp)
        monitor.attach()
        count_after_first = len(monitor._hooks)
        monitor.attach()
        assert len(monitor._hooks) == count_after_first
        monitor.detach()


# ── Nowe testy dla optymalizacji ──────────────────────────────────────────────

def test_hook_stores_tensor_not_float() -> None:
    """_make_hook odkłada tensor na urządzeniu -nie wywołuje .item() per hook."""
    model = create_model("lstm", 10, 16, output_size=5)
    with tempfile.TemporaryDirectory() as tmp:
        monitor = GradientMonitor(model, log_dir=tmp)
        monitor.attach()

        x = torch.randn(2, 5, 10, requires_grad=True)
        h = model.init_hidden(2, torch.device("cpu"))
        model(x, h)[0].mean().backward()

        assert len(monitor._current_norm_tensors) > 0, "Hook nie zapisał żadnych tensorów"
        for name, val in monitor._current_norm_tensors.items():
            assert isinstance(val, torch.Tensor), (
                f"{name}: oczekiwano torch.Tensor, dostano {type(val).__name__}"
            )
            assert val.numel() == 1, f"{name}: tensor normy powinien być skalarem"

        monitor.detach()


def test_save_is_async() -> None:
    """save() uruchamia daemon thread i zwraca ścieżkę natychmiast."""
    model = create_model("gru", 10, 16, output_size=5)
    with tempfile.TemporaryDirectory() as tmp:
        monitor = GradientMonitor(model, log_dir=tmp)
        monitor.attach()

        x = torch.randn(2, 5, 10, requires_grad=True)
        h = model.init_hidden(2, torch.device("cpu"))
        model(x, h)[0].mean().backward()
        monitor.step()

        path = monitor.save(epoch=0)

        assert isinstance(monitor._save_thread, threading.Thread), \
            "save() powinno uruchomić threading.Thread"

        # Ścieżka znana natychmiast -plik powstanie za chwilę
        assert path.parent.exists()
        assert path.suffix == ".npz"

        monitor._save_thread.join()
        assert path.exists(), "Plik .npz powinien istnieć po zakończeniu wątku"

        monitor.detach()


def test_npz_correct_data_after_thread_join() -> None:
    """Po thread.join() plik .npz zawiera dokładnie tyle wpisów ile było kroków TBPTT."""
    model = create_model("gru", 10, 16, output_size=5)
    with tempfile.TemporaryDirectory() as tmp:
        monitor = GradientMonitor(model, log_dir=tmp)
        monitor.attach()

        n_steps = 5
        for _ in range(n_steps):
            x = torch.randn(2, 5, 10, requires_grad=True)
            h = model.init_hidden(2, torch.device("cpu"))
            model(x, h)[0].mean().backward()
            monitor.step()

        path = monitor.save(epoch=1, experiment_id="test-async")
        monitor._save_thread.join()

        assert path.exists()
        with np.load(path) as data:
            assert len(data.files) > 0, "Plik .npz nie zawiera żadnych tablic"
            for key in data.files:
                assert len(data[key]) == n_steps, (
                    f"Klucz {key!r}: oczekiwano {n_steps} wartości, "
                    f"dostano {len(data[key])}"
                )

        monitor.detach()


def test_buffer_cleared_before_thread_completes() -> None:
    """Bufor jest czyszczony synchronicznie przez save() -przed zakończeniem zapisu."""
    model = create_model("gru", 10, 16, output_size=5)
    with tempfile.TemporaryDirectory() as tmp:
        monitor = GradientMonitor(model, log_dir=tmp)
        monitor.attach()

        x = torch.randn(2, 5, 10, requires_grad=True)
        h = model.init_hidden(2, torch.device("cpu"))
        model(x, h)[0].mean().backward()
        monitor.step()

        assert len(monitor._buffer) > 0, "Przed save() bufor powinien mieć dane"

        monitor.save(epoch=0)

        # Sprawdzamy natychmiast -bez join() -bufor musi być już czysty
        assert monitor._buffer == {}, \
            "Bufor powinien być wyczyszczony przez save() przed zakończeniem wątku"

        monitor._save_thread.join()
        monitor.detach()

