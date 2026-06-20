from __future__ import annotations

import json
import tempfile

import numpy as np
import torch

from ml_engine.model_registry import BaseRNNModel, create_model
from ml_engine.weight_tracker import WeightTrackerr


def _make_model(rnn_type: str = "gru") -> BaseRNNModel:
    return create_model(rnn_type, input_size=4, hidden_size=8, output_size=3)


def _total_params(model) -> int:
    return sum(p.numel() for _, p in model.named_parameters())


def test_names_and_shapes_match_named_parameters() -> None:
    model = _make_model()
    with tempfile.TemporaryDirectory() as tmp:
        tracker = WeightTrackerr(model, tmp, "exp")
        expected_names = [n for n, _ in model.named_parameters()]
        expected_shapes = [list(p.shape) for _, p in model.named_parameters()]
        assert tracker.names == expected_names
        assert tracker.shapes == expected_shapes


def test_step_accumulates_full_vectors() -> None:
    model = _make_model()
    with tempfile.TemporaryDirectory() as tmp:
        tracker = WeightTrackerr(model, tmp, "exp")
        assert tracker.n_steps == 0
        tracker.step()
        tracker.step()
        assert tracker.n_steps == 2
        P = _total_params(model)
        assert all(row.shape == (P,) for row in tracker.rows)


def test_step_records_current_weights() -> None:
    model = _make_model()
    with tempfile.TemporaryDirectory() as tmp:
        tracker = WeightTrackerr(model, tmp, "exp")
        tracker.step()
        with torch.no_grad():
            dict(model.named_parameters())[tracker.names[0]].data += 0.5
        tracker.step()
        # Pierwszy element wektora to pierwszy parametr; różnica == 0.5.
        assert tracker.rows[1][0] == np.float32(tracker.rows[0][0] + 0.5)


def test_save_npz_roundtrip_reconstructs_matrix() -> None:
    model = _make_model("lstm")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tracker = WeightTrackerr(model, tmp, "abc-123")
        tracker.step()
        tracker.step()
        tracker.step()
        tracker.save(epoch=2)
        path = tracker.flush()  # zapis jest asynchroniczny -flush gwarantuje plik

        assert path.exists()
        assert "abc_123" in path.name

        with np.load(str(path), allow_pickle=False) as data:
            W = data["W"]
            names = [str(n) for n in data["param_names"]]
            shapes = json.loads(str(data["param_shapes_json"]))
            epochs = data["epochs"]

        P = _total_params(model)
        assert W.shape == (3, P)
        assert names == tracker.names
        assert shapes == tracker.shapes
        assert list(epochs) == [0, 1, 2]
        # Ostatni wiersz == aktualne wagi modelu (spłaszczone w tej samej kolejności).
        flat_now = torch.cat(
            [dict(model.named_parameters())[n].detach().reshape(-1).float() for n in names]
        ).numpy()
        np.testing.assert_array_almost_equal(W[-1], flat_now)


def test_save_single_cumulative_file() -> None:
    model = _make_model()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tracker = WeightTrackerr(model, tmp, "exp")
        tracker.step()
        p1 = tracker.save(epoch=0)
        tracker.step()
        tracker.save(epoch=1)
        p2 = tracker.flush()  # zapis async -flush czeka i dopisuje komplet
        # Ta sama nazwa pliku, skumulowana zawartość.
        assert p1 == p2
        with np.load(str(p2), allow_pickle=False) as data:
            assert data["W"].shape[0] == 2


def test_subsample_records_every_n() -> None:
    model = _make_model()
    with tempfile.TemporaryDirectory() as tmp:
        tracker = WeightTrackerr(model, tmp, "exp", subsample=2)
        for _ in range(5):
            tracker.step()
        # Epoki 0, 2, 4 zarejestrowane; 1, 3 pominięte.
        assert tracker.n_steps == 3
        assert tracker.recorded_epochs == [0, 2, 4]


def test_flush_appends_final_weights() -> None:
    model = _make_model()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tracker = WeightTrackerr(model, tmp, "exp", subsample=10)
        for _ in range(3):  # tylko epoka 0 (0 % 10 == 0)
            tracker.step()
        assert tracker.n_steps == 1
        # Zmień wagi, by końcowy wektor różnił się od ostatniego zapisanego.
        with torch.no_grad():
            next(model.parameters()).data += 1.0
        path = tracker.flush()
        with np.load(str(path), allow_pickle=False) as data:
            W = data["W"]
        # Epoka 0 + doklejone faktyczne wagi końcowe (w_E) = 2 wiersze.
        assert W.shape[0] == 2


def test_empty_model_raises() -> None:
    import torch.nn as nn

    class _Empty(nn.Module):
        pass

    with tempfile.TemporaryDirectory() as tmp:
        try:
            WeightTrackerr(_Empty(), tmp, "e")
            raised = False
        except ValueError:
            raised = True
        assert raised
