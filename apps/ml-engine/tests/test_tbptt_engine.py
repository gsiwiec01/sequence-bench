from __future__ import annotations

from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml_engine.engine import TBPTTEngine
from ml_engine.model_registry import SequenceModel
from ml_engine.utils import set_seed


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
@pytest.mark.parametrize("T", [10, 50, 200])
def test_tbptt_1_T_equals_full_bptt(rnn_type: str, T: int) -> None:
    atol = 1e-5 if T <= 50 else 1e-4

    set_seed(42)
    model_tbptt = SequenceModel(
        rnn_type, input_size=10, hidden_size=32, output_size=5, task_type="seq2seq"
    )

    set_seed(42)
    model_bptt = SequenceModel(
        rnn_type, input_size=10, hidden_size=32, output_size=5, task_type="seq2seq"
    )

    x = torch.randn(2, T, 10)
    y = torch.randint(0, 5, (2, T))
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")

    engine = TBPTTEngine(
        model_tbptt,
        torch.optim.Adam(model_tbptt.parameters(), lr=1e-3),
        criterion,
        k1=1,
        k2=T,
        device=device,
    )
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    engine.train(loader, loader, max_epochs=1)

    opt = torch.optim.Adam(model_bptt.parameters(), lr=1e-3)
    h = model_bptt.init_hidden(2, device)
    out, _ = model_bptt(x, h)
    loss = criterion(out.reshape(-1, 5), y.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()

    for p1, p2 in zip(model_tbptt.parameters(), model_bptt.parameters(), strict=True):
        assert torch.allclose(p1, p2, atol=atol), (
            f"TBPTT(1,{T}) != BPTT for {rnn_type}, "
            f"max diff={(p1 - p2).abs().max():.2e}"
        )


def test_detach_prevents_gradient_flow() -> None:
    x = torch.randn(2, 5, requires_grad=True)
    h = x.detach()
    assert not h.requires_grad


def test_deterministic_runs() -> None:
    losses: list[list[float]] = []
    for _ in range(2):
        set_seed(0)
        model = SequenceModel("lstm", input_size=10, hidden_size=32, output_size=5)
        engine = TBPTTEngine(
            model,
            torch.optim.Adam(model.parameters()),
            torch.nn.CrossEntropyLoss(),
            k1=5,
            k2=5,
            device=torch.device("cpu"),
        )
        x = torch.randn(4, 20, 10)
        y = torch.randint(0, 5, (4,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        history = engine.train(loader, loader, max_epochs=2)
        losses.append([e["train_loss"] for e in history])
    assert losses[0] == losses[1]


def test_lstm_detach_returns_tuple() -> None:
    h = (torch.randn(1, 2, 8, requires_grad=True), torch.randn(1, 2, 8, requires_grad=True))
    detached = TBPTTEngine._detach_hidden(h)
    assert isinstance(detached, tuple) and len(detached) == 2
    assert not detached[0].requires_grad and not detached[1].requires_grad


def test_gradient_clipping_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    set_seed(0)
    model = SequenceModel("gru", input_size=4, hidden_size=8, output_size=3, task_type="seq2seq")
    engine = TBPTTEngine(
        model,
        torch.optim.SGD(model.parameters(), lr=1e-2),
        torch.nn.CrossEntropyLoss(),
        k1=4,
        k2=4,
        device=torch.device("cpu"),
        max_grad_norm=0.5,
    )
    x = torch.randn(2, 8, 4)
    y = torch.randint(0, 3, (2, 8))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    call_count = {"n": 0}
    real_clip = torch.nn.utils.clip_grad_norm_

    def spy_clip(*args: Any, **kwargs: Any) -> torch.Tensor:  # type: ignore[no-untyped-def]
        call_count["n"] += 1
        return real_clip(*args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy_clip)
    engine.train(loader, loader, max_epochs=1)

    assert call_count["n"] == 2


def test_early_stopping_triggers(monkeypatch: pytest.MonkeyPatch) -> None:
    set_seed(0)
    model = SequenceModel("gru", input_size=4, hidden_size=8, output_size=3)
    engine = TBPTTEngine(
        model,
        torch.optim.Adam(model.parameters(), lr=1e-3),
        torch.nn.CrossEntropyLoss(),
        k1=4,
        k2=4,
        device=torch.device("cpu"),
        early_stopping_metric="val_loss",
        early_stopping_mode="min",
    )
    x = torch.randn(4, 8, 4)
    y = torch.randint(0, 3, (4,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    monkeypatch.setattr(engine, "_eval_epoch", lambda _loader: {"val_loss": 1.0})

    history = engine.train(loader, loader, max_epochs=30, early_stopping_patience=2)

    assert len(history) == 3, f"Expected early stop at epoch 3, got {len(history)}"


def test_callback_invoked_each_epoch() -> None:
    set_seed(0)
    model = SequenceModel("gru", input_size=4, hidden_size=8, output_size=3)
    engine = TBPTTEngine(
        model,
        torch.optim.Adam(model.parameters(), lr=1e-3),
        torch.nn.CrossEntropyLoss(),
        k1=4,
        k2=4,
        device=torch.device("cpu"),
    )
    x = torch.randn(2, 8, 4)
    y = torch.randint(0, 3, (2,))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    received: list[tuple[int, dict[str, float]]] = []
    engine.train(
        loader, loader, max_epochs=3, early_stopping_patience=100,
        callback=lambda e, m: received.append((e, m)),
    )
    assert [e for e, _ in received] == [0, 1, 2]
    assert all("train_loss" in m and "val_loss" in m for _, m in received)


def test_invalid_k1_k2_rejected() -> None:
    model = SequenceModel("gru", input_size=4, hidden_size=8, output_size=3)
    with pytest.raises(ValueError):
        TBPTTEngine(
            model,
            torch.optim.Adam(model.parameters()),
            torch.nn.CrossEntropyLoss(),
            k1=5,
            k2=3,
            device=torch.device("cpu"),
        )

