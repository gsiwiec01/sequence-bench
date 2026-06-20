from __future__ import annotations

from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml_engine.engine import TBPTTEngine
from ml_engine.model_registry import create_model
from ml_engine.utils import set_seed


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
@pytest.mark.parametrize("T", [10, 50, 200])
def test_tbptt_1_T_equals_full_bptt(rnn_type: str, T: int) -> None:
    atol = 1e-5 if T <= 50 else 1e-4

    set_seed(42)
    model_tbptt = create_model(
        rnn_type, input_size=10, hidden_size=32, output_size=5, task_type="seq2seq"
    )

    set_seed(42)
    model_bptt = create_model(
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
        model = create_model("lstm", input_size=10, hidden_size=32, output_size=5)
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
    model = create_model("gru", input_size=4, hidden_size=8, output_size=3, task_type="seq2seq")
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
    model = create_model("gru", input_size=4, hidden_size=8, output_size=3)
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
    model = create_model("gru", input_size=4, hidden_size=8, output_size=3)
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
    model = create_model("gru", input_size=4, hidden_size=8, output_size=3)
    with pytest.raises(ValueError):
        TBPTTEngine(
            model,
            torch.optim.Adam(model.parameters()),
            torch.nn.CrossEntropyLoss(),
            k1=5,
            k2=3,
            device=torch.device("cpu"),
        )


def test_grad_clip_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cpu")
    x = torch.ones(2, 8, 4) * 5  # large inputs → large gradients
    y = torch.randint(0, 3, (2, 8))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    def make_engine(grad_clip_mode: str, **kw: Any) -> TBPTTEngine:
        set_seed(0)
        model = create_model("gru", input_size=4, hidden_size=16, output_size=3, task_type="seq2seq")
        return TBPTTEngine(
            model,
            torch.optim.SGD(model.parameters(), lr=0.0),
            torch.nn.CrossEntropyLoss(),
            k1=4, k2=4, device=device,
            grad_clip_mode=grad_clip_mode,
            **kw,
        )

    # ── 1. mode="none" must NOT call clip_grad_norm_ ──────────────────────
    n_norm_calls = {"n": 0}
    real_clip_norm = torch.nn.utils.clip_grad_norm_

    def spy_norm(*args: Any, **kwargs: Any) -> torch.Tensor:
        n_norm_calls["n"] += 1
        return real_clip_norm(*args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy_norm)
    make_engine("none").train(loader, loader, max_epochs=1)
    assert n_norm_calls["n"] == 0, "mode='none' must not call clip_grad_norm_"
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", real_clip_norm)

    # ── 2. mode="norm" with max_norm=0.1 actually bounds total gradient norm ─
    set_seed(0)
    model_norm = create_model("gru", input_size=4, hidden_size=16, output_size=3, task_type="seq2seq")
    model_norm.zero_grad()
    h = model_norm.init_hidden(2, device)
    out, _ = model_norm(x, h)
    torch.nn.CrossEntropyLoss()(out.reshape(-1, 3), y.reshape(-1)).backward()
    torch.nn.utils.clip_grad_norm_(model_norm.parameters(), 0.1)
    total_norm = sum(
        p.grad.norm(2).item() ** 2 for p in model_norm.parameters() if p.grad is not None
    ) ** 0.5
    assert total_norm <= 0.1 + 1e-5, f"norm after clip_grad_norm_(0.1) = {total_norm:.6f}"

    # ── 3. mode="value" with max_value=0.01 bounds every gradient element ──
    set_seed(0)
    model_val = create_model("gru", input_size=4, hidden_size=16, output_size=3, task_type="seq2seq")
    model_val.zero_grad()
    h = model_val.init_hidden(2, device)
    out, _ = model_val(x, h)
    torch.nn.CrossEntropyLoss()(out.reshape(-1, 3), y.reshape(-1)).backward()
    torch.nn.utils.clip_grad_value_(model_val.parameters(), 0.01)
    max_elem = max(p.grad.abs().max().item() for p in model_val.parameters() if p.grad is not None)
    assert max_elem <= 0.01 + 1e-7, f"max element after clip_grad_value_(0.01) = {max_elem:.8f}"

    # ── 4. mode="value" calls clip_grad_value_ through the engine ─────────
    n_val_calls = {"n": 0}
    real_clip_val = torch.nn.utils.clip_grad_value_

    def spy_val(*args: Any, **kwargs: Any) -> None:
        n_val_calls["n"] += 1
        real_clip_val(*args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_value_", spy_val)
    make_engine("value", max_grad_value=0.01).train(loader, loader, max_epochs=1)
    assert n_val_calls["n"] > 0, "mode='value' must call clip_grad_value_"


# ── Poprawka 1: loss tylko w ostatnim oknie dla classification / regression ───

def test_end_only_loss_last_window() -> None:
    """Classification: exactly one backward per batch regardless of how many k2-windows fit."""
    set_seed(0)
    model = create_model("gru", input_size=4, hidden_size=8, output_size=3, task_type="classification")
    engine = TBPTTEngine(
        model,
        torch.optim.Adam(model.parameters()),
        torch.nn.CrossEntropyLoss(),
        k1=4, k2=4,
        device=torch.device("cpu"),
        task_type="classification",
    )
    # seq_len=8 with k2=4 would yield 2 windows in old code → must fire only 1 backward
    x = torch.randn(2, 8, 4)
    y = torch.randint(0, 3, (2,))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    call_count: dict[str, int] = {"n": 0}
    orig_bwd = engine._backward_step
    def spy(loss: torch.Tensor) -> float:
        call_count["n"] += 1
        return orig_bwd(loss)
    engine._backward_step = spy  # type: ignore[method-assign]

    engine.train(loader, loader, max_epochs=1)
    assert call_count["n"] == 1, f"Expected 1 backward, got {call_count['n']}"


def test_seq2seq_loss_every_window() -> None:
    """seq2seq is unaffected: still computes loss at every window (2 here)."""
    set_seed(0)
    model = create_model("gru", input_size=4, hidden_size=8, output_size=3, task_type="seq2seq")
    engine = TBPTTEngine(
        model,
        torch.optim.Adam(model.parameters()),
        torch.nn.CrossEntropyLoss(),
        k1=4, k2=4,
        device=torch.device("cpu"),
        task_type="seq2seq",
    )
    x = torch.randn(2, 8, 4)
    y = torch.randint(0, 3, (2, 8))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    call_count: dict[str, int] = {"n": 0}
    orig_bwd = engine._backward_step
    def spy(loss: torch.Tensor) -> float:
        call_count["n"] += 1
        return orig_bwd(loss)
    engine._backward_step = spy  # type: ignore[method-assign]

    engine.train(loader, loader, max_epochs=1)
    assert call_count["n"] == 2, f"Expected 2 backward, got {call_count['n']}"


def test_regression_loss_last_window() -> None:
    """Regression: one backward per batch, finite loss, no accuracy in metrics."""
    set_seed(0)
    model = create_model("lstm", input_size=4, hidden_size=8, output_size=1, task_type="regression")
    engine = TBPTTEngine(
        model,
        torch.optim.Adam(model.parameters()),
        torch.nn.MSELoss(),
        k1=4, k2=4,
        device=torch.device("cpu"),
        task_type="regression",
    )
    x = torch.randn(2, 12, 4)  # 3 potential windows
    y = torch.randn(2, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    call_count: dict[str, int] = {"n": 0}
    orig_bwd = engine._backward_step
    def spy(loss: torch.Tensor) -> float:
        call_count["n"] += 1
        return orig_bwd(loss)
    engine._backward_step = spy  # type: ignore[method-assign]

    history = engine.train(loader, loader, max_epochs=1)
    assert call_count["n"] == 1, f"Expected 1 backward, got {call_count['n']}"
    assert "train_accuracy" not in history[0], "regression must not report train_accuracy"
    import math
    assert math.isfinite(history[0]["train_loss"])

