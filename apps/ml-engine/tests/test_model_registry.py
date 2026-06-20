from __future__ import annotations

import torch

from ml_engine.model_registry import create_model


def test_lstm_gru_same_interface() -> None:
    for rnn_type in ["lstm", "gru"]:
        model = create_model(rnn_type, input_size=10, hidden_size=32, output_size=5)
        device = torch.device("cpu")
        x = torch.randn(4, 20, 10)
        h = model.init_hidden(4, device)
        out, h_new = model(x, h)
        assert out.shape == (4, 5), f"{rnn_type}: output shape wrong"
        if rnn_type == "lstm":
            assert isinstance(h_new, tuple) and len(h_new) == 2


def test_forget_gate_bias_lstm() -> None:
    model = create_model("lstm", input_size=10, hidden_size=32)
    for name, param in model.rnn.named_parameters():
        if "bias" in name:
            n = param.size(0) // 4
            bias_vals = param.data[n : 2 * n]
            assert torch.allclose(bias_vals, torch.ones_like(bias_vals)), (
                "Forget gate bias must be 1.0"
            )

            break


def test_count_parameters() -> None:
    model = create_model("lstm", input_size=10, hidden_size=32)
    assert model.count_parameters() > 0


def test_vanilla_rnn_same_interface() -> None:
    for nonlinearity in ["tanh", "relu"]:
        model = create_model(
            "rnn", input_size=10, hidden_size=32, output_size=5, nonlinearity=nonlinearity
        )
        device = torch.device("cpu")
        x = torch.randn(4, 20, 10)
        h = model.init_hidden(4, device)

        assert isinstance(h, torch.Tensor), "RNN hidden must be a plain tensor, not a tuple"
        assert h.shape == (1, 4, 32), f"init_hidden shape wrong: {h.shape}"

        out, h_new = model(x, h)

        assert out.shape == (4, 5), f"rnn/{nonlinearity}: output shape wrong: {out.shape}"
        assert isinstance(h_new, torch.Tensor), "RNN h_new must be a plain tensor"
        assert h_new.shape == (1, 4, 32), f"rnn/{nonlinearity}: h_new shape wrong: {h_new.shape}"
