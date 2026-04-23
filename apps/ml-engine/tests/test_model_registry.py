from __future__ import annotations

import torch

from ml_engine.model_registry import SequenceModel


def test_lstm_gru_same_interface() -> None:
    for rnn_type in ["lstm", "gru"]:
        model = SequenceModel(rnn_type, input_size=10, hidden_size=32, output_size=5)
        device = torch.device("cpu")
        x = torch.randn(4, 20, 10)
        h = model.init_hidden(4, device)
        out, h_new = model(x, h)
        assert out.shape == (4, 5), f"{rnn_type}: output shape wrong"
        if rnn_type == "lstm":
            assert isinstance(h_new, tuple) and len(h_new) == 2


def test_forget_gate_bias_lstm() -> None:
    model = SequenceModel("lstm", input_size=10, hidden_size=32)
    for name, param in model.rnn.named_parameters():
        if "bias" in name:
            n = param.size(0) // 4
            bias_vals = param.data[n : 2 * n]
            assert torch.allclose(bias_vals, torch.ones_like(bias_vals)), (
                "Forget gate bias must be 1.0"
            )

            break


def test_count_parameters() -> None:
    model = SequenceModel("lstm", input_size=10, hidden_size=32)
    assert model.count_parameters() > 0
