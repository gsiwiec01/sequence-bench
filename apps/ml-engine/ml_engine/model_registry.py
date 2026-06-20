from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

class BaseRNNModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, h: object) -> tuple[torch.Tensor, object]:
        ...

    @abstractmethod
    def init_hidden(self, batch_size: int, device: torch.device) -> object:
        ...

    @abstractmethod
    def count_parameters(self) -> int:
        ...


class _SequenceModelBase(BaseRNNModel):
    task_type: str
    hidden_size: int
    num_layers: int
    rnn: nn.Module
    decoder: nn.Linear

    def forward(self, x: torch.Tensor, h: object) -> tuple[torch.Tensor, object]:
        out, h_new = self.rnn(x, h)
        if self.task_type in ("classification", "regression"):
            logits = self.decoder(out[:, -1, :])
        else:
            logits = self.decoder(out)
        return logits, h_new

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTMModel(_SequenceModelBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.2,
        output_size: int = 10,
        task_type: str = "classification",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.task_type = task_type
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.Linear(hidden_size, output_size)
        self._init_forget_gate_bias()

    def _init_forget_gate_bias(self) -> None:
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                n = param.size(0) // 4
                param.data[n : 2 * n].fill_(1.0)

    def init_hidden(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return zeros, zeros.clone()


class GRUModel(_SequenceModelBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.2,
        output_size: int = 10,
        task_type: str = "classification",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.task_type = task_type
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.Linear(hidden_size, output_size)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


class VanillaRNNModel(_SequenceModelBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.2,
        output_size: int = 10,
        task_type: str = "classification",
        nonlinearity: str = "tanh",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.task_type = task_type
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            nonlinearity=nonlinearity,
        )
        self.decoder = nn.Linear(hidden_size, output_size)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


def create_model(rnn_type: str, input_size: int, hidden_size: int = 256, **kwargs) -> BaseRNNModel:
    rnn_type_lower = rnn_type.lower()
    if rnn_type_lower == "lstm":
        kwargs.pop("nonlinearity", None)
        return LSTMModel(input_size=input_size, hidden_size=hidden_size, **kwargs)
    elif rnn_type_lower == "gru":
        kwargs.pop("nonlinearity", None)
        return GRUModel(input_size=input_size, hidden_size=hidden_size, **kwargs)
    elif rnn_type_lower == "rnn":
        return VanillaRNNModel(input_size=input_size, hidden_size=hidden_size, **kwargs)
    else:
        raise ValueError(f"Unsupported rnn_type: {rnn_type!r}. Choose 'lstm', 'gru', or 'rnn'.")
