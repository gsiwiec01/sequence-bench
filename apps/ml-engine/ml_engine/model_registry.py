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


class SequenceModel(BaseRNNModel):
    def __init__(
        self,
        rnn_type: str,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.2,
        output_size: int = 10,
        task_type: str = "classification",
        # task_type: "classification" (output[-1]) | "language_model" (all steps)
        #            "regression" (output[-1]) | "seq2seq" (all steps)
    ) -> None:
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.task_type = task_type

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.Linear(hidden_size, output_size)

        self._init_forget_gate_bias()

    def _init_forget_gate_bias(self) -> None:
        if self.rnn_type != "lstm":
            return
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                n = param.size(0) // 4
                param.data[n : 2 * n].fill_(1.0)

    def forward(self, x: torch.Tensor, h: object) -> tuple[torch.Tensor, object]:
        out, h_new = self.rnn(x, h)
        if self.task_type in ("classification", "regression"):
            logits = self.decoder(out[:, -1, :])
        else:
            logits = self.decoder(out)

        return logits, h_new

    def init_hidden(self, batch_size: int, device: torch.device) -> object:
        zeros = torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device
        )

        if self.rnn_type == "lstm":
            return zeros, zeros.clone()

        return zeros

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)