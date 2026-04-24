from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch.nn as nn


class GradientMonitor:
    def __init__(
        self,
        model: nn.Module,
        log_dir: str = "./gradients",
        max_file_size_gb: float = 1.0,
    ) -> None:
        self.model = model
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size_gb = max_file_size_gb

        self._hooks: list = []
        self._current_norms: dict[str, float] = {}
        self._buffer: dict[str, list[float]] = {}

    def attach(self) -> None:
        self.detach()
        for name, module in self.model.named_modules():
            handle = module.register_full_backward_hook(self._make_hook(name))
            self._hooks.append(handle)

    def _make_hook(self, layer_name: str):
        def hook(module, grad_input, grad_output):
            norms = [g.norm(2).item() for g in grad_output if g is not None]
            if norms:
                self._current_norms[layer_name] = float(np.mean(norms))
        return hook

    def step(self) -> None:
        for name, norm in self._current_norms.items():
            self._buffer.setdefault(name, []).append(norm)
        self._current_norms.clear()

    def save(self, epoch: int, experiment_id: Optional[str] = None) -> Path:
        self._warn_if_large(epoch)
        suffix = f"_{experiment_id}" if experiment_id else ""
        path = self.log_dir / f"gradients_epoch{epoch:04d}{suffix}.npz"
        np.savez_compressed(path, **{k: np.array(v) for k, v in self._buffer.items()})
        self._buffer.clear()
        return path

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _warn_if_large(self, epoch: int) -> None:
        n_layers = len(self._buffer)
        n_steps = max((len(v) for v in self._buffer.values()), default=0)
        size_gb = (n_layers * n_steps * 8) / 1024**3
        if size_gb > self.max_file_size_gb:
            warnings.warn(
                f"GradientMonitor: estimated size {size_gb:.2f} GB > "
                f"{self.max_file_size_gb} GB. Consider increasing "
                f"gradient_log_interval. Epoch: {epoch}",
                UserWarning,
                stacklevel=2,
            )

