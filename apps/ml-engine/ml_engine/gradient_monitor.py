from __future__ import annotations

import copy
import json
import threading
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
        lstm_gate_decompose: bool = False,
    ) -> None:
        self.model = model
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size_gb = max_file_size_gb
        self.lstm_gate_decompose = lstm_gate_decompose

        self._hooks: list = []
        self._current_norm_tensors: dict = {}  # tensors on device -.item() deferred to step()
        self._buffer: dict[str, list[float]] = {}
        self._lstm_hidden: dict[str, int] = {}
        self._save_thread: threading.Thread | None = None

    def attach(self) -> None:
        self.detach()
        self._lstm_hidden.clear()

        if self.lstm_gate_decompose:
            for mod_name, module in self.model.named_modules():
                if isinstance(module, nn.LSTM):
                    prefix = f"{mod_name}." if mod_name else ""
                    for layer_idx in range(module.num_layers):
                        for wname in ("weight_ih", "weight_hh"):
                            full = f"{prefix}{wname}_l{layer_idx}"
                            self._lstm_hidden[full] = module.hidden_size

        for param_name, param in self.model.named_parameters():
            if param.requires_grad:
                handle = param.register_hook(self._make_hook(param_name))
                self._hooks.append(handle)

    def _make_hook(self, param_name: str):
        def hook(grad):
            if grad is None:
                return
            hidden_size = self._lstm_hidden.get(param_name)
            if hidden_size and self.lstm_gate_decompose and grad.dim() >= 1 and grad.shape[0] == 4 * hidden_size:
                for gi, gate in enumerate(("i", "f", "g", "o")):
                    self._current_norm_tensors[f"{param_name}[{gate}]"] = (
                        grad[gi * hidden_size : (gi + 1) * hidden_size].detach().norm(2)
                    )
            else:
                self._current_norm_tensors[param_name] = grad.detach().norm(2)

        return hook

    def step(self) -> None:
        for name, tensor in self._current_norm_tensors.items():
            self._buffer.setdefault(name, []).append(tensor.item())
        self._current_norm_tensors.clear()

    def stats(self) -> dict[str, float | int | None]:
        if not self._buffer:
            return {"grad_norm_mean": None, "n_layers": 0, "n_steps": 0}
        all_vals = [v for vals in self._buffer.values() for v in vals]
        return {
            "grad_norm_mean": float(np.mean(all_vals)) if all_vals else None,
            "n_layers": len(self._buffer),
            "n_steps": max(len(v) for v in self._buffer.values()),
        }

    def save(self, epoch: int, experiment_id: Optional[str] = None) -> Path:
        self.last_stats = self.stats()
        self._warn_if_large(epoch)

        buffer_copy = copy.deepcopy(self._buffer)
        path = self._build_path(epoch, experiment_id)

        thread = threading.Thread(target=self._write_npz, args=(buffer_copy, path), daemon=True)
        thread.start()
        self._save_thread = thread

        self._buffer.clear()
        return path

    def _build_path(self, epoch: int, experiment_id: Optional[str] = None) -> Path:
        suffix = f"_{experiment_id}" if experiment_id else ""
        return self.log_dir / f"gradients_epoch{epoch:04d}{suffix}.npz"

    def _write_npz(self, buffer: dict, path: Path) -> None:
        encoded: dict[str, str] = {}
        arrays: dict[str, np.ndarray] = {}

        for original, values in buffer.items():
            safe = _encode_key(original)
            encoded[safe] = original
            arrays[safe] = np.array(values)

        np.savez_compressed(path, **arrays)
        path.with_suffix(".json").write_text(json.dumps(encoded, ensure_ascii=False))

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _warn_if_large(self, epoch: int) -> None:
        n = len(self._buffer)
        s = max((len(v) for v in self._buffer.values()), default=0)
        size_gb = (n * s * 8) / 1024**3
        if size_gb > self.max_file_size_gb:
            warnings.warn(
                f"GradientMonitor: estimated size {size_gb:.2f} GB > "
                f"{self.max_file_size_gb} GB at epoch {epoch}",
                UserWarning,
                stacklevel=2,
            )

def _encode_key(name: str) -> str:
    return name.replace(".", "__").replace("[", "_GATE_").replace("]", "")

def load_gradient_file(npz_path: str) -> dict[str, list[float]]:
    path = Path(npz_path)
    manifest_path = path.with_suffix(".json")

    key_map: dict[str, str] = {}
    if manifest_path.exists():
        key_map = json.loads(manifest_path.read_text())

    data = np.load(str(path), allow_pickle=False)
    return {
        key_map.get(safe, safe): data[safe].tolist()
        for safe in data.files
    }
