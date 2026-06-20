from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

class WeightTrackerr:
    def __init__(
        self,
        model: nn.Module,
        log_dir: str,
        experiment_id: str,
        subsample: int = 1,
    ) -> None:
        self.model = model
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id
        self.subsample = max(1, int(subsample))

        params = list(model.named_parameters())
        if not params:
            raise ValueError("Model nie ma żadnych parametrów do śledzenia.")

        self.names: list[str] = [name for name, _ in params]
        self.shapes: list[list[int]] = [list(p.shape) for _, p in params]
        self.rows: list[np.ndarray] = []
        self.recorded_epochs: list[int] = []
        self._epoch_count = 0

        safe_id = experiment_id.replace("-", "_")
        self._path = self.log_dir / f"weights_full_{safe_id}.npz"
        self._save_thread: threading.Thread | None = None

    def _current_vector(self) -> np.ndarray:
        with torch.no_grad():
            vec = torch.cat(
                [p.detach().reshape(-1).float().cpu() for _, p in self.model.named_parameters()]
            )
        return vec.numpy().astype(np.float32)

    def step(self) -> None:
        epoch = self._epoch_count
        self._epoch_count += 1
        if epoch % self.subsample != 0:
            return

        self.rows.append(self._current_vector())
        self.recorded_epochs.append(epoch)

    def _write(self, rows: list[np.ndarray], epochs: list[int], path: Path) -> None:
        if rows:
            W = np.stack(rows).astype(np.float32)
        else:
            W = np.zeros((0, len(self.names)), dtype=np.float32)

        np.savez_compressed(
            path,
            W=W,
            epochs=np.asarray(epochs, dtype=np.int64),
            param_names=np.array(self.names),
            param_shapes_json=np.array(json.dumps(self.shapes)),
        )

    def save(self) -> Path:
        if self._save_thread is not None and self._save_thread.is_alive():
            return self._path

        rows_snapshot = list(self.rows)
        epochs_snapshot = list(self.recorded_epochs)
        thread = threading.Thread(
            target=self._write, args=(rows_snapshot, epochs_snapshot, self._path), daemon=True
        )
        thread.start()
        self._save_thread = thread

        return self._path

    def flush(self) -> Path:
        if self._save_thread is not None and self._save_thread.is_alive():
            self._save_thread.join()

        final = self._current_vector()
        if not self.rows or not np.array_equal(self.rows[-1], final):
            self.rows.append(final)

            self.recorded_epochs.append(max(self._epoch_count - 1, 0))
        self._write(list(self.rows), list(self.recorded_epochs), self._path)

        return self._path

    @property
    def n_steps(self) -> int:
        return len(self.rows)


def load_trajectory(
    path: str | Path,
) -> tuple[np.ndarray, list[str], list[list[int]], list[int]]:
    with np.load(str(path), allow_pickle=False) as data:
        W = np.asarray(data["W"], dtype=np.float64)
        names = [str(n) for n in data["param_names"]]
        shapes = json.loads(str(data["param_shapes_json"]))
        epochs = (
            np.asarray(data["epochs"]).astype(int).tolist()
            if "epochs" in data
            else list(range(W.shape[0]))
        )

    return W, names, shapes, epochs
