from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml_engine.datasets.base import BaseDataModule, DatasetMetadata
from ml_engine.datasets.upload import UploadConfig, load_and_validate

class GenericSequenceDataModule(BaseDataModule):
    def __init__(self, file_path: str, config: UploadConfig, name: str = "custom"):
        self.name = name
        self.config = config
        data = load_and_validate(file_path, config)

        self._X_tr = _to_tensor(data["X_train"])
        self._X_val = _to_tensor(data["X_val"])
        self._X_te = _to_tensor(data["X_test"])
        self._y_tr = _to_tensor(data["y_train"])
        self._y_val = _to_tensor(data["y_val"])
        self._y_te = _to_tensor(data["y_test"])
        self._meta_raw = data["metadata"]

    def _make_loader(self, X: torch.Tensor, y: torch.Tensor | None, batch_size: int, shuffle: bool) -> DataLoader:
        ds = TensorDataset(X, y) if y is not None else TensorDataset(X)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def get_train_loader(self, batch_size: int, seed: int = 42) -> DataLoader:
        return self._make_loader(self._X_tr, self._y_tr, batch_size, shuffle=True)

    def get_val_loader(self, batch_size: int) -> DataLoader:
        return self._make_loader(self._X_val, self._y_val, batch_size, shuffle=False)

    def get_test_loader(self, batch_size: int) -> DataLoader:
        return self._make_loader(self._X_te, self._y_te, batch_size, shuffle=False)

    @property
    def metadata(self) -> DatasetMetadata:
        m = self._meta_raw
        return DatasetMetadata(
            name=self.name,
            sequence_length=m["T"],
            input_size=m["d"],
            output_size=self.config.n_classes or 1,
            n_classes=self.config.n_classes,
            n_train=m["n_train"],
            n_val=m["n_val"],
            n_test=m["n_test"],
            task_type=self.config.task_type,
        )


def _to_tensor(arr: np.ndarray | None) -> torch.Tensor | None:
    if arr is None:
        return None

    return torch.from_numpy(arr.astype(np.float32))
