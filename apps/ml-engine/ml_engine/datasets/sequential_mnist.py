from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from ml_engine.datasets.base import BaseDataModule, DatasetMetadata

logging.getLogger("torchvision").setLevel(logging.ERROR)
logging.getLogger("torchvision.datasets.utils").setLevel(logging.ERROR)

_CACHE_DIR = Path(os.environ.get("DATA_DIR", "/app/data")) / "mnist"


def _load_mnist_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        from torchvision.datasets import MNIST
        from torchvision import transforms
    except ImportError as e:
        raise ImportError("torchvision is required for sMNIST -install it in the ml-engine env") from e

    t = transforms.ToTensor()
    train_ds = MNIST(root=str(_CACHE_DIR), train=True,  download=True, transform=t)
    test_ds  = MNIST(root=str(_CACHE_DIR), train=False, download=True, transform=t)

    def _to_seq(ds) -> tuple[torch.Tensor, torch.Tensor]:
        loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
        x, y = next(iter(loader))          # x: (N, 1, 28, 28)
        x = x.view(len(ds), 784, 1)       # → (N, T, 1)
        return x, y

    x_tr, y_tr = _to_seq(train_ds)
    x_te, y_te = _to_seq(test_ds)
    return x_tr, y_tr, x_te, y_te


class SequentialMNISTDataModule(BaseDataModule):
    def __init__(
        self,
        permuted: bool = False,
        val_size: int = 5_000,
    ) -> None:
        self.permuted = permuted
        self.val_size = val_size
        self._data: tuple | None = None

    def _ensure_loaded(self) -> None:
        if self._data is not None:
            return

        x_tr, y_tr, x_te, y_te = _load_mnist_tensors()

        if self.permuted:
            torch.manual_seed(0)
            perm = torch.randperm(784)
            x_tr = x_tr[:, perm, :]
            x_te = x_te[:, perm, :]

        n_val = self.val_size
        x_val, y_val = x_tr[:n_val], y_tr[:n_val]
        x_tr,  y_tr  = x_tr[n_val:], y_tr[n_val:]

        self._data = (
            TensorDataset(x_tr, y_tr),
            TensorDataset(x_val, y_val),
            TensorDataset(x_te, y_te),
        )

    def get_train_loader(self, batch_size: int, seed: int = 42) -> DataLoader:
        self._ensure_loaded()
        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(
            self._data[0], batch_size=batch_size,
            shuffle=True, generator=g, pin_memory=True, num_workers=0,
        )

    def get_val_loader(self, batch_size: int) -> DataLoader:
        self._ensure_loaded()
        return DataLoader(self._data[1], batch_size=batch_size, pin_memory=True, num_workers=0)

    def get_test_loader(self, batch_size: int) -> DataLoader:
        self._ensure_loaded()
        return DataLoader(self._data[2], batch_size=batch_size, pin_memory=True, num_workers=0)

    @property
    def metadata(self) -> DatasetMetadata:
        self._ensure_loaded()
        n_tr = len(self._data[0])
        n_val = len(self._data[1])
        n_te  = len(self._data[2])

        return DatasetMetadata(
            name="pMNIST" if self.permuted else "sMNIST",
            sequence_length=784,
            input_size=1,
            output_size=10,
            n_classes=10,
            n_train=n_tr,
            n_val=n_val,
            n_test=n_te,
            task_type="classification",
        )
