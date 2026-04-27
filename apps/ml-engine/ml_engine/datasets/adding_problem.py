from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseDataModule, DatasetMetadata


class AddingProblemDataModule(BaseDataModule):
    def __init__(
            self,
            T: int = 200,
            n_train: int = 10_000,
            n_val: int = 1_000,
            n_test: int = 1_000,
    ) -> None:
        self.T = T
        self._n = {"train": n_train, "val": n_val, "test": n_test}

    def _generate(self, n: int, seed: int) -> TensorDataset:
        torch.manual_seed(seed)
        values = torch.rand(n, self.T)
        mask = torch.zeros(n, self.T)

        for i in range(n):
            idx = torch.randperm(self.T)[:2]
            mask[i, idx] = 1.0

        x = torch.stack([values, mask], dim=-1)  # (n, T, 2)
        y = (values * mask).sum(dim=1, keepdim=True)  # (n, 1)
        return TensorDataset(x, y)

    def get_train_loader(self, batch_size: int, seed: int = 42) -> DataLoader:
        return DataLoader(self._generate(self._n["train"], seed), batch_size=batch_size, shuffle=True)

    def get_val_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._generate(self._n["val"], seed=1), batch_size=batch_size)

    def get_test_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._generate(self._n["test"], seed=2), batch_size=batch_size)

    @property
    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name=f"adding_problem_T{self.T}",
            task_type="regression",
            T=self.T,
            input_size=2,
            output_size=1,
            metric_name="mse",
            n_train=self._n["train"],
            n_val=self._n["val"],
            n_test=self._n["test"],
        )
