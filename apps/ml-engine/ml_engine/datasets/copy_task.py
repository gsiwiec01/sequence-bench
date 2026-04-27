from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseDataModule, DatasetMetadata


class CopyTaskDataModule(BaseDataModule):
    def __init__(
            self,
            T: int = 100,
            n_classes: int = 8,
            copy_len: int = 10,
            n_train: int = 10_000,
            n_val: int = 1_000,
            n_test: int = 1_000,
    ) -> None:
        self.T = T
        self.n_classes = n_classes
        self.copy_len = copy_len
        self._n = {"train": n_train, "val": n_val, "test": n_test}

    def _generate(self, n: int, seed: int) -> TensorDataset:
        torch.manual_seed(seed)
        seq_len = self.T + self.copy_len + 1

        x = torch.zeros(n, seq_len, self.n_classes + 2)
        y = torch.zeros(n, seq_len, dtype=torch.long)

        tokens = torch.randint(0, self.n_classes, (n, self.copy_len))

        for i in range(self.copy_len):
            x[:, i, tokens[:, i]] = 1.0

        x[:, self.T + self.copy_len, self.n_classes] = 1.0
        y[:, self.T + 1:] = tokens

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
            name=f"copy_task_T{self.T}",
            task_type="seq2seq",
            T=self.T + self.copy_len + 1,
            input_size=self.n_classes + 2,
            output_size=self.n_classes + 1,
            metric_name="accuracy",
            n_train=self._n["train"],
            n_val=self._n["val"],
            n_test=self._n["test"],
        )
