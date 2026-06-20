from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from ml_engine.datasets.base import BaseDataModule, DatasetMetadata


class CopyTaskDataModule(BaseDataModule):
    def __init__(
            self,
            T: int = 120,
            n_classes: int = 8,
            copy_len: int = 10,
            n_train: int = 10_000,
            n_val: int = 1_000,
            n_test: int = 1_000,
    ) -> None:
        if T < 2 * copy_len + 1:
            raise ValueError(f"T={T} must be >= 2*copy_len+1={2 * copy_len + 1}")

        self.T = T
        self.n_classes = n_classes
        self.copy_len = copy_len
        self._n = {"train": n_train, "val": n_val, "test": n_test}

    def _generate(self, n: int, seed: int) -> TensorDataset:
        torch.manual_seed(seed)
        trigger_pos = self.T - self.copy_len - 1

        x = torch.zeros(n, self.T, self.n_classes + 2)
        y = torch.zeros(n, self.T, dtype=torch.long)

        tokens = torch.randint(0, self.n_classes, (n, self.copy_len))

        x[:, : self.copy_len, : self.n_classes].scatter_(2, tokens.unsqueeze(-1), 1.0)

        x[:, trigger_pos, self.n_classes] = 1.0
        # shift tokens by +1 - class 0 = blank
        y[:, trigger_pos + 1:] = tokens + 1

        return TensorDataset(x, y)

    def get_train_loader(self, batch_size: int, seed: int = 42) -> DataLoader:
        return DataLoader(self._generate(self._n["train"], seed), batch_size=batch_size,
                          shuffle=True, pin_memory=True, num_workers=0)

    def get_val_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._generate(self._n["val"], seed=1), batch_size=batch_size,
                          pin_memory=True, num_workers=0)

    def get_test_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._generate(self._n["test"], seed=2), batch_size=batch_size,
                          pin_memory=True, num_workers=0)

    @property
    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name=f"copy_task_T{self.T}",
            sequence_length=self.T,
            input_size=self.n_classes + 2,
            output_size=self.n_classes + 1,
            n_classes=self.n_classes + 1,
            n_train=self._n["train"],
            n_val=self._n["val"],
            n_test=self._n["test"],
            task_type="seq2seq",
        )
