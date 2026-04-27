from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch.utils.data import DataLoader


@dataclass
class DatasetMetadata:
    name: str
    task_type: str  # "classification" | "regression" | "language_model" | "seq2seq"
    T: int  # nominal sequence length
    input_size: int  # d — number of input features
    output_size: int  # number of classes or output dimension
    metric_name: str  # "accuracy" | "mse" | "perplexity" | "auc"
    n_train: int
    n_val: int
    n_test: int


class BaseDataModule(ABC):
    @abstractmethod
    def get_train_loader(self, batch_size: int, seed: int) -> DataLoader: ...

    @abstractmethod
    def get_val_loader(self, batch_size: int) -> DataLoader: ...

    @abstractmethod
    def get_test_loader(self, batch_size: int) -> DataLoader: ...

    @property
    @abstractmethod
    def metadata(self) -> DatasetMetadata: ...
