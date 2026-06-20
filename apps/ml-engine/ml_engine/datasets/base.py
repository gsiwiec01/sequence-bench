from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from torch.utils.data import DataLoader


@dataclass
class DatasetMetadata:
    name: str
    sequence_length: int  # nominal T
    input_size: int  # d -number of input features
    output_size: int  # output dimension (= n_classes for classification, 1 for regression)
    n_classes: int | None = None  # nullable; set for classification-like datasets
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    task_type: str = "classification"


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
