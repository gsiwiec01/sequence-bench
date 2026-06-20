from typing import Any, Callable

def _make_copy_task(T: int):
    from ml_engine.datasets.copy_task import CopyTaskDataModule
    return CopyTaskDataModule(T=T)


def _make_adding_problem(T: int):
    from ml_engine.datasets.adding_problem import AddingProblemDataModule
    return AddingProblemDataModule(T=T)


def _make_sequential_mnist(permuted: bool):
    from ml_engine.datasets.sequential_mnist import SequentialMNISTDataModule
    return SequentialMNISTDataModule(permuted=permuted)


DATASET_REGISTRY: dict[str, Callable[[], Any]] = {
    "copy_task_T30":       lambda: _make_copy_task(30),
    "copy_task_T120":      lambda: _make_copy_task(120),
    "copy_task_T220":      lambda: _make_copy_task(220),
    "adding_problem_T200": lambda: _make_adding_problem(200),
    "adding_problem_T500": lambda: _make_adding_problem(500),
    "sMNIST":              lambda: _make_sequential_mnist(False),
}