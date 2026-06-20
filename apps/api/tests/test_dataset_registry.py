from api.builtin_datasets import BUILTIN_DATASETS
from worker.dataset_registry import DATASET_REGISTRY

def test_all_builtins_registered():
    missing = [ds["name"] for ds in BUILTIN_DATASETS if ds["name"] not in DATASET_REGISTRY]
    assert not missing, f"Missing from DATASET_REGISTRY: {missing}"

def test_registry_factories_are_callable():
    for name, factory in DATASET_REGISTRY.items():
        assert callable(factory), f"Factory for {name!r} is not callable"

def test_no_extra_keys_without_builtin_entry():
    builtin_names = {ds["name"] for ds in BUILTIN_DATASETS}
    orphaned = [name for name in DATASET_REGISTRY if name not in builtin_names]
    assert not orphaned, f"Keys in DATASET_REGISTRY without BUILTIN_DATASETS entry: {orphaned}"
