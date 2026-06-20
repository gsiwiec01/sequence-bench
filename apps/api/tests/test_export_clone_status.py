"""Tests for export CSV, clone-config, and system status endpoints (pure logic)."""
import csv
import io
import pytest

def _build_export_rows(exps, ds_map: dict, T_map: dict) -> list[list]:
    rows = []
    for e in exps:
        T = T_map.get(e["dataset_id"], 1)
        rows.append([
            e["id"],
            ds_map.get(e["dataset_id"], e["dataset_id"]),
            e["architecture"],
            e["k1"],
            e["k2"],
            round(e["k2"] / T, 4),
            e["seed"],
            e.get("best_metric"),
            e.get("convergence_epoch"),
            e.get("total_training_time_s"),
            e.get("n_parameters"),
            e["status"],
        ])
    return rows

COLUMNS = [
    "experiment_id", "dataset", "architecture", "k1", "k2", "k2_ratio",
    "seed", "best_metric", "convergence_epoch", "total_training_time_s",
    "n_parameters", "status",
]

def test_export_csv_columns():
    exps = [
        {"id": "aaa", "dataset_id": "ds1", "architecture": "lstm", "k1": 5, "k2": 50,
         "seed": 42, "best_metric": 0.95, "convergence_epoch": 30,
         "total_training_time_s": 120.5, "n_parameters": 50000, "status": "completed"},
    ]
    ds_map = {"ds1": "adding_problem"}
    T_map = {"ds1": 100}
    rows = _build_export_rows(exps, ds_map, T_map)
    assert len(rows) == 1
    row = rows[0]
    assert len(row) == len(COLUMNS)

def test_export_csv_k2_ratio():
    exps = [
        {"id": "bbb", "dataset_id": "ds1", "architecture": "gru", "k1": 10, "k2": 25,
         "seed": 43, "best_metric": None, "convergence_epoch": None,
         "total_training_time_s": None, "n_parameters": None, "status": "failed"},
    ]
    T_map = {"ds1": 100}
    rows = _build_export_rows(exps, {}, T_map)
    k2_ratio = rows[0][5]
    assert k2_ratio == pytest.approx(0.25)

def test_export_csv_roundtrip():
    exps = [
        {"id": "ccc", "dataset_id": "ds2", "architecture": "rnn", "k1": 1, "k2": 10,
         "seed": 0, "best_metric": 0.5, "convergence_epoch": 10,
         "total_training_time_s": 60.0, "n_parameters": 1000, "status": "completed"},
    ]
    ds_map = {"ds2": "copy_task"}
    T_map = {"ds2": 50}
    rows = _build_export_rows(exps, ds_map, T_map)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(COLUMNS)
    writer.writerows(rows)
    buf.seek(0)
    reader = csv.DictReader(buf)
    parsed = list(reader)

    assert len(parsed) == 1
    assert parsed[0]["experiment_id"] == "ccc"
    assert parsed[0]["dataset"] == "copy_task"
    assert float(parsed[0]["k2_ratio"]) == pytest.approx(0.2)

def test_export_csv_empty():
    rows = _build_export_rows([], {}, {})
    assert rows == []

def _build_clone_config(exp: dict) -> dict:
    return {
        "dataset_id": exp["dataset_id"],
        "architecture": exp["architecture"],
        "k1": exp["k1"],
        "k2": exp["k2"],
        "seed": exp["seed"],
        "task_type": exp["task_type"],
        "hyperparams": exp["hyperparams"],
        "additional_metrics": exp.get("additional_metrics", []),
    }

def test_clone_config_all_fields_present():
    exp = {
        "dataset_id": "ds1", "architecture": "lstm", "k1": 5, "k2": 50,
        "seed": 42, "task_type": "classification",
        "hyperparams": {"hidden_size": 256, "num_layers": 1, "dropout": 0.2},
        "additional_metrics": ["f1_macro"],
    }
    cfg = _build_clone_config(exp)
    required_keys = {
        "dataset_id", "architecture", "k1", "k2", "seed",
        "task_type", "hyperparams", "additional_metrics",
    }
    assert required_keys == set(cfg.keys())

def test_clone_config_values_match_source():
    exp = {
        "dataset_id": "ds99", "architecture": "gru", "k1": 10, "k2": 80,
        "seed": 123, "task_type": "regression",
        "hyperparams": {"hidden_size": 128, "learning_rate": 0.001},
        "additional_metrics": [],
    }
    cfg = _build_clone_config(exp)
    assert cfg["dataset_id"] == "ds99"
    assert cfg["architecture"] == "gru"
    assert cfg["k1"] == 10
    assert cfg["k2"] == 80
    assert cfg["seed"] == 123
    assert cfg["hyperparams"]["hidden_size"] == 128

def _build_status(gpu_available, gpu_name, total_mb, used_mb, active, queued, workers):
    free_mb = (total_mb - used_mb) if (gpu_available and total_mb is not None) else None
    return {
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "gpu_memory_total_mb": total_mb,
        "gpu_memory_used_mb": used_mb,
        "gpu_memory_free_mb": free_mb,
        "active_experiments": active,
        "queued_experiments": queued,
        "celery_workers_active": workers,
    }

def test_system_status_gpu_present():
    status = _build_status(True, "RTX 5070 Ti", 16384, 4821, 2, 5, 1)
    assert status["gpu_available"] is True
    assert status["gpu_name"] == "RTX 5070 Ti"
    assert status["gpu_memory_free_mb"] == 16384 - 4821
    assert status["active_experiments"] == 2
    assert status["queued_experiments"] == 5
    assert status["celery_workers_active"] == 1

def test_system_status_no_gpu():
    status = _build_status(False, None, None, None, 0, 3, 0)
    assert status["gpu_available"] is False
    assert status["gpu_name"] is None
    assert status["gpu_memory_free_mb"] is None
    assert status["queued_experiments"] == 3

def test_system_status_vram_pct():
    status = _build_status(True, "A100", 40960, 32768, 1, 0, 2)
    pct = round(status["gpu_memory_used_mb"] / status["gpu_memory_total_mb"] * 100)
    assert pct == 80

def test_system_status_all_keys_present():
    status = _build_status(False, None, None, None, 0, 0, 0)
    required = {
        "gpu_available", "gpu_name", "gpu_memory_total_mb",
        "gpu_memory_used_mb", "gpu_memory_free_mb",
        "active_experiments", "queued_experiments", "celery_workers_active",
    }
    assert required == set(status.keys())
