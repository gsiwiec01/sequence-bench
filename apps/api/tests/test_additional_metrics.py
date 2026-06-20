import numpy as np
import pytest

from api.metrics_registry import AVAILABLE_METRICS, compute_additional_metrics

def test_metrics_validation_unknown_metric():
    unknown = "nonexistent_metric"
    for task_type, allowed in AVAILABLE_METRICS.items():
        assert unknown not in allowed, f"{unknown!r} niespodziewanie w {task_type}"


def test_available_metrics_all_task_types():
    expected_types = {"classification", "regression", "language_model", "seq2seq", "forecasting"}
    assert expected_types == set(AVAILABLE_METRICS.keys())


def test_available_metrics_non_empty():
    for task_type, metrics in AVAILABLE_METRICS.items():
        assert len(metrics) > 0, f"Brak metryk dla task_type={task_type}"

def test_additional_metrics_classification_f1():
    preds = np.array([0, 1, 2, 0, 1, 2])
    targets = np.array([0, 1, 2, 1, 0, 2])
    results = compute_additional_metrics(preds, targets, ["f1_macro", "f1_weighted"], "classification")
    assert "f1_macro" in results
    assert "f1_weighted" in results
    assert 0.0 <= results["f1_macro"] <= 1.0
    assert 0.0 <= results["f1_weighted"] <= 1.0


def test_additional_metrics_classification_precision_recall():
    preds = np.array([0, 0, 1, 1, 2, 2])
    targets = np.array([0, 1, 1, 2, 2, 0])
    results = compute_additional_metrics(
        preds, targets, ["precision_macro", "recall_macro", "mcc"], "classification"
    )
    assert "precision_macro" in results
    assert "recall_macro" in results
    assert "mcc" in results
    assert -1.0 <= results["mcc"] <= 1.0


def test_additional_metrics_classification_accuracy():
    preds = np.array([0, 1, 2, 0])
    targets = np.array([0, 1, 2, 0])
    results = compute_additional_metrics(preds, targets, ["accuracy"], "classification")
    assert results["accuracy"] == pytest.approx(1.0)


def test_additional_metrics_auc_macro_binary():
    np.random.seed(42)
    targets = np.array([0, 0, 1, 1, 0, 1])
    logits = np.array([
        [2.0, -1.0],
        [1.5, -0.5],
        [-0.5, 1.5],
        [-1.0, 2.0],
        [0.5, -0.5],
        [-0.5, 0.5],
    ])
    preds = logits.argmax(axis=-1)
    results = compute_additional_metrics(preds, targets, ["auc_macro"], "classification", logits=logits)
    assert "auc_macro" in results
    assert 0.0 <= results["auc_macro"] <= 1.0


def test_additional_metrics_auc_skipped_without_logits():
    preds = np.array([0, 1, 0, 1])
    targets = np.array([0, 1, 1, 0])
    results = compute_additional_metrics(preds, targets, ["auc_macro"], "classification")
    assert "auc_macro" not in results

def test_additional_metrics_regression_basic():
    preds = np.array([1.0, 2.0, 3.0, 4.0])
    targets = np.array([1.1, 1.9, 3.2, 3.8])
    results = compute_additional_metrics(preds, targets, ["mse", "mae", "r2"], "regression")
    assert "mse" in results
    assert "mae" in results
    assert "r2" in results
    assert results["mse"] >= 0.0
    assert results["mae"] >= 0.0
    assert results["r2"] <= 1.0


def test_additional_metrics_regression_perfect():
    preds = np.array([1.0, 2.0, 3.0])
    targets = np.array([1.0, 2.0, 3.0])
    results = compute_additional_metrics(preds, targets, ["mse", "r2"], "regression")
    assert results["mse"] == pytest.approx(0.0, abs=1e-9)
    assert results["r2"] == pytest.approx(1.0)


def test_additional_metrics_regression_mape():
    preds = np.array([100.0, 200.0])
    targets = np.array([110.0, 190.0])
    results = compute_additional_metrics(preds, targets, ["mape"], "regression")
    assert "mape" in results
    assert results["mape"] >= 0.0

def test_additional_metrics_language_model_cross_entropy():
    np.random.seed(0)
    n = 50
    n_classes = 10
    logits = np.random.randn(n, n_classes)
    targets = np.random.randint(0, n_classes, n)
    preds = logits.argmax(axis=-1)
    results = compute_additional_metrics(
        preds, targets, ["cross_entropy", "perplexity"], "language_model", logits=logits
    )
    assert "cross_entropy" in results
    assert "perplexity" in results
    assert results["cross_entropy"] > 0.0
    assert results["perplexity"] > 1.0

def test_empty_metric_names_returns_empty():
    preds = np.array([0, 1])
    targets = np.array([0, 1])
    results = compute_additional_metrics(preds, targets, [], "classification")
    assert results == {}


def test_unknown_metric_name_is_silently_skipped():
    preds = np.array([0, 1])
    targets = np.array([0, 1])
    results = compute_additional_metrics(preds, targets, ["totally_unknown"], "classification")
    assert "totally_unknown" not in results
