from __future__ import annotations

import numpy as np

ES_MODE: dict[str, str] = {
    "val_loss": "min",
    "mse": "min",
    "mae": "min",
    "mape": "min",
    "cross_entropy": "min",
    "perplexity": "min",
    "accuracy": "max",
    "f1_macro": "max",
    "f1_weighted": "max",
    "precision_macro": "max",
    "recall_macro": "max",
    "auc_macro": "max",
    "mcc": "max",
    "r2": "max",
}

AVAILABLE_METRICS: dict[str, list[str]] = {
    "classification": [
        "accuracy", "f1_macro", "f1_weighted", "precision_macro",
        "recall_macro", "auc_macro", "mcc",
    ],
    "regression": ["mse", "mae", "r2", "mape"],
    "language_model": ["perplexity", "cross_entropy"],
    "seq2seq": ["accuracy", "f1_macro", "cross_entropy"],
    "forecasting": ["mse", "mae", "mape", "r2"],
}


def compute_additional_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    metric_names: list[str],
    task_type: str,
    logits: np.ndarray | None = None,
) -> dict[str, float]:
    from sklearn.metrics import (
        f1_score,
        matthews_corrcoef,
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
    )

    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    results: dict[str, float] = {}
    cross_entropy: float | None = None

    for name in metric_names:
        try:
            if name == "accuracy":
                results[name] = float(np.mean(preds == targets))

            elif name == "f1_macro":
                results[name] = float(
                    f1_score(targets, preds, average="macro", zero_division=0)
                )
            elif name == "f1_weighted":
                results[name] = float(
                    f1_score(targets, preds, average="weighted", zero_division=0)
                )
            elif name == "precision_macro":
                results[name] = float(
                    precision_score(targets, preds, average="macro", zero_division=0)
                )
            elif name == "recall_macro":
                results[name] = float(
                    recall_score(targets, preds, average="macro", zero_division=0)
                )
            elif name == "mcc":
                results[name] = float(matthews_corrcoef(targets, preds))

            elif name == "auc_macro":
                if logits is not None and len(logits):
                    probs = _softmax(logits)
                    classes = np.unique(targets)
                    if len(classes) == 2:
                        results[name] = float(roc_auc_score(targets, probs[:, 1]))
                    else:
                        results[name] = float(
                            roc_auc_score(
                                targets, probs, multi_class="ovr", average="macro"
                            )
                        )

            elif name == "mse":
                results[name] = float(mean_squared_error(targets, preds))
            elif name == "mae":
                results[name] = float(mean_absolute_error(targets, preds))
            elif name == "r2":
                results[name] = float(r2_score(targets, preds))
            elif name == "mape":
                results[name] = float(mean_absolute_percentage_error(targets, preds))

            elif name == "cross_entropy":
                if logits is not None and len(logits):
                    log_probs = np.log(_softmax(logits) + 1e-12)
                    idx = targets.astype(int)
                    ce = float(-np.mean(log_probs[np.arange(len(idx)), idx]))
                    cross_entropy = ce
                    results[name] = ce

            elif name == "perplexity":
                if cross_entropy is not None:
                    results[name] = float(np.exp(cross_entropy))
                elif logits is not None and len(logits):
                    log_probs = np.log(_softmax(logits) + 1e-12)
                    idx = targets.astype(int)
                    ce = float(-np.mean(log_probs[np.arange(len(idx)), idx]))
                    results[name] = float(np.exp(ce))

        except Exception:
            pass

    return results
