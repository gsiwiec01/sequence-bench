from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class UploadConfig:
    normalize: str              # "zscore" | "minmax" | "none"
    nan_strategy: str           # "error" | "interpolate" | "ffill" | "drop"
    n_classes: Optional[int] = None   # dla klasyfikacji; określa output_size
    # task_type jest zachowane dla wstecznej kompatybilności -nie wpływa na przetwarzanie danych
    task_type: str = "unknown"
    sliding_window: Optional[int] = None
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    target_column: Optional[str] = None  # dla CSV z kolumną celu


MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB


def load_and_validate(file_path: str, config: UploadConfig) -> dict:
    path = Path(file_path)

    if path.stat().st_size > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"Plik za duży: {path.stat().st_size / 1e6:.1f} MB > 500 MB")

    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        X = data["X"]
        y = data.get("y")
    elif suffix == ".npy":
        X = np.load(path, allow_pickle=False)
        y = None
    elif suffix == ".csv":
        X, y = _load_csv(path, config.target_column)
    else:
        raise ValueError(f"Nieobsługiwany format: {suffix}. Użyj .npz, .npy lub .csv")

    _validate_array(X, "X")
    if y is not None:
        _validate_array(y, "y")

    if X.ndim == 2:
        X = X[:, :, np.newaxis]

    N, T, d = X.shape
    if N < 10:
        raise ValueError(f"Za mało próbek: {N} < 10")
    if T <= 1:
        raise ValueError(f"Sekwencja za krótka: T={T}")

    X = _handle_nan(X, config.nan_strategy)

    n_train = int(N * config.train_split)
    n_val = int(N * config.val_split)
    X_tr, X_val, X_te = X[:n_train], X[n_train:n_train + n_val], X[n_train + n_val:]
    if y is not None:
        y_tr, y_val, y_te = y[:n_train], y[n_train:n_train + n_val], y[n_train + n_val:]
    else:
        y_tr = y_val = y_te = None

    X_tr, X_val, X_te = _normalize(X_tr, X_val, X_te, config.normalize)

    if config.sliding_window:
        X_tr, y_tr = _apply_sliding_window(X_tr, y_tr, config.sliding_window)
        X_val, y_val = _apply_sliding_window(X_val, y_val, config.sliding_window)
        X_te, y_te = _apply_sliding_window(X_te, y_te, config.sliding_window)

    return {
        "X_train": X_tr, "X_val": X_val, "X_test": X_te,
        "y_train": y_tr, "y_val": y_val, "y_test": y_te,
        "metadata": {"N": N, "T": T, "d": d, "n_train": len(X_tr),
                     "n_val": len(X_val), "n_test": len(X_te)},
    }


def _validate_array(arr: np.ndarray, name: str) -> None:
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name}: dane muszą być numeryczne (dtype={arr.dtype})")
    if np.any(np.isinf(arr)):
        raise ValueError(f"{name}: dane zawierają Inf")


def _handle_nan(X: np.ndarray, strategy: str) -> np.ndarray:
    if not np.any(np.isnan(X)):
        return X

    if strategy == "error":
        raise ValueError("Dane zawierają NaN. Użyj nan_strategy='interpolate', 'ffill' lub 'drop'")

    if strategy == "drop":
        mask = ~np.any(np.isnan(X.reshape(X.shape[0], -1)), axis=1)
        return X[mask]

    if strategy == "ffill":
        for i in range(X.shape[0]):
            for dim in range(X.shape[2]):
                s = X[i, :, dim]
                nan_idx = np.where(np.isnan(s))[0]
                for idx in nan_idx:
                    X[i, idx, dim] = X[i, idx - 1, dim] if idx > 0 else 0.0
        return X

    if strategy == "interpolate":
        from scipy.interpolate import interp1d
        for i in range(X.shape[0]):
            for dim in range(X.shape[2]):
                s = X[i, :, dim]
                t = np.arange(len(s))
                valid = ~np.isnan(s)
                if valid.sum() > 1:
                    f = interp1d(t[valid], s[valid], bounds_error=False, fill_value="extrapolate")
                    X[i, :, dim] = f(t)
        return X

    raise ValueError(f"Nieznana nan_strategy: {strategy}")


def _normalize(
    X_tr: np.ndarray, X_val: np.ndarray, X_te: np.ndarray, method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if method == "none":
        return X_tr, X_val, X_te

    if method == "zscore":
        mu = X_tr.mean(axis=(0, 1), keepdims=True)
        sigma = X_tr.std(axis=(0, 1), keepdims=True) + 1e-8
        return (X_tr - mu) / sigma, (X_val - mu) / sigma, (X_te - mu) / sigma

    if method == "minmax":
        mn = X_tr.min(axis=(0, 1), keepdims=True)
        mx = X_tr.max(axis=(0, 1), keepdims=True)
        rng = mx - mn + 1e-8
        return (X_tr - mn) / rng, (X_val - mn) / rng, (X_te - mn) / rng

    raise ValueError(f"Nieznana normalize: {method}")


def _apply_sliding_window(
    X: np.ndarray, y: np.ndarray | None, window_size: int
) -> tuple[np.ndarray, np.ndarray | None]:
    if X is None:
        return X, y

    N, T, d = X.shape
    xs, ys = [], []
    for i in range(N):
        for start in range(0, T - window_size + 1):
            xs.append(X[i, start: start + window_size])
            if y is not None:
                ys.append(y[i])

    X_out = np.array(xs)
    y_out = np.array(ys) if ys else None

    return X_out, y_out


def _load_csv(path: Path, target_column: str | None) -> tuple[np.ndarray, np.ndarray | None]:
    df = pd.read_csv(path)
    if target_column and target_column in df.columns:
        y = df[target_column].values
        X = df.drop(columns=[target_column]).values
    else:
        X = df.values
        y = None

    return X, y
