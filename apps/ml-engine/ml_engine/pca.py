from __future__ import annotations

import numpy as np

def pca_directions(W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    W = np.asarray(W, dtype=np.float64)
    if W.shape[0] < 3:
        raise ValueError(
            f"Trajektoria ma {W.shape[0]} epok; PCA-2D wymaga >= 3 "
            "(wycentrowana macierz musi mieć rangę >= 2)."
        )
    w_end = W[-1]
    Wc = W - w_end

    _, S, Vt = np.linalg.svd(Wc, full_matrices=False)
    if Vt.shape[0] < 2:
        raise ValueError("Wycentrowana trajektoria ma rangę < 2 - brak 2 kierunków PCA.")

    d1 = Vt[0]
    d2 = Vt[1]

    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 - (d2 @ d1) * d1
    d2 = d2 / np.linalg.norm(d2)

    total = float(np.sum(S ** 2))
    explained = float((S[0] ** 2 + S[1] ** 2) / total) if total > 0 else 0.0

    return d1, d2, w_end, explained

def project_trajectory(
    W: np.ndarray, w_end: np.ndarray, d1: np.ndarray, d2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    Wc = np.asarray(W, dtype=np.float64) - w_end
    return Wc @ d1, Wc @ d2

def grid_axes(
    a_t: np.ndarray, b_t: np.ndarray, resolution: int = 25, margin: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    def axis(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        lo, hi = float(v.min()), float(v.max())
        span = hi - lo
        if span < 1e-12:
            span = 1.0

        pad = span * margin
        ax = np.linspace(lo - pad, hi + pad, resolution)

        j = int(np.argmin(np.abs(ax)))
        return ax - ax[j]

    return axis(a_t), axis(b_t)
