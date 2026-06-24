"""Forecast accuracy metrics.

WAPE (weighted absolute percentage error) is the headline metric: it behaves well
with intermittent / low-count demand where MAPE explodes on near-zero actuals.
"""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted absolute percentage error = sum|err| / sum|actual|."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = float(np.sum(np.abs(y_true)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)
