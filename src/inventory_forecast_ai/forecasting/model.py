"""CatBoost demand model: a single global model across all SKUs and horizons.

One model handles every horizon because the horizon ``h`` is just another input
feature (see ``features.py``). A Poisson loss is used because the target is
non-negative count data (units sold); it keeps predictions non-negative and matches
the demand-generating process, so the prediction for a row is the expected demand
``lambda`` for that (SKU, target-day, horizon).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from inventory_forecast_ai.forecasting.features import (
    CATEGORICAL_FEATURES,
    FEATURES,
    TARGET,
)

DEFAULT_PARAMS: dict = {
    "loss_function": "Poisson",
    "iterations": 400,
    "depth": 6,
    "learning_rate": 0.08,
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
    # "CPU" keeps training portable and deterministic. Switch to "GPU" only for very
    # large catalogues, where the CUDA speed-up outweighs its init overhead.
    "task_type": "CPU",
}


def build_params(task_type: str = "CPU") -> dict:
    """Return training params with the chosen compute device (CPU/GPU)."""
    return {**DEFAULT_PARAMS, "task_type": task_type}


def _to_pool(frame: pd.DataFrame, *, with_label: bool = True) -> Pool:
    """Build a CatBoost Pool, casting categorical columns to string."""
    features = frame[FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        features[col] = features[col].astype(str)
    label = frame[TARGET] if with_label else None
    return Pool(features, label=label, cat_features=CATEGORICAL_FEATURES)


def train_model(train_df: pd.DataFrame, params: dict | None = None) -> CatBoostRegressor:
    """Fit a CatBoost model on the multi-horizon feature frame."""
    model = CatBoostRegressor(**(params or DEFAULT_PARAMS))
    model.fit(_to_pool(train_df))
    return model


def predict(model: CatBoostRegressor, frame: pd.DataFrame) -> np.ndarray:
    """Predict expected demand (lambda) per row, clipped to non-negative values."""
    preds = model.predict(_to_pool(frame, with_label=False))
    return np.clip(preds, 0.0, None)


def predict_path(model: CatBoostRegressor, forecast_frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the predicted expected demand ``lam`` to a forward-path frame.

    ``forecast_frame`` is the output of ``build_forecast_frame`` (``horizon`` rows
    per SKU). The result is sorted by ``(sku_id, horizon)`` so downstream scenario
    generation can pivot it straight into a per-SKU demand path.
    """
    out = forecast_frame.copy()
    out["lam"] = predict(model, out)
    return out.sort_values(["sku_id", "horizon"]).reset_index(drop=True)
