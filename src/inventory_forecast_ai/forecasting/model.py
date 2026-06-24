"""CatBoost demand model: a single global model across all SKUs.

A Poisson loss is used because the target is non-negative count data (units sold);
it keeps predictions non-negative and matches the demand-generating process.
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
}


def _to_pool(frame: pd.DataFrame, *, with_label: bool = True) -> Pool:
    """Build a CatBoost Pool, casting categorical columns to string."""
    features = frame[FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        features[col] = features[col].astype(str)
    label = frame[TARGET] if with_label else None
    return Pool(features, label=label, cat_features=CATEGORICAL_FEATURES)


def train_model(train_df: pd.DataFrame, params: dict | None = None) -> CatBoostRegressor:
    """Fit a CatBoost model on the engineered feature frame."""
    model = CatBoostRegressor(**(params or DEFAULT_PARAMS))
    model.fit(_to_pool(train_df))
    return model


def predict(model: CatBoostRegressor, frame: pd.DataFrame) -> np.ndarray:
    """Predict expected demand, clipped to non-negative values."""
    preds = model.predict(_to_pool(frame, with_label=False))
    return np.clip(preds, 0.0, None)
