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
    # Upper bound only: when an eval set is supplied, early stopping selects the actual
    # number of trees (see ``train_model``), so this is a ceiling, not a tuned constant.
    "iterations": 2000,
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


def train_model(
    train_df: pd.DataFrame,
    params: dict | None = None,
    *,
    eval_df: pd.DataFrame | None = None,
    early_stopping_rounds: int = 50,
) -> CatBoostRegressor:
    """Fit a CatBoost model on the multi-horizon feature frame.

    When ``eval_df`` is given (a temporal validation tail — see
    ``features.temporal_train_val_split``), training uses early stopping: ``iterations``
    becomes an upper bound and the number of trees is chosen where the validation loss
    stops improving, and the best iteration is kept (``use_best_model``). Without an
    eval set it is a plain fit at the configured ``iterations``.
    """
    model = CatBoostRegressor(**(params or DEFAULT_PARAMS))
    if eval_df is not None:
        model.fit(
            _to_pool(train_df),
            eval_set=_to_pool(eval_df),
            early_stopping_rounds=early_stopping_rounds,
            use_best_model=True,
        )
    else:
        model.fit(_to_pool(train_df))
    return model


def best_iteration(model: CatBoostRegressor) -> int:
    """Number of trees kept after early stopping — the honest, selected iteration count."""
    return int(model.tree_count_)


def validation_curve(model: CatBoostRegressor) -> dict[str, list[float]]:
    """Per-iteration validation loss recorded during an early-stopping fit.

    Returns ``{metric_name: [loss per iteration]}`` (empty if the model was trained
    without an eval set). Used to log the selection curve to MLflow.
    """
    evals = model.get_evals_result()
    val = evals.get("validation") or evals.get("validation_0") or {}
    return {name: [float(x) for x in values] for name, values in val.items()}


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
