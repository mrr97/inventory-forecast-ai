"""CatBoost demand model: a single global model across all SKUs and horizons.

One model handles every horizon because the horizon ``h`` is just another input
feature (see ``features.py``). A Poisson loss is used because the target is
non-negative count data (units sold); it keeps predictions non-negative and matches
the demand-generating process, so the prediction for a row is the expected demand
``lambda`` for that (SKU, target-day, horizon).
"""

from __future__ import annotations

import itertools

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


# Deliberately tiny grid over the two hyperparameters that matter most for a gradient-boosted
# model: tree depth (model capacity) and learning rate. A small, explicit grid — not an
# exhaustive sweep — is the right size here: with ~22 SKUs the marginal value of fine tuning is
# low, so this is a sanity check that the defaults are not badly off, evaluated honestly on a
# temporal validation tail (never a shuffled CV, which would leak the future into the past).
DEFAULT_SEARCH_GRID: dict[str, list] = {
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.08],
}


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
    plot_file: str | None = None,
) -> CatBoostRegressor:
    """Fit a CatBoost model on the multi-horizon feature frame.

    When ``eval_df`` is given (a temporal validation tail — see
    ``features.temporal_train_val_split``), training uses early stopping: ``iterations``
    becomes an upper bound and the number of trees is chosen where the validation loss
    stops improving, and the best iteration is kept (``use_best_model``). Without an
    eval set it is a plain fit at the configured ``iterations``.

    If ``plot_file`` is given, CatBoost writes its native interactive training-progress
    chart to that HTML file (``fit(plot_file=...)``) — no manual plotting.
    """
    model = CatBoostRegressor(**(params or DEFAULT_PARAMS))
    fit_kwargs: dict = {}
    if plot_file is not None:
        fit_kwargs["plot_file"] = plot_file
    if eval_df is not None:
        model.fit(
            _to_pool(train_df),
            eval_set=_to_pool(eval_df),
            early_stopping_rounds=early_stopping_rounds,
            use_best_model=True,
            **fit_kwargs,
        )
    else:
        model.fit(_to_pool(train_df), **fit_kwargs)
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


def _validation_loss(model: CatBoostRegressor) -> float:
    """Best (lowest) validation loss reached during an early-stopping fit."""
    val = model.get_best_score().get("validation", {})
    return float(next(iter(val.values())))


def search_hyperparameters(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    base_params: dict,
    grid: dict[str, list] | None = None,
    early_stopping_rounds: int = 50,
) -> tuple[dict, list[dict]]:
    """Tiny temporal grid search over ``grid`` (default ``DEFAULT_SEARCH_GRID``).

    Each combination is trained with early stopping on the temporal validation tail
    ``val_df`` and scored by its best validation loss; the combination with the lowest loss
    wins. Because ``val_df`` is a strict future slice of ``train_df`` (see
    ``features.temporal_train_val_split``), selection never sees data it is scored on.

    Returns ``(best, trials)`` where ``best`` has keys ``params`` (the winning params with
    the early-stopping-selected ``iterations`` baked in), ``combo`` (just the searched keys)
    and ``val_loss``; ``trials`` lists every combination's ``{**combo, iterations, val_loss}``
    for logging.
    """
    grid = grid or DEFAULT_SEARCH_GRID
    combos = [dict(zip(grid, values, strict=True)) for values in itertools.product(*grid.values())]

    best: dict | None = None
    trials: list[dict] = []
    for combo in combos:
        params = {**base_params, **combo}
        model = train_model(
            train_df, params, eval_df=val_df, early_stopping_rounds=early_stopping_rounds
        )
        loss = _validation_loss(model)
        chosen = best_iteration(model)
        trials.append({**combo, "iterations": chosen, "val_loss": loss})
        if best is None or loss < best["val_loss"]:
            best = {"params": {**params, "iterations": chosen}, "combo": combo, "val_loss": loss}

    assert best is not None  # grid is never empty
    return best, trials


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
