"""End-to-end training pipeline: features -> multi-horizon backtest -> MLflow -> model."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
import structlog

from inventory_forecast_ai.config import Settings
from inventory_forecast_ai.forecasting.backtest import BacktestResult, backtest_forecast
from inventory_forecast_ai.forecasting.features import (
    build_training_frame,
    temporal_train_val_split,
)
from inventory_forecast_ai.forecasting.model import (
    DEFAULT_PARAMS,
    best_iteration,
    build_params,
    train_model,
    validation_curve,
)

log = structlog.get_logger(__name__)


def _log_to_mlflow(
    settings: Settings,
    result: BacktestResult,
    *,
    best_iter: int,
    val_curve: dict[str, list[float]],
) -> None:
    """Record params, pooled backtest metrics and the early-stopping selection in MLflow."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "model": "catboost",
                "loss_function": DEFAULT_PARAMS["loss_function"],
                # `iterations` is only a ceiling; early stopping chooses `best_iteration`.
                "iterations_cap": DEFAULT_PARAMS["iterations"],
                "depth": DEFAULT_PARAMS["depth"],
                "learning_rate": DEFAULT_PARAMS["learning_rate"],
                "horizon": result.horizon,
                "backtest_holdout_days": settings.backtest_holdout_days,
                "early_stopping_val_days": settings.early_stopping_val_days,
                "early_stopping_rounds": settings.early_stopping_rounds,
            }
        )
        # The iteration count early stopping settled on, plus the validation-loss curve
        # that drove the choice (one point per iteration -> a curve in the MLflow UI).
        mlflow.log_metric("best_iteration", best_iter)
        for metric_name, values in val_curve.items():
            for step, value in enumerate(values):
                mlflow.log_metric(f"val_{metric_name}", value, step=step)
        mlflow.log_metric("wape_forward", result.wape["forward"])
        mlflow.log_metric("wape_trailing", result.wape["trailing"])
        mlflow.log_metric("mae_forward", result.mae["forward"])
        mlflow.log_metric("wape_promo_forward", result.wape_promo["forward"])
        mlflow.log_metric("wape_promo_trailing", result.wape_promo["trailing"])
        mlflow.log_metric("improvement_vs_trailing", result.improvement_vs_trailing())
        mlflow.log_metric("promo_improvement_vs_trailing", result.promo_improvement_vs_trailing())


def _save_forecast_metrics(
    settings: Settings, result: BacktestResult, *, best_iteration: int | None = None
) -> None:
    """Persist backtest metrics (incl. the per-horizon forecast CV) for downstream use.

    The ``forecast_cv`` array is what the scenario generator reads to widen its samples
    by the *measured* per-horizon forecast uncertainty. ``best_iteration`` records the
    number of trees early stopping selected for the deployed model.
    """
    payload = {
        "horizon": result.horizon,
        "best_iteration": best_iteration,
        "wape": result.wape,
        "mae": result.mae,
        "wape_promo": result.wape_promo,
        "improvement_vs_trailing": result.improvement_vs_trailing(),
        "promo_improvement_vs_trailing": result.promo_improvement_vs_trailing(),
        "per_horizon_wape": result.per_horizon_wape,
        "forecast_cv": result.forecast_cv,
        "n_origins": result.n_origins,
        "n_rows": result.n_rows,
    }
    path = settings.forecast_metrics_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_and_evaluate(settings: Settings) -> BacktestResult:
    """Backtest the forward forecast vs. the trailing mean, log to MLflow, save the model.

    Returns the backtest result so the caller can present a summary.
    """
    demand = pd.read_csv(settings.demand_path)
    skus = pd.read_csv(settings.skus_path)

    params = build_params(settings.catboost_task_type)
    result = backtest_forecast(
        demand,
        skus,
        holdout_days=settings.backtest_holdout_days,
        params=params,
        val_days=settings.early_stopping_val_days,
        early_stopping_rounds=settings.early_stopping_rounds,
    )

    # Pick the iteration count honestly: early stopping on a temporal validation tail
    # (the most recent `early_stopping_val_days` of history). `iterations` is only a cap.
    features = build_training_frame(demand, skus)
    train_part, val_part = temporal_train_val_split(
        features, val_days=settings.early_stopping_val_days
    )
    es_model = train_model(
        train_part,
        params,
        eval_df=val_part,
        early_stopping_rounds=settings.early_stopping_rounds,
    )
    chosen_iterations = best_iteration(es_model)
    log.info(
        "early_stopping_selected",
        best_iteration=chosen_iterations,
        iterations_cap=params["iterations"],
        val_rows=len(val_part),
    )

    _log_to_mlflow(
        settings, result, best_iter=chosen_iterations, val_curve=validation_curve(es_model)
    )
    _save_forecast_metrics(settings, result, best_iteration=chosen_iterations)

    # Refit on ALL history at the selected iteration count: recent data is the most
    # informative for forecasting, and the validation tail has already done its job.
    log.info("features_built", rows=len(features), skus=skus["sku_id"].nunique())
    final_model = train_model(features, {**params, "iterations": chosen_iterations})
    model_path: Path = settings.model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(model_path))
    log.info("model_saved", path=str(model_path))

    return result
