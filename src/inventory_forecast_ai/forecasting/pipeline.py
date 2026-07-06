"""End-to-end training pipeline: features -> multi-horizon backtest -> MLflow -> model."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
import structlog

from inventory_forecast_ai.config import Settings
from inventory_forecast_ai.forecasting.backtest import BacktestResult, backtest_forecast
from inventory_forecast_ai.forecasting.features import build_training_frame
from inventory_forecast_ai.forecasting.model import DEFAULT_PARAMS, build_params, train_model

log = structlog.get_logger(__name__)


def _log_to_mlflow(settings: Settings, result: BacktestResult) -> None:
    """Record params and pooled backtest metrics in the local MLflow store."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "model": "catboost",
                "loss_function": DEFAULT_PARAMS["loss_function"],
                "iterations": DEFAULT_PARAMS["iterations"],
                "depth": DEFAULT_PARAMS["depth"],
                "learning_rate": DEFAULT_PARAMS["learning_rate"],
                "horizon": result.horizon,
                "backtest_holdout_days": settings.backtest_holdout_days,
            }
        )
        mlflow.log_metric("wape_forward", result.wape["forward"])
        mlflow.log_metric("wape_trailing", result.wape["trailing"])
        mlflow.log_metric("mae_forward", result.mae["forward"])
        mlflow.log_metric("wape_promo_forward", result.wape_promo["forward"])
        mlflow.log_metric("wape_promo_trailing", result.wape_promo["trailing"])
        mlflow.log_metric("improvement_vs_trailing", result.improvement_vs_trailing())
        mlflow.log_metric("promo_improvement_vs_trailing", result.promo_improvement_vs_trailing())


def _save_forecast_metrics(settings: Settings, result: BacktestResult) -> None:
    """Persist backtest metrics (incl. the per-horizon forecast CV) for downstream use.

    The ``forecast_cv`` array is what the scenario generator reads to widen its samples
    by the *measured* per-horizon forecast uncertainty.
    """
    payload = {
        "horizon": result.horizon,
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
        demand, skus, holdout_days=settings.backtest_holdout_days, params=params
    )
    _log_to_mlflow(settings, result)
    _save_forecast_metrics(settings, result)

    # Final model trained on all available history, for downstream optimization.
    features = build_training_frame(demand, skus)
    log.info("features_built", rows=len(features), skus=skus["sku_id"].nunique())
    final_model = train_model(features, params)
    model_path: Path = settings.model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(model_path))
    log.info("model_saved", path=str(model_path))

    return result
