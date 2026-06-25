"""End-to-end training pipeline: features -> backtest -> MLflow -> saved model."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
import structlog

from inventory_forecast_ai.config import Settings
from inventory_forecast_ai.forecasting.backtest import BacktestResult, walk_forward
from inventory_forecast_ai.forecasting.features import build_features
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
                "backtest_folds": settings.backtest_folds,
                "backtest_fold_days": settings.backtest_fold_days,
            }
        )
        for name in result.wape:
            mlflow.log_metric(f"{name}_wape", result.wape[name])
            mlflow.log_metric(f"{name}_mae", result.mae[name])
        mlflow.log_metric("wape_improvement_vs_naive", result.wape_improvement_vs("naive"))
        mlflow.log_metric(
            "wape_improvement_vs_seasonal_naive",
            result.wape_improvement_vs("seasonal_naive"),
        )


def train_and_evaluate(settings: Settings) -> BacktestResult:
    """Build features, backtest CatBoost vs. baselines, log to MLflow, save the model.

    Returns the backtest result so the caller can present a summary.
    """
    demand = pd.read_csv(settings.demand_path)
    skus = pd.read_csv(settings.skus_path)
    features = build_features(demand, skus)
    log.info("features_built", rows=len(features), skus=skus["sku_id"].nunique())

    params = build_params(settings.catboost_task_type)
    result = walk_forward(
        features,
        folds=settings.backtest_folds,
        fold_days=settings.backtest_fold_days,
        params=params,
    )
    _log_to_mlflow(settings, result)

    # Final model trained on all available history, for downstream optimization (F3).
    final_model = train_model(features, params)
    model_path: Path = settings.model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(model_path))
    log.info("model_saved", path=str(model_path))

    return result
