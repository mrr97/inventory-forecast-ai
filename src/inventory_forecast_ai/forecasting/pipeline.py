"""End-to-end training pipeline: features -> multi-horizon backtest -> MLflow -> model."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
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
    DEFAULT_SEARCH_GRID,
    best_iteration,
    build_params,
    search_hyperparameters,
    train_model,
    validation_curve,
)

log = structlog.get_logger(__name__)


def _search_pre_holdout(
    demand: pd.DataFrame, skus: pd.DataFrame, base_params: dict, *, settings: Settings
) -> tuple[dict, list[dict]]:
    """Run the small temporal grid search on **pre-holdout** data only.

    Selecting depth / learning_rate on data strictly before the backtest window keeps the
    backtest honest: the reported metrics are for a model whose hyperparameters were never
    tuned against the held-out tail they are scored on.
    """
    demand = demand.copy()
    demand["date"] = pd.to_datetime(demand["date"])
    unique_days = np.sort(demand["date"].unique())
    holdout_start = unique_days[len(unique_days) - settings.backtest_holdout_days]
    pre = build_training_frame(demand[demand["date"] < holdout_start], skus)
    pre_train, pre_val = temporal_train_val_split(pre, val_days=settings.early_stopping_val_days)
    return search_hyperparameters(
        pre_train,
        pre_val,
        base_params=base_params,
        early_stopping_rounds=settings.early_stopping_rounds,
    )


def _log_to_mlflow(
    settings: Settings,
    result: BacktestResult,
    *,
    tuned_params: dict,
    best_iter: int,
    val_curve: dict[str, list[float]],
    trials: list[dict],
    training_curve_path: Path,
) -> None:
    """Record params, pooled backtest metrics, the search and the early-stopping selection."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "model": "catboost",
                "loss_function": DEFAULT_PARAMS["loss_function"],
                # `iterations` is only a ceiling; early stopping chooses `best_iteration`.
                "iterations_cap": DEFAULT_PARAMS["iterations"],
                # depth / learning_rate come from the grid search when it ran.
                "depth": tuned_params["depth"],
                "learning_rate": tuned_params["learning_rate"],
                "hyperparameter_search": settings.hyperparameter_search,
                "horizon": result.horizon,
                "backtest_holdout_days": settings.backtest_holdout_days,
                "early_stopping_val_days": settings.early_stopping_val_days,
                "early_stopping_rounds": settings.early_stopping_rounds,
            }
        )
        # The grid search: log every trial's validation loss for the record.
        if trials:
            mlflow.log_dict(
                {"grid": DEFAULT_SEARCH_GRID, "trials": trials}, "hyperparameter_search.json"
            )
        # CatBoost's own training-progress chart for the final model (native fit(plot_file=...)).
        mlflow.log_artifact(str(training_curve_path))
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

    # Optional small temporal grid search over depth / learning_rate on PRE-HOLDOUT data,
    # so the chosen hyperparameters never see the backtest window. They are then used
    # everywhere below (backtest and deployed model) for a coherent set of numbers.
    trials: list[dict] = []
    if settings.hyperparameter_search and settings.early_stopping_val_days > 0:
        best, trials = _search_pre_holdout(demand, skus, params, settings=settings)
        params = {**params, **best["combo"]}  # keep `iterations` as the ceiling
        log.info(
            "hyperparameter_search_done",
            best=best["combo"],
            val_loss=round(best["val_loss"], 5),
            trials=len(trials),
        )

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

    # Refit on ALL history at the selected iteration count: recent data is the most
    # informative for forecasting, and the validation tail has already done its job.
    # CatBoost writes its native training-progress chart straight to HTML (fit(plot_file=...)),
    # which needs file writing enabled and a train_dir; both live under the gitignored models/.
    log.info("features_built", rows=len(features), skus=skus["sku_id"].nunique())
    model_dir: Path = settings.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    training_curve_path = model_dir / "final_training_curve.html"
    final_model = train_model(
        features,
        {
            **params,
            "iterations": chosen_iterations,
            "allow_writing_files": True,
            "train_dir": str(model_dir / "catboost_info"),
        },
        plot_file=str(training_curve_path),
    )

    _log_to_mlflow(
        settings,
        result,
        tuned_params=params,
        best_iter=chosen_iterations,
        val_curve=validation_curve(es_model),
        trials=trials,
        training_curve_path=training_curve_path,
    )
    _save_forecast_metrics(settings, result, best_iteration=chosen_iterations)

    model_path: Path = settings.model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(model_path))
    log.info("model_saved", path=str(model_path))

    return result
