"""Run the optimization end-to-end and persist the Pareto frontier."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import structlog

from inventory_forecast_ai.config import Settings
from inventory_forecast_ai.forecasting.features import HORIZON
from inventory_forecast_ai.optimization.optimize import OptimizationResult, optimize_policies
from inventory_forecast_ai.optimization.scenarios import build_demand_scenarios

log = structlog.get_logger(__name__)


def save_pareto(result: OptimizationResult, path: Path) -> None:
    """Write the frontier, baseline and per-point policies as JSON for the UI."""
    payload = {
        "sku_ids": result.sku_ids,
        "baseline": {
            "cost": result.baseline_cost,
            "stockout": result.baseline_stockout,
            "service": 1.0 - result.baseline_stockout,
        },
        "cost_saving_pct": None if np.isnan(result.cost_saving_pct) else result.cost_saving_pct,
        "points": [
            {
                "cost": float(c),
                "stockout": float(so),
                "service": float(1.0 - so),
                "policy": policy.tolist(),
            }
            for c, so, policy in zip(
                result.pareto_cost,
                result.pareto_stockout,
                result.pareto_policies,
                strict=True,
            )
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _log_to_mlflow(settings: Settings, result: OptimizationResult) -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    with mlflow.start_run(run_name="optimize"):
        mlflow.log_params(
            {
                "optimizer": "NSGA2",
                "pop_size": settings.opt_pop_size,
                "generations": settings.opt_generations,
                "scenarios": settings.opt_scenarios,
                "horizon_days": HORIZON,
                "inventory_budget": settings.inventory_budget,
            }
        )
        mlflow.log_metric("pareto_points", len(result.pareto_cost))
        mlflow.log_metric("baseline_service", 1.0 - result.baseline_stockout)
        mlflow.log_metric("baseline_cost", result.baseline_cost)
        if not np.isnan(result.cost_saving_pct):
            mlflow.log_metric("cost_saving_pct", result.cost_saving_pct)


def run_optimization(settings: Settings) -> OptimizationResult:
    """Build scenarios, run NSGA-II, persist the frontier and log to MLflow."""
    scenarios = build_demand_scenarios(settings)
    result = optimize_policies(
        scenarios,
        pop_size=settings.opt_pop_size,
        generations=settings.opt_generations,
        seed=settings.opt_seed,
        budget=settings.inventory_budget,
    )
    save_pareto(result, settings.pareto_path)
    _log_to_mlflow(settings, result)
    log.info("pareto_saved", path=str(settings.pareto_path))
    return result
