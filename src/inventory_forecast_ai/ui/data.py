"""Data loaders for the dashboard (kept out of the Dash layout for testability)."""

from __future__ import annotations

import json

import pandas as pd
from catboost import CatBoostRegressor

from inventory_forecast_ai.config import Settings
from inventory_forecast_ai.forecasting.features import build_features
from inventory_forecast_ai.forecasting.model import predict


def load_forecast_frame(settings: Settings, *, tail_days: int = 90) -> pd.DataFrame:
    """Return recent actual-vs-predicted demand per SKU for the forecast view."""
    demand = pd.read_csv(settings.demand_path)
    skus = pd.read_csv(settings.skus_path)
    feats = build_features(demand, skus)

    model = CatBoostRegressor()
    model.load_model(str(settings.model_path))
    feats = feats.assign(predicted=predict(model, feats), actual=feats["units_sold"])

    recent = feats.sort_values("date").groupby("sku_id", observed=True).tail(tail_days)
    return recent[["date", "sku_id", "actual", "predicted"]].reset_index(drop=True)


def load_pareto(settings: Settings) -> dict | None:
    """Load the saved Pareto frontier, or None if `optimize` has not run yet."""
    path = settings.pareto_path
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_report(settings: Settings) -> dict | None:
    """Load the consolidated scorecard, or None if `metrics` has not run yet."""
    path = settings.report_path
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
