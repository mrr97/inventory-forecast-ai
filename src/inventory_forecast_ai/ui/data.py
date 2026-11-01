"""Data loaders for the dashboard (kept out of the Dash layout for testability)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from inventory_forecast_ai.config import Settings
from inventory_forecast_ai.forecasting.features import TARGET, build_forecast_frame
from inventory_forecast_ai.forecasting.model import predict_path


def load_forecast_frame(settings: Settings, *, history_days: int = 60) -> pd.DataFrame:
    """Return recent history plus the forward forecast path per SKU for the chart.

    Each SKU gets ``history_days`` of recent actuals (``forecast`` NaN) followed by the
    model's forward path over the protection window, where ``actual`` is what really
    happened and ``forecast`` is the predicted demand. Plotting both shows the model
    projecting *forward*, not fitting the past.
    """
    demand = pd.read_csv(settings.demand_path)
    demand["date"] = pd.to_datetime(demand["date"])
    skus = pd.read_csv(settings.skus_path)

    model = CatBoostRegressor()
    model.load_model(str(settings.model_path))
    path = predict_path(model, build_forecast_frame(demand, skus))

    frames: list[pd.DataFrame] = []
    for sku_id, fc in path.groupby("sku_id", observed=True):
        window_start = fc["date"].min()
        hist = (
            demand[(demand["sku_id"] == sku_id) & (demand["date"] < window_start)]
            .sort_values("date")
            .tail(history_days)
        )
        frames.append(
            pd.DataFrame(
                {
                    "date": hist["date"],
                    "sku_id": sku_id,
                    "actual": hist["units_sold"].astype(float),
                    "forecast": np.nan,
                }
            )
        )
        frames.append(
            pd.DataFrame(
                {
                    "date": fc["date"],
                    "sku_id": sku_id,
                    "actual": fc[TARGET].astype(float),
                    "forecast": fc["lam"].to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def load_pareto(settings: Settings) -> dict | None:
    """Load the saved Pareto frontier, or None if `optimize` has not run yet."""
    path = settings.pareto_path
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_product_names(settings: Settings) -> dict[str, str]:
    """Map saved SKU identifiers to their display names."""
    skus = pd.read_csv(settings.skus_path)
    return dict(zip(skus["sku_id"], skus["name"], strict=True))


def load_report(settings: Settings) -> dict | None:
    """Load the consolidated scorecard, or None if `metrics` has not run yet."""
    path = settings.report_path
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
