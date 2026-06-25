"""Turn the demand forecast into Monte-Carlo demand scenarios.

The CatBoost model was trained with a Poisson loss, so its prediction for a given
day is the expected demand ``lambda``. We sample ``Poisson(lambda)`` to obtain a
distribution of plausible demand paths over the planning horizon — exactly the
uncertainty the optimizer needs to weigh service against cost. Using Poisson here
is consistent with the model that produced ``lambda``.

The horizon is the most recent ``horizon_days`` of history per SKU: a representative
demand period whose model means drive the scenarios (we are generating plausible
demand, not measuring forecast accuracy, so this is leakage-free by construction).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog
from catboost import CatBoostRegressor

from inventory_forecast_ai.config import Settings
from inventory_forecast_ai.forecasting.features import build_features
from inventory_forecast_ai.forecasting.model import predict

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class DemandScenarios:
    """Sampled demand plus the per-SKU economics the simulator needs."""

    sku_ids: list[str]
    demand: np.ndarray  # (n_sku, n_scenarios, horizon), integer units
    mean_daily: np.ndarray  # (n_sku,) average forecast demand, for policy bounds
    lead_time: np.ndarray  # (n_sku,) int days
    holding: np.ndarray  # (n_sku,) EUR / unit / day
    order_cost: np.ndarray  # (n_sku,) EUR / order
    unit_cost: np.ndarray  # (n_sku,) EUR / unit
    min_order_qty: np.ndarray  # (n_sku,) units

    @property
    def n_sku(self) -> int:
        return len(self.sku_ids)


def build_demand_scenarios(
    settings: Settings,
    *,
    horizon: int | None = None,
    n_scenarios: int | None = None,
    seed: int | None = None,
) -> DemandScenarios:
    """Load data + model, forecast the planning horizon, sample Poisson scenarios."""
    horizon = horizon if horizon is not None else settings.opt_horizon_days
    n_scenarios = n_scenarios if n_scenarios is not None else settings.opt_scenarios
    seed = seed if seed is not None else settings.opt_seed

    demand_df = pd.read_csv(settings.demand_path)
    skus = pd.read_csv(settings.skus_path).set_index("sku_id")

    features = build_features(demand_df, skus.reset_index())
    model = CatBoostRegressor()
    model.load_model(str(settings.model_path))
    features = features.assign(lam=predict(model, features))

    # Last `horizon` days per SKU form the planning window.
    window = (
        features.sort_values("date")
        .groupby("sku_id", observed=True)
        .tail(horizon)
        .sort_values(["sku_id", "date"])
    )
    sku_ids = sorted(window["sku_id"].unique())
    lam = np.stack([window.loc[window["sku_id"] == s, "lam"].to_numpy() for s in sku_ids])

    rng = np.random.default_rng(seed)
    demand = rng.poisson(lam[:, None, :].repeat(n_scenarios, axis=1)).astype(float)

    econ = skus.loc[sku_ids]
    scenarios = DemandScenarios(
        sku_ids=sku_ids,
        demand=demand,
        mean_daily=lam.mean(axis=1),
        lead_time=econ["lead_time_days"].to_numpy().astype(int),
        holding=econ["holding_cost_per_unit_per_day"].to_numpy(),
        order_cost=econ["order_cost"].to_numpy(),
        unit_cost=econ["unit_cost"].to_numpy(),
        min_order_qty=econ["min_order_qty"].to_numpy().astype(float),
    )
    log.info(
        "demand_scenarios_built",
        skus=scenarios.n_sku,
        scenarios=n_scenarios,
        horizon=horizon,
    )
    return scenarios
