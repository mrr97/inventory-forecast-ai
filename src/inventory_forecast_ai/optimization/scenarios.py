"""Turn the *forward* demand forecast into Monte-Carlo demand scenarios.

The model predicts the expected demand ``lambda`` for each of the next ``HORIZON``
days (the forward path from ``build_forecast_frame``). We sample demand around that
path so the optimizer sees the range of plausible futures it must provision for —
this is what makes the (s, S) policy anticipate promotions and seasonality instead
of chasing the recent past.

Two sources of uncertainty are injected:

1. **Demand randomness given lambda** — the Poisson count noise.
2. **Forecast uncertainty** — ``lambda`` is itself an estimate, and it gets less
   reliable further out. We widen the Poisson mean by a per-horizon factor (a
   Gamma multiplier, giving a Negative-Binomial-style spread) so scenarios are not
   over-confident. The widening grows with the horizon.

   NOTE (F2 provisional): the per-horizon spread is a placeholder that grows like
   ``sqrt(h)``. F4 replaces ``_provisional_forecast_cv`` with the **measured**
   per-horizon forecast error (backtest residuals), which is the honest calibration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog
from catboost import CatBoostRegressor

from inventory_forecast_ai.config import Settings
from inventory_forecast_ai.forecasting.features import HORIZON, build_forecast_frame
from inventory_forecast_ai.forecasting.model import predict_path

log = structlog.get_logger(__name__)

# Fallback base coefficient of variation at horizon 1, used only when no measured
# per-horizon forecast error is available yet (before a backtest has been run).
_PROVISIONAL_CV_BASE: float = 0.15


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


def _provisional_forecast_cv(horizon: int) -> np.ndarray:
    """Fallback per-horizon forecast CV: grows like ``sqrt(h)`` (used only pre-backtest)."""
    h = np.arange(1, horizon + 1, dtype=float)
    return _PROVISIONAL_CV_BASE * np.sqrt(h)


def _load_forecast_cv(settings: Settings, horizon: int) -> np.ndarray:
    """Read the *measured* per-horizon forecast CV from the backtest, or fall back.

    ``train`` persists ``forecast_cv`` (forecast error net of Poisson noise, per horizon)
    in the forecast-metrics artifact. When it is present and the right length we widen
    scenarios by the real uncertainty; otherwise we use the provisional curve.
    """
    path = settings.forecast_metrics_path
    if path.exists():
        cv = json.loads(path.read_text(encoding="utf-8")).get("forecast_cv")
        if cv is not None and len(cv) == horizon:
            return np.asarray(cv, dtype=float)
        log.warning("forecast_cv_missing_or_mismatched", using="provisional")
    return _provisional_forecast_cv(horizon)


def _sample_demand(
    rng: np.random.Generator, lam: np.ndarray, cv: np.ndarray, n_scenarios: int
) -> np.ndarray:
    """Sample widened-Poisson demand scenarios.

    Args:
        lam: (n_sku, horizon) expected demand per day of the forward path.
        cv: (horizon,) forecast coefficient of variation per horizon.
        n_scenarios: number of Monte-Carlo paths.

    Returns:
        (n_sku, n_scenarios, horizon) non-negative integer demand. For each scenario
        we draw a Gamma multiplier with mean 1 and the given per-horizon CV (Gamma-
        Poisson = Negative Binomial), then sample Poisson of the widened mean.
    """
    n_sku, horizon = lam.shape
    lam_b = lam[:, None, :]  # (n_sku, 1, horizon)
    cv_b = np.clip(cv, 1e-6, None)[None, None, :]  # (1, 1, horizon)
    # Gamma factor: mean 1, variance cv**2  ->  shape = 1/cv**2, scale = cv**2.
    shape = 1.0 / cv_b**2
    scale = cv_b**2
    factor = rng.gamma(shape=shape, scale=scale, size=(n_sku, n_scenarios, horizon))
    return rng.poisson(np.clip(lam_b * factor, 0.0, None)).astype(float)


def build_demand_scenarios(
    settings: Settings,
    *,
    horizon: int = HORIZON,
    n_scenarios: int | None = None,
    seed: int | None = None,
    origin_date=None,
) -> DemandScenarios:
    """Load data + model, forecast the forward window, sample widened-Poisson scenarios."""
    n_scenarios = n_scenarios if n_scenarios is not None else settings.opt_scenarios
    seed = seed if seed is not None else settings.opt_seed

    demand_df = pd.read_csv(settings.demand_path)
    skus = pd.read_csv(settings.skus_path).set_index("sku_id")

    model = CatBoostRegressor()
    model.load_model(str(settings.model_path))

    forecast = build_forecast_frame(
        demand_df, skus.reset_index(), origin_date=origin_date, horizon=horizon
    )
    path = predict_path(model, forecast)  # sorted by (sku_id, horizon)

    sku_ids = sorted(path["sku_id"].unique())
    lam = (
        path.pivot(index="sku_id", columns="horizon", values="lam")
        .loc[sku_ids]
        .to_numpy()
    )  # (n_sku, horizon)

    rng = np.random.default_rng(seed)
    cv = _load_forecast_cv(settings, horizon)
    demand = _sample_demand(rng, lam, cv, n_scenarios)

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
        forward_from=str(forecast["date"].min().date()),
    )
    return scenarios
