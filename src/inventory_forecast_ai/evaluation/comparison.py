"""Decision-level proof: forward forecast vs. trailing mean on a seasonal stretch.

The forecast-accuracy backtest shows the forward model is *more accurate*; this shows
that accuracy turns into *better inventory decisions*. On the most seasonal stretch of
the data (the steepest demand ramp), we roll the origin weekly and, at each origin,
build the **same textbook (s, S) safety-stock policy** two ways:

- **forward** — fed the model's forward forecast for the protection window,
- **trailing** — fed the flat trailing 28-day mean (the retrovisor).

Both policies are then simulated against the **actual** demand that followed, and we
report the realized operating cost and service. Isolating a single policy formula (only
the demand estimate differs) is the cleanest apples-to-apples: it attributes the whole
gap to *looking forward vs. looking back*. The model is trained only on data before the
stretch, so the evaluation is out-of-sample.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import structlog

from inventory_forecast_ai.config import Settings
from inventory_forecast_ai.forecasting.features import (
    HORIZON,
    TARGET,
    build_forecast_frame,
    build_training_frame,
    temporal_train_val_split,
)
from inventory_forecast_ai.forecasting.model import build_params, predict_path, train_model
from inventory_forecast_ai.optimization.optimize import baseline_policy
from inventory_forecast_ai.optimization.scenarios import DemandScenarios
from inventory_forecast_ai.optimization.simulator import simulate_policies

log = structlog.get_logger(__name__)


@dataclass
class ComparisonResult:
    """Realized forward-vs-trailing outcome on the seasonal stretch.

    The headline is *total economic cost* = operating cost (holding + ordering) plus the
    **cost of stockouts** (lost sales valued at unit margin). Operating cost alone is
    misleading: understocking looks cheap precisely because it fails to serve demand, so
    the lost-margin term is what makes the comparison honest (the newsvendor view).
    """

    window_start: str
    window_end: str
    n_origins: int
    forward_cost: float  # total economic cost (operating + lost margin)
    trailing_cost: float
    forward_operating: float
    trailing_operating: float
    forward_service: float
    trailing_service: float
    per_origin: list[dict]

    @property
    def cost_reduction(self) -> float:
        """Fractional realized total-cost reduction of forward vs. trailing."""
        return float((self.trailing_cost - self.forward_cost) / self.trailing_cost)

    @property
    def service_gain(self) -> float:
        """Percentage-point service gain of forward vs. trailing."""
        return float(self.forward_service - self.trailing_service)


def _pick_seasonal_window(
    dates: np.ndarray, totals: np.ndarray, window_days: int, min_history: int
) -> int:
    """Return the origin index of the steepest demand ramp (most seasonal stretch)."""
    n = len(dates)
    best_i, best_slope = min_history, -1.0
    for i in range(min_history, n - window_days - HORIZON):
        head = totals[i : i + 14].mean()
        tail = totals[i + window_days - 14 : i + window_days].mean()
        slope = abs(tail - head)
        if slope > best_slope:
            best_i, best_slope = i, slope
    return best_i


def _matrix(frame: pd.DataFrame, sku_ids: list[str], col: str) -> np.ndarray:
    """Pivot a forecast/actual frame column into a (n_sku, horizon) matrix."""
    return frame.pivot(index="sku_id", columns="horizon", values=col).loc[sku_ids].to_numpy()


def _scenarios(
    sku_ids: list[str], econ: pd.DataFrame, demand: np.ndarray, mean_daily: np.ndarray
) -> DemandScenarios:
    """Assemble a DemandScenarios (demand shape (n_sku, n_scen, horizon))."""
    return DemandScenarios(
        sku_ids=sku_ids,
        demand=demand,
        mean_daily=mean_daily,
        lead_time=econ["lead_time_days"].to_numpy().astype(int),
        holding=econ["holding_cost_per_unit_per_day"].to_numpy(),
        order_cost=econ["order_cost"].to_numpy(),
        unit_cost=econ["unit_cost"].to_numpy(),
        min_order_qty=econ["min_order_qty"].to_numpy().astype(float),
    )


def compare_forward_vs_trailing(
    demand: pd.DataFrame,
    skus: pd.DataFrame,
    *,
    window_days: int = 60,
    step_days: int = 7,
    min_history: int = 400,
    params: dict | None = None,
    val_days: int = 0,
    early_stopping_rounds: int = 50,
) -> ComparisonResult:
    """Roll the origin through the most seasonal stretch and compare realized outcomes."""
    demand = demand.copy()
    demand["date"] = pd.to_datetime(demand["date"])
    dates = np.sort(demand["date"].unique())
    totals = demand.groupby("date")["units_sold"].sum().sort_index().to_numpy()

    start_i = _pick_seasonal_window(dates, totals, window_days, min_history)
    window_start = dates[start_i]

    # Honest OOS: train only on data before the stretch. Early-stop like the deployed
    # model when a validation tail is requested, so the compared forecast matches it.
    train_df = build_training_frame(demand[demand["date"] < window_start], skus)
    p = params or build_params("CPU")
    if val_days > 0:
        tr, val = temporal_train_val_split(train_df, val_days=val_days)
        model = train_model(tr, p, eval_df=val, early_stopping_rounds=early_stopping_rounds)
    else:
        model = train_model(train_df, p)

    econ = skus.set_index("sku_id")
    origin_idxs = range(start_i, start_i + window_days, step_days)
    per_origin: list[dict] = []

    for oi in origin_idxs:
        if oi + HORIZON >= len(dates):
            break
        origin = dates[oi]
        fc = predict_path(model, build_forecast_frame(demand, skus, origin_date=origin))
        sku_ids = sorted(fc["sku_id"].unique())
        econ_o = econ.loc[sku_ids]

        lam = _matrix(fc, sku_ids, "lam")  # (n_sku, H) forward forecast
        actual = _matrix(fc, sku_ids, TARGET)  # (n_sku, H) what really happened
        trailing = _matrix(fc, sku_ids, "roll_mean_28")[:, 0]  # flat trailing mean per SKU

        # Same (s, S) safety-stock rule, fed forward vs. trailing demand estimates.
        fwd_pol = _scenarios(sku_ids, econ_o, actual[:, None, :], lam.mean(axis=1))
        trail_pol = _scenarios(sku_ids, econ_o, actual[:, None, :], trailing)
        s_f, big_s_f = baseline_policy(fwd_pol)
        s_t, big_s_t = baseline_policy(trail_pol)

        # Simulate both against the ACTUAL demand that followed.
        actual_scn = _scenarios(sku_ids, econ_o, actual[:, None, :], lam.mean(axis=1))
        m_f = simulate_policies(s_f[None, :], big_s_f[None, :], actual_scn)
        m_t = simulate_policies(s_t[None, :], big_s_t[None, :], actual_scn)

        # Value stockouts at the demand-weighted unit margin (lost sales = lost margin).
        total_demand = float(actual.sum())
        demand_by_sku = actual.sum(axis=1)
        margin = (econ_o["price"] - econ_o["unit_cost"]).to_numpy()
        avg_margin = float(np.average(margin, weights=demand_by_sku)) if total_demand else 0.0
        op_f, op_t = float(m_f["operating_cost"][0]), float(m_t["operating_cost"][0])
        svc_f = float(1.0 - m_f["stockout_rate"][0])
        svc_t = float(1.0 - m_t["stockout_rate"][0])
        lost_cost_f = (1.0 - svc_f) * total_demand * avg_margin
        lost_cost_t = (1.0 - svc_t) * total_demand * avg_margin

        per_origin.append(
            {
                "origin": str(pd.Timestamp(origin).date()),
                "forward_cost": op_f + lost_cost_f,
                "trailing_cost": op_t + lost_cost_t,
                "forward_operating": op_f,
                "trailing_operating": op_t,
                "forward_service": svc_f,
                "trailing_service": svc_t,
            }
        )

    def _avg(key: str) -> float:
        return float(np.mean([p[key] for p in per_origin]))

    result = ComparisonResult(
        window_start=str(pd.Timestamp(window_start).date()),
        window_end=str(pd.Timestamp(dates[start_i + window_days]).date()),
        n_origins=len(per_origin),
        forward_cost=_avg("forward_cost"),
        trailing_cost=_avg("trailing_cost"),
        forward_operating=_avg("forward_operating"),
        trailing_operating=_avg("trailing_operating"),
        forward_service=_avg("forward_service"),
        trailing_service=_avg("trailing_service"),
        per_origin=per_origin,
    )
    log.info(
        "forward_vs_trailing_done",
        window=f"{result.window_start}..{result.window_end}",
        origins=result.n_origins,
        cost_reduction=round(result.cost_reduction, 4),
        service_gain=round(result.service_gain, 4),
    )
    return result


def save_comparison(result: ComparisonResult, path) -> None:
    """Persist the comparison (incl. derived reductions) as JSON for the scorecard/UI."""
    payload = asdict(result)
    payload["cost_reduction"] = result.cost_reduction
    payload["service_gain"] = result.service_gain
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_comparison(settings: Settings) -> ComparisonResult:
    """Run the seasonal-stretch comparison end-to-end and persist it."""
    demand = pd.read_csv(settings.demand_path)
    skus = pd.read_csv(settings.skus_path)
    result = compare_forward_vs_trailing(
        demand,
        skus,
        params=build_params(settings.catboost_task_type),
        val_days=settings.early_stopping_val_days,
        early_stopping_rounds=settings.early_stopping_rounds,
    )
    save_comparison(result, settings.comparison_path)
    log.info("comparison_saved", path=str(settings.comparison_path))
    return result
