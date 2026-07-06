"""Rolling-origin, multi-horizon backtest of the forward forecast.

A single model is trained on everything before a held-out tail; then the origin is
rolled day by day through the holdout and, at each origin ``t``, the model forecasts
the whole protection window ``t+1 .. t+HORIZON``. Predictions are scored against the
actuals — pooled across origins and SKUs — per horizon. The model never sees the
holdout in training, so the evaluation is honest and out-of-sample.

Two things come out of it:

- **Forward vs. trailing accuracy.** The forward model is compared against the
  "retrovisor" baseline (the trailing 28-day mean, flat across the window). We report
  the overall error and, crucially, the error **on promotion days**, where the trailing
  mean is blind and the forward model should win clearly.
- **Per-horizon forecast uncertainty.** For each horizon we estimate the forecast-error
  spread *after removing the Poisson count noise*, so the scenario generator can widen
  its samples by the **measured** uncertainty (replacing F2's provisional placeholder).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog

from inventory_forecast_ai.forecasting.features import (
    HORIZON,
    TARGET,
    build_forecast_frame,
    build_training_frame,
)
from inventory_forecast_ai.forecasting.metrics import mae, wape
from inventory_forecast_ai.forecasting.model import predict, train_model

log = structlog.get_logger(__name__)


@dataclass
class BacktestResult:
    """Pooled, out-of-sample forward-vs-trailing accuracy plus per-horizon uncertainty."""

    horizon: int
    n_origins: int
    n_rows: int
    wape: dict[str, float]  # {"forward", "trailing"}, pooled over all horizons
    mae: dict[str, float]
    wape_promo: dict[str, float]  # error on promotion days only
    per_horizon_wape: dict[str, list[float]]  # {"forward": [...], "trailing": [...]}
    forecast_cv: list[float]  # per-horizon forecast CV for scenario widening

    def improvement_vs_trailing(self) -> float:
        """Relative overall WAPE reduction of the forward model vs. the trailing mean."""
        base = self.wape["trailing"]
        return float((base - self.wape["forward"]) / base) if base else float("nan")

    def promo_improvement_vs_trailing(self) -> float:
        """Relative WAPE reduction on promotion days (where the trailing mean is blind)."""
        base = self.wape_promo["trailing"]
        return float((base - self.wape_promo["forward"]) / base) if base else float("nan")


def _forecast_cv_by_horizon(
    actual: np.ndarray, pred: np.ndarray, horizon_idx: np.ndarray, horizon: int
) -> list[float]:
    """Per-horizon forecast coefficient of variation, net of Poisson count noise.

    The residual variance mixes two things: the model's forecast error *and* the
    irreducible Poisson noise (variance ~ mean). The scenario sampler already adds the
    Poisson noise itself, so here we subtract it out to avoid double-counting, leaving
    only the forecast-uncertainty component as a multiplicative CV on lambda.
    """
    cv: list[float] = []
    for h in range(1, horizon + 1):
        m = horizon_idx == h
        pred_h = pred[m]
        mean_pred = float(np.mean(pred_h)) if pred_h.size else 0.0
        mse = float(np.mean((actual[m] - pred_h) ** 2)) if pred_h.size else 0.0
        forecast_var = max(mse - mean_pred, 0.0)  # remove Poisson variance (~ mean)
        cv.append(float(np.sqrt(forecast_var) / mean_pred) if mean_pred > 1e-9 else 0.0)
    return cv


def backtest_forecast(
    demand: pd.DataFrame,
    skus: pd.DataFrame,
    *,
    holdout_days: int,
    horizon: int = HORIZON,
    params: dict | None = None,
) -> BacktestResult:
    """Roll the origin through the held-out tail and score the forward path per horizon."""
    demand = demand.copy()
    demand["date"] = pd.to_datetime(demand["date"])
    unique_days = np.sort(demand["date"].unique())
    holdout_start = unique_days[len(unique_days) - holdout_days]

    train_df = build_training_frame(demand[demand["date"] < holdout_start], skus, horizon=horizon)
    model = train_model(train_df, params)

    # Origins are holdout days that still have a full protection window of actuals ahead.
    last_origin = unique_days[len(unique_days) - 1 - horizon]
    origins = [d for d in unique_days if holdout_start <= d <= last_origin]

    actual: list[np.ndarray] = []
    pred_f: list[np.ndarray] = []
    pred_t: list[np.ndarray] = []
    hz: list[np.ndarray] = []
    promo: list[np.ndarray] = []
    for origin in origins:
        fc = build_forecast_frame(demand, skus, origin_date=origin, horizon=horizon)
        actual.append(fc[TARGET].to_numpy(dtype=float))
        pred_f.append(predict(model, fc))
        pred_t.append(fc["roll_mean_28"].to_numpy(dtype=float))  # trailing mean = retrovisor
        hz.append(fc["horizon"].to_numpy(dtype=int))
        promo.append(fc["on_promo"].to_numpy(dtype=bool))

    y = np.concatenate(actual)
    f = np.concatenate(pred_f)
    t = np.concatenate(pred_t)
    h_idx = np.concatenate(hz)
    on_promo = np.concatenate(promo)

    per_h = {
        "forward": [float(wape(y[h_idx == h], f[h_idx == h])) for h in range(1, horizon + 1)],
        "trailing": [float(wape(y[h_idx == h], t[h_idx == h])) for h in range(1, horizon + 1)],
    }
    result = BacktestResult(
        horizon=horizon,
        n_origins=len(origins),
        n_rows=int(y.size),
        wape={"forward": float(wape(y, f)), "trailing": float(wape(y, t))},
        mae={"forward": float(mae(y, f)), "trailing": float(mae(y, t))},
        wape_promo={
            "forward": float(wape(y[on_promo], f[on_promo])),
            "trailing": float(wape(y[on_promo], t[on_promo])),
        },
        per_horizon_wape=per_h,
        forecast_cv=_forecast_cv_by_horizon(y, f, h_idx, horizon),
    )
    log.info(
        "backtest_done",
        origins=result.n_origins,
        rows=result.n_rows,
        wape_forward=round(result.wape["forward"], 4),
        wape_trailing=round(result.wape["trailing"], 4),
        improvement_vs_trailing=round(result.improvement_vs_trailing(), 4),
    )
    return result
