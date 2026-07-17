"""Feature engineering for multi-horizon demand forecasting.

The model predicts demand up to ``HORIZON`` days ahead with a **single** global
model that takes the **horizon ``h`` as an input feature** (the "direct" approach
with the horizon as a variable). To forecast a day, we query the same model once
per horizon ``h = 1 .. HORIZON``; no recursion, so forecast error does not compound.

Every training row pairs two time references that must never be mixed:

- **Origin features** (``ORIGIN_FEATURES``): recent demand levels known *at the
  moment we forecast* (origin day ``t``) — lagged demand and rolling stats up to
  and including ``t``. These describe the state at decision time.
- **Target-day features**: everything knowable in advance about the day being
  predicted (``t + h``) — calendar, and the *planned* promotion / price. Treating
  promotions and price as known ahead is what lets the forecast anticipate them.

The target is the units sold on the target day ``t + h``. Keeping the origin
features strictly at ``t`` and only pulling *plannable* covariates from ``t + h``
keeps the setup leak-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Forecast horizon in days = protection interval (weekly review + max lead time).
# Fixed for all SKUs: simpler to build/explain, and a superset of every SKU's own
# protection interval, so nothing is lost (each SKU consumes only the slice it needs
# downstream, and the simulator already respects per-SKU lead time).
HORIZON: int = 14

# Recent-demand features anchored at the origin day t (knowable when we forecast).
ORIGIN_FEATURES: list[str] = [
    "recent_demand",  # units sold on the origin day t
    "recent_demand_7d",  # units sold at t-7 (same weekday, a week earlier)
    "recent_demand_14d",  # t-14
    "recent_demand_28d",  # t-28
    "roll_mean_7",  # mean units over the 7 days ending at t
    "roll_mean_28",  # mean units over the 28 days ending at t
    "roll_std_7",  # std of units over the 7 days ending at t
]

# Numeric features describing the target day t+h (all knowable in advance).
TARGET_NUMERIC: list[str] = ["day_of_year", "week_of_year", "price"]

NUMERIC_FEATURES: list[str] = ["horizon", *ORIGIN_FEATURES, *TARGET_NUMERIC]

# Categorical features: sku_id / category are static; the rest describe the target day.
CATEGORICAL_FEATURES: list[str] = [
    "sku_id",
    "category",
    "dow",
    "month",
    "is_weekend",
    "on_promo",
]

FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "units_sold"

# Target-day covariate columns copied from the target row when expanding horizons.
_TARGET_COVARIATES: list[str] = [
    "dow",
    "month",
    "is_weekend",
    "day_of_year",
    "week_of_year",
    "on_promo",
    "price",
]


def _prepare(demand: pd.DataFrame, skus: pd.DataFrame) -> pd.DataFrame:
    """Join demand with SKU attributes and compute per-day origin + calendar columns.

    Every row is annotated *both* as a potential origin (recent-demand features,
    as-of that day) and as a potential target (its own calendar / promo / price).
    The horizon expansion later shifts the origin features onto the target rows.
    """
    df = demand.merge(skus[["sku_id", "category"]], on="sku_id", how="left")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)

    grouped = df.groupby("sku_id", observed=True)["units_sold"]
    # Origin features are as-of day t (inclusive): the target is t+h with h>=1, so
    # including day t's own demand is not leakage.
    df["recent_demand"] = df["units_sold"]
    df["recent_demand_7d"] = grouped.shift(7)
    df["recent_demand_14d"] = grouped.shift(14)
    df["recent_demand_28d"] = grouped.shift(28)
    df["roll_mean_7"] = grouped.transform(lambda s: s.rolling(7).mean())
    df["roll_mean_28"] = grouped.transform(lambda s: s.rolling(28).mean())
    df["roll_std_7"] = grouped.transform(lambda s: s.rolling(7).std())

    # Calendar features of the (target) day.
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = df["dow"] >= 5
    return df


def build_training_frame(
    demand: pd.DataFrame, skus: pd.DataFrame, *, horizon: int = HORIZON
) -> pd.DataFrame:
    """Expand the daily history into multi-horizon training rows.

    For every SKU and every horizon ``h = 1 .. horizon`` a target day ``t+h`` is
    paired with the origin features from ``t`` (``origin.shift(h)``). Warm-up rows
    whose origin lacks the longest lag / rolling window are dropped.

    Returns a frame with all ``FEATURES`` + ``TARGET`` (plus ``date`` = target day).
    """
    base = _prepare(demand, skus)
    frames: list[pd.DataFrame] = []

    for sku_id, g in base.groupby("sku_id", observed=True, sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        origin = g[ORIGIN_FEATURES]
        for h in range(1, horizon + 1):
            shifted = origin.shift(h)  # origin features from h days before the target
            block = g[["date", "category", *_TARGET_COVARIATES, TARGET]].copy()
            block["sku_id"] = sku_id
            block["horizon"] = h
            for col in ORIGIN_FEATURES:
                block[col] = shifted[col].to_numpy()
            frames.append(block)

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=ORIGIN_FEATURES).reset_index(drop=True)
    return out


def temporal_train_val_split(
    frame: pd.DataFrame, *, val_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split multi-horizon rows by **target date**: the last ``val_days`` days = validation.

    Used to feed CatBoost early stopping an honest, out-of-time validation set. Because
    the split is on the target day ``t+h``, no target day in the validation tail appears
    in the training rows — the model is validated strictly on the future, mirroring how it
    is used. Rows whose target lands on/after the cutoff go to validation; the rest train.
    """
    dates = np.sort(frame["date"].unique())
    if len(dates) <= val_days:
        raise ValueError(
            f"Not enough distinct target days ({len(dates)}) for a {val_days}-day "
            "validation tail; lower early_stopping_val_days or add history."
        )
    cutoff = dates[len(dates) - val_days]
    train = frame[frame["date"] < cutoff].reset_index(drop=True)
    val = frame[frame["date"] >= cutoff].reset_index(drop=True)
    return train, val


def build_forecast_frame(
    demand: pd.DataFrame,
    skus: pd.DataFrame,
    *,
    origin_date=None,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """Build the forward path (``horizon`` rows per SKU) from a single origin.

    Origin features are taken as-of ``origin_date`` (default: the latest day that
    still leaves ``horizon`` days of runway in ``demand``, so target days exist for
    evaluation). Target-day covariates for ``t+1 .. t+horizon`` are read from
    ``demand`` — in production these rows carry the *planned* promo/price (and a
    missing ``units_sold``); here they come from the data so the path can be scored.

    Returns a frame with all ``FEATURES`` (plus ``date`` and ``TARGET`` for scoring).
    """
    base = _prepare(demand, skus)
    rows: list[dict] = []

    for sku_id, g in base.groupby("sku_id", observed=True, sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        if origin_date is None:
            oi = len(g) - 1 - horizon
        else:
            match = g.index[g["date"] == pd.Timestamp(origin_date)]
            if len(match) == 0:
                raise ValueError(f"origin_date {origin_date!r} not found for SKU {sku_id}")
            oi = int(match[0])
        if oi < 0:
            raise ValueError(f"Not enough history before the origin for SKU {sku_id}")

        origin_feats = g.loc[oi, ORIGIN_FEATURES]
        if origin_feats.isna().any():
            raise ValueError(f"Origin day lacks warm-up history for SKU {sku_id}")

        for h in range(1, horizon + 1):
            ti = oi + h
            if ti >= len(g):
                raise ValueError(
                    f"Missing target-day covariates for horizon {h} on SKU {sku_id}: "
                    "append the planned promo/price rows for the future window."
                )
            tgt = g.loc[ti]
            row = {
                "sku_id": sku_id,
                "category": tgt["category"],
                "horizon": h,
                "date": tgt["date"],
                TARGET: tgt[TARGET],
            }
            for col in _TARGET_COVARIATES:
                row[col] = tgt[col]
            for col in ORIGIN_FEATURES:
                row[col] = origin_feats[col]
            rows.append(row)

    return pd.DataFrame(rows)
