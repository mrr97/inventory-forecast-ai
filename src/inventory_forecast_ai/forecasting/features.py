"""Feature engineering for one-step-ahead demand forecasting.

The target is next-day units sold. To keep the backtest honest (no leakage in a
one-step-ahead rolling evaluation), every feature is knowable at prediction time:

- **Calendar** features come from the date itself.
- **Promotions / price** are treated as planned and known in advance.
- **Lags and rolling stats** are shifted so they only ever look at the past.

Categorical columns are kept as-is and handed to CatBoost natively.
"""

from __future__ import annotations

import pandas as pd

LAGS: tuple[int, ...] = (1, 7, 14, 28)

NUMERIC_FEATURES: list[str] = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "roll_mean_7",
    "roll_mean_28",
    "roll_std_7",
    "price",
    "day_of_year",
    "week_of_year",
]

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


def build_features(demand: pd.DataFrame, skus: pd.DataFrame) -> pd.DataFrame:
    """Join demand with SKU attributes and engineer the model features.

    Args:
        demand: long-form daily sales (date, sku_id, units_sold, on_promo, price).
        skus: SKU master (must contain sku_id, category).

    Returns:
        A frame with all FEATURES + TARGET, sorted by (sku_id, date), with the
        initial warm-up rows that lack lags dropped.
    """
    df = demand.merge(skus[["sku_id", "category"]], on="sku_id", how="left")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)

    grouped = df.groupby("sku_id", observed=True)["units_sold"]
    for lag in LAGS:
        df[f"lag_{lag}"] = grouped.shift(lag)
    # Rolling stats use shift(1) so the current day never leaks into its own feature.
    df["roll_mean_7"] = grouped.transform(lambda s: s.shift(1).rolling(7).mean())
    df["roll_mean_28"] = grouped.transform(lambda s: s.shift(1).rolling(28).mean())
    df["roll_std_7"] = grouped.transform(lambda s: s.shift(1).rolling(7).std())

    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = df["dow"] >= 5

    # Drop warm-up rows that don't have the longest lag / rolling window yet.
    df = df.dropna(subset=["lag_28", "roll_mean_28", "roll_std_7"]).reset_index(drop=True)
    return df
