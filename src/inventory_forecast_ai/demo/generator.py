"""Deterministic synthetic demand generator.

Produces a small, reproducible retail dataset: a SKU master table plus two years
of daily unit sales. The demand signal is intentionally *learnable* — it combines
weekly and yearly seasonality, a mild trend, and price promotions — wrapped in
Poisson noise so a model has something real to capture above a naive baseline.

Everything is driven by a single seed, so the same seed always yields byte-for-byte
identical CSVs (verified in tests).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import structlog

from inventory_forecast_ai.models import Category, Sku

log = structlog.get_logger(__name__)

# Category -> (yearly peak day-of-year, yearly amplitude). Amplitude is the
# fractional swing around the mean (0.3 = +/-30% across the year).
_SEASONALITY: dict[Category, tuple[int, float]] = {
    Category.BEVERAGES: (200, 0.35),  # summer peak
    Category.SNACKS: (340, 0.20),  # mild end-of-year peak
    Category.HOUSEHOLD: (15, 0.10),  # fairly flat
    Category.PERSONAL_CARE: (60, 0.15),
    Category.STATIONERY: (250, 0.40),  # back-to-school (early September)
}

_CATEGORY_NOUNS: dict[Category, list[str]] = {
    Category.BEVERAGES: ["Cola", "Water", "Juice", "Iced Tea", "Energy Drink", "Coffee"],
    Category.SNACKS: ["Chips", "Cookies", "Nuts", "Crackers", "Chocolate", "Popcorn"],
    Category.HOUSEHOLD: ["Detergent", "Bleach", "Sponges", "Trash Bags", "Wipes"],
    Category.PERSONAL_CARE: ["Shampoo", "Soap", "Toothpaste", "Lotion", "Razors"],
    Category.STATIONERY: ["Notebook", "Pens", "Markers", "Folders", "Glue"],
}


def _make_skus(rng: np.random.Generator, count: int) -> list[Sku]:
    """Draw `count` SKUs with reproducible economics and demand levels."""
    categories = list(Category)
    skus: list[Sku] = []
    for i in range(count):
        category = categories[i % len(categories)]
        noun = rng.choice(_CATEGORY_NOUNS[category])
        unit_cost = round(float(rng.uniform(0.5, 12.0)), 2)
        markup = float(rng.uniform(1.3, 2.2))
        price = round(unit_cost * markup, 2)
        skus.append(
            Sku(
                sku_id=f"SKU-{1001 + i:04d}",
                name=f"{noun} {category.value.replace('_', ' ').title()} #{1001 + i}",
                category=category,
                unit_cost=unit_cost,
                price=price,
                # Holding cost ~ a small daily fraction of the unit cost.
                holding_cost_per_unit_per_day=round(
                    unit_cost * float(rng.uniform(0.001, 0.004)), 4
                ),
                order_cost=round(float(rng.uniform(20.0, 80.0)), 2),
                lead_time_days=int(rng.integers(2, 8)),
                min_order_qty=int(rng.choice([0, 0, 10, 20, 50])),
                base_demand=round(float(rng.uniform(5.0, 60.0)), 2),
            )
        )
    return skus


def _daily_demand(
    rng: np.random.Generator,
    sku: Sku,
    dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (units, on_promo, price) arrays for one SKU over `dates`."""
    n = len(dates)
    day_of_week = dates.dayofweek.to_numpy()  # 0=Mon .. 6=Sun
    day_of_year = dates.dayofyear.to_numpy()
    t = np.arange(n)

    # Weekly seasonality: per-SKU multipliers, retail tends to peak on weekends.
    dow_factor = rng.uniform(0.85, 1.15, size=7)
    dow_factor[5] *= 1.25  # Saturday
    dow_factor[6] *= 1.15  # Sunday
    weekly = dow_factor[day_of_week]

    # Yearly seasonality from the SKU's category.
    peak_doy, amplitude = _SEASONALITY[sku.category]
    yearly = 1.0 + amplitude * np.cos(2.0 * np.pi * (day_of_year - peak_doy) / 365.0)

    # Mild linear trend over the whole horizon (+/- a few percent per year).
    trend = 1.0 + float(rng.uniform(-0.05, 0.15)) * (t / max(n - 1, 1))

    # Promotions: a handful of multi-day events with a price cut and demand uplift.
    on_promo = np.zeros(n, dtype=bool)
    n_events = int(rng.integers(6, 14))
    for _ in range(n_events):
        start = int(rng.integers(0, n))
        length = int(rng.integers(3, 11))
        on_promo[start : start + length] = True
    promo_uplift = np.where(on_promo, float(rng.uniform(1.4, 2.4)), 1.0)
    # Price drops ~15-30% during promotions.
    promo_discount = float(rng.uniform(0.15, 0.30))
    price = np.where(on_promo, round(sku.price * (1 - promo_discount), 2), sku.price)

    mean = sku.base_demand * weekly * yearly * trend * promo_uplift
    units = rng.poisson(np.clip(mean, 0.1, None)).astype(int)
    return units, on_promo, price


def generate_demo_data(
    skus_path,
    demand_path,
    *,
    count: int,
    seed: int,
    start_date: date,
    end_date: date,
) -> tuple[int, int]:
    """Generate the SKU master and daily demand CSVs.

    Returns:
        (n_skus, n_demand_rows) actually written.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date, freq="D")

    skus = _make_skus(rng, count)
    sku_records = [s.model_dump() for s in skus]
    sku_df = pd.DataFrame(sku_records)

    frames: list[pd.DataFrame] = []
    for sku in skus:
        units, on_promo, price = _daily_demand(rng, sku, dates)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "sku_id": sku.sku_id,
                    "units_sold": units,
                    "on_promo": on_promo,
                    "price": price,
                }
            )
        )
    demand_df = pd.concat(frames, ignore_index=True)
    demand_df["date"] = demand_df["date"].dt.strftime("%Y-%m-%d")

    skus_path.parent.mkdir(parents=True, exist_ok=True)
    demand_path.parent.mkdir(parents=True, exist_ok=True)
    sku_df.to_csv(skus_path, index=False)
    demand_df.to_csv(demand_path, index=False)

    log.info(
        "generated_demo_data",
        skus=len(sku_df),
        demand_rows=len(demand_df),
        date_range=f"{start_date.isoformat()}..{end_date.isoformat()}",
    )
    return len(sku_df), len(demand_df)
