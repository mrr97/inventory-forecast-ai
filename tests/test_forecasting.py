"""Tests for feature engineering, metrics and the walk-forward backtest."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from inventory_forecast_ai.demo.generator import generate_demo_data
from inventory_forecast_ai.forecasting.backtest import walk_forward
from inventory_forecast_ai.forecasting.features import (
    FEATURES,
    TARGET,
    build_features,
)
from inventory_forecast_ai.forecasting.metrics import mae, wape


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small but multi-month dataset so lags/rolling windows are populated."""
    d = tmp_path_factory.mktemp("data")
    generate_demo_data(
        d / "skus.csv",
        d / "demand.csv",
        count=6,
        seed=42,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 6, 30),
    )
    return pd.read_csv(d / "demand.csv"), pd.read_csv(d / "skus.csv")


def test_metrics_basic():
    y = np.array([10.0, 0.0, 5.0])
    assert mae(y, y) == 0.0
    assert wape(y, y) == 0.0
    assert wape(np.array([10.0, 10.0]), np.array([8.0, 14.0])) == pytest.approx(6 / 20)
    assert np.isnan(wape(np.array([0.0, 0.0]), np.array([1.0, 2.0])))


def test_build_features_has_all_columns_and_no_leakage_nans(demo):
    demand, skus = demo
    feats = build_features(demand, skus)
    for col in [*FEATURES, TARGET]:
        assert col in feats.columns
    # Warm-up rows dropped -> no NaNs left in the lag/rolling features.
    assert not feats[["lag_1", "lag_7", "lag_28", "roll_mean_28", "roll_std_7"]].isna().any().any()
    # Lags must equal the actual prior values within each SKU.
    one = feats[feats["sku_id"] == feats["sku_id"].iloc[0]].sort_values("date")
    assert (one["lag_1"].to_numpy()[1:] == one[TARGET].to_numpy()[:-1]).all()


def test_walk_forward_structure_and_improvement_metric(demo):
    demand, skus = demo
    feats = build_features(demand, skus)
    # Tiny model keeps this a fast plumbing check; the real accuracy gain is shown
    # by the full `train` run, not asserted here on an under-powered config.
    result = walk_forward(
        feats,
        folds=2,
        fold_days=14,
        params={
            "loss_function": "Poisson",
            "iterations": 50,
            "depth": 4,
            "random_seed": 42,
            "verbose": False,
            "allow_writing_files": False,
        },
    )
    assert result.n_folds == 2
    assert result.n_test_rows > 0
    assert set(result.wape) == {"naive", "seasonal_naive", "catboost"}
    assert all(v >= 0 for v in result.wape.values())
    assert len(result.per_fold) == 2
    # Improvement metric is consistent with the raw WAPEs.
    expected = (result.wape["naive"] - result.wape["catboost"]) / result.wape["naive"]
    assert result.wape_improvement_vs("naive") == pytest.approx(expected)
