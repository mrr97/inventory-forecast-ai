"""Tests for multi-horizon feature engineering, metrics and the rolling backtest."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from inventory_forecast_ai.demo.generator import generate_demo_data
from inventory_forecast_ai.forecasting.backtest import backtest_forecast
from inventory_forecast_ai.forecasting.features import (
    FEATURES,
    TARGET,
    build_forecast_frame,
    build_training_frame,
)
from inventory_forecast_ai.forecasting.metrics import mae, wape
from inventory_forecast_ai.forecasting.model import search_hyperparameters

_TINY = {
    "loss_function": "Poisson",
    "iterations": 40,
    "depth": 4,
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
}


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Several months of data so lags/rolling windows and a holdout are populated."""
    d = tmp_path_factory.mktemp("data")
    generate_demo_data(
        d / "skus.csv",
        d / "demand.csv",
        count=4,
        seed=42,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 8, 31),
    )
    return pd.read_csv(d / "demand.csv"), pd.read_csv(d / "skus.csv")


def test_metrics_basic():
    y = np.array([10.0, 0.0, 5.0])
    assert mae(y, y) == 0.0
    assert wape(y, y) == 0.0
    assert wape(np.array([10.0, 10.0]), np.array([8.0, 14.0])) == pytest.approx(6 / 20)
    assert np.isnan(wape(np.array([0.0, 0.0]), np.array([1.0, 2.0])))


def test_training_frame_columns_and_no_leakage(demo):
    demand, skus = demo
    train = build_training_frame(demand, skus, horizon=7)
    for col in [*FEATURES, TARGET]:
        assert col in train.columns
    # Warm-up origins dropped -> no NaNs left in any model feature.
    assert not train[list(FEATURES)].isna().any().any()
    assert set(train["horizon"]) == set(range(1, 8))
    # For horizon 1 the origin is t-1, so recent_demand at t must equal the actual at t-1.
    h1 = train[train["horizon"] == 1]
    one = h1[h1["sku_id"] == h1["sku_id"].iloc[0]].sort_values("date")
    assert (one["recent_demand"].to_numpy()[1:] == one[TARGET].to_numpy()[:-1]).all()


def test_forecast_frame_is_one_path_per_sku(demo):
    demand, skus = demo
    fc = build_forecast_frame(demand, skus, horizon=7)
    assert set(fc["horizon"]) == set(range(1, 8))
    per_sku = fc.groupby("sku_id").size()
    assert (per_sku == 7).all()
    for col in FEATURES:
        assert col in fc.columns


def test_hyperparameter_search_picks_a_config(demo):
    demand, skus = demo
    from inventory_forecast_ai.forecasting.features import temporal_train_val_split

    frame = build_training_frame(demand, skus, horizon=7)
    train, val = temporal_train_val_split(frame, val_days=14)
    grid = {"depth": [4, 6], "learning_rate": [0.1]}
    best, trials = search_hyperparameters(
        train, val, base_params=_TINY, grid=grid, early_stopping_rounds=10
    )

    assert len(trials) == 2  # 2 depths x 1 learning rate
    assert best["combo"]["depth"] in (4, 6)
    assert best["combo"]["learning_rate"] == 0.1
    # The winner has the lowest validation loss and carries a concrete iteration count.
    assert best["val_loss"] == min(t["val_loss"] for t in trials)
    assert best["params"]["iterations"] >= 1


def test_backtest_reports_forward_and_trailing(demo):
    demand, skus = demo
    result = backtest_forecast(demand, skus, holdout_days=30, horizon=7, params=_TINY)
    assert result.horizon == 7
    assert result.n_origins > 0
    assert set(result.wape) == {"forward", "trailing"}
    assert all(v >= 0 for v in result.wape.values())
    assert len(result.forecast_cv) == 7
    assert all(cv >= 0 for cv in result.forecast_cv)
    assert len(result.per_horizon_wape["forward"]) == 7
    # Improvement metric is consistent with the raw WAPEs.
    expected = (result.wape["trailing"] - result.wape["forward"]) / result.wape["trailing"]
    assert result.improvement_vs_trailing() == pytest.approx(expected)
