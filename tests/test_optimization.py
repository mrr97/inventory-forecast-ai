"""Tests for the inventory simulator and the NSGA-II optimization."""

import numpy as np

from inventory_forecast_ai.optimization.optimize import baseline_policy, optimize_policies
from inventory_forecast_ai.optimization.scenarios import DemandScenarios
from inventory_forecast_ai.optimization.simulator import simulate_policies


def _scenarios(demand: np.ndarray) -> DemandScenarios:
    n_sku = demand.shape[0]
    return DemandScenarios(
        sku_ids=[f"SKU-{1001 + i:04d}" for i in range(n_sku)],
        demand=demand,
        mean_daily=demand.mean(axis=(1, 2)),
        lead_time=np.full(n_sku, 2, dtype=int),
        holding=np.full(n_sku, 0.05),
        order_cost=np.full(n_sku, 30.0),
        unit_cost=np.full(n_sku, 5.0),
        min_order_qty=np.zeros(n_sku),
    )


def test_zero_demand_gives_no_stockout_and_known_holding_cost():
    horizon = 10
    sc = _scenarios(np.zeros((1, 4, horizon)))
    s = np.array([[3.0]])
    big_s = np.array([[10.0]])
    out = simulate_policies(s, big_s, sc)
    assert out["stockout_rate"][0] == 0.0
    # No sales -> stock stays at S every day -> holding = S * h * horizon.
    assert np.isclose(out["operating_cost"][0], 10.0 * 0.05 * horizon)


def test_more_stock_never_increases_stockout():
    rng = np.random.default_rng(0)
    demand = rng.poisson(8.0, size=(3, 6, 15)).astype(float)
    sc = _scenarios(demand)
    low = simulate_policies(np.full((1, 3), 5.0), np.full((1, 3), 15.0), sc)
    high = simulate_policies(np.full((1, 3), 30.0), np.full((1, 3), 80.0), sc)
    assert high["stockout_rate"][0] <= low["stockout_rate"][0] + 1e-9
    assert 0.0 <= high["stockout_rate"][0] <= 1.0
    assert 0.0 <= low["stockout_rate"][0] <= 1.0


def test_baseline_policy_orders_above_reorder_point():
    demand = np.random.default_rng(1).poisson(10.0, size=(4, 5, 20)).astype(float)
    sc = _scenarios(demand)
    s, big_s = baseline_policy(sc)
    assert np.all(big_s > s)
    assert np.all(s > 0)


def test_optimize_returns_a_valid_pareto_frontier():
    demand = np.random.default_rng(2).poisson(10.0, size=(4, 8, 20)).astype(float)
    sc = _scenarios(demand)
    result = optimize_policies(sc, pop_size=20, generations=8, seed=42)

    assert len(result.pareto_cost) > 0
    assert result.pareto_cost.shape[0] == result.pareto_stockout.shape[0]
    assert np.all(np.isfinite(result.pareto_cost))
    assert np.all((result.pareto_stockout >= 0) & (result.pareto_stockout <= 1))
    # Frontier is sorted by ascending cost; service must not get worse as cost rises.
    assert np.all(np.diff(result.pareto_stockout) <= 1e-6)
    assert np.isfinite(result.baseline_cost)
