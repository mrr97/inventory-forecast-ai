"""Continuous-review (s, S) inventory simulator.

Given a candidate policy per SKU — reorder point ``s`` and order-up-to level ``S`` —
the simulator plays the inventory forward day by day over every demand scenario and
returns the metrics the optimizer trades off:

- **operating cost**: holding cost (carrying stock) + ordering cost (fixed cost per
  replenishment), averaged over scenarios,
- **stockout rate**: unmet demand / total demand (lost sales, not back-ordered),
- **average investment**: mean capital tied up in stock (for an optional budget cap).

Each policy is evaluated for the whole population at once: state arrays are shaped
``(pop, n_scenarios)`` and we loop over SKUs (outer) and days (inner). Replenishment
orders arrive after the SKU's lead time.
"""

from __future__ import annotations

import numpy as np

from inventory_forecast_ai.optimization.scenarios import DemandScenarios


def simulate_policies(
    s: np.ndarray,
    big_s: np.ndarray,
    scenarios: DemandScenarios,
) -> dict[str, np.ndarray]:
    """Evaluate a population of policies.

    Args:
        s: reorder points, shape (pop, n_sku).
        big_s: order-up-to levels, shape (pop, n_sku); must satisfy big_s >= s.
        scenarios: sampled demand and per-SKU economics.

    Returns:
        Dict of ``operating_cost``, ``stockout_rate`` and ``avg_investment``,
        each an array of shape (pop,).
    """
    pop = s.shape[0]
    demand = scenarios.demand  # (n_sku, K, T)
    n_sku, n_scen, horizon = demand.shape

    operating_cost = np.zeros(pop)
    total_lost = np.zeros(pop)
    investment_accum = np.zeros(pop)

    for j in range(n_sku):
        lead = int(scenarios.lead_time[j])
        holding = float(scenarios.holding[j])
        order_cost = float(scenarios.order_cost[j])
        unit_cost = float(scenarios.unit_cost[j])
        moq = float(scenarios.min_order_qty[j])

        s_j = s[:, j][:, None]  # (pop, 1) -> broadcast over scenarios
        big_s_j = big_s[:, j][:, None]
        dem_j = demand[j]  # (K, T)

        # Start fully stocked (just replenished); pipeline holds in-transit arrivals.
        on_hand = np.repeat(big_s_j, n_scen, axis=1).astype(float)  # (pop, K)
        pipeline = np.zeros((pop, n_scen, horizon + lead))

        for t in range(horizon):
            on_hand += pipeline[:, :, t]  # receive arrivals due today

            d = dem_j[None, :, t]  # (1, K) -> broadcast over pop
            sales = np.minimum(on_hand, d)
            total_lost += (d - sales).sum(axis=1)
            on_hand -= sales

            operating_cost += on_hand.sum(axis=1) * holding
            investment_accum += (on_hand * unit_cost).sum(axis=1)

            outstanding = pipeline[:, :, t + 1 :].sum(axis=2)
            inv_position = on_hand + outstanding
            need = inv_position <= s_j
            qty = np.where(need, np.maximum(big_s_j - inv_position, moq), 0.0)
            qty = np.maximum(qty, 0.0)

            operating_cost += need.sum(axis=1) * order_cost
            pipeline[:, :, t + lead] += qty

    # total_lost is summed over the same (sku, scenario, day) units as demand,
    # so the unit fill-rate complement is simply lost / demand.
    total_demand = float(demand.sum())
    stockout_rate = total_lost / total_demand if total_demand > 0 else np.zeros(pop)

    return {
        "operating_cost": operating_cost / n_scen,
        "stockout_rate": stockout_rate,
        "avg_investment": investment_accum / n_scen / horizon,
    }
