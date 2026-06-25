"""Multi-objective optimization of (s, S) policies with NSGA-II (pymoo).

Decision variables per SKU are the reorder point ``s`` and the gap ``g = S - s``
(so ``S = s + g`` is always >= ``s`` by construction). The two objectives — operating
cost and stockout rate — genuinely conflict, so the optimizer returns a **Pareto
frontier**: the set of policies where cost cannot drop without service getting worse.

A simple textbook policy (safety stock from the demand mean and lead time) is
evaluated on the same simulator to anchor a headline saving: at the baseline's
service level, how much cheaper is the best optimized policy?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from inventory_forecast_ai.optimization.scenarios import DemandScenarios
from inventory_forecast_ai.optimization.simulator import simulate_policies

log = structlog.get_logger(__name__)

# Service factor (~95% one-sided normal) for the baseline safety stock.
_BASELINE_Z = 1.64


@dataclass
class OptimizationResult:
    """Pareto frontier plus the baseline anchor and the headline saving."""

    sku_ids: list[str]
    pareto_cost: np.ndarray  # (n_points,)
    pareto_stockout: np.ndarray  # (n_points,)
    pareto_policies: np.ndarray  # (n_points, 2 * n_sku) -> [s..., S...]
    baseline_cost: float
    baseline_stockout: float
    cost_saving_pct: float  # vs baseline, at >= baseline service level

    @property
    def pareto_service(self) -> np.ndarray:
        return 1.0 - self.pareto_stockout


def _policy_bounds(scenarios: DemandScenarios) -> tuple[np.ndarray, np.ndarray]:
    """Per-variable lower/upper bounds for [s..., gap...]."""
    mean_d = np.maximum(scenarios.mean_daily, 0.1)
    lead = scenarios.lead_time
    n = scenarios.n_sku
    # s up to demand over (lead time + a fortnight); gap up to ~3 weeks of demand.
    s_upper = mean_d * (lead + 14)
    gap_upper = mean_d * 21
    xl = np.zeros(2 * n)
    xu = np.concatenate([s_upper, gap_upper])
    return xl, xu


def baseline_policy(scenarios: DemandScenarios) -> tuple[np.ndarray, np.ndarray]:
    """Textbook safety-stock policy: s covers lead-time demand + safety stock."""
    mean_d = scenarios.mean_daily
    lead = scenarios.lead_time
    sigma = np.sqrt(np.maximum(mean_d, 0.0))  # Poisson std ~ sqrt(mean)
    safety = _BASELINE_Z * sigma * np.sqrt(np.maximum(lead, 1))
    s = mean_d * lead + safety
    big_s = s + mean_d * 7  # one week of cover above the reorder point
    return s, big_s


class _InventoryProblem(Problem):
    def __init__(self, scenarios: DemandScenarios, budget: float | None):
        self._scenarios = scenarios
        self._budget = budget
        xl, xu = _policy_bounds(scenarios)
        super().__init__(
            n_var=2 * scenarios.n_sku,
            n_obj=2,
            n_ieq_constr=1 if budget is not None else 0,
            xl=xl,
            xu=xu,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        n = self._scenarios.n_sku
        s = x[:, :n]
        big_s = s + x[:, n:]
        metrics = simulate_policies(s, big_s, self._scenarios)
        out["F"] = np.column_stack([metrics["operating_cost"], metrics["stockout_rate"]])
        if self._budget is not None:
            out["G"] = (metrics["avg_investment"] - self._budget)[:, None]


def _saving_vs_baseline(
    pareto_cost: np.ndarray,
    pareto_stockout: np.ndarray,
    baseline_cost: float,
    baseline_stockout: float,
) -> float:
    """Best cost reduction among Pareto policies that match or beat baseline service."""
    eligible = pareto_cost[pareto_stockout <= baseline_stockout + 1e-9]
    if eligible.size == 0 or baseline_cost <= 0:
        return float("nan")
    return float((baseline_cost - eligible.min()) / baseline_cost)


def optimize_policies(
    scenarios: DemandScenarios,
    *,
    pop_size: int,
    generations: int,
    seed: int,
    budget: float | None = None,
) -> OptimizationResult:
    """Run NSGA-II and assemble the Pareto frontier and baseline comparison."""
    problem = _InventoryProblem(scenarios, budget)
    algorithm = NSGA2(pop_size=pop_size)
    res = minimize(
        problem,
        algorithm,
        ("n_gen", generations),
        seed=seed,
        verbose=False,
    )

    n = scenarios.n_sku
    s_opt = res.X[:, :n]
    big_s_opt = s_opt + res.X[:, n:]
    policies = np.column_stack([s_opt, big_s_opt])

    s0, big_s0 = baseline_policy(scenarios)
    base = simulate_policies(s0[None, :], big_s0[None, :], scenarios)
    baseline_cost = float(base["operating_cost"][0])
    baseline_stockout = float(base["stockout_rate"][0])

    # Sort the frontier by cost for stable, plottable output.
    order = np.argsort(res.F[:, 0])
    pareto_cost = res.F[order, 0]
    pareto_stockout = res.F[order, 1]
    policies = policies[order]

    saving = _saving_vs_baseline(pareto_cost, pareto_stockout, baseline_cost, baseline_stockout)
    log.info(
        "optimization_done",
        pareto_points=len(pareto_cost),
        baseline_service=round(1 - baseline_stockout, 4),
        cost_saving_pct=None if np.isnan(saving) else round(saving, 4),
    )
    return OptimizationResult(
        sku_ids=scenarios.sku_ids,
        pareto_cost=pareto_cost,
        pareto_stockout=pareto_stockout,
        pareto_policies=policies,
        baseline_cost=baseline_cost,
        baseline_stockout=baseline_stockout,
        cost_saving_pct=saving,
    )
