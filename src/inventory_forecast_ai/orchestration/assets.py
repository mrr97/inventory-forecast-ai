"""The pipeline as a Dagster asset graph.

Each asset is a stage of the demand-to-decision pipeline and wraps the same function
the CLI calls, so there is a single source of truth. Stages hand off through files on
disk (CSV, model, JSON), so dependencies are expressed with ``deps=[...]`` rather than
passing data in memory:

    demo_data -> trained_model -> pareto_frontier -> scorecard

Materializing ``scorecard`` (or ``*``) runs the whole chain in order.
"""
# No `from __future__ import annotations`: Dagster introspects the real `context`
# type at runtime, and PEP 563 stringized annotations would break that validation.

from dagster import AssetExecutionContext, MaterializeResult, MetadataValue, asset

from inventory_forecast_ai.config import get_settings


@asset
def demo_data(context: AssetExecutionContext) -> MaterializeResult:
    """Generate the reproducible synthetic SKU master and daily demand history."""
    from inventory_forecast_ai.demo.generator import generate_demo_data

    settings = get_settings()
    n_skus, n_rows = generate_demo_data(
        settings.skus_path,
        settings.demand_path,
        count=settings.demo_sku_count,
        seed=settings.demo_seed,
        start_date=settings.demo_start_date,
        end_date=settings.demo_end_date,
    )
    context.log.info("generated demo data: %d SKUs, %d rows", n_skus, n_rows)
    return MaterializeResult(metadata={"skus": n_skus, "demand_rows": n_rows})


@asset(deps=[demo_data])
def trained_model(context: AssetExecutionContext) -> MaterializeResult:
    """Walk-forward backtest CatBoost vs. baselines, log to MLflow, save the model."""
    from inventory_forecast_ai.forecasting.pipeline import train_and_evaluate

    settings = get_settings()
    result = train_and_evaluate(settings)
    return MaterializeResult(
        metadata={
            "wape_forward": round(result.wape["forward"], 4),
            "improvement_vs_trailing": MetadataValue.float(result.improvement_vs_trailing()),
            "n_rows": result.n_rows,
        }
    )


@asset(deps=[trained_model])
def pareto_frontier(context: AssetExecutionContext) -> MaterializeResult:
    """Optimize (s, S) policies with NSGA-II and persist the cost-vs-service frontier."""
    from inventory_forecast_ai.optimization.pipeline import run_optimization

    settings = get_settings()
    result = run_optimization(settings)
    metadata: dict = {
        "pareto_points": len(result.pareto_cost),
        "baseline_cost": round(result.baseline_cost, 1),
    }
    if result.cost_saving_pct == result.cost_saving_pct:  # not NaN
        metadata["cost_saving_pct"] = MetadataValue.float(result.cost_saving_pct)
    return MaterializeResult(metadata=metadata)


@asset(deps=[pareto_frontier])
def scorecard(context: AssetExecutionContext) -> MaterializeResult:
    """Consolidate forecast accuracy + cost savings into the report.json scorecard."""
    from inventory_forecast_ai.evaluation.report import run_metrics

    settings = get_settings()
    report = run_metrics(settings)
    return MaterializeResult(
        metadata={
            "headline_forecast": report["headline"]["forecast"],
            "headline_optimization": report["headline"]["optimization"],
        }
    )


ALL_ASSETS = [demo_data, trained_model, pareto_frontier, scorecard]
