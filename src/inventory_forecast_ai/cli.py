"""Command-line interface for Inventory Forecast AI."""

from typing import Annotated

import typer

from inventory_forecast_ai.config import get_settings
from inventory_forecast_ai.log import configure_logging

app = typer.Typer(
    name="inventory-forecast-ai",
    help="Demand forecasting + multi-objective inventory optimization.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Inventory Forecast AI command-line interface."""
    # Present so Typer keeps named subcommands even when only one exists yet.


@app.command(name="generate-demo")
def generate_demo(
    count: Annotated[
        int | None,
        typer.Option("--count", min=1, help="How many SKUs to generate (default from settings)."),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Random seed (fixed for reproducible output)."),
    ] = None,
) -> None:
    """Generate the synthetic SKU master and daily demand history."""
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.json_logs)

    from inventory_forecast_ai.demo.generator import generate_demo_data

    n_skus, n_rows = generate_demo_data(
        settings.skus_path,
        settings.demand_path,
        count=count if count is not None else settings.demo_sku_count,
        seed=seed if seed is not None else settings.demo_seed,
        start_date=settings.demo_start_date,
        end_date=settings.demo_end_date,
    )

    typer.echo("")
    typer.echo(f"Generated {n_skus} SKUs and {n_rows} daily demand rows")
    typer.echo(f"  SKU master: {settings.skus_path}")
    typer.echo(f"  Demand:     {settings.demand_path}")


@app.command()
def train() -> None:
    """Backtest the forward forecast vs. the trailing mean, log to MLflow, save model."""
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.json_logs)

    from inventory_forecast_ai.forecasting.pipeline import train_and_evaluate

    result = train_and_evaluate(settings)

    typer.echo("")
    typer.echo(
        f"Rolling-origin backtest — {result.n_origins} origins, {result.n_rows} forecasts "
        f"over a {result.horizon}-day horizon"
    )
    typer.echo(f"{'model':<16}{'WAPE':>10}{'MAE':>10}")
    for name in ("forward", "trailing"):
        typer.echo(f"{name:<16}{result.wape[name]:>10.4f}{result.mae[name]:>10.3f}")
    typer.echo("")
    typer.echo(
        f"Forward vs trailing: {result.improvement_vs_trailing():.1%} lower WAPE overall, "
        f"{result.promo_improvement_vs_trailing():.1%} on promo days"
    )
    typer.echo(f"Saved model -> {settings.model_path}")


@app.command()
def optimize() -> None:
    """Optimize (s, S) policies with NSGA-II, build the Pareto frontier, save it."""
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.json_logs)

    from inventory_forecast_ai.optimization.pipeline import run_optimization

    result = run_optimization(settings)

    cheapest = result.pareto_cost.argmin()
    safest = result.pareto_stockout.argmin()
    typer.echo("")
    typer.echo(f"Pareto frontier: {len(result.pareto_cost)} non-dominated policies")
    typer.echo(f"{'policy':<18}{'op. cost':>12}{'service':>10}")
    typer.echo(
        f"{'baseline':<18}{result.baseline_cost:>12.0f}{1 - result.baseline_stockout:>10.1%}"
    )
    typer.echo(
        f"{'cheapest':<18}{result.pareto_cost[cheapest]:>12.0f}"
        f"{result.pareto_service[cheapest]:>10.1%}"
    )
    typer.echo(
        f"{'best service':<18}{result.pareto_cost[safest]:>12.0f}"
        f"{result.pareto_service[safest]:>10.1%}"
    )
    typer.echo("")
    if result.cost_saving_pct == result.cost_saving_pct:  # not NaN
        typer.echo(
            f"At the baseline's service level, the optimizer cuts operating cost by "
            f"{result.cost_saving_pct:.1%}."
        )
    typer.echo(f"Saved frontier -> {settings.pareto_path}")


@app.command()
def metrics() -> None:
    """Consolidate forecast accuracy + cost savings into a single scorecard (report.json)."""
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.json_logs)

    from inventory_forecast_ai.evaluation.report import run_metrics

    report = run_metrics(settings)
    fc = report["forecast"]
    opt = report["optimization"]

    typer.echo("")
    typer.echo("Forecast accuracy (rolling-origin backtest)")
    typer.echo(
        f"  forward WAPE {fc['wape']['forward']:.3f} vs trailing {fc['wape']['trailing']:.3f}"
    )
    typer.echo(
        f"  improvement: {fc['improvement_vs_trailing']:.1%} overall, "
        f"{fc['promo_improvement_vs_trailing']:.1%} on promo days"
    )
    typer.echo("")
    typer.echo("Inventory cost vs. service (Pareto frontier)")
    typer.echo(f"  {opt['n_pareto_points']} non-dominated policies")
    typer.echo(f"  baseline: €{opt['baseline_cost']:.0f} at {opt['baseline_service']:.1%} service")
    if opt["cost_saving_pct"] is not None:
        typer.echo(f"  saving at baseline service: {opt['cost_saving_pct']:.1%}")
    typer.echo("")
    typer.echo(f"Saved scorecard -> {settings.report_path}")


@app.command()
def compare() -> None:
    """Decision-level forward-vs-trailing cost comparison on the most seasonal stretch."""
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.json_logs)

    from inventory_forecast_ai.evaluation.comparison import run_comparison

    res = run_comparison(settings)

    typer.echo("")
    typer.echo(
        f"Seasonal stretch {res.window_start}..{res.window_end} — {res.n_origins} weekly origins"
    )
    typer.echo(
        f"{'policy':<10}{'total cost':>13}{'operating':>12}{'lost sales':>12}{'service':>11}"
    )
    for name, tot, op, svc in (
        ("forward", res.forward_cost, res.forward_operating, res.forward_service),
        ("trailing", res.trailing_cost, res.trailing_operating, res.trailing_service),
    ):
        typer.echo(f"{name:<10}{tot:>13.0f}{op:>12.0f}{tot - op:>12.0f}{svc:>11.1%}")
    typer.echo("")
    typer.echo(
        f"Forward cuts total cost {res.cost_reduction:.1%} (fewer lost sales) "
        f"at {res.service_gain * 100:+.1f} pp service."
    )
    typer.echo(f"Saved comparison -> {settings.comparison_path}")


@app.command()
def pipeline() -> None:
    """Run the whole Dagster pipeline: generate-demo -> train -> optimize -> metrics."""
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.json_logs)

    from inventory_forecast_ai.orchestration.definitions import run_pipeline

    typer.echo("")
    typer.echo(
        "Materializing asset graph: demo_data -> {trained_model -> pareto_frontier, "
        "forward_vs_trailing} -> scorecard"
    )
    success = run_pipeline()
    if not success:
        typer.echo("Pipeline failed.")
        raise typer.Exit(code=1)
    typer.echo("")
    typer.echo("Pipeline complete. Scorecard at " + str(settings.report_path))


@app.command()
def dashboard() -> None:
    """Launch the Dash UI: demand forecast and the cost-vs-service Pareto frontier."""
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.json_logs)

    from inventory_forecast_ai.ui.app import create_app

    app_ = create_app(settings)
    typer.echo("")
    typer.echo(f"Dashboard running at http://{settings.ui_host}:{settings.ui_port}")
    typer.echo("Press Ctrl+C to stop.")
    app_.run(host=settings.ui_host, port=settings.ui_port, debug=False)


if __name__ == "__main__":
    app()
