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


if __name__ == "__main__":
    app()
