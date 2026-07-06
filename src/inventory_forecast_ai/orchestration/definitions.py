"""Dagster code location: the asset graph plus a job that materializes the whole pipeline.

`dagster dev` (or the `pyproject.toml [tool.dagster]` entry) loads this module to serve
the asset graph in the Dagster UI. `run_pipeline()` materializes everything headlessly,
which is what the `pipeline` CLI command and the Docker `pipeline` service call.
"""

from __future__ import annotations

from dagster import Definitions, ScheduleDefinition, define_asset_job

from inventory_forecast_ai.orchestration.assets import ALL_ASSETS

inventory_pipeline_job = define_asset_job(name="inventory_pipeline", selection="*")

# Weekly recompute: re-run the whole demand-to-policy pipeline every Monday morning so
# the (s, S) policies always cover the upcoming protection interval with a fresh forecast
# (planned promotions/prices for the week are known by then). This is the operational
# cadence — a single, fixed schedule, not an event-driven one.
weekly_recompute_schedule = ScheduleDefinition(
    name="weekly_inventory_recompute",
    job=inventory_pipeline_job,
    cron_schedule="0 6 * * 1",  # Mondays at 06:00
    execution_timezone="Europe/Madrid",
)

defs = Definitions(
    assets=ALL_ASSETS,
    jobs=[inventory_pipeline_job],
    schedules=[weekly_recompute_schedule],
)


def run_pipeline() -> bool:
    """Materialize the full asset graph in process. Returns True on success."""
    from dagster import materialize

    result = materialize(ALL_ASSETS)
    return result.success
