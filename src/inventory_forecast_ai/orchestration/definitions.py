"""Dagster code location: the asset graph plus a job that materializes the whole pipeline.

`dagster dev` (or the `pyproject.toml [tool.dagster]` entry) loads this module to serve
the asset graph in the Dagster UI. `run_pipeline()` materializes everything headlessly,
which is what the `pipeline` CLI command and the Docker `pipeline` service call.
"""

from __future__ import annotations

from dagster import Definitions, define_asset_job

from inventory_forecast_ai.orchestration.assets import ALL_ASSETS

inventory_pipeline_job = define_asset_job(name="inventory_pipeline", selection="*")

defs = Definitions(assets=ALL_ASSETS, jobs=[inventory_pipeline_job])


def run_pipeline() -> bool:
    """Materialize the full asset graph in process. Returns True on success."""
    from dagster import materialize

    result = materialize(ALL_ASSETS)
    return result.success
