"""Hermetic tests for the Dagster asset graph (structure, not heavy execution)."""

from dagster import AssetKey

from inventory_forecast_ai.orchestration.assets import ALL_ASSETS
from inventory_forecast_ai.orchestration.definitions import (
    defs,
    inventory_pipeline_job,
    weekly_recompute_schedule,
)


def _asset_keys() -> set[str]:
    keys: set[str] = set()
    for a in ALL_ASSETS:
        keys |= {k.to_user_string() for k in a.keys}
    return keys


def test_all_stages_present():
    assert _asset_keys() == {
        "demo_data",
        "trained_model",
        "pareto_frontier",
        "forward_vs_trailing",
        "scorecard",
    }


def test_pipeline_dependencies():
    deps: dict[str, set[str]] = {}
    for a in ALL_ASSETS:
        for key, ins in a.asset_deps.items():
            deps[key.to_user_string()] = {dep.to_user_string() for dep in ins}

    assert deps["demo_data"] == set()
    assert deps["trained_model"] == {"demo_data"}
    assert deps["pareto_frontier"] == {"trained_model"}
    assert deps["forward_vs_trailing"] == {"demo_data"}
    assert deps["scorecard"] == {"pareto_frontier", "forward_vs_trailing"}


def test_definitions_expose_the_pipeline_job():
    assert inventory_pipeline_job.name == "inventory_pipeline"
    assert defs.get_assets_def(AssetKey("scorecard")) is not None


def test_weekly_recompute_schedule():
    assert weekly_recompute_schedule.name == "weekly_inventory_recompute"
    assert weekly_recompute_schedule.cron_schedule == "0 6 * * 1"
