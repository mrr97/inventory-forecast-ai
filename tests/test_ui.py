"""Hermetic tests for the Dash UI: figure builders, data loaders, app construction.

The app is built with the data layer stubbed out so the test needs neither a trained
CatBoost model nor a running server — it verifies the layout and callbacks wire up.
"""

import json

import pandas as pd
import plotly.graph_objects as go

from inventory_forecast_ai.ui import app as ui_app
from inventory_forecast_ai.ui.app import _forecast_figure, _pareto_figure, create_app
from inventory_forecast_ai.ui.data import load_pareto


def _forecast_frame() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=5)
    rows = []
    for sku in ("SKU-1001", "SKU-1002"):
        for d in dates:
            rows.append({"date": d, "sku_id": sku, "actual": 10.0, "predicted": 9.0})
    return pd.DataFrame(rows)


def _pareto_dict() -> dict:
    return {
        "sku_ids": ["SKU-1001", "SKU-1002"],
        "baseline": {"cost": 5000.0, "stockout": 0.002, "service": 0.998},
        "cost_saving_pct": 0.12,
        "points": [
            {"cost": 4000.0, "stockout": 0.05, "service": 0.95, "policy": {}},
            {"cost": 4500.0, "stockout": 0.02, "service": 0.98, "policy": {}},
        ],
    }


def test_forecast_figure_has_actual_and_forecast_traces():
    fig = _forecast_figure(_forecast_frame(), "SKU-1001")
    assert isinstance(fig, go.Figure)
    names = {trace.name for trace in fig.data}
    assert {"Actual", "Forecast"} <= names


def test_pareto_figure_plots_frontier_and_baseline():
    fig = _pareto_figure(_pareto_dict())
    names = {trace.name for trace in fig.data}
    assert {"Optimized policies", "Baseline policy"} <= names


def test_load_pareto_missing_returns_none(tmp_path):
    class _S:
        pareto_path = tmp_path / "nope.json"

    assert load_pareto(_S()) is None


def test_load_pareto_reads_saved_file(tmp_path):
    path = tmp_path / "pareto.json"
    path.write_text(json.dumps(_pareto_dict()), encoding="utf-8")

    class _S:
        pareto_path = path

    loaded = load_pareto(_S())
    assert loaded["cost_saving_pct"] == 0.12


def _report_dict() -> dict:
    return {
        "forecast": {
            "wape": {"naive": 0.227, "seasonal_naive": 0.271, "catboost": 0.137},
            "improvement_vs_naive": 0.396,
        },
        "optimization": {
            "cost_saving_pct": 0.12,
            "baseline_service": 0.998,
            "n_pareto_points": 69,
        },
        "headline": {
            "forecast": "CatBoost cuts forecast error (WAPE) 40% vs. naive",
            "optimization": "At the baseline's service level, optimization cuts operating cost 12%",
        },
    }


def test_create_app_builds_with_stubbed_data(monkeypatch):
    monkeypatch.setattr(ui_app, "load_forecast_frame", lambda settings: _forecast_frame())
    monkeypatch.setattr(ui_app, "load_pareto", lambda settings: _pareto_dict())
    monkeypatch.setattr(ui_app, "load_report", lambda settings: _report_dict())

    app = create_app(settings=object())
    assert app.layout is not None
    # Headline reflects the saving from the stubbed Pareto.
    assert "12%" in str(app.layout)


def test_scorecard_renders_metric_cards():
    children = ui_app._scorecard_children(_report_dict())
    rendered = str(children)
    assert "0.137" in rendered  # forecast WAPE card
    assert "12%" in rendered  # cost saving card


def test_scorecard_handles_missing_report():
    children = ui_app._scorecard_children(None)
    assert "metrics" in str(children)
