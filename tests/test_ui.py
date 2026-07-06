"""Hermetic tests for the Dash UI: figure builders, data loaders, app construction.

The app is built with the data layer stubbed out so the test needs neither a trained
CatBoost model nor a running server — it verifies the layout and callbacks wire up.
"""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from inventory_forecast_ai.ui import app as ui_app
from inventory_forecast_ai.ui.app import _forecast_figure, _pareto_figure, create_app
from inventory_forecast_ai.ui.data import load_pareto


def _forecast_frame() -> pd.DataFrame:
    """Recent history (forecast NaN) followed by a short forward path per SKU."""
    hist = pd.date_range("2025-01-01", periods=5)
    fwd = pd.date_range("2025-01-06", periods=3)
    rows = []
    for sku in ("SKU-1001", "SKU-1002"):
        for d in hist:
            rows.append({"date": d, "sku_id": sku, "actual": 10.0, "forecast": np.nan})
        for d in fwd:
            rows.append({"date": d, "sku_id": sku, "actual": 11.0, "forecast": 9.0})
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


def _report_dict() -> dict:
    return {
        "forecast": {
            "wape": {"forward": 0.131, "trailing": 0.253},
            "improvement_vs_trailing": 0.482,
            "promo_improvement_vs_trailing": 0.753,
        },
        "optimization": {
            "cost_saving_pct": 0.12,
            "baseline_service": 0.998,
            "n_pareto_points": 18,
        },
        "decision": {"cost_reduction": 0.188, "service_gain": 0.0108},
        "headline": {
            "forecast": "Forward cuts WAPE 48% vs. trailing, 75% on promo days",
            "optimization": "At baseline service, optimization cuts cost 12%",
            "decision": "On the seasonal stretch, planning ahead cuts cost 19%",
        },
    }


def test_forecast_figure_has_actual_and_forward_traces():
    fig = _forecast_figure(_forecast_frame(), "SKU-1001")
    assert isinstance(fig, go.Figure)
    names = {trace.name for trace in fig.data}
    assert {"Actual", "Forward forecast"} <= names


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


def test_create_app_builds_with_stubbed_data(monkeypatch):
    monkeypatch.setattr(ui_app, "load_forecast_frame", lambda settings: _forecast_frame())
    monkeypatch.setattr(ui_app, "load_pareto", lambda settings: _pareto_dict())
    monkeypatch.setattr(ui_app, "load_report", lambda settings: _report_dict())

    app = create_app(settings=object())
    assert app.layout is not None
    assert "12%" in str(app.layout)


def test_scorecard_renders_metric_cards():
    children = ui_app._scorecard_children(_report_dict())
    rendered = str(children)
    assert "0.131" in rendered  # forward WAPE card
    assert "12%" in rendered  # cost saving card
    assert "19%" in rendered  # seasonal-stretch cost-cut card


def test_scorecard_handles_missing_report():
    children = ui_app._scorecard_children(None)
    assert "metrics" in str(children)
