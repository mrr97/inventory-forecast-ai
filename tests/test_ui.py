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
from inventory_forecast_ai.ui.data import load_pareto, load_product_names


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
        "sku_ids": ["SKU-1002", "SKU-1001"],
        "baseline": {"cost": 5000.0, "stockout": 0.002, "service": 0.998},
        "cost_saving_pct": 0.12,
        "points": [
            {
                "cost": 4000.0,
                "stockout": 0.05,
                "service": 0.95,
                "policy": [12.25, 3.5, 84.5, 40.75],
            },
            {
                "cost": 4500.0,
                "stockout": 0.02,
                "service": 0.98,
                "policy": [20.0, 7.0, 95.0, 55.0],
            },
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
    monkeypatch.setattr(ui_app, "load_product_names", lambda settings: {})

    app = create_app(settings=object())
    assert app.layout is not None
    assert "12%" in str(app.layout)


def test_load_product_names_preserves_sku_association(tmp_path):
    class _S:
        skus_path = tmp_path / "skus.csv"

    _S.skus_path.write_text("sku_id,name\nSKU-1002,Popcorn\nSKU-1001,Cola\n", encoding="utf-8")
    assert load_product_names(_S()) == {"SKU-1002": "Popcorn", "SKU-1001": "Cola"}


def test_policy_selection_updates_product_targets_and_clears_them_for_baseline(monkeypatch):
    monkeypatch.setattr(ui_app, "load_forecast_frame", lambda settings: _forecast_frame())
    monkeypatch.setattr(ui_app, "load_pareto", lambda settings: _pareto_dict())
    monkeypatch.setattr(ui_app, "load_report", lambda settings: _report_dict())
    monkeypatch.setattr(
        ui_app, "load_product_names", lambda settings: {"SKU-1001": "Cola", "SKU-1002": "Popcorn"}
    )
    client = create_app(settings=object()).server.test_client()

    def select(click):
        response = client.post(
            "/_dash-update-component",
            json={
                "output": "pareto-detail.children",
                "outputs": {"id": "pareto-detail", "property": "children"},
                "inputs": [{"id": "pareto-graph", "property": "clickData", "value": click}],
                "changedPropIds": ["pareto-graph.clickData"],
                "state": [],
            },
        )
        assert response.status_code == 200
        return response.get_json()["response"]["pareto-detail"]["children"]

    def table_rows(component):
        if isinstance(component, dict):
            if component.get("type") == "Tbody":
                return [
                    [cell["props"]["children"] for cell in row["props"]["children"]]
                    for row in component["props"]["children"]
                ]
            return table_rows(component.get("props", {}).get("children"))
        if isinstance(component, list):
            return next((rows for child in component if (rows := table_rows(child))), [])
        return []

    assert "Click a point" in select(None)
    first = select({"points": [{"customdata": 0}]})
    assert table_rows(first) == [
        ["SKU-1002", "Popcorn", "12.25", "84.50"],
        ["SKU-1001", "Cola", "3.50", "40.75"],
    ]
    assert "€4000" in str(first)
    second = select({"points": [{"customdata": 1}]})
    assert table_rows(second) == [
        ["SKU-1002", "Popcorn", "20.00", "95.00"],
        ["SKU-1001", "Cola", "7.00", "55.00"],
    ]
    assert "€4500" in str(second)
    baseline = select({"points": [{"curveNumber": 1, "pointNumber": 0}]})
    assert "Baseline policy selected" in baseline
    assert table_rows(baseline) == []


def test_scorecard_renders_metric_cards():
    children = ui_app._scorecard_children(_report_dict())
    rendered = str(children)
    assert "0.131" in rendered  # forward WAPE card
    assert "12%" in rendered  # cost saving card
    assert "19%" in rendered  # seasonal-stretch cost-cut card


def test_scorecard_handles_missing_report():
    children = ui_app._scorecard_children(None)
    assert "metrics" in str(children)
