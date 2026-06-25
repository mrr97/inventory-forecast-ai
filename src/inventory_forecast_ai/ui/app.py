"""Dash app: two views — demand forecast and the cost-vs-service Pareto frontier."""

from __future__ import annotations

import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from inventory_forecast_ai.config import Settings, get_settings
from inventory_forecast_ai.ui.data import load_forecast_frame, load_pareto

_TEAL = "#0f9d8f"
_SLATE = "#334155"
_AMBER = "#f59e0b"


def _forecast_figure(frame, sku_id: str) -> go.Figure:
    one = frame[frame["sku_id"] == sku_id]
    fig = go.Figure()
    fig.add_scatter(
        x=one["date"], y=one["actual"], name="Actual", mode="lines", line={"color": _SLATE}
    )
    fig.add_scatter(
        x=one["date"],
        y=one["predicted"],
        name="Forecast",
        mode="lines",
        line={"color": _TEAL, "dash": "dash"},
    )
    fig.update_layout(
        margin={"t": 30, "r": 10, "b": 30, "l": 40},
        legend={"orientation": "h", "y": 1.1},
        yaxis_title="Units / day",
        template="plotly_white",
    )
    return fig


def _pareto_figure(pareto: dict) -> go.Figure:
    costs = [p["cost"] for p in pareto["points"]]
    service = [p["service"] * 100 for p in pareto["points"]]
    fig = go.Figure()
    fig.add_scatter(
        x=costs,
        y=service,
        mode="lines+markers",
        name="Optimized policies",
        line={"color": _TEAL},
        marker={"size": 7},
        customdata=list(range(len(costs))),
        hovertemplate="Cost €%{x:.0f}<br>Service %{y:.1f}%<extra></extra>",
    )
    base = pareto["baseline"]
    fig.add_scatter(
        x=[base["cost"]],
        y=[base["service"] * 100],
        mode="markers",
        name="Baseline policy",
        marker={"color": _AMBER, "size": 14, "symbol": "star"},
        hovertemplate="Baseline<br>Cost €%{x:.0f}<br>Service %{y:.1f}%<extra></extra>",
    )
    fig.update_layout(
        margin={"t": 30, "r": 10, "b": 40, "l": 50},
        legend={"orientation": "h", "y": 1.12},
        xaxis_title="Operating cost (€, lower is better)",
        yaxis_title="Service level (%)",
        template="plotly_white",
    )
    return fig


def create_app(settings: Settings | None = None) -> Dash:
    """Build the Dash app. Data is loaded once at construction for snappy callbacks."""
    settings = settings or get_settings()
    forecast = load_forecast_frame(settings)
    pareto = load_pareto(settings)
    sku_ids = sorted(forecast["sku_id"].unique())

    app = Dash(__name__, title="Inventory Forecast AI")

    saving = pareto.get("cost_saving_pct") if pareto else None
    headline = (
        f"At the baseline's service level, optimization cuts operating cost by {saving:.0%}."
        if saving
        else "Run `inventory-forecast-ai optimize` to build the Pareto frontier."
    )

    forecast_tab = dcc.Tab(
        label="Demand forecast",
        children=[
            html.P("Actual vs. CatBoost forecast for the selected SKU.", style={"color": _SLATE}),
            dcc.Dropdown(
                id="sku-dropdown",
                options=[{"label": s, "value": s} for s in sku_ids],
                value=sku_ids[0],
                clearable=False,
                style={"maxWidth": "320px"},
            ),
            dcc.Graph(id="forecast-graph"),
        ],
    )

    pareto_children = [
        html.P(headline, style={"fontWeight": 600, "color": _TEAL}),
    ]
    if pareto:
        pareto_children += [
            dcc.Graph(id="pareto-graph", figure=_pareto_figure(pareto)),
            html.Div(
                id="pareto-detail",
                children="Click a point on the frontier to see that policy.",
                style={"color": _SLATE, "marginTop": "8px"},
            ),
        ]
    pareto_tab = dcc.Tab(label="Cost vs. service (Pareto)", children=pareto_children)

    app.layout = html.Div(
        style={"maxWidth": "920px", "margin": "0 auto", "fontFamily": "system-ui"},
        children=[
            html.H2("Inventory Forecast AI", style={"color": _SLATE}),
            dcc.Tabs([forecast_tab, pareto_tab]),
        ],
    )

    @app.callback(Output("forecast-graph", "figure"), Input("sku-dropdown", "value"))
    def _update_forecast(sku_id: str):
        return _forecast_figure(forecast, sku_id)

    if pareto:

        @app.callback(Output("pareto-detail", "children"), Input("pareto-graph", "clickData"))
        def _show_policy(click_data):
            if not click_data:
                return "Click a point on the frontier to see that policy."
            point = click_data["points"][0]
            idx = point.get("customdata")
            if idx is None:
                return "Baseline policy selected."
            chosen = pareto["points"][idx]
            return (
                f"Selected policy → operating cost €{chosen['cost']:.0f}, "
                f"service {chosen['service'] * 100:.1f}%."
            )

    return app


def main() -> None:
    """Entry point for the `dashboard` CLI command."""
    settings = get_settings()
    app = create_app(settings)
    app.run(host=settings.ui_host, port=settings.ui_port, debug=False)
