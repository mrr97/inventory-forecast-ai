"""Dash app: demand forecast, the cost-vs-service Pareto frontier, and a scorecard."""

from __future__ import annotations

import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from inventory_forecast_ai.config import Settings, get_settings
from inventory_forecast_ai.ui.data import (
    load_forecast_frame,
    load_pareto,
    load_product_names,
    load_report,
)

_TEAL = "#0f9d8f"
_SLATE = "#334155"
_AMBER = "#f59e0b"


def _forecast_figure(frame, sku_id: str) -> go.Figure:
    one = frame[frame["sku_id"] == sku_id]
    fig = go.Figure()
    fig.add_scatter(
        x=one["date"], y=one["actual"], name="Actual", mode="lines", line={"color": _SLATE}
    )
    # Forecast is populated only over the forward window (NaN in the history part).
    fig.add_scatter(
        x=one["date"],
        y=one["forecast"],
        name="Forward forecast",
        mode="lines",
        line={"color": _TEAL, "dash": "dash"},
        connectgaps=False,
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


def _metric_card(label: str, value: str, hint: str) -> html.Div:
    return html.Div(
        style={
            "flex": "1 1 200px",
            "padding": "16px",
            "border": f"1px solid {_SLATE}22",
            "borderRadius": "10px",
            "background": "#f8fafc",
        },
        children=[
            html.Div(value, style={"fontSize": "28px", "fontWeight": 700, "color": _TEAL}),
            html.Div(label, style={"fontWeight": 600, "color": _SLATE, "marginTop": "4px"}),
            html.Div(hint, style={"fontSize": "12px", "color": _SLATE}),
        ],
    )


def _scorecard_children(report: dict | None) -> list:
    if not report:
        return [
            html.P(
                "Run `inventory-forecast-ai metrics` to build the scorecard.",
                style={"color": _SLATE},
            )
        ]
    fc = report["forecast"]
    opt = report["optimization"]
    decision = report.get("decision")
    saving = opt["cost_saving_pct"]
    cards_children = [
        _metric_card(
            "Forecast error (WAPE)",
            f"{fc['wape']['forward']:.3f}",
            f"{fc['improvement_vs_trailing']:.0%} better than the trailing mean",
        ),
        _metric_card(
            "Cost saving",
            "—" if saving is None else f"{saving:.0%}",
            "at the baseline's service level",
        ),
    ]
    if decision is not None:
        cards_children.append(
            _metric_card(
                "Seasonal-stretch cost cut",
                f"{decision['cost_reduction']:.0%}",
                f"{decision['service_gain'] * 100:+.1f} pp service vs. trailing",
            )
        )
    cards = html.Div(
        style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "marginTop": "8px"},
        children=cards_children,
    )
    paragraphs = [
        html.P(report["headline"]["forecast"] + ".", style={"color": _SLATE}),
        html.P(report["headline"]["optimization"] + ".", style={"color": _SLATE}),
    ]
    if "decision" in report["headline"]:
        paragraphs.append(html.P(report["headline"]["decision"] + ".", style={"color": _SLATE}))
    return [*paragraphs, cards]


def _policy_detail(pareto: dict, click_data: dict | None, product_names: dict[str, str]):
    if not click_data or not click_data.get("points"):
        return "Click a point on the frontier to see replenishment levels for each product."
    idx = click_data["points"][0].get("customdata")
    if idx is None:
        return "Baseline policy selected. Select an optimized policy to see product-level targets."

    chosen = pareto["points"][idx]
    sku_ids = pareto["sku_ids"]
    n = len(sku_ids)
    policy = chosen["policy"]  # Saved as [s for every SKU, S for every SKU].
    rows = []
    cell_style = {"padding": "10px 12px", "borderBottom": "1px solid #e2e8f0"}
    number_style = {**cell_style, "textAlign": "right", "fontVariantNumeric": "tabular-nums"}
    for sku_id, reorder, target in zip(sku_ids, policy[:n], policy[n:], strict=True):
        rows.append(
            html.Tr(
                [
                    html.Th(sku_id, scope="row", style={**cell_style, "fontWeight": 400}),
                    html.Td(product_names.get(sku_id, sku_id), style=cell_style),
                    html.Td(f"{reorder:,.2f}", style=number_style),
                    html.Td(f"{target:,.2f}", style=number_style),
                ]
            )
        )

    return html.Div(
        [
            html.P(
                f"Selected policy → operating cost €{chosen['cost']:.0f}, "
                f"service {chosen['service'] * 100:.1f}%.",
                style={"fontWeight": 600},
            ),
            html.P(
                "Reorder when stock on hand plus incoming orders is at or below the reorder "
                "point (s). Order enough to reach the target level (S), subject to the "
                "product's minimum order quantity. All quantities are in units, displayed "
                "to two decimal places."
            ),
            html.Div(
                html.Table(
                    [
                        html.Caption(
                            "Replenishment levels by product",
                            style={"textAlign": "left", "fontWeight": 600, "padding": "8px 12px"},
                        ),
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th(label, scope="col", style=cell_style)
                                    for label in (
                                        "SKU",
                                        "Product",
                                        "Reorder point (s)",
                                        "Replenish up to (S)",
                                    )
                                ],
                                style={"background": "#f8fafc"},
                            )
                        ),
                        html.Tbody(rows),
                    ],
                    style={"width": "100%", "borderCollapse": "collapse", "textAlign": "left"},
                ),
                style={"overflowX": "auto"},
            ),
        ]
    )


def create_app(settings: Settings | None = None) -> Dash:
    """Build the Dash app. Data is loaded once at construction for snappy callbacks."""
    settings = settings or get_settings()
    forecast = load_forecast_frame(settings)
    pareto = load_pareto(settings)
    report = load_report(settings)
    product_names = load_product_names(settings) if pareto else {}
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
            html.P(
                "Recent demand and the model's forward forecast for the selected SKU.",
                style={"color": _SLATE},
            ),
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
                children=_policy_detail(pareto, None, product_names),
                style={"color": _SLATE, "marginTop": "8px"},
            ),
        ]
    pareto_tab = dcc.Tab(label="Cost vs. service (Pareto)", children=pareto_children)
    scorecard_tab = dcc.Tab(label="Scorecard", children=_scorecard_children(report))

    app.layout = html.Div(
        style={"maxWidth": "920px", "margin": "0 auto", "fontFamily": "system-ui"},
        children=[
            html.H2("Inventory Forecast AI", style={"color": _SLATE}),
            dcc.Tabs([scorecard_tab, forecast_tab, pareto_tab]),
        ],
    )

    @app.callback(Output("forecast-graph", "figure"), Input("sku-dropdown", "value"))
    def _update_forecast(sku_id: str):
        return _forecast_figure(forecast, sku_id)

    if pareto:

        @app.callback(Output("pareto-detail", "children"), Input("pareto-graph", "clickData"))
        def _show_policy(click_data):
            return _policy_detail(pareto, click_data, product_names)

    return app


def main() -> None:
    """Entry point for the `dashboard` CLI command."""
    settings = get_settings()
    app = create_app(settings)
    app.run(host=settings.ui_host, port=settings.ui_port, debug=False)
