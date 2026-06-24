# Inventory Forecast AI

> Predict demand, then explore the **cost-vs-service trade-off frontier** to decide how much to order.

![CI](https://img.shields.io/badge/CI-ruff%20%2B%20pytest-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.12-blue)

**Status:** Phases 1–2 of 5 complete. **CatBoost cuts forecast error (WAPE) by ~40% vs. a naive baseline and ~49% vs. a seasonal-naive baseline**, in a 3-fold walk-forward backtest.

---

## What it does

A compact, end-to-end inventory-planning pipeline:

1. **Forecast demand** per SKU with [CatBoost](https://catboost.ai/), benchmarked against a naive baseline using walk-forward backtesting.
2. **Track experiments** with MLflow (local file store, no server).
3. **Optimize ordering policies** with a **multi-objective metaheuristic** ([pymoo](https://pymoo.org/), NSGA-II). Each candidate policy is scored by **simulating** it against the forecasted (uncertain) demand, so the result is a **Pareto frontier** of total cost vs. service level — not a single point.
4. **Visualize** the forecast and the trade-off frontier in a Dash dashboard.
5. **Orchestrate** the whole pipeline with a minimal Dagster job.

### Why a metaheuristic instead of an exact solver?

The order policy is evaluated by **simulation over forecasted demand scenarios**. Service level (fill rate) is non-linear and effectively black-box, which breaks the clean linear program and genuinely justifies a population-based multi-objective optimizer. The deliverable is a frontier the business can pick from, reflecting that real demand is uncertain.

## The two objectives

| Objective | Meaning |
|-----------|---------|
| **Minimize total cost** | Holding + ordering + purchase cost |
| **Maximize service level** | Fill rate / minimize stockouts |

These genuinely conflict — more safety stock improves service but costs more — which is exactly what makes a Pareto frontier informative. (An optional third objective, e.g. number of orders, may be added later.)

## Demo dataset

Two years of daily sales for ~22 fictional SKUs across five categories, with weekly and yearly seasonality, price promotions, and Poisson noise — fully reproducible from a single seed.

```bash
uv sync --dev
uv run inventory-forecast-ai generate-demo
# -> data/skus.csv      (SKU master + economics)
# -> data/demand.csv    (daily units_sold, on_promo, price)
```

## Forecasting

A single **global CatBoost** model across all SKUs (Poisson loss for count demand),
with calendar features, lags (1/7/14/28), rolling stats, price and promotion flags.
It is benchmarked one-step-ahead with a **walk-forward backtest** against two
baselines — *naive* (yesterday) and *seasonal-naive* (same weekday last week) —
and every run is tracked in **MLflow** (local SQLite backend, no server).

```bash
uv run inventory-forecast-ai train       # backtest + MLflow + save model
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # optional: inspect runs
```

| Model | WAPE | MAE |
|-------|-----:|----:|
| naive | 0.227 | 8.98 |
| seasonal-naive | 0.271 | 10.72 |
| **CatBoost** | **0.137** | **5.43** |

→ **~40% lower WAPE than naive, ~49% lower than seasonal-naive.**

## Project layout

```
src/inventory_forecast_ai/
  config.py        # Pydantic settings (.env)
  log.py           # structlog setup
  models.py        # typed domain models (Sku, Category)
  demo/generator.py# deterministic synthetic data
  forecasting/     # features, baselines, CatBoost model, walk-forward backtest
  cli.py           # typer CLI
tests/             # pytest
```

## Roadmap

- [x] **F1** — Repo scaffold + reproducible synthetic dataset
- [x] **F2** — Features + CatBoost forecast vs. baselines + walk-forward backtest + MLflow
- [ ] **F3** — pymoo multi-objective optimization (simulation-based) + Dash (forecast + Pareto views)
- [ ] **F4** — Metrics: forecast accuracy + cost savings vs. a simple policy
- [ ] **F5** — README polish + CI + Dagster job wiring everything together

---

## Resumen (ES)

Pipeline de planificación de inventario de principio a fin: **predice la demanda** por SKU con CatBoost (frente a un baseline naive, con backtesting walk-forward), registra los experimentos en **MLflow** local, y **optimiza las políticas de pedido** con una **metaheurística multi-objetivo** (pymoo / NSGA-II). Cada política se evalúa **por simulación** sobre la demanda prevista (incierta), así que el resultado es una **frontera de Pareto** coste-vs-nivel de servicio, no un único óptimo. Forecast y frontera se muestran en un dashboard Dash, y todo se encadena con un job mínimo de **Dagster**.

**Por qué metaheurística y no solver exacto:** el nivel de servicio se mide simulando y es no-lineal/caja-negra, lo que rompe el programa lineal y justifica un optimizador poblacional multi-objetivo. La demanda real es incierta; entregar una frontera de decisión es más honesto y más útil que un único número.

## Licencia

MIT
