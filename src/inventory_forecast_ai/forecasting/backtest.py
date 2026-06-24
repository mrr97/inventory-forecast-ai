"""Walk-forward backtest comparing CatBoost against two naive baselines.

Evaluation is one-step-ahead over the most recent ``folds * fold_days`` days. For
each fold we train on everything before the fold (expanding window) and score the
fold's actuals against:

- **naive**: yesterday's demand (``lag_1``),
- **seasonal_naive**: same weekday last week (``lag_7``) — the honest stand-in for
  a classical seasonal time-series model,
- **catboost**: the global gradient-boosted model.

Predictions from all folds are pooled before computing the final metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import structlog

from inventory_forecast_ai.forecasting.features import TARGET
from inventory_forecast_ai.forecasting.metrics import mae, wape
from inventory_forecast_ai.forecasting.model import predict, train_model

log = structlog.get_logger(__name__)

MODELS = ("naive", "seasonal_naive", "catboost")


@dataclass
class BacktestResult:
    """Pooled, one-step-ahead accuracy per model plus per-fold detail."""

    mae: dict[str, float]
    wape: dict[str, float]
    n_folds: int
    n_test_rows: int
    per_fold: list[dict] = field(default_factory=list)

    def wape_improvement_vs(self, baseline: str) -> float:
        """Relative WAPE reduction of CatBoost vs. the given baseline (0..1)."""
        base = self.wape[baseline]
        return float((base - self.wape["catboost"]) / base) if base else float("nan")


def walk_forward(
    features: pd.DataFrame,
    *,
    folds: int,
    fold_days: int,
    params: dict | None = None,
) -> BacktestResult:
    """Run the expanding-window, one-step-ahead backtest."""
    df = features.sort_values(["date", "sku_id"]).reset_index(drop=True)
    # Work in terms of the sorted unique calendar days; the last `folds * fold_days`
    # days form the contiguous test blocks (oldest fold first).
    unique_days = np.sort(df["date"].unique())

    pooled: dict[str, list[np.ndarray]] = {m: [] for m in MODELS}
    pooled_true: list[np.ndarray] = []
    per_fold: list[dict] = []

    for k in range(folds):
        start_idx = len(unique_days) - (folds - k) * fold_days
        end_idx = start_idx + fold_days
        if start_idx <= 0:
            continue
        test_days = unique_days[start_idx:end_idx]
        test_start, test_end = test_days[0], test_days[-1]

        train_df = df[df["date"] < test_start]
        test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)]
        if train_df.empty or test_df.empty:
            continue

        y_true = test_df[TARGET].to_numpy()
        model = train_model(train_df, params)
        preds = {
            "naive": test_df["lag_1"].to_numpy(),
            "seasonal_naive": test_df["lag_7"].to_numpy(),
            "catboost": predict(model, test_df),
        }

        pooled_true.append(y_true)
        fold_metrics = {"fold": k, "test_rows": len(test_df)}
        for name, p in preds.items():
            pooled[name].append(p)
            fold_metrics[f"{name}_wape"] = wape(y_true, p)
        per_fold.append(fold_metrics)
        log.info(
            "backtest_fold",
            fold=k,
            test_window=f"{pd.Timestamp(test_start).date()}..{pd.Timestamp(test_end).date()}",
            rows=len(test_df),
            catboost_wape=round(fold_metrics["catboost_wape"], 4),
        )

    y_all = np.concatenate(pooled_true)
    mae_by = {m: mae(y_all, np.concatenate(pooled[m])) for m in MODELS}
    wape_by = {m: wape(y_all, np.concatenate(pooled[m])) for m in MODELS}
    return BacktestResult(
        mae=mae_by,
        wape=wape_by,
        n_folds=len(per_fold),
        n_test_rows=len(y_all),
        per_fold=per_fold,
    )
