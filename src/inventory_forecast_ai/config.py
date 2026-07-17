"""Application settings loaded from environment variables / .env file."""

from datetime import date
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration. Every value can be overridden via env vars or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Synthetic demo dataset (reproducible by seed)
    data_dir: Path = Path("data")
    skus_path: Path = Path("data/skus.csv")
    demand_path: Path = Path("data/demand.csv")
    demo_seed: int = 42
    demo_sku_count: int = 22
    # Two full years of daily history, ending on a fixed date for reproducibility.
    demo_start_date: date = date(2024, 1, 1)
    demo_end_date: date = date(2025, 12, 31)

    # MLflow experiment tracking. Local SQLite backend = no server, single file.
    # (MLflow deprecated the bare file store; SQLite is the recommended local backend.)
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "demand-forecast"

    # Trained model artifact (CatBoost) — produced by `train`, consumed by `optimize`.
    model_dir: Path = Path("models")
    model_path: Path = Path("models/catboost_demand.cbm")

    # Rolling-origin backtest: train on all but the last `holdout_days`, then roll the
    # origin through the holdout scoring the full forward path (t+1 .. t+HORIZON).
    backtest_holdout_days: int = 120

    # Early stopping: reserve the most recent `early_stopping_val_days` of (target-day)
    # history as a temporal validation tail. CatBoost picks the number of iterations by
    # early stopping on it, so the model's `iterations` is only an upper bound — no
    # hand-tuned magic number. Set `early_stopping_val_days = 0` to fall back to a plain
    # fit at the ceiling.
    early_stopping_val_days: int = 28
    early_stopping_rounds: int = 50

    # Small temporal grid search over depth / learning_rate (see model.DEFAULT_SEARCH_GRID).
    # Selected on the same temporal validation tail as early stopping, on pre-holdout data
    # so it never sees the backtest window. A sanity check on the defaults, not an
    # exhaustive sweep. Set False to skip it and train at the default hyperparameters.
    hyperparameter_search: bool = True

    # CatBoost compute device. Defaults to "GPU": training always runs on CUDA with no
    # other code change. Override to "CPU" (e.g. `CATBOOST_TASK_TYPE=CPU`) on machines
    # without a GPU or for fully portable/deterministic runs. See the README "Scaling" note.
    catboost_task_type: str = "GPU"

    # Multi-objective inventory optimization. The planning horizon is fixed at the
    # protection interval (weekly review + max lead time) in `forecasting.features.HORIZON`
    # — a single source of truth — so it is not duplicated here.
    opt_scenarios: int = 50  # Monte-Carlo demand scenarios per evaluation
    opt_pop_size: int = 80  # NSGA-II population
    opt_generations: int = 60  # NSGA-II generations
    opt_seed: int = 42
    # Optional shared cap on average inventory investment (EUR). When set it couples
    # the SKUs, turning the problem into a genuine joint allocation. None = unconstrained.
    inventory_budget: float | None = None
    pareto_path: Path = Path("data/optimization/pareto.json")

    # Consolidated metrics scorecard. `train` writes the forecast metrics, `optimize`
    # writes the Pareto frontier, `compare` writes the decision comparison, and `metrics`
    # merges them into a single report the dashboard reads and the README cites.
    forecast_metrics_path: Path = Path("data/metrics/forecast.json")
    report_path: Path = Path("data/metrics/report.json")
    # Decision-level forward-vs-trailing comparison on the most seasonal stretch.
    comparison_path: Path = Path("data/metrics/comparison.json")

    # Dash UI
    ui_host: str = "127.0.0.1"
    ui_port: int = 8050

    # Logging
    log_level: str = "INFO"
    json_logs: bool = False


def get_settings() -> Settings:
    """Build a fresh Settings instance (reads `.env` if present)."""
    return Settings()
