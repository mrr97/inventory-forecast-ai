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

    # MLflow experiment tracking (local file store, no server) — wired up in F2
    # Local SQLite backend = no server, single file. (MLflow deprecated the bare
    # file store; SQLite is the recommended serverless local backend.)
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "demand-forecast"

    # Trained model artifact (CatBoost) — produced in F2, consumed in F3
    model_dir: Path = Path("models")
    model_path: Path = Path("models/catboost_demand.cbm")

    # Walk-forward backtest: evaluate the last `folds * fold_days` days, one-step-ahead.
    backtest_folds: int = 3
    backtest_fold_days: int = 28

    # CatBoost compute device. "CPU" is the default and keeps the project fully
    # portable (runs anywhere with `docker compose up`, deterministic). For very
    # large catalogues (hundreds of thousands of SKUs) set "GPU" — CatBoost will
    # train on CUDA with no other code change. Not worth it at this data size, where
    # GPU init overhead outweighs the compute. See the README "Scaling" note.
    catboost_task_type: str = "CPU"

    # Multi-objective inventory optimization (F3).
    opt_horizon_days: int = 28  # planning horizon simulated per policy
    opt_scenarios: int = 50  # Monte-Carlo demand scenarios per evaluation
    opt_pop_size: int = 80  # NSGA-II population
    opt_generations: int = 60  # NSGA-II generations
    opt_seed: int = 42
    # Optional shared cap on average inventory investment (EUR). When set it couples
    # the SKUs, turning the problem into a genuine joint allocation. None = unconstrained.
    inventory_budget: float | None = None
    pareto_path: Path = Path("data/optimization/pareto.json")

    # Consolidated metrics scorecard (F4). `train` writes the forecast metrics,
    # `optimize` writes the Pareto frontier, and `metrics` merges both into a single
    # report the dashboard reads and the README cites.
    forecast_metrics_path: Path = Path("data/metrics/forecast.json")
    report_path: Path = Path("data/metrics/report.json")

    # Dash UI — wired up in F3
    ui_host: str = "127.0.0.1"
    ui_port: int = 8050

    # Logging
    log_level: str = "INFO"
    json_logs: bool = False


def get_settings() -> Settings:
    """Build a fresh Settings instance (reads `.env` if present)."""
    return Settings()
