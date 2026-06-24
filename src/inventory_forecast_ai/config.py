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
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment: str = "demand-forecast"

    # Dash UI — wired up in F3
    ui_host: str = "127.0.0.1"
    ui_port: int = 8050

    # Logging
    log_level: str = "INFO"
    json_logs: bool = False


def get_settings() -> Settings:
    """Build a fresh Settings instance (reads `.env` if present)."""
    return Settings()
