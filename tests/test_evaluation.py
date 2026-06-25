"""Tests for the consolidated metrics scorecard (F4)."""

import json

import pytest

from inventory_forecast_ai.evaluation.report import (
    MissingArtifactError,
    build_report,
    save_report,
)


class _Settings:
    def __init__(self, forecast_path, pareto_path, report_path):
        self.forecast_metrics_path = forecast_path
        self.pareto_path = pareto_path
        self.report_path = report_path


def _forecast_json() -> dict:
    return {
        "wape": {"naive": 0.227, "seasonal_naive": 0.271, "catboost": 0.137},
        "mae": {"naive": 8.98, "seasonal_naive": 10.72, "catboost": 5.43},
        "improvement_vs_naive": 0.396,
        "improvement_vs_seasonal_naive": 0.494,
        "n_folds": 3,
        "n_test_rows": 1848,
    }


def _pareto_json() -> dict:
    return {
        "sku_ids": ["SKU-1001"],
        "baseline": {"cost": 4561.0, "stockout": 0.0022, "service": 0.9978},
        "cost_saving_pct": 0.12,
        "points": [
            {"cost": 3072.0, "stockout": 0.107, "service": 0.893, "policy": [1, 2]},
            {"cost": 4567.0, "stockout": 0.0, "service": 1.0, "policy": [9, 9]},
        ],
    }


def _settings(tmp_path) -> _Settings:
    fpath = tmp_path / "forecast.json"
    ppath = tmp_path / "pareto.json"
    fpath.write_text(json.dumps(_forecast_json()), encoding="utf-8")
    ppath.write_text(json.dumps(_pareto_json()), encoding="utf-8")
    return _Settings(fpath, ppath, tmp_path / "report.json")


def test_build_report_consolidates_both_halves(tmp_path):
    report = build_report(_settings(tmp_path))

    assert report["forecast"]["wape"]["catboost"] == 0.137
    assert report["optimization"]["n_pareto_points"] == 2
    assert report["optimization"]["cost_saving_pct"] == 0.12
    # Cheapest and best-service points are picked correctly from the frontier.
    assert report["optimization"]["cheapest_cost"] == 3072.0
    assert report["optimization"]["best_service"] == 1.0
    assert "40%" in report["headline"]["forecast"]
    assert "12%" in report["headline"]["optimization"]


def test_build_report_handles_missing_cost_saving(tmp_path):
    settings = _settings(tmp_path)
    data = json.loads(settings.pareto_path.read_text())
    data["cost_saving_pct"] = None
    settings.pareto_path.write_text(json.dumps(data), encoding="utf-8")

    report = build_report(settings)
    assert report["optimization"]["cost_saving_pct"] is None
    assert "no cost saving" in report["headline"]["optimization"].lower()


def test_build_report_errors_clearly_when_artifact_missing(tmp_path):
    settings = _Settings(
        tmp_path / "absent.json", tmp_path / "absent2.json", tmp_path / "report.json"
    )
    with pytest.raises(MissingArtifactError, match="train"):
        build_report(settings)


def test_save_report_roundtrips(tmp_path):
    report = build_report(_settings(tmp_path))
    out = tmp_path / "nested" / "report.json"
    save_report(report, out)
    assert json.loads(out.read_text(encoding="utf-8")) == report
