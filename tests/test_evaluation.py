"""Tests for the consolidated metrics scorecard (forecast + optimization + decision)."""

import json

import pytest

from inventory_forecast_ai.evaluation.report import (
    MissingArtifactError,
    build_report,
    save_report,
)


class _Settings:
    def __init__(self, forecast_path, pareto_path, report_path, comparison_path):
        self.forecast_metrics_path = forecast_path
        self.pareto_path = pareto_path
        self.report_path = report_path
        self.comparison_path = comparison_path


def _forecast_json() -> dict:
    return {
        "horizon": 14,
        "wape": {"forward": 0.131, "trailing": 0.253},
        "mae": {"forward": 5.43, "trailing": 10.72},
        "wape_promo": {"forward": 0.118, "trailing": 0.477},
        "improvement_vs_trailing": 0.482,
        "promo_improvement_vs_trailing": 0.753,
        "per_horizon_wape": {"forward": [0.13] * 14, "trailing": [0.25] * 14},
        "forecast_cv": [0.09] * 14,
        "n_origins": 106,
        "n_rows": 32648,
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


def _comparison_json() -> dict:
    return {
        "window_start": "2025-08-09",
        "window_end": "2025-10-08",
        "n_origins": 9,
        "forward_cost": 2749.0,
        "trailing_cost": 3385.0,
        "forward_operating": 2361.0,
        "trailing_operating": 2346.0,
        "forward_service": 0.993,
        "trailing_service": 0.983,
        "cost_reduction": 0.188,
        "service_gain": 0.0108,
        "per_origin": [],
    }


def _settings(tmp_path, *, with_comparison: bool = True) -> _Settings:
    fpath = tmp_path / "forecast.json"
    ppath = tmp_path / "pareto.json"
    cpath = tmp_path / "comparison.json"
    fpath.write_text(json.dumps(_forecast_json()), encoding="utf-8")
    ppath.write_text(json.dumps(_pareto_json()), encoding="utf-8")
    if with_comparison:
        cpath.write_text(json.dumps(_comparison_json()), encoding="utf-8")
    return _Settings(fpath, ppath, tmp_path / "report.json", cpath)


def test_build_report_consolidates_all_parts(tmp_path):
    report = build_report(_settings(tmp_path))

    assert report["forecast"]["wape"]["forward"] == 0.131
    assert report["optimization"]["n_pareto_points"] == 2
    assert report["decision"]["cost_reduction"] == 0.188
    assert report["optimization"]["cheapest_cost"] == 3072.0
    assert "48%" in report["headline"]["forecast"]
    assert "75%" in report["headline"]["forecast"]
    assert "12%" in report["headline"]["optimization"]
    assert "19%" in report["headline"]["decision"]


def test_build_report_without_comparison_degrades_gracefully(tmp_path):
    report = build_report(_settings(tmp_path, with_comparison=False))
    assert "decision" not in report
    assert "decision" not in report["headline"]


def test_build_report_errors_clearly_when_artifact_missing(tmp_path):
    settings = _Settings(
        tmp_path / "absent.json",
        tmp_path / "absent2.json",
        tmp_path / "report.json",
        tmp_path / "absent3.json",
    )
    with pytest.raises(MissingArtifactError, match="train"):
        build_report(settings)


def test_save_report_roundtrips(tmp_path):
    report = build_report(_settings(tmp_path))
    out = tmp_path / "nested" / "report.json"
    save_report(report, out)
    assert json.loads(out.read_text(encoding="utf-8")) == report
