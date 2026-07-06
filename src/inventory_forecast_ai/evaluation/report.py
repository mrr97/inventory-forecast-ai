"""Build the consolidated metrics scorecard from the persisted artifacts.

The report merges the persisted outputs into a single JSON the dashboard reads and the
README cites:

- forecast accuracy (WAPE forward vs. trailing, overall and on promo days), by ``train``,
- inventory cost vs. service (Pareto frontier + saving vs. the baseline policy), by
  ``optimize``,
- the decision-level forward-vs-trailing comparison on the seasonal stretch, by
  ``compare`` (optional — folded in when present).

It deliberately *reads artifacts* rather than recomputing, so the scorecard reflects
exactly the committed model, frontier and comparison, and is cheap to regenerate.
"""

from __future__ import annotations

import json
from pathlib import Path

from inventory_forecast_ai.config import Settings


class MissingArtifactError(FileNotFoundError):
    """Raised when a required upstream artifact has not been produced yet."""


def _load_json(path: Path, *, produced_by: str) -> dict:
    if not path.exists():
        raise MissingArtifactError(
            f"{path} not found — run `inventory-forecast-ai {produced_by}` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _forecast_block(forecast: dict) -> dict:
    return {
        "horizon": forecast["horizon"],
        "wape": forecast["wape"],  # {"forward", "trailing"}
        "mae": forecast["mae"],
        "wape_promo": forecast["wape_promo"],
        "improvement_vs_trailing": forecast["improvement_vs_trailing"],
        "promo_improvement_vs_trailing": forecast["promo_improvement_vs_trailing"],
        "n_origins": forecast["n_origins"],
        "n_rows": forecast["n_rows"],
    }


def _optimization_block(pareto: dict) -> dict:
    points = pareto["points"]
    cheapest = min(points, key=lambda p: p["cost"])
    best_service = max(points, key=lambda p: p["service"])
    return {
        "n_pareto_points": len(points),
        "baseline_cost": pareto["baseline"]["cost"],
        "baseline_service": pareto["baseline"]["service"],
        "cost_saving_pct": pareto["cost_saving_pct"],
        "cheapest_cost": cheapest["cost"],
        "cheapest_service": cheapest["service"],
        "best_service": best_service["service"],
        "best_service_cost": best_service["cost"],
    }


def _decision_block(comparison: dict) -> dict:
    return {
        "window_start": comparison["window_start"],
        "window_end": comparison["window_end"],
        "cost_reduction": comparison["cost_reduction"],
        "service_gain": comparison["service_gain"],
        "forward_cost": comparison["forward_cost"],
        "trailing_cost": comparison["trailing_cost"],
    }


def _headline(forecast: dict, optimization: dict, decision: dict | None) -> dict[str, str]:
    fc = (
        f"Forward forecast cuts error (WAPE) {forecast['improvement_vs_trailing']:.0%} vs. the "
        f"trailing mean, {forecast['promo_improvement_vs_trailing']:.0%} on promo days"
    )
    saving = optimization["cost_saving_pct"]
    opt = (
        f"At the baseline's service level, optimization cuts operating cost {saving:.0%}"
        if saving is not None
        else "Pareto frontier built; no cost saving computed"
    )
    headline = {"forecast": fc, "optimization": opt}
    if decision is not None:
        headline["decision"] = (
            f"On the seasonal stretch, planning ahead cuts total cost "
            f"{decision['cost_reduction']:.0%} at {decision['service_gain'] * 100:+.1f} pp service "
            f"vs. the trailing mean"
        )
    return headline


def build_report(settings: Settings) -> dict:
    """Consolidate the forecast, optimization (and optional decision) artifacts.

    The decision-level forward-vs-trailing comparison is optional: it is folded in when
    its artifact exists, so the scorecard degrades gracefully before ``compare`` is run.
    """
    forecast = _load_json(settings.forecast_metrics_path, produced_by="train")
    pareto = _load_json(settings.pareto_path, produced_by="optimize")

    forecast_block = _forecast_block(forecast)
    optimization_block = _optimization_block(pareto)

    decision_block = None
    comparison_path = getattr(settings, "comparison_path", None)
    if comparison_path is not None and comparison_path.exists():
        decision_block = _decision_block(json.loads(comparison_path.read_text(encoding="utf-8")))

    report = {
        "forecast": forecast_block,
        "optimization": optimization_block,
        "headline": _headline(forecast_block, optimization_block, decision_block),
    }
    if decision_block is not None:
        report["decision"] = decision_block
    return report


def save_report(report: dict, path: Path) -> None:
    """Write the consolidated scorecard as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_metrics(settings: Settings) -> dict:
    """Build the scorecard and persist it to ``settings.report_path``."""
    report = build_report(settings)
    save_report(report, settings.report_path)
    return report
