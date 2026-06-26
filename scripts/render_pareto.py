"""Render the Pareto frontier as a static hero image for the README.

Reads the committed ``data/optimization/pareto.json`` (produced by
``inventory-forecast-ai optimize``) and writes ``docs/pareto.png`` — the cost-vs-service
trade-off frontier, with the textbook safety-stock baseline and the optimizer's
equal-service policy highlighted. Regenerate with:

    uv run python scripts/render_pareto.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PARETO_PATH = ROOT / "data" / "optimization" / "pareto.json"
OUT_PATH = ROOT / "docs" / "pareto.png"

_INK = "#1e293b"
_FRONTIER = "#2563eb"
_BASELINE = "#dc2626"
_PICK = "#059669"


def main() -> None:
    data = json.loads(PARETO_PATH.read_text(encoding="utf-8"))
    baseline = data["baseline"]
    points = sorted(data["points"], key=lambda p: p["cost"])
    saving = data["cost_saving_pct"]

    costs = [p["cost"] for p in points]
    services = [p["service"] * 100 for p in points]

    # The optimizer's equal-service policy: cheapest frontier point that still meets
    # the baseline's service level — this is where the cost saving is quoted.
    at_service = min(
        (p for p in points if p["service"] >= baseline["service"]),
        key=lambda p: p["cost"],
        default=None,
    )

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    ax.plot(costs, services, "-", color=_FRONTIER, alpha=0.35, zorder=1)
    ax.scatter(
        costs,
        services,
        s=34,
        color=_FRONTIER,
        edgecolor="white",
        linewidth=0.6,
        zorder=2,
        label=f"Pareto frontier ({len(points)} policies)",
    )
    ax.scatter(
        [baseline["cost"]],
        [baseline["service"] * 100],
        s=170,
        marker="X",
        color=_BASELINE,
        edgecolor="white",
        linewidth=1.0,
        zorder=4,
        label="Safety-stock baseline",
    )
    if at_service is not None:
        ax.scatter(
            [at_service["cost"]],
            [at_service["service"] * 100],
            s=170,
            marker="*",
            color=_PICK,
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
            label=f"Optimizer @ baseline service (-{saving * 100:.0f}% cost)",
        )
        ax.annotate(
            f"same service,\n-{saving * 100:.0f}% cost",
            xy=(at_service["cost"], at_service["service"] * 100),
            xytext=(
                at_service["cost"] + (max(costs) - min(costs)) * 0.06,
                at_service["service"] * 100 - 6,
            ),
            color=_PICK,
            fontsize=10,
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": _PICK},
        )

    ax.set_xlabel("Operating cost (EUR)", color=_INK, fontsize=11)
    ax.set_ylabel("Service level (%)", color=_INK, fontsize=11)
    ax.set_title(
        "Inventory ordering: cost vs. service trade-off frontier",
        color=_INK,
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, alpha=0.18)
    ax.tick_params(colors=_INK)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"Wrote {OUT_PATH.relative_to(ROOT)}  ({len(points)} frontier policies)")


if __name__ == "__main__":
    main()
