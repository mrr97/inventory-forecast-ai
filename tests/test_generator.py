"""Tests for the deterministic synthetic demand generator."""

from datetime import date

import pandas as pd
import pytest

from inventory_forecast_ai.demo.generator import generate_demo_data
from inventory_forecast_ai.models import Category, Sku

START = date(2024, 1, 1)
END = date(2024, 3, 31)  # short range keeps tests fast


def _generate(tmp_path, *, count=8, seed=42):
    skus_path = tmp_path / "skus.csv"
    demand_path = tmp_path / "demand.csv"
    n_skus, n_rows = generate_demo_data(
        skus_path,
        demand_path,
        count=count,
        seed=seed,
        start_date=START,
        end_date=END,
    )
    return skus_path, demand_path, n_skus, n_rows


def test_row_counts_match_skus_and_days(tmp_path):
    _, _, n_skus, n_rows = _generate(tmp_path, count=8)
    n_days = (END - START).days + 1
    assert n_skus == 8
    assert n_rows == 8 * n_days


def test_same_seed_is_byte_for_byte_reproducible(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    skus_a, demand_a, _, _ = _generate(a, seed=42)
    skus_b, demand_b, _, _ = _generate(b, seed=42)
    assert skus_a.read_bytes() == skus_b.read_bytes()
    assert demand_a.read_bytes() == demand_b.read_bytes()


def test_different_seed_changes_demand(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _, demand_a, _, _ = _generate(a, seed=42)
    _, demand_b, _, _ = _generate(b, seed=7)
    assert demand_a.read_bytes() != demand_b.read_bytes()


def test_sku_master_validates_against_model(tmp_path):
    skus_path, _, _, _ = _generate(tmp_path, count=10)
    df = pd.read_csv(skus_path)
    # Every row must round-trip through the typed model.
    skus = [Sku(**row) for row in df.to_dict(orient="records")]
    assert len(skus) == 10
    assert {s.category for s in skus} <= set(Category)
    assert df["sku_id"].is_unique


def test_demand_is_non_negative_integer_with_promo_flag(tmp_path):
    _, demand_path, _, _ = _generate(tmp_path)
    df = pd.read_csv(demand_path)
    assert (df["units_sold"] >= 0).all()
    assert df["units_sold"].dtype.kind in "iu"
    assert df["on_promo"].dtype == bool
    # Promotions should exist but be a minority of days.
    assert 0 < df["on_promo"].mean() < 0.5


def test_promo_days_have_lower_price(tmp_path):
    _, demand_path, _, _ = _generate(tmp_path)
    df = pd.read_csv(demand_path)
    for _sku_id, group in df.groupby("sku_id"):
        if group["on_promo"].any() and (~group["on_promo"]).any():
            assert group.loc[group["on_promo"], "price"].max() <= (
                group.loc[~group["on_promo"], "price"].min() + 1e-9
            )


@pytest.mark.parametrize("seed", [0, 1, 99])
def test_runs_for_various_seeds(tmp_path, seed):
    _, _, n_skus, n_rows = _generate(tmp_path, seed=seed)
    assert n_skus == 8
    assert n_rows > 0
