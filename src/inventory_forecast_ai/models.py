"""Typed domain models shared across the pipeline.

These Pydantic models are the contract at the boundaries (data generation,
optimization inputs, API/UI). Heavy numeric work happens on pandas/numpy; these
models validate and document the small, structured records that travel between
stages.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class Category(StrEnum):
    """Product family. Drives demand seasonality patterns in the demo data."""

    BEVERAGES = "beverages"
    SNACKS = "snacks"
    HOUSEHOLD = "household"
    PERSONAL_CARE = "personal_care"
    STATIONERY = "stationery"


class Sku(BaseModel):
    """A stock-keeping unit and its economics.

    The cost fields feed the inventory optimizer: holding cost penalises
    excess stock, ordering cost penalises frequent replenishment, and the lead
    time governs how early an order must be placed.
    """

    sku_id: str = Field(pattern=r"^SKU-\d{4}$")
    name: str
    category: Category
    unit_cost: float = Field(gt=0, description="Purchase cost per unit (EUR).")
    price: float = Field(gt=0, description="Sale price per unit (EUR).")
    holding_cost_per_unit_per_day: float = Field(
        gt=0, description="Cost of carrying one unit in stock for one day (EUR)."
    )
    order_cost: float = Field(
        gt=0, description="Fixed cost incurred per replenishment order (EUR)."
    )
    lead_time_days: int = Field(ge=1, le=14, description="Days from order to delivery.")
    min_order_qty: int = Field(ge=0, description="Minimum units per replenishment order.")
    base_demand: float = Field(gt=0, description="Average daily units sold, before seasonality.")
