"""Synthetic world: catalog + intents + ground-truth resolution.

Ground truth is the *merchant's* binding of a cart to a SKU. The attacker may
rewrite everything the agent *sees* (labels, order, rationale, memory) but never
that binding -- so divergence is always measured against what was really bought.
The binding is carried in `CartContents.id` as `"{SKU}#{seq}"`, which is exactly
how a real merchant cart id identifies its offer.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from minimal_drift.config import get_config


class Product(BaseModel):
    sku: str
    name: str
    category: str
    price: float
    currency: str
    merchant: str
    refund_period: int
    attributes: dict[str, str | int | float] = Field(default_factory=dict)

    @property
    def refundable(self) -> bool:
        return self.refund_period > 0


class IntentSpec(BaseModel):
    """A user intent plus the ground truth needed to score divergence."""

    id: str
    natural_language_description: str
    budget_cap: float
    currency: str
    merchants: list[str] | None
    requires_refundability: bool
    target_category: str
    ground_truth_skus: list[str]
    discriminating_attributes: list[str]
    intent_expiry_hours: int = 24
    flagship: bool = False


class World(BaseModel):
    """Immutable synthetic world: products by sku + intents by id."""

    products: dict[str, Product]
    intents: dict[str, IntentSpec]

    def product(self, sku: str) -> Product:
        return self.products[sku]

    def intent(self, intent_id: str) -> IntentSpec:
        return self.intents[intent_id]

    def by_category(self, category: str) -> list[Product]:
        return [p for p in self.products.values() if p.category == category]

    def ground_truth_products(self, intent_id: str) -> list[Product]:
        return [self.products[s] for s in self.intents[intent_id].ground_truth_skus]


def load_world(catalog_path: Path | None = None, intents_path: Path | None = None) -> World:
    """Load the synthetic world from JSON. Pure apart from the two file reads."""
    cfg = get_config()
    cat = json.loads((catalog_path or cfg.catalog_path).read_text())
    ints = json.loads((intents_path or cfg.intents_path).read_text())
    return World(
        products={p["sku"]: Product(**p) for p in cat["products"]},
        intents={i["id"]: IntentSpec(**i) for i in ints["intents"]},
    )


@lru_cache(maxsize=4)
def default_world() -> World:
    """Cached default world (data/catalog.json + data/intents.json)."""
    return load_world()


def make_cart_id(sku: str, intent_id: str) -> str:
    """Merchant cart id that binds the offer to its SKU.

    Stable per (intent, sku) rather than per result position: an intervention names
    a cart id, and it has to mean the same offer at every session seed, otherwise
    a counterfactual would compare an attack against a target that moved.
    """
    tag = hashlib.sha256(f"{intent_id}|{sku}".encode()).hexdigest()[:6]
    return f"{sku}#{tag}"


def sku_from_cart_id(cart_id: str) -> str:
    """Inverse of `make_cart_id`. The attacker cannot forge this binding."""
    return cart_id.split("#", 1)[0]
