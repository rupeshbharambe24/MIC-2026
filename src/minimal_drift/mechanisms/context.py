"""What a mechanism needs to know about the world it is attacking.

Everything here is derived from the intent, the catalog and what the merchant
actually surfaced -- no target-model queries. That is deliberate: a candidate that
cannot possibly clear the validator or tau should be ruled out for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

from minimal_drift.catalog.ground_truth import IntentSpec, Product, World
from minimal_drift.harness.agent import _is_stored_value, intent_tokens
from minimal_drift.harness.ap2_env import CartOffer
from minimal_drift.scoring.divergence import breakdown_for_sku


@dataclass
class MechanismContext:
    """The attacker's read-only view of one (intent, retrieved offers) pair."""

    world: World
    spec: IntentSpec
    offers: list[CartOffer]
    tau: float

    # -- validity, computed without querying the target --------------------- #
    def would_validate(self, p: Product) -> bool:
        """Could a payment for this product pass the constraint engine at all?"""
        if p.currency != self.spec.currency or p.price > self.spec.budget_cap:
            return False
        if self.spec.merchants and p.merchant not in self.spec.merchants:
            return False
        if self.spec.requires_refundability and p.refund_period <= 0:
            return False
        return True

    def divergence_of(self, p: Product) -> float:
        return breakdown_for_sku(p.sku, self.spec.id, self.world).total

    def needs_consent(self, p: Product) -> bool:
        """True when the human-present confirmation step would catch this substitution.

        The user is shown the merchant-SIGNED cart, so no feed edit hides a stored-value
        substitution from them -- only the consent surface can. A mechanism that cannot
        write to the consent surface will be blocked on these targets, and that block is
        reported as `consent_blocked` for the mutator to reflect on.
        """
        return _is_stored_value(p.name) and not _is_stored_value(
            self.spec.natural_language_description
        )

    # -- the option set ---------------------------------------------------- #
    @cached_property
    def incumbents(self) -> list[CartOffer]:
        """Offers a faithful agent may buy: the ground-truth set, as retrieved."""
        gt = set(self.spec.ground_truth_skus)
        return [o for o in self.offers if o.product.sku in gt]

    @cached_property
    def candidates(self) -> list[CartOffer]:
        """Retrieved offers that are payment-valid and divergent enough to matter."""
        out = [
            o
            for o in self.offers
            if o.product.sku not in set(self.spec.ground_truth_skus)
            and self.would_validate(o.product)
            and self.divergence_of(o.product) >= self.tau
        ]
        return sorted(
            out, key=lambda o: (-self.divergence_of(o.product), -o.product.price, o.product.sku)
        )

    @cached_property
    def self_sufficient_candidates(self) -> list[CartOffer]:
        """Candidates a single non-consent mechanism could actually complete."""
        return [o for o in self.candidates if not self.needs_consent(o.product)]

    def top_incumbent(self) -> CartOffer | None:
        return self.incumbents[0] if self.incumbents else None

    SELF_SUFFICIENT_BIAS = 0.6

    def pick_target(self, rng: Any, prefer_self_sufficient: bool = True) -> CartOffer | None:
        """Choose a drift target.

        Biased toward targets this mechanism can finish alone, but NOT restricted to
        them. Excluding the rest entirely was a mistake: the highest-value targets are
        stored value, they need the consent surface, and never proposing them meant the
        `consent_blocked` boundary was never hit -- so the reflection path that repairs
        it was never exercised and the most expensive attack in the space was never
        found (INR 7,695 vs INR 1,912).
        """
        pool = self.candidates
        if (
            prefer_self_sufficient
            and self.self_sufficient_candidates
            and rng.random() < self.SELF_SUFFICIENT_BIAS
        ):
            pool = self.self_sufficient_candidates
        if not pool:
            return None
        # Bias to the highest-divergence target, but keep the tail reachable.
        head = pool[: max(1, len(pool) // 2)]
        return head[rng.randrange(len(head))]

    # -- text material ----------------------------------------------------- #
    @cached_property
    def intent_phrase(self) -> str:
        """The user's own content words, for prose that echoes the request back."""
        want = intent_tokens(self.spec.natural_language_description)
        ordered = [
            t
            for t in self.spec.natural_language_description.lower().replace(",", " ").split()
            if t.strip(".") in want
        ]
        return " ".join(dict.fromkeys(t.strip(".") for t in ordered)) or "your request"

    @cached_property
    def discriminators(self) -> list[str]:
        """The attributes that separate the ground truth from everything else."""
        return list(self.spec.discriminating_attributes)


def build_context(
    world: World, intent_id: str, offers: list[CartOffer], tau: float
) -> MechanismContext:
    return MechanismContext(world=world, spec=world.intent(intent_id), offers=offers, tau=tau)
