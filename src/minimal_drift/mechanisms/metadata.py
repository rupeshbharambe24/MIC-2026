"""Metadata substitution: make a different object *look* like the requested one.

Causal story: the agent judges relevance from the product feed. Rewrite a drift
target's label to echo the user's own words, or spoof the one attribute the intent
turns on, and the target wins on relevance rather than on position or persuasion.

The merchant still signs the truth, so this never touches the mandate -- and it never
reaches the confirmation screen either, which is exactly why a stored-value target
still needs the consent surface.
"""

from __future__ import annotations

from typing import Any

from minimal_drift.mechanisms import text
from minimal_drift.mechanisms.base import Intervention, Mechanism, Operator


class MetadataMechanism(Mechanism):
    name = "metadata"

    def operators(self) -> list[str]:
        return ["relabel", "set_attribute"]

    def propose(self, parent: Intervention | None, rng: Any, ctx: Any) -> Intervention:
        target = ctx.pick_target(rng)
        if target is None:
            return Intervention(mechanism=self.name, intent_id=ctx.spec.id)
        ops: list[Operator] = []
        if rng.random() < 0.75:
            ops.append(
                Operator(
                    kind="relabel",
                    target=target.cart_id,
                    payload=text.label(rng, ctx.intent_phrase, target.product.merchant),
                )
            )
        if ctx.discriminators and (not ops or rng.random() < 0.6):
            key = ctx.discriminators[rng.randrange(len(ctx.discriminators))]
            want = _spoof_value(ctx, key)
            ops.append(Operator(kind="set_attribute", target=target.cart_id, key=key, payload=want))
        return Intervention(
            mechanism=self.name,
            intent_id=ctx.spec.id,
            note=f"semantic substitution on {target.product.sku}",
            ops=ops,
        )


def _spoof_value(ctx: Any, key: str) -> str:
    """Copy the value the ground truth carries for `key`, so the fake matches the real."""
    for o in ctx.incumbents:
        if key in o.product.attributes:
            return str(o.product.attributes[key])
    return "true"
