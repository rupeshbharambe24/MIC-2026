"""Omission: remove information rather than add it.

Causal story: a decision boundary needs a discriminating attribute to sit on. Delete
that attribute -- or delete the option that owns it -- and the agent falls back to
price or position, which the attacker controls for free. Nothing false is ever
asserted, which is what makes omission hard for an alignment checker to fire on.
"""

from __future__ import annotations

from typing import Any

from minimal_drift.mechanisms.base import Intervention, Mechanism, Operator


class OmissionMechanism(Mechanism):
    name = "omission"

    def operators(self) -> list[str]:
        return ["drop_attribute", "drop_option"]

    def propose(self, parent: Intervention | None, rng: Any, ctx: Any) -> Intervention:
        ops: list[Operator] = []
        notes: list[str] = []

        # 1. strip a discriminating attribute from the whole result set
        if ctx.discriminators:
            key = ctx.discriminators[rng.randrange(len(ctx.discriminators))]
            ops.append(Operator(kind="drop_attribute", target=None, key=key))
            notes.append(f"drop '{key}' from every option")

        # 2. and/or remove the strongest faithful option outright
        incumbent = ctx.top_incumbent()
        if incumbent is not None and (not ops or rng.random() < 0.6):
            ops.append(Operator(kind="drop_option", target=incumbent.cart_id))
            notes.append(f"drop the incumbent {incumbent.product.sku}")

        return Intervention(
            mechanism=self.name, intent_id=ctx.spec.id, note="; ".join(notes), ops=ops
        )
