"""Ranking manipulation: change the ORDER, change nothing else.

Causal story: agents over-weight the first result. This mechanism spends its whole
budget on position -- promote a drift target, optionally demote the incumbent -- and
is the control that tells us how much of an attack is really just position bias.
"""

from __future__ import annotations

from typing import Any

from minimal_drift.mechanisms.base import Intervention, Mechanism, Operator


class RankingMechanism(Mechanism):
    name = "ranking"

    def operators(self) -> list[str]:
        return ["reorder"]

    def propose(self, parent: Intervention | None, rng: Any, ctx: Any) -> Intervention:
        target = ctx.pick_target(rng)
        if target is None:
            return Intervention(mechanism=self.name, intent_id=ctx.spec.id)
        order = [target.cart_id]
        # Optionally stack a second promoted entry so the incumbent falls further.
        others = [o for o in ctx.candidates if o.cart_id != target.cart_id]
        if others and rng.random() < 0.4:
            order.append(others[rng.randrange(len(others))].cart_id)
        return Intervention(
            mechanism=self.name,
            intent_id=ctx.spec.id,
            note=f"promote {target.product.sku} to position 1",
            ops=[Operator(kind="reorder", payload=order)],
        )
