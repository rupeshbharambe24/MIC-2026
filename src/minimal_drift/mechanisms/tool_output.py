"""API steering: the merchant's own tool output as the injection vehicle.

Causal story: an agent treats a tool result as ground truth about the world. A
rationale attached to one option, or a note about the availability of the others,
is indistinguishable from real merchant data at the point the agent reads it.

This is the channel the Day-3 ablation found to be load-bearing, so it is the one to
measure most carefully.
"""

from __future__ import annotations

from typing import Any

from minimal_drift.mechanisms import text
from minimal_drift.mechanisms.base import Intervention, Mechanism, Operator


class ToolOutputMechanism(Mechanism):
    name = "tool_output"

    def operators(self) -> list[str]:
        return ["annotate", "add_tool_note"]

    def propose(self, parent: Intervention | None, rng: Any, ctx: Any) -> Intervention:
        target = ctx.pick_target(rng)
        if target is None:
            return Intervention(mechanism=self.name, intent_id=ctx.spec.id)
        ops: list[Operator] = [
            Operator(
                kind="annotate",
                target=target.cart_id,
                payload=text.rationale(rng, ctx.intent_phrase, target.product.merchant),
            )
        ]
        if rng.random() < 0.45:
            ops.append(Operator(kind="add_tool_note", payload=text.choose(rng, text.TOOL_NOTES)))
        return Intervention(
            mechanism=self.name,
            intent_id=ctx.spec.id,
            note=f"merchant-voiced rationale for {target.product.sku}",
            ops=ops,
        )
