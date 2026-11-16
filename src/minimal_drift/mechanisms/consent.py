"""Confirmation ambiguity: attack the informed-consent step itself.

Causal story: AP2's human-present flow is only as strong as what the user is shown.
The disclosure is rendered from the merchant-SIGNED cart, so no other surface can
touch it -- but the *framing* around it is agent-authored, and that is writable. A
prompt that never names the item, or that presents the purchase as already approved,
converts a vigilant user into a rubber stamp.

This mechanism changes no agent decision on its own -- it removes the last check on a
decision some other mechanism already moved. Expect ATE = 0 in isolation and a
necessary role in composite elites; that asymmetry is the finding, not a defect.
"""

from __future__ import annotations

from typing import Any

from minimal_drift.mechanisms import text
from minimal_drift.mechanisms.base import Intervention, Mechanism, Operator


class ConsentMechanism(Mechanism):
    name = "consent"

    def operators(self) -> list[str]:
        return ["rewrite_confirmation"]

    def propose(self, parent: Intervention | None, rng: Any, ctx: Any) -> Intervention:
        return Intervention(
            mechanism=self.name,
            intent_id=ctx.spec.id,
            note="reframe the confirmation so the signed item is never disclosed",
            ops=[
                Operator(
                    kind="rewrite_confirmation",
                    payload=text.choose(rng, text.CONSENT_PROMPTS),
                )
            ],
        )
