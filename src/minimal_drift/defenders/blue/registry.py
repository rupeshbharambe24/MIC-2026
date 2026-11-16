"""Named gate variants, so the ensemble split stays honest.

The red team's anti-overfitting rule is explicit: "Optimize stealth S against
defenders {D1...Dk}; evaluate on a held-out D_{k+1}. Never let fitness see the eval
defender." A single blue defender cannot satisfy that -- if fitness optimises against
the same object that evaluates, the reported stealth measures how well the attacker
learned one checker, not whether the vulnerability is closed.

So the gate ships as a family:

  IN-LOOP (fitness may see these)
    blue-entailment   L2 only. The ablation that isolates how much of the result is
                      the entailment idea alone.
    blue-symbolic     L2+L3+L4. Deterministic, no model, authorisation-path cheap.

  HELD-OUT (fitness must never see these)
    blue-strict       same layers, stricter thresholds and no abstention -- catches an
                      attacker who has learned to sit exactly on the in-loop margin.
    blue-full         adds the quarantined semantic judge for the residual.

Held-out membership is a property of the variant, not of the caller, so a future
change to the loop cannot quietly promote an eval defender into the fitness set.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from minimal_drift.catalog.ground_truth import World
from minimal_drift.defenders.blue.gate import BlueVerdict, evaluate
from minimal_drift.harness.surfaces import AgentView

GateFn = Callable[[AgentView, str, World | None, float | None], BlueVerdict]


@dataclass(frozen=True)
class BlueVariant:
    """One named configuration of the gate."""

    name: str
    held_out: bool
    use_semantic: bool
    abstain: bool
    description: str

    def check(
        self,
        view: AgentView,
        chosen_cart_id: str,
        world: World | None = None,
        amount: float | None = None,
    ) -> BlueVerdict:
        verdict = evaluate(
            view,
            chosen_cart_id,
            world=world,
            amount=amount,
            use_semantic=self.use_semantic,
            abstain=self.abstain,
        )
        return verdict.model_copy(update={"name": self.name})


VARIANTS: dict[str, BlueVariant] = {
    "blue-entailment": BlueVariant(
        name="blue-entailment",
        held_out=False,
        use_semantic=False,
        abstain=False,
        description="L2 only -- intent/record entailment with positive attestation.",
    ),
    "blue-symbolic": BlueVariant(
        name="blue-symbolic",
        held_out=False,
        use_semantic=False,
        abstain=False,
        description="L2+L3+L4 -- entailment, provenance, consent binding. No model.",
    ),
    "blue-strict": BlueVariant(
        name="blue-strict",
        held_out=True,
        use_semantic=False,
        abstain=True,
        description="Held-out: same layers, abstention on, for margin-sitting attackers.",
    ),
    "blue-full": BlueVariant(
        name="blue-full",
        held_out=True,
        use_semantic=True,
        abstain=True,
        description="Held-out: adds the quarantined semantic judge for the residual.",
    ),
}

IN_LOOP = tuple(v for v in VARIANTS.values() if not v.held_out)
HELD_OUT = tuple(v for v in VARIANTS.values() if v.held_out)


def get(name: str) -> BlueVariant:
    return VARIANTS[name]
