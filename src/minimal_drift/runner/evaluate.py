"""The Evaluator: turn a candidate delta into a causal verdict, cheaply.

This is the only place that spends target-model queries, and it enforces the
objective's constraint order so a bad candidate costs as little as possible:

    1. VALIDITY GATE   -- apply delta, run one seed, check V(delta) == 1.
                          Invalid candidates are discarded here and cost the
                          attacker no counterfactual budget at all.
    2. COUNTERFACTUAL  -- matched-pair ATE over N seeds against a cached clean arm.
    3. IMPACT          -- economic impact and stealth on the perturbed arm.

The clean arm is cached per (intent, seed) because it never depends on delta;
without that cache every candidate would pay for its own control group twice.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel

from minimal_drift.attribution.ate import ATEResult, ate
from minimal_drift.config import get_config
from minimal_drift.harness.ap2_env import AP2Env, RunResult
from minimal_drift.mechanisms.base import Intervention


class Evaluation(BaseModel):
    """Everything known about one candidate delta after evaluation."""

    signature: str
    mechanism: str
    stage: str
    surfaces: tuple[str, ...]
    n_fields: int
    cost: float
    valid: bool
    invalid_reasons: list[str] = []
    ate: ATEResult | None = None
    economics: float = 0.0
    queries: int = 0
    boundary_reason: str = ""
    chosen_sku: str = ""

    @property
    def attributable(self) -> bool:
        return bool(self.valid and self.ate and self.ate.attributable)


@dataclass
class Evaluator:
    """Stateful only in its caches and its query counter."""

    env: AP2Env
    seeds: Sequence[int] = field(default_factory=lambda: tuple(range(1, get_config().n_seeds + 1)))
    tau: float | None = None
    queries: int = 0
    _clean: dict[tuple[str, int], RunResult] = field(default_factory=dict)
    _cache: dict[tuple[str, str], Evaluation] = field(default_factory=dict)

    # -- clean arm --------------------------------------------------------- #
    def clean_run(self, intent_id: str, seed: int) -> RunResult:
        key = (intent_id, seed)
        if key not in self._clean:
            self._clean[key] = self.env.run(intent_id, seed)
            self.queries += 1
        return self._clean[key]

    def clean_arm(self, intent_id: str) -> list[RunResult]:
        return [self.clean_run(intent_id, s) for s in self.seeds]

    def baseline_epsilon(self, intent_id: str) -> float:
        arm = self.clean_arm(intent_id)
        return sum(r.divergence for r in arm) / len(arm)

    # -- evaluation -------------------------------------------------------- #
    def probe(self, delta: Intervention, intent_id: str) -> RunResult:
        """One perturbed run at the first seed: the cheapest possible validity check."""
        self.queries += 1
        return self.env.run(intent_id, self.seeds[0], stack=delta.stack(), keep_artifacts=True)

    def evaluate(self, delta: Intervention, intent_id: str | None = None) -> Evaluation:
        """Validity gate, then counterfactual ATE. Memoised on (intent, delta signature)."""
        intent_id = intent_id or delta.intent_id
        key = (intent_id, delta.signature())
        if key in self._cache:
            return self._cache[key]

        base = dict(
            signature=delta.signature(),
            mechanism=delta.mechanism,
            stage=delta.stage,
            surfaces=delta.surfaces,
            n_fields=delta.n_fields,
            cost=delta.cost(),
        )

        # 1. validity gate -- zero counterfactual budget spent on invalid candidates
        first = self.probe(delta, intent_id)
        if not first.valid:
            out = Evaluation(
                **base,
                valid=False,
                invalid_reasons=first.validation.reasons,
                queries=1,
                boundary_reason=first.boundary_reason,
                chosen_sku=first.purchased_sku,
            )
            self._cache[key] = out
            return out

        # 2. counterfactual over the remaining seeds
        pert = [first] + [
            self.env.run(intent_id, s, stack=delta.stack()) for s in list(self.seeds)[1:]
        ]
        self.queries += len(pert) - 1
        res = ate(self.clean_arm(intent_id), pert, intent_id, tau=self.tau)

        # 3. impact on the perturbed arm
        econ = [p.economics.impact for p in pert if p.economics]
        drifted = [p for p in pert if p.divergence > 0]
        out = Evaluation(
            **base,
            valid=True,
            ate=res,
            economics=round(sum(econ) / len(econ), 2) if econ else 0.0,
            queries=len(pert),
            boundary_reason=(drifted[0] if drifted else pert[0]).boundary_reason,
            chosen_sku=(drifted[0] if drifted else pert[0]).purchased_sku,
        )
        self._cache[key] = out
        return out

    def ate_only(self, delta: Intervention, intent_id: str) -> ATEResult:
        """Adapter for `attribution.ablation`, which wants delta -> ATEResult."""
        ev = self.evaluate(delta, intent_id)
        return ev.ate or ATEResult(
            ate=0.0,
            ci95=(0.0, 0.0),
            baseline_error=self.baseline_epsilon(intent_id),
            attributable=False,
            n_pairs=0,
            tau=self.tau or get_config().tau,
            p_positive=0.0,
        )
