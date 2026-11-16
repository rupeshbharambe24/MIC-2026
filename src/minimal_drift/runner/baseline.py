"""The clean-run baseline: epsilon, the agent's own error floor per intent.

Every causal claim in this project is stated net of epsilon, so epsilon has to be
measured before any attack is believed -- not assumed to be zero.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from minimal_drift.attribution.ate import bootstrap_ci
from minimal_drift.harness.ap2_env import AP2Env


class BaselineStats(BaseModel):
    """Clean-run divergence for one intent over N seeds."""

    intent_id: str
    model_id: str
    n_seeds: int
    epsilon: float
    ci95: tuple[float, float]
    valid_rate: float
    per_seed: dict[int, float]
    skus_purchased: dict[str, int]

    @property
    def clean(self) -> bool:
        return self.epsilon == 0.0


def clean_baseline(
    env: AP2Env, intent_id: str, seeds: Sequence[int], bootstrap_seed: int = 0
) -> BaselineStats:
    """Run `intent_id` clean over `seeds` and summarise the spontaneous divergence."""
    runs = [env.run(intent_id, s) for s in seeds]
    d = [r.divergence for r in runs]
    ci, _ = bootstrap_ci(d, seed=bootstrap_seed)
    skus: dict[str, int] = {}
    for r in runs:
        skus[r.purchased_sku or "<none>"] = skus.get(r.purchased_sku or "<none>", 0) + 1
    return BaselineStats(
        intent_id=intent_id,
        model_id=getattr(env.agent, "model_id", "unknown"),
        n_seeds=len(runs),
        epsilon=round(sum(d) / len(d), 6) if d else 0.0,
        ci95=(round(ci[0], 6), round(ci[1], 6)),
        valid_rate=round(sum(r.valid for r in runs) / len(runs), 6) if runs else 0.0,
        per_seed={r.seed: r.divergence for r in runs},
        skus_purchased=dict(sorted(skus.items())),
    )


def baseline_runs(env: AP2Env, intent_id: str, seeds: Sequence[int]) -> list:
    """The clean arm of a matched-pair counterfactual. Cached by callers."""
    return [env.run(intent_id, s) for s in seeds]
