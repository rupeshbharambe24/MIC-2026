"""Counterfactual attribution: did the intervention *cause* the drift?

Neyman-Rubin matched-pair average treatment effect. The unit is a session seed;
the pair is (clean run, perturbed run) at the same seed -- same paraphrase, same
retrieval jitter, same tie-breaking -- so the only difference between the two arms
is delta. That pairing is what lets us subtract the agent's own error floor
instead of arguing about it.

    ATE      = mean over seeds of ( D_perturbed - D_clean )
    CI95     = percentile bootstrap over the paired differences
    epsilon  = mean D_clean  (the clean-run spontaneous divergence)

An effect is `attributable` only if all three hold:
    lower CI > 0        the effect is not noise
    ATE >= tau          the effect is large enough to matter
    ATE > epsilon       the effect exceeds what the agent does on its own

Pure functions: no I/O, no LLM, no global RNG.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel

from minimal_drift.config import get_config


@runtime_checkable
class HasDivergence(Protocol):
    seed: int
    divergence: float


class ATEResult(BaseModel):
    """The causal claim, with everything needed to audit it."""

    ate: float
    ci95: tuple[float, float]
    baseline_error: float
    attributable: bool
    n_pairs: int
    tau: float
    p_positive: float  # bootstrap share of resamples with a positive effect

    @property
    def lower_ci(self) -> float:
        return self.ci95[0]


def _divergences(runs: Sequence[Any]) -> dict[int, float]:
    """Accept RunResult-likes (matched by seed) or a bare sequence of D values."""
    if not runs:
        return {}
    if isinstance(runs[0], (int, float)):
        return {i: float(v) for i, v in enumerate(runs)}
    return {int(r.seed): float(r.divergence) for r in runs}


def bootstrap_ci(
    values: Sequence[float],
    seed: int = 0,
    resamples: int | None = None,
    alpha: float = 0.05,
) -> tuple[tuple[float, float], float]:
    """Percentile bootstrap CI for the mean of `values`, plus P(mean > 0).

    Returns ((lo, hi), p_positive). Deterministic in `seed`. A degenerate sample
    (n < 2, or zero variance) returns the point estimate as both bounds, which is
    the honest answer rather than a fabricated interval.
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return (0.0, 0.0), 0.0
    mean = float(arr.mean())
    if n < 2 or float(arr.std()) == 0.0:
        return (mean, mean), 1.0 if mean > 0 else 0.0
    rng = np.random.default_rng(seed)
    k = resamples if resamples is not None else get_config().bootstrap_resamples
    means = arr[rng.integers(0, n, size=(k, n))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi)), float((means > 0).mean())


def ate(
    run_clean: Sequence[Any],
    run_pert: Sequence[Any],
    intent_id: str = "",
    n_seeds: int | None = None,
    tau: float | None = None,
    bootstrap_seed: int = 0,
) -> ATEResult:
    """Neyman-Rubin matched-pair ATE of an intervention on divergence D over N seeds.

    `intent_id` is carried for logging only -- the math is per-pair. Unmatched seeds
    are dropped rather than silently zero-filled.
    """
    cfg = get_config()
    tau = cfg.tau if tau is None else tau
    clean, pert = _divergences(run_clean), _divergences(run_pert)
    seeds = sorted(set(clean) & set(pert))
    if n_seeds is not None:
        seeds = seeds[:n_seeds]
    if not seeds:
        return ATEResult(
            ate=0.0,
            ci95=(0.0, 0.0),
            baseline_error=0.0,
            attributable=False,
            n_pairs=0,
            tau=tau,
            p_positive=0.0,
        )

    diffs = [pert[s] - clean[s] for s in seeds]
    epsilon = float(np.mean([clean[s] for s in seeds]))
    effect = float(np.mean(diffs))
    ci, p_pos = bootstrap_ci(diffs, seed=bootstrap_seed)
    return ATEResult(
        ate=round(effect, 6),
        ci95=(round(ci[0], 6), round(ci[1], 6)),
        baseline_error=round(epsilon, 6),
        attributable=bool(ci[0] > 0.0 and effect >= tau and effect > epsilon),
        n_pairs=len(seeds),
        tau=tau,
        p_positive=round(p_pos, 4),
    )
