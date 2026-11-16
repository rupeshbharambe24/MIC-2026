"""Localise the drift: which part of delta actually did the work.

Two complementary ablations over the same `evaluate` callable:

`single_stage`   -- apply ONLY the operators belonging to one causal surface stage.
                    Answers "does the ranking edit alone cause the drift?"
`leave_one_out`  -- apply delta minus one operator. Answers "is this operator
                    necessary?" -- the drop in ATE is that operator's contribution.

`evaluate(delta) -> ATEResult` is injected by the caller, so this module stays pure
of environment, model, and I/O concerns and is testable with synthetic run data.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from minimal_drift.attribution.ate import ATEResult
from minimal_drift.mechanisms.base import Intervention, Stage

Evaluate = Callable[[Intervention], ATEResult]


class StageEffect(BaseModel):
    stage: str
    n_ops: int
    result: ATEResult
    share_of_full: float  # stage ATE / full ATE


class OperatorEffect(BaseModel):
    index: int
    kind: str
    stage: str
    target: str | None
    result_without: ATEResult  # ATE of delta with this operator removed
    drop: float  # full ATE - ATE without
    necessary: bool  # removing it stops the effect being attributable


class Localisation(BaseModel):
    """Where the causal work happens, and which operators are load-bearing."""

    full: ATEResult
    stages: list[StageEffect]
    operators: list[OperatorEffect]

    @property
    def dominant_stage(self) -> str | None:
        """The stage whose operators alone reproduce the most of the full effect.

        `None` when no single stage produces any effect on its own -- the drift is
        then *conjunctive*, and naming one stage as dominant would be a tie-break
        artefact dressed up as a causal claim.
        """
        pool = [s for s in self.stages if s.result.attributable] or [
            s for s in self.stages if s.result.ate > 0
        ]
        return max(pool, key=lambda s: s.result.ate).stage if pool else None

    @property
    def conjunctive(self) -> bool:
        """True when the effect exists only as a combination of stages."""
        return self.full.attributable and self.dominant_stage is None

    @property
    def necessary_operators(self) -> list[int]:
        return [o.index for o in self.operators if o.necessary]


def single_stage(delta: Intervention, evaluate: Evaluate, full: ATEResult) -> list[StageEffect]:
    """ATE of each causal stage of `delta` in isolation."""
    out: list[StageEffect] = []
    for stage in delta.stages:
        sub = delta.only_stage(stage)
        res = evaluate(sub)
        out.append(
            StageEffect(
                stage=stage,
                n_ops=len(sub.ops),
                result=res,
                share_of_full=round(res.ate / full.ate, 4) if full.ate else 0.0,
            )
        )
    return out


def leave_one_out(delta: Intervention, evaluate: Evaluate, full: ATEResult) -> list[OperatorEffect]:
    """ATE of `delta` with each operator removed in turn."""
    out: list[OperatorEffect] = []
    for i, op in enumerate(delta.ops):
        res = evaluate(delta.without(i))
        out.append(
            OperatorEffect(
                index=i,
                kind=op.kind,
                stage=op.stage,
                target=op.target,
                result_without=res,
                drop=round(full.ate - res.ate, 6),
                necessary=full.attributable and not res.attributable,
            )
        )
    return out


def localise(delta: Intervention, evaluate: Evaluate) -> Localisation:
    """Full attribution report for `delta`: stage isolation + leave-one-out."""
    full = evaluate(delta)
    return Localisation(
        full=full,
        stages=single_stage(delta, evaluate, full),
        operators=leave_one_out(delta, evaluate, full),
    )


def stage_of_effect(loc: Localisation) -> Stage | None:
    """Convenience for reporting: the stage the drift localises to."""
    s = loc.dominant_stage
    return s  # type: ignore[return-value]
