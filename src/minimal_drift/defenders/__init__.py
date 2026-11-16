"""Defenders, and the stealth metric computed from them.

S = 1 - fire_rate. Two are kept strictly apart: `task_alignment` is IN the loop and
fitness may optimise against it; `held_out` is never visible to fitness and gives an
out-of-sample detectability estimate.
"""

from __future__ import annotations

from collections.abc import Sequence

from minimal_drift.defenders import held_out, task_alignment
from minimal_drift.defenders.held_out import HeldOutVerdict
from minimal_drift.defenders.task_alignment import DefenderVerdict


def stealth(verdicts: Sequence[DefenderVerdict | HeldOutVerdict]) -> float:
    """S = 1 - fire_rate over a set of runs. 1.0 means never detected."""
    if not verdicts:
        return 1.0
    return round(1.0 - sum(v.fired for v in verdicts) / len(verdicts), 6)


__all__ = ["DefenderVerdict", "HeldOutVerdict", "held_out", "stealth", "task_alignment"]
