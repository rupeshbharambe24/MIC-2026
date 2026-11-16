"""Delta debugging: shrink a working attack to its irreducible core, |delta*|.

`shrink` removes operators while the attack still holds, and returns the smallest
delta it can certify. Two passes, in this order:

  1. **ddmin-style subset removal** -- try dropping whole halves, then quarters, and
     so on. This escapes the local minimum a greedy pass falls into when two operators
     are only necessary *together*.
  2. **greedy, costliest-first** -- then remove single operators, most expensive first,
     so the surviving core is minimal in C(delta) and not merely in operator count.

`keep(delta) -> bool` is supplied by the caller and must encode the FULL constraint
set (validity AND attributability over N seeds). Everything `shrink` returns is
therefore certified against the same bar as the attack it came from, and is always a
subset of the input: |delta*| <= |delta|.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from minimal_drift.mechanisms.base import Intervention

Keep = Callable[[Intervention], bool]


class ShrinkReport(BaseModel):
    """What the minimiser did, so |delta*| is auditable rather than asserted."""

    n_fields_before: int
    n_fields_after: int
    cost_before: float
    cost_after: float
    removed_ops: list[str]
    keep_calls: int

    @property
    def reduced(self) -> bool:
        return self.n_fields_after < self.n_fields_before


def shrink(delta: Intervention, keep: Keep) -> Intervention:
    """Remove components while `keep` stays True. Returns the irreducible core delta*."""
    return shrink_with_report(delta, keep)[0]


def shrink_with_report(delta: Intervention, keep: Keep) -> tuple[Intervention, ShrinkReport]:
    """`shrink`, plus a record of what was removed and how many checks it cost."""
    calls = 0

    def ok(candidate: Intervention) -> bool:
        nonlocal calls
        calls += 1
        return keep(candidate)

    before_fields, before_cost = delta.n_fields, delta.cost()
    removed: list[str] = []

    if not delta.ops or not ok(delta):
        # Nothing to shrink, or the input does not hold in the first place.
        return delta, ShrinkReport(
            n_fields_before=before_fields,
            n_fields_after=before_fields,
            cost_before=before_cost,
            cost_after=before_cost,
            removed_ops=[],
            keep_calls=calls,
        )

    current = delta

    # -- pass 1: ddmin subset removal -------------------------------------- #
    granularity = 2
    while len(current.ops) > 1:
        chunk = max(1, len(current.ops) // granularity)
        shrunk = False
        for start in range(0, len(current.ops), chunk):
            keep_idx = [i for i in range(len(current.ops)) if not (start <= i < start + chunk)]
            if not keep_idx:
                continue
            candidate = current.subset(keep_idx)
            if ok(candidate):
                removed += [
                    current.ops[i].kind for i in range(len(current.ops)) if i not in keep_idx
                ]
                current = candidate
                granularity = 2
                shrunk = True
                break
        if not shrunk:
            if chunk == 1:
                break
            granularity = min(len(current.ops), granularity * 2)

    # -- pass 2: greedy, costliest first ----------------------------------- #
    changed = True
    while changed and len(current.ops) > 1:
        changed = False
        order = sorted(
            range(len(current.ops)),
            key=lambda i: (-current.ops[i].fields, -current.ops[i].tokens, -current.ops[i].turns),
        )
        for i in order:
            candidate = current.without(i)
            if ok(candidate):
                removed.append(current.ops[i].kind)
                current = candidate
                changed = True
                break

    assert current.is_subset_of(delta), "shrink must return a subset of its input"
    return current, ShrinkReport(
        n_fields_before=before_fields,
        n_fields_after=current.n_fields,
        cost_before=before_cost,
        cost_after=current.cost(),
        removed_ops=removed,
        keep_calls=calls,
    )
