"""MAP-Elites archive over BEHAVIOURAL descriptors.

    descriptor = (mechanism_class, causal_surface_stage)

Behavioural, not parametric: two interventions land in the same cell when they tell
the same causal story, regardless of how their text differs. That is what makes
"cells filled" a coverage claim about *strategies* rather than about string variety.

Elite selection inside a cell is LEXICOGRAPHIC, in this order:

    1. min C(delta)    the objective is minimality; nothing outranks it
    2. max ATE         then the size of the proven effect
    3. max log(1+E)    then economic impact, log-scaled so a big number cannot
                       buy its way past a worse ATE
    4. max R           then transfer

Ties on every key keep the incumbent, so the archive is stable under re-runs.

The archive also records FAILURES per cell with their boundary reason. That record is
what the mutator reflects on -- a cell that keeps dying at the consent step needs a
different strategy, not another sample of the same one.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from pydantic import BaseModel

from minimal_drift.attribution.ate import ATEResult
from minimal_drift.mechanisms.base import STAGES, Intervention

Cell = tuple[str, str]


class Elite(BaseModel):
    """The best intervention found for one behavioural cell."""

    cell: Cell
    intervention: Intervention
    ate: ATEResult
    R: float = 0.0
    E: float = 0.0
    stealth_in_loop: float = 1.0
    stealth_held_out: float = 1.0
    delta_star: int = 0  # |delta*| after minimisation
    generation: int = 0

    @property
    def cost(self) -> float:
        return self.intervention.cost()

    @property
    def n_fields(self) -> int:
        return self.intervention.n_fields

    @property
    def fitness(self) -> float:
        """Reward used for QD-score and bandit updates: effect per unit cost, transferred."""
        if not self.ate.attributable or self.cost <= 0:
            return 0.0
        return round((self.ate.ate / self.cost) * max(self.R, 0.0), 6)

    def key(self) -> tuple[float, float, float, float]:
        """Lexicographic elite key; SMALLER is better."""
        return (self.cost, -self.ate.ate, -math.log1p(max(self.E, 0.0)), -self.R)


@dataclass
class Archive:
    """Sparse MAP-Elites grid plus a per-cell failure ledger."""

    mechanisms: tuple[str, ...]
    stages: tuple[str, ...] = STAGES
    elites: dict[Cell, Elite] = field(default_factory=dict)
    failures: dict[Cell, Counter] = field(default_factory=dict)
    last_failures: dict[Cell, str] = field(default_factory=dict)
    attempts: dict[Cell, int] = field(default_factory=dict)

    # -- geometry ---------------------------------------------------------- #
    @property
    def cells(self) -> list[Cell]:
        return [(m, s) for m in self.mechanisms for s in self.stages]

    @property
    def capacity(self) -> int:
        return len(self.mechanisms) * len(self.stages)

    def reachable_cells(self) -> list[Cell]:
        """Cells a mechanism can reach with its OWN operators.

        The nominal grid is mechanisms x stages, but most of it is empty by
        construction: `ranking` only writes to the ranking stage, so
        (ranking, confirmation) can never be filled by the ranking mechanism alone.
        Reporting coverage against the nominal 30 would understate the search and
        invite exactly the binning artefact Day 6 checks for, so both denominators are
        reported. Cross-mechanism splicing during reflection can still reach cells
        outside this set -- which is why it is a floor, not a cap.
        """
        from minimal_drift.mechanisms import MECHANISMS
        from minimal_drift.mechanisms.base import OP_SURFACE

        out: list[Cell] = []
        for m in self.mechanisms:
            mech = MECHANISMS.get(m)
            if mech is None:
                continue
            for stage in {OP_SURFACE[k][1] for k in mech.operators()}:
                if stage in self.stages:
                    out.append((m, stage))
        return sorted(set(out))

    @property
    def reachable_capacity(self) -> int:
        return len(self.reachable_cells())

    def descriptor(self, delta: Intervention) -> Cell:
        """Behavioural descriptor, derived from the operators actually present."""
        return (delta.effective_mechanism, delta.stage)

    # -- writes ------------------------------------------------------------ #
    def upsert(
        self,
        cell: Cell,
        intervention: Intervention,
        ate: ATEResult,
        R: float,
        E: float,
        stealth_in_loop: float = 1.0,
        stealth_held_out: float = 1.0,
        delta_star: int | None = None,
        generation: int = 0,
    ) -> bool:
        """Return True if this delta becomes the new elite under the lexicographic rule."""
        self.attempts[cell] = self.attempts.get(cell, 0) + 1
        candidate = Elite(
            cell=cell,
            intervention=intervention,
            ate=ate,
            R=R,
            E=E,
            stealth_in_loop=stealth_in_loop,
            stealth_held_out=stealth_held_out,
            delta_star=delta_star if delta_star is not None else intervention.n_fields,
            generation=generation,
        )
        incumbent = self.elites.get(cell)
        if incumbent is not None and incumbent.key() <= candidate.key():
            return False
        self.elites[cell] = candidate
        return True

    def record_fail(self, cell: Cell, boundary_reason: str) -> None:
        """Log why a candidate for `cell` did not make it. Feeds mutator reflection."""
        self.attempts[cell] = self.attempts.get(cell, 0) + 1
        self.failures.setdefault(cell, Counter())[boundary_reason] += 1
        self.last_failures[cell] = boundary_reason

    # -- reads ------------------------------------------------------------- #
    def elite(self, cell: Cell) -> Elite | None:
        return self.elites.get(cell)

    def dominant_failure(self, cell: Cell) -> str | None:
        """The boundary this cell keeps dying at, over its whole history."""
        c = self.failures.get(cell)
        if not c:
            return None
        return c.most_common(1)[0][0]

    def last_failure(self, cell: Cell) -> str | None:
        """The boundary this cell died at MOST RECENTLY.

        This is what drives reflection, not `dominant_failure`. A rare but highly
        informative boundary -- being stopped at the user's confirmation screen -- gets
        drowned out by a common one if you reflect on the mode, and the single most
        valuable repair in the space (add a consent operator) then never gets tried.
        """
        return self.last_failures.get(cell)

    @property
    def filled(self) -> int:
        return len(self.elites)

    @property
    def coverage(self) -> float:
        """Filled share of the NOMINAL mechanism x stage grid."""
        return round(self.filled / self.capacity, 6) if self.capacity else 0.0

    @property
    def filled_reachable(self) -> int:
        """Filled cells that lie INSIDE the reachable set.

        Reported separately from `filled` because cross-mechanism splicing can land an
        elite outside that set -- so `filled` can exceed `reachable_capacity`, and using
        it as the numerator would print things like "8/7".
        """
        return len(set(self.elites) & set(self.reachable_cells()))

    @property
    def reachable_coverage(self) -> float:
        """Filled share of the cells a mechanism can reach with its own operators."""
        cap = self.reachable_capacity
        return round(self.filled_reachable / cap, 6) if cap else 0.0

    @property
    def qd_score(self) -> float:
        """Sum of elite fitness over filled cells: the standard QD summary."""
        return round(sum(e.fitness for e in self.elites.values()), 6)

    @property
    def min_delta_star(self) -> int | None:
        return min((e.delta_star for e in self.elites.values()), default=None)

    def median_delta_star(self) -> float | None:
        """The headline number: median |delta*| across filled cells."""
        vals = sorted(e.delta_star for e in self.elites.values())
        if not vals:
            return None
        mid = len(vals) // 2
        return float(vals[mid]) if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0

    def mechanisms_covered(self) -> set[str]:
        return {m for m, _ in self.elites}

    def stages_covered(self) -> set[str]:
        return {s for _, s in self.elites}

    def summary(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "reachable_capacity": self.reachable_capacity,
            "filled": self.filled,
            "filled_reachable": self.filled_reachable,
            "coverage": self.coverage,
            "reachable_coverage": self.reachable_coverage,
            "off_grid_cells": sorted(set(self.elites) - set(self.reachable_cells())),
            "qd_score": self.qd_score,
            "median_delta_star": self.median_delta_star(),
            "min_delta_star": self.min_delta_star,
            "mechanisms_covered": sorted(self.mechanisms_covered()),
            "stages_covered": sorted(self.stages_covered()),
            "failures": {f"{m}|{s}": dict(c) for (m, s), c in sorted(self.failures.items())},
        }

    def to_json(self) -> dict[str, object]:
        return {
            "summary": self.summary(),
            "elites": {
                f"{m}|{s}": {
                    "delta": e.intervention.model_dump(mode="json"),
                    "n_fields": e.n_fields,
                    "delta_star": e.delta_star,
                    "cost": e.cost,
                    "ate": e.ate.model_dump(mode="json"),
                    "R": e.R,
                    "E": e.E,
                    "stealth_in_loop": e.stealth_in_loop,
                    "stealth_held_out": e.stealth_held_out,
                    "fitness": e.fitness,
                    "generation": e.generation,
                }
                for (m, s), e in sorted(self.elites.items())
            },
        }
