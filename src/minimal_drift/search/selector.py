"""UCB1 bandit over archive cells.

Each cell is an arm. Pulling it means "spend a generation trying to improve this
behavioural niche". Reward is the loop's fitness signal, clipped to [0, 1] so the
confidence term keeps its usual meaning.

Cells with no attack surface at all (a mechanism that cannot reach a stage) return
zero reward forever, and UCB1 stops visiting them on its own -- there is no need to
hand-prune the grid, and hand-pruning would hide which niches are genuinely empty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from minimal_drift.search.archive import Cell


@dataclass
class UCB1:
    """Standard UCB1. Deterministic given the update sequence."""

    cells: list[Cell]
    c: float = 1.4  # exploration constant
    counts: dict[Cell, int] = field(default_factory=dict)
    rewards: dict[Cell, float] = field(default_factory=dict)
    t: int = 0

    def __post_init__(self) -> None:
        for cell in self.cells:
            self.counts.setdefault(cell, 0)
            self.rewards.setdefault(cell, 0.0)

    def mean(self, cell: Cell) -> float:
        n = self.counts[cell]
        return self.rewards[cell] / n if n else 0.0

    def score(self, cell: Cell) -> float:
        n = self.counts[cell]
        if n == 0:
            return math.inf
        return self.mean(cell) + self.c * math.sqrt(math.log(max(self.t, 1)) / n)

    def ucb_pick(self) -> Cell:
        """The next cell to spend a generation on. Ties break on cell order."""
        return max(self.cells, key=lambda cell: (self.score(cell), -self.cells.index(cell)))

    def update(self, cell: Cell, reward: float) -> None:
        self.t += 1
        self.counts[cell] += 1
        self.rewards[cell] += max(0.0, min(1.0, reward))

    def state(self) -> dict[str, dict[str, float]]:
        return {
            f"{m}|{s}": {"pulls": self.counts[(m, s)], "mean_reward": round(self.mean((m, s)), 6)}
            for (m, s) in self.cells
        }
