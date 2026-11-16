"""Day 5 gate: the lexicographic elite rule, the descriptor, and the bandit."""

from __future__ import annotations

import pytest

from minimal_drift.attribution.ate import ate
from minimal_drift.mechanisms import MECHANISM_NAMES
from minimal_drift.mechanisms.base import Intervention, Operator
from minimal_drift.search.archive import Archive, Elite
from minimal_drift.search.selector import UCB1


def make_ate(value: float, tau: float = 0.3):
    return ate([0.0] * 12, [value] * 12, tau=tau)


def delta(*ops: Operator, mechanism: str = "ranking") -> Intervention:
    return Intervention(mechanism=mechanism, ops=list(ops), intent_id="intent_shoes_10k")


def reorder(*ids: str) -> Operator:
    return Operator(kind="reorder", payload=list(ids))


def annotate(target: str, text: str) -> Operator:
    return Operator(kind="annotate", target=target, payload=text)


@pytest.fixture()
def archive():
    return Archive(mechanisms=MECHANISM_NAMES)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def test_grid_geometry(archive):
    assert archive.capacity == 6 * 5 == 30
    assert archive.filled == 0 and archive.coverage == 0.0


def test_reachable_capacity_is_smaller_than_the_nominal_grid(archive):
    """Most of mechanisms x stages is empty by construction, so both are reported."""
    reachable = archive.reachable_cells()
    assert 0 < len(reachable) < archive.capacity
    assert ("ranking", "ranking") in reachable
    assert ("ranking", "confirmation") not in reachable  # reorder cannot reach consent
    assert ("consent", "confirmation") in reachable
    assert {"retrieval", "presentation"} <= {s for m, s in reachable if m == "omission"}


def test_descriptor_uses_the_operators_actually_present(archive):
    """A minimised delta is filed under what it does, not under what proposed it."""
    d = Intervention(mechanism="consent", ops=[reorder("A#1")])
    assert d.effective_mechanism == "ranking"
    assert archive.descriptor(d) == ("ranking", "ranking")

    kept = Intervention(
        mechanism="consent",
        ops=[Operator(kind="rewrite_confirmation", payload="ok"), reorder("A#1")],
    )
    assert kept.effective_mechanism == "consent"  # its own operator survived
    assert archive.descriptor(kept) == ("consent", "ranking")  # earliest stage touched


# --------------------------------------------------------------------------- #
# the lexicographic elite rule, one key at a time
# --------------------------------------------------------------------------- #
def test_first_candidate_always_fills_an_empty_cell(archive):
    assert archive.upsert(("ranking", "ranking"), delta(reorder("A#1")), make_ate(0.5), 1.0, 100.0)
    assert archive.filled == 1 and archive.coverage == pytest.approx(1 / 30, abs=1e-6)


def test_lower_cost_wins_even_with_a_much_worse_ate(archive):
    cell = ("ranking", "ranking")
    dear = delta(reorder("A#1"), annotate("A#1", "x"))
    cheap = delta(reorder("A#1"))
    archive.upsert(cell, dear, make_ate(0.9), 1.0, 100_000.0)
    assert archive.upsert(cell, cheap, make_ate(0.35), 0.5, 0.0)
    assert archive.elite(cell).intervention.n_fields == 1


def test_higher_ate_wins_at_equal_cost(archive):
    cell = ("ranking", "ranking")
    archive.upsert(cell, delta(reorder("A#1")), make_ate(0.4), 1.0, 0.0)
    assert archive.upsert(cell, delta(reorder("B#1")), make_ate(0.8), 1.0, 0.0)
    assert archive.elite(cell).ate.ate == pytest.approx(0.8)
    assert not archive.upsert(cell, delta(reorder("C#1")), make_ate(0.6), 1.0, 0.0)


def test_impact_breaks_ties_at_equal_cost_and_ate(archive):
    cell = ("ranking", "ranking")
    archive.upsert(cell, delta(reorder("A#1")), make_ate(0.5), 1.0, 10.0)
    assert archive.upsert(cell, delta(reorder("B#1")), make_ate(0.5), 1.0, 9000.0)
    assert archive.elite(cell).E == 9000.0


def test_transfer_breaks_ties_last(archive):
    cell = ("ranking", "ranking")
    archive.upsert(cell, delta(reorder("A#1")), make_ate(0.5), 0.4, 100.0)
    assert archive.upsert(cell, delta(reorder("B#1")), make_ate(0.5), 0.9, 100.0)
    assert archive.elite(cell).R == 0.9


def test_a_full_tie_keeps_the_incumbent(archive):
    """Stability under re-runs: nothing changes without a strict improvement."""
    cell = ("ranking", "ranking")
    a = delta(reorder("A#1"))
    archive.upsert(cell, a, make_ate(0.5), 1.0, 100.0)
    assert not archive.upsert(cell, delta(reorder("B#1")), make_ate(0.5), 1.0, 100.0)
    assert archive.elite(cell).intervention.ops[0].payload == ["A#1"]


def test_impact_is_log_scaled_so_it_cannot_outrank_ate(archive):
    cell = ("ranking", "ranking")
    archive.upsert(cell, delta(reorder("A#1")), make_ate(0.9), 1.0, 0.0)
    assert not archive.upsert(cell, delta(reorder("B#1")), make_ate(0.4), 1.0, 10_000_000.0)


def test_elite_key_ordering_is_total():
    def elite(cost_ops, a, e, r):
        return Elite(
            cell=("ranking", "ranking"),
            intervention=delta(*cost_ops),
            ate=make_ate(a),
            R=r,
            E=e,
        )

    cheap = elite([reorder("A#1")], 0.4, 0.0, 0.0)
    dear = elite([reorder("A#1"), annotate("A#1", "x")], 0.9, 1e6, 1.0)
    assert cheap.key() < dear.key()


# --------------------------------------------------------------------------- #
# fitness / QD / |delta*|
# --------------------------------------------------------------------------- #
def test_non_attributable_elites_contribute_zero_fitness(archive):
    cell = ("ranking", "ranking")
    archive.upsert(cell, delta(reorder("A#1")), make_ate(0.1), 1.0, 500.0)  # below tau
    assert archive.elite(cell).fitness == 0.0
    assert archive.qd_score == 0.0


def test_fitness_is_effect_per_cost_times_transfer(archive):
    cell = ("ranking", "ranking")
    d = delta(reorder("A#1"))
    archive.upsert(cell, d, make_ate(0.8), 0.5, 0.0)
    e = archive.elite(cell)
    assert e.fitness == pytest.approx((0.8 / d.cost()) * 0.5, abs=1e-6)


def test_median_delta_star_is_the_headline_number(archive):
    for i, n in enumerate([1, 1, 3, 5]):
        ops = [reorder(f"A{i}#1")] + [annotate(f"A{i}#1", f"x{j}") for j in range(n - 1)]
        archive.upsert(
            ("ranking", ("retrieval", "ranking", "presentation", "memory", "confirmation")[i]),
            delta(*ops),
            make_ate(0.9),
            1.0,
            0.0,
            delta_star=n,
        )
    assert archive.median_delta_star() == 2.0  # (1+3)/2 over [1,1,3,5]
    assert archive.min_delta_star == 1


def test_failures_are_recorded_and_the_last_one_drives_reflection(archive):
    cell = ("metadata", "presentation")
    for reason in ("no_drift", "no_drift", "no_drift", "consent_blocked"):
        archive.record_fail(cell, reason)
    assert archive.dominant_failure(cell) == "no_drift"
    assert archive.last_failure(cell) == "consent_blocked"
    assert archive.attempts[cell] == 4


def test_summary_and_json_round_trip(archive):
    archive.upsert(("ranking", "ranking"), delta(reorder("A#1")), make_ate(0.9), 1.0, 5.0)
    archive.record_fail(("consent", "confirmation"), "no_drift")
    s = archive.summary()
    assert s["filled"] == 1 and s["capacity"] == 30
    assert s["reachable_capacity"] < s["capacity"]
    assert s["failures"]["consent|confirmation"] == {"no_drift": 1}
    import json

    json.dumps(archive.to_json())  # must be serialisable


# --------------------------------------------------------------------------- #
# UCB1
# --------------------------------------------------------------------------- #
def test_ucb_tries_every_arm_before_repeating():
    cells = [("m", s) for s in ("a", "b", "c")]
    sel = UCB1(cells=cells)
    seen = set()
    for _ in range(len(cells)):
        c = sel.ucb_pick()
        seen.add(c)
        sel.update(c, 0.0)
    assert seen == set(cells)


def test_ucb_concentrates_on_the_rewarding_arm():
    cells = [("m", "good"), ("m", "bad")]
    sel = UCB1(cells=cells)
    for _ in range(60):
        c = sel.ucb_pick()
        sel.update(c, 1.0 if c[1] == "good" else 0.0)
    assert sel.counts[("m", "good")] > sel.counts[("m", "bad")]
    assert sel.mean(("m", "good")) > sel.mean(("m", "bad"))


def test_ucb_rewards_are_clipped_to_unit_range():
    sel = UCB1(cells=[("m", "a")])
    sel.update(("m", "a"), 12.0)
    sel.update(("m", "a"), -5.0)
    assert sel.mean(("m", "a")) == 0.5


def test_ucb_abandons_a_dead_arm_but_never_prunes_it():
    """Zero-reward niches decay on their own; hand-pruning would hide empty niches."""
    cells = [("m", "live"), ("m", "dead")]
    sel = UCB1(cells=cells)
    for _ in range(80):
        c = sel.ucb_pick()
        sel.update(c, 1.0 if c[1] == "live" else 0.0)
    assert sel.counts[("m", "dead")] >= 1
    assert sel.counts[("m", "dead")] < sel.counts[("m", "live")]
    assert ("m", "dead") in sel.cells


def test_ucb_state_is_reportable():
    sel = UCB1(cells=[("ranking", "ranking")])
    sel.update(("ranking", "ranking"), 0.25)
    assert sel.state()["ranking|ranking"] == {"pulls": 1, "mean_reward": 0.25}
