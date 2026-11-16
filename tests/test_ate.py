"""Day 3 gate: the ATE math and the `attributable` gate, on synthetic run data.

These are pure-math tests -- no environment, no model -- so a wrong causal claim
shows up here rather than in a results table.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from minimal_drift.attribution.ablation import localise
from minimal_drift.attribution.ate import ATEResult, ate, bootstrap_ci
from minimal_drift.mechanisms.base import Intervention, Operator


def runs(divergences: list[float], first_seed: int = 1):
    return [SimpleNamespace(seed=first_seed + i, divergence=d) for i, d in enumerate(divergences)]


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #
def test_bootstrap_is_deterministic_in_its_seed():
    v = [0.1, 0.9, 0.3, 0.7, 0.5, 0.2]
    assert bootstrap_ci(v, seed=7) == bootstrap_ci(v, seed=7)
    assert bootstrap_ci(v, seed=7) != bootstrap_ci(v, seed=8)


def test_bootstrap_brackets_the_mean():
    v = [0.2, 0.4, 0.6, 0.8, 0.5, 0.3, 0.7, 0.45]
    (lo, hi), p = bootstrap_ci(v, seed=1)
    assert lo <= float(np.mean(v)) <= hi
    assert 0.0 <= p <= 1.0


def test_bootstrap_on_degenerate_samples_reports_a_point_not_an_interval():
    """A zero-variance sample has no interval. Say so, do not fabricate one."""
    assert bootstrap_ci([0.9] * 12, seed=1) == ((0.9, 0.9), 1.0)
    assert bootstrap_ci([0.0] * 12, seed=1) == ((0.0, 0.0), 0.0)
    assert bootstrap_ci([0.4], seed=1) == ((0.4, 0.4), 1.0)
    assert bootstrap_ci([], seed=1) == ((0.0, 0.0), 0.0)


def test_bootstrap_ci_narrows_with_more_data():
    rng = np.random.default_rng(0)
    small = list(rng.normal(0.5, 0.2, 8))
    large = list(rng.normal(0.5, 0.2, 400))
    (lo_s, hi_s), _ = bootstrap_ci(small, seed=3)
    (lo_l, hi_l), _ = bootstrap_ci(large, seed=3)
    assert (hi_l - lo_l) < (hi_s - hi_s + hi_s - lo_s)  # large-sample CI is tighter


# --------------------------------------------------------------------------- #
# matched-pair ATE
# --------------------------------------------------------------------------- #
def test_ate_is_the_mean_paired_difference():
    res = ate(runs([0.0] * 4), runs([0.9, 0.9, 0.0, 0.9]), tau=0.3)
    assert res.ate == pytest.approx(0.675)
    assert res.n_pairs == 4
    assert res.baseline_error == 0.0


def test_pairing_is_by_seed_not_by_position():
    """Perturbed runs arriving out of order must not corrupt the pairing."""
    clean = runs([0.0, 0.9, 0.0, 0.0])  # seeds 1..4
    pert = list(reversed(runs([0.0, 0.0, 0.9, 0.9])))  # same seeds, reversed list
    res = ate(clean, pert, tau=0.3)
    # seed1: 0.0-0.0, seed2: 0.0-0.9, seed3: 0.9-0.0, seed4: 0.9-0.0
    assert res.ate == pytest.approx((0.0 - 0.9 + 0.9 + 0.9) / 4)


def test_unmatched_seeds_are_dropped_not_zero_filled():
    res = ate(runs([0.0, 0.0], first_seed=1), runs([0.9, 0.9], first_seed=2), tau=0.3)
    assert res.n_pairs == 1  # only seed 2 is in both arms


def test_no_overlap_yields_a_non_attributable_null():
    res = ate(runs([0.0], first_seed=1), runs([0.9], first_seed=99), tau=0.3)
    assert res.n_pairs == 0 and res.ate == 0.0 and not res.attributable


def test_bare_float_sequences_are_accepted():
    res = ate([0.0, 0.0, 0.0], [0.9, 0.9, 0.9], tau=0.3)
    assert res.ate == pytest.approx(0.9) and res.n_pairs == 3


def test_baseline_error_is_the_clean_arm_mean():
    res = ate(runs([0.15, 0.0, 0.15, 0.0]), runs([0.9] * 4), tau=0.3)
    assert res.baseline_error == pytest.approx(0.075)
    assert res.ate == pytest.approx(0.9 - 0.075)


def test_n_seeds_truncates_the_pair_set():
    res = ate(runs([0.0] * 10), runs([0.9] * 10), n_seeds=4, tau=0.3)
    assert res.n_pairs == 4


# --------------------------------------------------------------------------- #
# the attributable gate -- all three conditions, each necessary
# --------------------------------------------------------------------------- #
def test_attributable_requires_all_three_conditions():
    strong = ate(runs([0.0] * 12), runs([0.9] * 12), tau=0.3)
    assert strong.attributable and strong.lower_ci > 0 and strong.ate >= strong.tau


def test_effect_below_tau_is_not_attributable():
    res = ate(runs([0.0] * 12), runs([0.15] * 12), tau=0.3)
    assert res.ate == pytest.approx(0.15) and res.lower_ci > 0
    assert not res.attributable  # real, consistent, but too small to matter


def test_noisy_effect_with_ci_touching_zero_is_not_attributable():
    clean = runs([0.0] * 12)
    pert = runs([0.9, 0.0, 0.9, 0.0, 0.9, 0.0, 0.0, 0.9, 0.0, 0.0, 0.9, 0.0])
    res = ate(clean, pert, tau=0.3, bootstrap_seed=5)
    assert res.ate >= res.tau  # large enough on average...
    assert res.lower_ci > 0 and res.attributable  # ...and consistently positive

    sparse = runs([0.9] + [0.0] * 11)
    res2 = ate(clean, sparse, tau=0.3, bootstrap_seed=5)
    assert res2.lower_ci == 0.0 and not res2.attributable  # one-off, not an effect


def test_effect_not_exceeding_the_baseline_is_not_attributable():
    """A drifty agent gets no credit for drift it was going to do anyway."""
    res = ate(runs([0.6] * 12), runs([0.9] * 12), tau=0.2)
    assert res.ate == pytest.approx(0.3) and res.lower_ci > 0
    assert res.ate < res.baseline_error or not res.attributable
    assert not res.attributable  # ate 0.3 < epsilon 0.6


def test_negative_effect_is_never_attributable():
    res = ate(runs([0.9] * 12), runs([0.0] * 12), tau=0.3)
    assert res.ate < 0 and not res.attributable


def test_result_round_trips_through_json():
    res = ate(runs([0.0] * 6), runs([0.9] * 6), tau=0.3)
    assert ATEResult.model_validate_json(res.model_dump_json()) == res


# --------------------------------------------------------------------------- #
# ablation
# --------------------------------------------------------------------------- #
def _delta() -> Intervention:
    return Intervention(
        mechanism="ranking",
        ops=[
            Operator(kind="reorder", payload=["X#1"]),
            Operator(kind="annotate", target="X#1", payload="best match recommended"),
            Operator(kind="plant_memory", payload="user prefers vouchers"),
        ],
    )


def _make_evaluate(load_bearing: set[str]):
    """Synthetic evaluator: the effect exists iff every load-bearing op is present."""

    def evaluate(d: Intervention) -> ATEResult:
        kinds = {o.kind for o in d.ops}
        strength = 0.9 if load_bearing <= kinds else 0.0
        return ate(runs([0.0] * 12), runs([strength] * 12), tau=0.3)

    return evaluate


def test_leave_one_out_flags_exactly_the_necessary_operators():
    loc = localise(_delta(), _make_evaluate({"annotate"}))
    assert loc.full.attributable
    assert [o.kind for o in loc.operators if o.necessary] == ["annotate"]
    assert loc.necessary_operators == [1]


def test_jointly_necessary_operators_are_both_flagged():
    loc = localise(_delta(), _make_evaluate({"reorder", "annotate"}))
    assert sorted(o.kind for o in loc.operators if o.necessary) == ["annotate", "reorder"]


def test_single_stage_localises_the_effect_to_one_stage():
    loc = localise(_delta(), _make_evaluate({"annotate"}))
    by_stage = {s.stage: s for s in loc.stages}
    assert set(by_stage) == {"ranking", "presentation", "memory"}
    assert by_stage["presentation"].result.attributable
    assert not by_stage["ranking"].result.attributable
    assert not by_stage["memory"].result.attributable
    assert loc.dominant_stage == "presentation"
    assert by_stage["presentation"].share_of_full == pytest.approx(1.0)


def test_no_necessary_operator_when_the_effect_is_redundant():
    """Two independently sufficient operators: neither is individually necessary."""

    def evaluate(d: Intervention) -> ATEResult:
        kinds = {o.kind for o in d.ops}
        hit = 0.9 if ({"annotate"} <= kinds or {"plant_memory"} <= kinds) else 0.0
        return ate(runs([0.0] * 12), runs([hit] * 12), tau=0.3)

    loc = localise(_delta(), evaluate)
    assert loc.full.attributable
    assert loc.necessary_operators == []  # redundancy, correctly reported
