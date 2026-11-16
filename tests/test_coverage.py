"""Day 6 gate: external-taxonomy coverage and the binning-sensitivity check."""

from __future__ import annotations

import pytest

from minimal_drift.mechanisms import MECHANISM_NAMES
from minimal_drift.metrics.coverage import (
    MAESTRO_LAYERS,
    MECHANISM_MAPPING,
    OUT_OF_SCOPE,
    OWASP_THREATS,
    band,
    binning_sensitivity,
    maestro_coverage,
    owasp_coverage,
    unique_strategies,
)

ALL_CELLS = [
    ("ranking", "ranking"),
    ("omission", "retrieval"),
    ("metadata", "presentation"),
    ("memory", "memory"),
    ("consent", "confirmation"),
    ("tool_output", "presentation"),
]


# --------------------------------------------------------------------------- #
# the mapping itself
# --------------------------------------------------------------------------- #
def test_every_mechanism_is_mapped():
    assert set(MECHANISM_MAPPING) == set(MECHANISM_NAMES)


def test_the_mapping_only_references_real_taxonomy_items():
    """A typo'd threat id would silently inflate coverage."""
    for mechanism, (threats, layers) in MECHANISM_MAPPING.items():
        assert threats and layers, mechanism
        assert set(threats) <= set(OWASP_THREATS), mechanism
        assert set(layers) <= set(MAESTRO_LAYERS), mechanism


def test_out_of_scope_items_are_declared_with_a_reason():
    assert set(OUT_OF_SCOPE) <= set(OWASP_THREATS)
    assert all(reason for reason in OUT_OF_SCOPE.values())


def test_no_mechanism_claims_an_out_of_scope_threat():
    """Otherwise the denominator and the numerator would disagree."""
    for mechanism, (threats, _) in MECHANISM_MAPPING.items():
        assert not set(threats) & set(OUT_OF_SCOPE), mechanism


# --------------------------------------------------------------------------- #
# coverage
# --------------------------------------------------------------------------- #
def test_empty_archive_covers_nothing():
    c = owasp_coverage([])
    assert c.covered_items == [] and c.coverage_of_in_scope == 0.0
    assert c.coverage_of_all == 0.0


def test_coverage_uses_an_external_denominator_not_our_grid():
    c = owasp_coverage(ALL_CELLS)
    assert c.total_items == len(OWASP_THREATS) == 15
    assert c.in_scope_items == 15 - len(OUT_OF_SCOPE)
    assert c.in_scope_items < c.total_items


def test_full_mechanism_coverage_does_not_claim_full_taxonomy_coverage():
    """Six mechanisms cannot possibly cover 15 threats; the report must not pretend."""
    c = owasp_coverage(ALL_CELLS)
    assert 0.0 < c.coverage_of_all < 1.0
    assert c.uncovered_in_scope or c.coverage_of_in_scope == 1.0


def test_coverage_is_monotone_in_filled_cells():
    small = owasp_coverage(ALL_CELLS[:2])
    big = owasp_coverage(ALL_CELLS)
    assert set(small.covered_items) <= set(big.covered_items)
    assert small.coverage_of_in_scope <= big.coverage_of_in_scope


def test_consent_is_what_covers_the_human_in_the_loop_threats():
    without = set(owasp_coverage([c for c in ALL_CELLS if c[0] != "consent"]).covered_items)
    with_ = set(owasp_coverage(ALL_CELLS).covered_items)
    assert {"T10", "T15"} <= (with_ - without)


def test_memory_is_what_covers_memory_poisoning():
    without = set(owasp_coverage([c for c in ALL_CELLS if c[0] != "memory"]).covered_items)
    assert "T1" not in without
    assert "T1" in set(owasp_coverage(ALL_CELLS).covered_items)


def test_maestro_coverage_declares_model_internals_out_of_scope():
    c = maestro_coverage(ALL_CELLS)
    assert "L1" in c.out_of_scope and "L4" in c.out_of_scope
    assert c.in_scope_items == len(MAESTRO_LAYERS) - 2
    assert 0.0 < c.coverage_of_in_scope <= 1.0


def test_unique_strategies_counts_distinct_cells():
    assert unique_strategies(ALL_CELLS) == 6
    assert unique_strategies(ALL_CELLS + ALL_CELLS) == 6


# --------------------------------------------------------------------------- #
# binning sensitivity
# --------------------------------------------------------------------------- #
def test_bands_partition_the_impact_range():
    edges = (500.0, 2000.0)
    assert band(0.0, edges) == "<=500"
    assert band(500.0, edges) == "<=500"
    assert band(500.01, edges) == "<=2000"
    assert band(9999.0, edges) == ">2000"


def test_baseline_variant_matches_the_archive_grid():
    b = binning_sensitivity(ALL_CELLS, mechanisms=6)
    base = next(v for v in b.variants if v.name.startswith("baseline"))
    assert base.capacity == 30 and base.filled == 6


def test_merging_stages_raises_coverage_which_is_the_point_of_the_check():
    """Coarser bins flatter the coverage number. Reporting both is the guard."""
    b = binning_sensitivity(ALL_CELLS, mechanisms=6)
    base = next(v for v in b.variants if v.name.startswith("baseline"))
    merged = next(v for v in b.variants if v.name.startswith("merged"))
    assert merged.capacity < base.capacity
    assert merged.coverage > base.coverage


def test_split_and_economic_variants_appear_only_with_their_inputs():
    without = binning_sensitivity(ALL_CELLS, mechanisms=6)
    assert not any("split" in v.name for v in without.variants)
    assert not any("impact band" in v.name for v in without.variants)

    with_ = binning_sensitivity(
        ALL_CELLS,
        elite_op_kinds={c: ["annotate"] for c in ALL_CELLS},
        elite_impacts={c: 1500.0 for c in ALL_CELLS},
        mechanisms=6,
    )
    assert any("split" in v.name for v in with_.variants)
    assert sum("impact band" in v.name for v in with_.variants) == 2


def test_spread_is_reported_and_flags_an_unstable_grid():
    b = binning_sensitivity(ALL_CELLS, mechanisms=6)
    assert b.spread == pytest.approx(
        max(v.coverage for v in b.variants) - min(v.coverage for v in b.variants)
    )
    assert isinstance(b.stable, bool)


def test_no_variants_means_no_spread():
    b = binning_sensitivity([], mechanisms=0)
    assert b.spread == 0.0
