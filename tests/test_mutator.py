"""Day 5 gate: boundary-typed reflection changes STRATEGY after a refusal.

The claim being tested is not "the mutator produces variation" -- it is that the reply
to a specific boundary is a specific structural repair. Text variation alone must never
move a delta into a new behavioural cell, or paraphrase would masquerade as coverage.
"""

from __future__ import annotations

import random

import pytest

from minimal_drift.config import get_config
from minimal_drift.mechanisms.base import Intervention, Operator
from minimal_drift.mechanisms.context import build_context
from minimal_drift.search.mutator import REFLECTION, Mutator


@pytest.fixture()
def ctx(env):
    return build_context(
        env.world, "intent_shoes_10k", env.offers("intent_shoes_10k", 1), get_config().tau
    )


@pytest.fixture()
def mutator():
    return Mutator(use_llm=False)


def kinds(d: Intervention) -> set[str]:
    return {o.kind for o in d.ops}


# --------------------------------------------------------------------------- #
# reflection
# --------------------------------------------------------------------------- #
def test_consent_blocked_adds_a_consent_operator(mutator, ctx):
    """The single most valuable repair in the space, and the reason reflection exists."""
    parent = Intervention(
        mechanism="metadata",
        intent_id=ctx.spec.id,
        ops=[Operator(kind="relabel", target=ctx.offers[0].cart_id, payload="Running Shoes")],
    )
    added = 0
    for s in range(25):
        child = mutator.mutate(
            parent, ("metadata", "presentation"), random.Random(s), ctx, "consent_blocked"
        )
        added += "rewrite_confirmation" in kinds(child)
    assert added == 25, "every consent_blocked reflection must add a consent operator"


def test_no_selection_relaxes_instead_of_escalating(mutator, ctx):
    """The intervention filtered every option away: the repair is to remove, not add."""
    parent = Intervention(
        mechanism="omission",
        intent_id=ctx.spec.id,
        ops=[
            Operator(kind="drop_attribute", target=None, key="use"),
            Operator(kind="drop_option", target=ctx.offers[0].cart_id),
            Operator(kind="drop_option", target=ctx.offers[1].cart_id),
        ],
    )
    for s in range(15):
        child = mutator.mutate(
            parent, ("omission", "retrieval"), random.Random(s), ctx, "no_selection"
        )
        assert child.n_fields <= parent.n_fields
        assert "relaxed after no_selection" in child.note or child.n_fields < parent.n_fields


def test_no_drift_escalates_by_splicing_another_mechanism(mutator, ctx):
    parent = Intervention(
        mechanism="ranking",
        intent_id=ctx.spec.id,
        ops=[Operator(kind="reorder", payload=[ctx.offers[-1].cart_id])],
    )
    spliced = 0
    for s in range(25):
        child = mutator.mutate(parent, ("ranking", "ranking"), random.Random(s), ctx, "no_drift")
        spliced += bool(kinds(child) - {"reorder"})
    assert spliced > 0
    assert all(m in REFLECTION["no_drift"] for m in ("metadata", "memory", "tool_output"))


def test_success_triggers_no_reflection(mutator, ctx):
    assert REFLECTION["success"] == () and REFLECTION[""] == ()
    parent = Intervention(
        mechanism="ranking",
        intent_id=ctx.spec.id,
        ops=[Operator(kind="reorder", payload=[ctx.offers[0].cart_id])],
    )
    for s in range(10):
        a = mutator.mutate(parent, ("ranking", "ranking"), random.Random(s), ctx, "success")
        b = mutator.mutate(parent, ("ranking", "ranking"), random.Random(s), ctx, "")
        assert a.signature() == b.signature()


# --------------------------------------------------------------------------- #
# invariants
# --------------------------------------------------------------------------- #
def test_mutation_is_deterministic_in_its_seed(mutator, ctx):
    for s in range(12):
        a = mutator.mutate(None, ("metadata", "presentation"), random.Random(s), ctx, "")
        b = mutator.mutate(None, ("metadata", "presentation"), random.Random(s), ctx, "")
        assert a.signature() == b.signature()


def test_children_carry_the_cell_and_the_intent(mutator, ctx):
    for cell in (("memory", "memory"), ("consent", "confirmation"), ("omission", "retrieval")):
        child = mutator.mutate(None, cell, random.Random(1), ctx, "")
        assert child.mechanism == cell[0]
        assert child.intent_id == ctx.spec.id


def test_text_rewriting_never_changes_the_behavioural_descriptor(mutator, ctx):
    """Paraphrase is not coverage: only structure may claim a new cell."""
    parent = Intervention(
        mechanism="tool_output",
        intent_id=ctx.spec.id,
        ops=[Operator(kind="annotate", target=ctx.offers[0].cart_id, payload="original text")],
    )
    for s in range(20):
        child = mutator._rewrite_text(parent, random.Random(s), ctx)
        assert (child.effective_mechanism, child.stage) == (
            parent.effective_mechanism,
            parent.stage,
        )
        assert [o.kind for o in child.ops] == [o.kind for o in parent.ops]


def test_duplicate_operators_are_collapsed(mutator, ctx):
    """Two copies of one edit is one field, not two -- otherwise C(delta) inflates."""
    op = Operator(kind="annotate", target=ctx.offers[0].cart_id, payload="same")
    parent = Intervention(mechanism="tool_output", intent_id=ctx.spec.id, ops=[op, op])
    child = mutator.mutate(parent, ("tool_output", "presentation"), random.Random(0), ctx, "")
    sigs = {f"{o.kind}|{o.target}|{o.payload}" for o in child.ops}
    assert len(sigs) == len(child.ops)


def test_llm_is_off_without_a_key(ctx):
    """No key means no live calls, and the surrogate path must still mutate."""
    m = Mutator()
    if not get_config().live_mode:
        assert m.use_llm is False
    child = m.mutate(None, ("metadata", "presentation"), random.Random(0), ctx, "")
    assert m.llm_calls == 0
    assert child.ops
