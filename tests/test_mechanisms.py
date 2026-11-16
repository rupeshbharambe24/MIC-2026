"""Day 4 gate: every mechanism emits a well-formed delta, and C(delta) is honest."""

from __future__ import annotations

import random

import pytest

from minimal_drift.config import CostWeights, get_config
from minimal_drift.mechanisms import MECHANISM_NAMES, MECHANISMS
from minimal_drift.mechanisms.base import OP_SURFACE, STAGES, Intervention, Operator
from minimal_drift.mechanisms.context import build_context
from minimal_drift.validator.constraint_engine import validate

SEEDS = range(20)


@pytest.fixture()
def contexts(env):
    cfg = get_config()
    return {i: build_context(env.world, i, env.offers(i, 1), cfg.tau) for i in env.world.intents}


# --------------------------------------------------------------------------- #
# well-formedness
# --------------------------------------------------------------------------- #
def test_registry_holds_exactly_the_six_mechanisms():
    assert set(MECHANISM_NAMES) == {
        "ranking",
        "omission",
        "metadata",
        "memory",
        "consent",
        "tool_output",
    }


@pytest.mark.parametrize("name", MECHANISM_NAMES)
def test_every_proposal_is_well_formed(name, contexts):
    mech = MECHANISMS[name]
    seen_any_ops = False
    for intent_id, ctx in contexts.items():
        for s in SEEDS:
            d = mech.propose(None, random.Random(s), ctx)
            assert d.mechanism == name
            assert d.intent_id == intent_id
            for op in d.ops:
                assert op.kind in mech.operators(), f"{name} emitted {op.kind}"
                assert op.surface in OP_SURFACE[op.kind]
                assert op.stage in STAGES
                op.to_injection()  # must compile
            if d.ops:
                seen_any_ops = True
                assert d.stage in STAGES
                assert d.surfaces and all(
                    s in ("metadata", "tool", "ranking", "memory", "consent") for s in d.surfaces
                )
    assert seen_any_ops, f"{name} never produced an intervention"


@pytest.mark.parametrize("name", MECHANISM_NAMES)
def test_proposals_are_deterministic_in_their_seed(name, contexts):
    ctx = contexts["intent_shoes_10k"]
    for s in SEEDS:
        a = MECHANISMS[name].propose(None, random.Random(s), ctx)
        b = MECHANISMS[name].propose(None, random.Random(s), ctx)
        assert a.signature() == b.signature()


@pytest.mark.parametrize("name", MECHANISM_NAMES)
def test_applying_a_proposal_never_corrupts_the_view(name, contexts, env):
    """Interventions rewrite what the agent SEES. Nothing else may change."""
    for intent_id, ctx in contexts.items():
        clean = env.clean_view(intent_id, seed=1)
        for s in list(SEEDS)[:6]:
            d = MECHANISMS[name].propose(None, random.Random(s), ctx)
            out = d.stack().apply(clean)
            assert {o.cart_id for o in out.options} <= {o.cart_id for o in clean.options}
            for o in out.options:
                orig = clean.option(o.cart_id)
                assert orig is not None
                # price, currency, merchant and refund_period are merchant-signed facts
                assert (o.price, o.currency, o.merchant, o.refund_period) == (
                    orig.price,
                    orig.currency,
                    orig.merchant,
                    orig.refund_period,
                )
            assert out.budget_cap == clean.budget_cap
            assert out.intent_text == clean.intent_text
            assert clean.confirmation_prompt is None  # input untouched


# --------------------------------------------------------------------------- #
# the validity gate
# --------------------------------------------------------------------------- #
def test_no_intervention_can_produce_an_invalid_signed_payment(contexts, env):
    """An intervention may cause drift, but never an out-of-mandate payment.

    Whatever the attacker writes, the merchant signs the truth and the constraint
    engine sees it -- so any run that reaches a payment must validate. Runs blocked
    earlier (consent, no eligible option) carry a reason instead.
    """
    for intent_id, ctx in contexts.items():
        for name in MECHANISM_NAMES:
            for s in list(SEEDS)[:5]:
                d = MECHANISMS[name].propose(None, random.Random(s), ctx)
                if not d.ops:
                    continue
                r = env.run(intent_id, 1, stack=d.stack())
                if r.boundary_reason in ("consent_blocked", "no_selection"):
                    assert not r.valid and r.validation.reasons
                else:
                    assert r.valid, (name, intent_id, r.validation.reasons)


def test_candidates_are_prefiltered_to_valid_targets(contexts, env):
    """Targets that could not possibly validate are ruled out for zero queries."""
    for intent_id, ctx in contexts.items():
        spec = env.world.intent(intent_id)
        for o in ctx.candidates:
            p = o.product
            assert p.price <= spec.budget_cap and p.currency == spec.currency
            if spec.requires_refundability:
                assert p.refund_period > 0
            if spec.merchants:
                assert p.merchant in spec.merchants
            assert ctx.divergence_of(p) >= get_config().tau


def test_refundability_requirement_removes_every_stored_value_target(env):
    """Why `requires_refundability` is a real defence, not decoration."""
    cfg = get_config()
    strict = [i for i, s in env.world.intents.items() if s.requires_refundability]
    assert strict, "expected some refundability-constrained intents"
    for intent_id in strict:
        ctx = build_context(env.world, intent_id, env.offers(intent_id, 1), cfg.tau)
        assert all(o.product.refund_period > 0 for o in ctx.candidates)
        assert all(o.product.category not in ("gift_card", "store_credit") for o in ctx.candidates)


# --------------------------------------------------------------------------- #
# C(delta) monotonicity
# --------------------------------------------------------------------------- #
def op(kind: str, **kw) -> Operator:
    return Operator(kind=kind, **kw)


def test_cost_is_monotonic_in_fields():
    a = Intervention(mechanism="m", ops=[op("reorder", payload=["X"])])
    b = Intervention(
        mechanism="m", ops=[op("reorder", payload=["X"]), op("relabel", target="X", payload="y")]
    )
    assert b.n_fields > a.n_fields and b.cost() > a.cost()


def test_cost_is_monotonic_in_tokens():
    short = Intervention(mechanism="m", ops=[op("add_tool_note", payload="a b")])
    long = Intervention(mechanism="m", ops=[op("add_tool_note", payload=" ".join(["w"] * 400))])
    assert long.n_tokens > short.n_tokens and long.cost() > short.cost()


def test_cost_is_monotonic_in_turns():
    a = Intervention(mechanism="m", ops=[op("plant_memory", payload="x", turns=0)])
    b = Intervention(mechanism="m", ops=[op("plant_memory", payload="x", turns=3)])
    assert b.n_turns > a.n_turns and b.cost() > a.cost()


def test_cost_is_monotonic_in_surfaces():
    one = Intervention(mechanism="m", ops=[op("relabel", target="X", payload="a")])
    two = Intervention(
        mechanism="m",
        ops=[op("relabel", target="X", payload="a"), op("plant_memory", payload="a")],
    )
    assert len(two.surfaces) > len(one.surfaces) and two.cost() > one.cost()


def test_cost_weights_are_respected():
    d = Intervention(mechanism="m", ops=[op("relabel", target="X", payload="a b c")])
    cheap = d.cost(CostWeights(field=1.0, per_100_tokens=0.0, turn=0.0, surface=0.0))
    dear = d.cost(CostWeights(field=10.0, per_100_tokens=0.0, turn=0.0, surface=0.0))
    assert dear == pytest.approx(10 * cheap)


def test_blanket_omission_costs_more_than_a_targeted_one():
    targeted = Intervention(mechanism="m", ops=[op("drop_attribute", target="X#1", key="use")])
    blanket = Intervention(mechanism="m", ops=[op("drop_attribute", target=None, key="use")])
    assert blanket.n_fields > targeted.n_fields


# --------------------------------------------------------------------------- #
# delta algebra (used by the minimiser and the ablation)
# --------------------------------------------------------------------------- #
def test_without_and_subset_produce_subsets():
    d = Intervention(
        mechanism="m",
        ops=[
            op("reorder", payload=["X"]),
            op("annotate", target="X", payload="hi"),
            op("plant_memory", payload="m"),
        ],
    )
    for i in range(len(d.ops)):
        assert d.without(i).is_subset_of(d)
        assert len(d.without(i).ops) == len(d.ops) - 1
    assert d.subset([0, 2]).is_subset_of(d)
    assert not d.is_subset_of(d.without(0))


def test_stage_is_the_earliest_stage_touched():
    d = Intervention(
        mechanism="m",
        ops=[op("reorder", payload=["X"]), op("drop_option", target="Y")],
    )
    assert d.stages == ("retrieval", "ranking")
    assert d.stage == "retrieval"


def test_only_stage_and_only_surface_partition_the_ops():
    d = Intervention(
        mechanism="m",
        ops=[
            op("reorder", payload=["X"]),
            op("relabel", target="X", payload="a"),
            op("rewrite_confirmation", payload="ok"),
        ],
    )
    assert sum(len(d.only_stage(s).ops) for s in d.stages) == len(d.ops)
    assert sum(len(d.only_surface(s).ops) for s in d.surfaces) == len(d.ops)


def test_signature_distinguishes_different_deltas():
    a = Intervention(mechanism="m", ops=[op("relabel", target="X", payload="a")])
    b = Intervention(mechanism="m", ops=[op("relabel", target="X", payload="b")])
    assert a.signature() != b.signature()
    assert a.signature() == a.model_copy(deep=True).signature()


def test_memory_constraints_are_parsed_and_enforced(env):
    """A planted requirement acts as a filter, which is what makes memory dangerous."""
    from minimal_drift.harness.agent import SurrogateAgent, memory_constraints
    from minimal_drift.harness.surfaces import SurfaceStack, plant_memory

    assert memory_constraints(["requires: style=retro_running"]) == [("style", "retro_running")]
    assert memory_constraints(["saved preference: shoes"]) == []

    view = env.clean_view("intent_shoes_10k", seed=1)
    agent = SurrogateAgent()
    before = [o.cart_id for o in view.options if agent.eligible(view, o)]
    after_view = (
        SurfaceStack().add("memory", plant_memory("requires: style=retro_running")).apply(view)
    )
    after = [o.cart_id for o in after_view.options if agent.eligible(after_view, o)]
    assert len(after) < len(before)
    assert all("SNK" in c or "retro" in c for c in after) or after == []


def test_consent_alone_changes_no_agent_decision(env, contexts):
    """Documented, and load-bearing for the report: consent removes a check, it does
    not cause a choice. Expect identical selections with and without it."""
    ctx = contexts["intent_shoes_10k"]
    d = MECHANISMS["consent"].propose(None, random.Random(0), ctx)
    for seed in range(1, 8):
        clean = env.run("intent_shoes_10k", seed)
        with_consent = env.run("intent_shoes_10k", seed, stack=d.stack())
        assert with_consent.chosen_cart_id == clean.chosen_cart_id


def test_validator_is_the_gate_not_the_mechanisms(world):
    """Sanity: the mechanisms do not get to decide validity. `validate` does."""
    from minimal_drift.harness.ap2_env import (
        CartOffer,
        build_intent_mandate,
        sign_cart,
        sign_payment,
    )

    spec = world.intent("intent_trail_shoes")  # requires_refundability
    cart = sign_cart(CartOffer(cart_id="GFT-303#x", product=world.product("GFT-303")))
    res = validate(build_intent_mandate(spec), cart, sign_payment(cart))
    assert not res.valid
