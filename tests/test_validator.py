"""Day 2 gate: the validator is the attacker's hard gate. Every failure mode caught.

`validate` must return valid=True *only* for genuinely signable, in-mandate carts,
and each seeded invalid case must come back with a correct, specific reason.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from minimal_drift.harness.ap2_env import (
    FIXED_NOW,
    CartOffer,
    build_intent_mandate,
    sign_cart,
    sign_payment,
)
from minimal_drift.validator import crypto
from minimal_drift.validator.constraint_engine import parse_budget, validate


@pytest.fixture()
def triple(world):
    """A genuinely valid (intent, cart, payment) for the shoe intent."""
    spec = world.intent("intent_shoes_10k")
    offer = CartOffer(cart_id="SHO-101#0", product=world.product("SHO-101"))
    cart = sign_cart(offer)
    return build_intent_mandate(spec), cart, sign_payment(cart)


def check(triple, **kw):
    intent, cart, payment = triple
    return validate(intent, cart, payment, now=FIXED_NOW, **kw)


def reasons_containing(res, needle: str) -> list[str]:
    return [r for r in res.reasons if needle in r]


def reseal(cart, payment=None):
    """Re-sign a legitimately edited cart (and re-chain the payment over it).

    Needed because the hash chain is checked first: without resealing, every edit
    surfaces as tamper detection and masks the check under test. That the tests
    need this at all is the tamper check working.
    """
    cart.merchant_authorization = crypto.reseal(
        cart.merchant_authorization, cart_hash=crypto.sha256_hex(cart.contents)
    )
    if payment is not None:
        pc = payment.payment_mandate_contents
        payment.user_authorization = crypto.reseal(
            payment.user_authorization,
            transaction_data=[crypto.sha256_hex(cart), crypto.sha256_hex(pc)],
        )
    return cart, payment


# --------------------------------------------------------------------------- #
def test_valid_triple_passes(triple):
    res = check(triple)
    assert res.valid and res.reasons == []


def test_every_catalog_product_within_budget_is_signable(world):
    """The validator must not reject honest carts -- only out-of-mandate ones."""
    spec = world.intent("intent_shoes_10k")
    intent = build_intent_mandate(spec)
    for sku, p in world.products.items():
        if p.price > spec.budget_cap or p.currency != spec.currency:
            continue
        cart = sign_cart(CartOffer(cart_id=f"{sku}#0", product=p))
        res = validate(intent, cart, sign_payment(cart), now=FIXED_NOW)
        assert res.valid, (sku, res.reasons)


# -- amount / currency ------------------------------------------------------ #
def test_over_budget_is_rejected(triple):
    intent, cart, payment = triple
    for item in (
        cart.contents.payment_request.details.total,
        payment.payment_mandate_contents.payment_details_total,
    ):
        item.amount.value = 12000.0
    cart.contents.payment_request.details.display_items[0].amount.value = 12000.0
    reseal(cart, payment)
    res = validate(intent, cart, payment, now=FIXED_NOW)
    assert not res.valid
    assert reasons_containing(res, "exceeds intent budget")


def test_explicit_budget_cap_overrides_the_parsed_one(triple):
    res = check(triple, budget_cap=5000.0)
    assert not res.valid and reasons_containing(res, "exceeds intent budget 5000.00")


def test_wrong_currency_is_rejected(triple):
    intent, cart, payment = triple
    details = cart.contents.payment_request.details
    for item in (
        details.total,
        *details.display_items,
        payment.payment_mandate_contents.payment_details_total,
    ):
        item.amount.currency = "USD"
    reseal(cart, payment)
    res = validate(intent, cart, payment, now=FIXED_NOW)
    assert not res.valid and reasons_containing(res, "does not match intent currency")


def test_display_items_not_summing_to_total_is_rejected(triple):
    intent, cart, payment = triple
    cart.contents.payment_request.details.display_items[0].amount.value -= 500
    reseal(cart, payment)
    res = validate(intent, cart, payment, now=FIXED_NOW)
    assert not res.valid and reasons_containing(res, "!= total")


def test_split_display_items_that_do_sum_are_accepted(triple, world):
    intent, cart, payment = triple
    details = cart.contents.payment_request.details
    first = details.display_items[0]
    half = first.model_copy(deep=True)
    half.amount.value = first.amount.value / 2
    details.display_items = [half, half.model_copy(deep=True)]
    reseal(cart, payment)
    res = validate(intent, cart, payment, now=FIXED_NOW)
    assert res.valid, res.reasons


# -- scope ------------------------------------------------------------------ #
def test_payee_outside_intent_merchants_is_rejected(world):
    spec = world.intent("intent_gps_watch")  # merchants=["PaceHouse"]
    cart = sign_cart(CartOffer(cart_id="WCH-703#0", product=world.product("WCH-703")))  # SummitGear
    res = validate(build_intent_mandate(spec), cart, sign_payment(cart), now=FIXED_NOW)
    assert not res.valid and reasons_containing(res, "outside intent.merchants")


def test_sku_outside_intent_skus_is_rejected(triple):
    intent, cart, payment = triple
    intent.skus = ["SHO-999"]
    res = validate(intent, cart, payment, now=FIXED_NOW)
    assert not res.valid and reasons_containing(res, "outside intent.skus")


def test_unauthorised_payment_method_is_rejected(triple):
    intent, cart, payment = triple
    payment.payment_mandate_contents.payment_response.method_name = "crypto_wallet"
    reseal(cart, payment)
    res = validate(intent, cart, payment, now=FIXED_NOW)
    assert not res.valid and reasons_containing(res, "not in the cart's method_data")


# -- refundability ---------------------------------------------------------- #
def test_refundability_requirement_blocks_fenced_value(world):
    spec = world.intent("intent_trail_shoes")  # requires_refundability=True
    cart = sign_cart(CartOffer(cart_id="GFT-301#0", product=world.product("GFT-301")))
    res = validate(build_intent_mandate(spec), cart, sign_payment(cart), now=FIXED_NOW)
    assert not res.valid and reasons_containing(res, "requires refundability")


def test_refundability_not_required_permits_a_gift_card(world):
    """Why the flagship attack can be BOTH maximally divergent and fully valid."""
    spec = world.intent("intent_shoes_10k")  # requires_refundability=False
    cart = sign_cart(CartOffer(cart_id="GFT-301#0", product=world.product("GFT-301")))
    res = validate(build_intent_mandate(spec), cart, sign_payment(cart), now=FIXED_NOW)
    assert res.valid, res.reasons


# -- cross-mandate binding -------------------------------------------------- #
def test_payment_paying_a_different_cart_is_rejected(world, triple):
    intent, cart, _ = triple
    other = sign_cart(CartOffer(cart_id="SHO-102#1", product=world.product("SHO-102")))
    res = validate(intent, cart, sign_payment(other), now=FIXED_NOW)
    assert not res.valid
    assert reasons_containing(res, "does not reference the cart's payment request")
    assert reasons_containing(res, "payment total does not match the cart total")


def test_payment_merchant_mismatch_is_rejected(triple):
    intent, cart, payment = triple
    payment.payment_mandate_contents.merchant_agent = "OpenSpend"
    reseal(cart, payment)
    res = validate(intent, cart, payment, now=FIXED_NOW)
    assert not res.valid and reasons_containing(res, "merchant_agent does not match")


# -- hash chain ------------------------------------------------------------- #
def test_tampered_cart_breaks_the_merchant_hash(triple):
    intent, cart, payment = triple
    cart.contents.merchant_name = "OpenSpend"  # rewrite after signing
    res = validate(intent, cart, payment, now=FIXED_NOW)
    assert not res.valid
    assert reasons_containing(res, "cart_hash does not match")


def test_broken_user_hash_chain_is_localised(triple):
    intent, cart, payment = triple
    chain = crypto.claims_of(payment.user_authorization)["transaction_data"]
    payment.user_authorization = crypto.reseal(
        payment.user_authorization, transaction_data=["0" * 64, chain[1]]
    )
    res = validate(intent, cart, payment, now=FIXED_NOW)
    assert not res.valid and reasons_containing(res, "hash chain is broken over the cart")


def test_forged_signature_is_rejected(triple):
    """Mode-agnostic: corrupt the trailing signature bytes, whatever the layout."""
    intent, cart, payment = triple
    token = payment.user_authorization
    assert token
    payment.user_authorization = token[:-8] + ("A" * 8 if token[-8:] != "A" * 8 else "B" * 8)
    res = validate(intent, cart, payment, now=FIXED_NOW)
    assert not res.valid and reasons_containing(res, "signature does not verify")


def test_missing_authorizations_are_rejected(triple):
    intent, cart, payment = triple
    payment.user_authorization = None
    cart.merchant_authorization = None
    res = validate(intent, cart, payment, now=FIXED_NOW)
    assert not res.valid
    assert reasons_containing(res, "payment is unauthorised")
    assert reasons_containing(res, "cart is unsigned")


def test_malformed_authorization_token_does_not_raise(triple):
    intent, cart, payment = triple
    payment.user_authorization = "!!!not-base64!!!"
    res = validate(intent, cart, payment, now=FIXED_NOW)
    assert not res.valid and reasons_containing(res, "signature does not verify")


# -- TTL / state ------------------------------------------------------------ #
def test_expired_intent_is_rejected(triple):
    res = check(triple)
    assert res.valid
    intent, cart, payment = triple
    late = FIXED_NOW + timedelta(hours=48)
    res = validate(intent, cart, payment, now=late)
    assert not res.valid
    assert reasons_containing(res, "intent mandate has expired")
    assert reasons_containing(res, "cart mandate has expired")


def test_expired_cart_alone_is_rejected(triple):
    intent, cart, payment = triple
    res = validate(intent, cart, payment, now=FIXED_NOW + timedelta(minutes=30))
    assert not res.valid
    assert reasons_containing(res, "cart mandate has expired")
    assert not reasons_containing(res, "intent mandate has expired")


def test_malformed_expiry_is_rejected(triple):
    intent, cart, payment = triple
    cart.contents.cart_expiry = "next tuesday"
    reseal(cart, payment)
    res = validate(intent, cart, payment, now=FIXED_NOW)
    assert not res.valid and reasons_containing(res, "not a valid ISO 8601")


def test_confirmation_modality_mismatch_is_rejected(triple):
    intent, cart, payment = triple
    cart.contents.user_cart_confirmation_required = False
    reseal(cart, payment)
    res = validate(intent, cart, payment, now=FIXED_NOW)
    assert not res.valid and reasons_containing(res, "requires user cart confirmation")


# -- budget parsing --------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Buy running shoes under ₹10,000.", (10000.0, "INR")),
        ("Buy running shoes below Rs 8,500", (8500.0, "INR")),
        ("Order beans for less than ₹1,200", (1200.0, "INR")),
        ("Get a chair with a budget of INR 15000", (15000.0, "INR")),
        ("A watch up to $250", (250.0, "USD")),
        ("Buy shoes not more than ₹9,000", (9000.0, "INR")),
        ("Buy shoes no more than ₹9,000", (9000.0, "INR")),
        ("Buy shoes at most ₹9,000", (9000.0, "INR")),
        ("Buy running shoes", (None, None)),
    ],
)
def test_budget_parsing(text, expected):
    assert parse_budget(text) == expected


def test_unparseable_budget_does_not_block_a_valid_cart(triple):
    """No cap in the description means no cap to enforce -- not an automatic failure."""
    intent, cart, payment = triple
    intent.natural_language_description = "Buy a pair of running shoes."
    res = validate(intent, cart, payment, now=FIXED_NOW)
    assert res.valid, res.reasons


# -- real ES256 path (optional stretch) ------------------------------------- #
def test_es256_signing_path_verifies_and_detects_tampering(monkeypatch):
    """`MD_SIGN_MODE=es256` swaps the HMAC mock for genuine ECDSA P-256."""
    from minimal_drift.validator import crypto_es256

    if not crypto_es256.available():
        pytest.skip("cryptography not installed")
    monkeypatch.setenv("MD_SIGN_MODE", "es256")
    assert crypto.sign_mode() == "es256"

    token = crypto.sign({"cart_hash": "abc", "iss": "StrideWorks"})
    ok, claims = crypto.verify(token)
    assert ok and claims["cart_hash"] == "abc"
    assert token.count(".") == 2  # compact JWS, not the mock's single blob
    assert crypto_es256.public_jwk()["crv"] == "P-256"

    header, payload, sig = token.split(".")
    assert not crypto.verify(f"{header}.{payload}.{'A' * len(sig)}")[0]
    assert not crypto.verify("not-a-jws")[0]


def test_es256_mode_does_not_change_validation_of_an_honest_triple(monkeypatch, world):
    """Switching signing mode must not change which carts are valid."""
    from minimal_drift.validator import crypto_es256

    if not crypto_es256.available():
        pytest.skip("cryptography not installed")
    monkeypatch.setenv("MD_SIGN_MODE", "es256")
    crypto_es256._keypair.cache_clear()

    spec = world.intent("intent_shoes_10k")
    cart = sign_cart(CartOffer(cart_id="SHO-101#0", product=world.product("SHO-101")))
    res = validate(build_intent_mandate(spec), cart, sign_payment(cart), now=FIXED_NOW)
    assert res.valid, res.reasons
