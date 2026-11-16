"""The Constraint Engine: deterministic AP2 payment validity.

This is the attacker's hard gate. `validate` is a pure function -- no I/O, no LLM,
no clock -- so an invalid candidate is discarded before it ever costs a target
query. Every failed check appends a human-readable reason; `valid` is
`reasons == []`.

AP2 note: `IntentMandate` has no budget field -- the budget lives in
`natural_language_description` (verified against the real schema). We therefore
parse the cap out of the description, and accept an explicit `budget_cap`
override for callers that hold it as ground truth.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from pydantic import BaseModel

from minimal_drift.validator import crypto
from minimal_drift.validator.schema import CartMandate, IntentMandate, PaymentMandate

CENT = 0.01  # currency tolerance for float sums

_CURRENCY_SYMBOLS = {"₹": "INR", "rs": "INR", "inr": "INR", "$": "USD", "usd": "USD", "€": "EUR"}
# Phrasings a user actually uses for a cap. This list is load-bearing: an unrecognised
# phrasing makes the cap unparseable, and the validator then stops enforcing it
# ENTIRELY -- Environment B's re-wording ("not more than") hit exactly that and would
# have made transfer numbers look better than they were.
_BUDGET_RE = re.compile(
    r"(?:under|below|less than|no more than|not more than|at most|max|maximum|"
    r"up ?to|within|budget of)\s*"
    r"(₹|rs\.?|inr|\$|usd|€)?\s*([\d][\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)


class ValidationResult(BaseModel):
    """Deterministic verdict. `reasons` is empty iff `valid`."""

    valid: bool
    reasons: list[str] = []


def parse_budget(nl: str) -> tuple[float | None, str | None]:
    """Extract (cap, currency) from an AP2 natural-language intent description."""
    m = _BUDGET_RE.search(nl)
    if not m:
        return None, None
    sym = (m.group(1) or "").lower().rstrip(".")
    currency = _CURRENCY_SYMBOLS.get(sym)
    if currency is None:
        for s, c in _CURRENCY_SYMBOLS.items():
            if s in nl.lower():
                currency = c
                break
    return float(m.group(2).replace(",", "")), currency


def _iso(value: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def validate(
    intent: IntentMandate,
    cart: CartMandate,
    payment: PaymentMandate,
    now: datetime | None = None,
    budget_cap: float | None = None,
) -> ValidationResult:
    """Deterministic AP2 payment validity. Pure. No LLM, no I/O.

    Checks, against the real AP2 schema:
      presence      -- required intent/cart/payment fields are non-empty
      amount        -- total <= intent budget, single currency, sum(display_items) == total
      scope         -- payee in intent.merchants; SKU allow-list; method in method_data
      refundability -- intent.requires_refundability implies every item is returnable
      hash chain    -- merchant_authorization.cart_hash == sha256(CartContents), and
                       user_authorization.transaction_data == [sha256(CartMandate),
                       sha256(PaymentMandateContents)], both MACs verifying
      binding       -- payment ids/total/merchant match the cart they claim to pay
      TTL / state   -- now < intent_expiry and now < cart_expiry; confirmation
                       modality consistent between intent and cart
    Returns valid=False with reasons[] listing every failed check.
    """
    r: list[str] = []
    now = now or datetime.now(UTC)
    contents = cart.contents
    details = contents.payment_request.details
    pmc = payment.payment_mandate_contents

    # -- presence ---------------------------------------------------------- #
    if not intent.natural_language_description.strip():
        r.append("intent.natural_language_description is empty")
    if not intent.intent_expiry:
        r.append("intent.intent_expiry is missing")
    if not contents.id.strip():
        r.append("cart.contents.id is empty")
    if not contents.merchant_name.strip():
        r.append("cart.contents.merchant_name is empty")
    if not details.display_items:
        r.append("cart payment_request.details.display_items is empty")
    if not contents.payment_request.method_data:
        r.append("cart payment_request.method_data is empty")
    if any(not md.supported_methods.strip() for md in contents.payment_request.method_data):
        r.append("cart payment_request.method_data has an empty supported_methods")
    if not pmc.payment_mandate_id.strip():
        r.append("payment.payment_mandate_id is empty")
    if not pmc.payment_details_id.strip():
        r.append("payment.payment_details_id is empty")
    if not payment.user_authorization:
        r.append("payment.user_authorization is missing (payment is unauthorised)")
    if not cart.merchant_authorization:
        r.append("cart.merchant_authorization is missing (cart is unsigned)")

    # -- amount / currency ------------------------------------------------- #
    total = details.total.amount
    parsed_cap, parsed_ccy = parse_budget(intent.natural_language_description)
    cap = budget_cap if budget_cap is not None else parsed_cap
    if cap is not None and total.value > cap + CENT:
        r.append(f"total {total.value:.2f} exceeds intent budget {cap:.2f}")
    if parsed_ccy and parsed_ccy != total.currency:
        r.append(f"cart currency {total.currency} does not match intent currency {parsed_ccy}")
    if any(i.amount.currency != total.currency for i in details.display_items):
        r.append("display_items mix currencies with the total")
    line_sum = sum(i.amount.value for i in details.display_items)
    if abs(line_sum - total.value) > CENT:
        r.append(f"display_items sum {line_sum:.2f} != total {total.value:.2f}")
    if total.value <= 0:
        r.append("total amount is not positive")

    # -- scope ------------------------------------------------------------- #
    if intent.merchants and contents.merchant_name not in intent.merchants:
        r.append(f"payee {contents.merchant_name!r} is outside intent.merchants")
    if intent.skus:
        sku = contents.id.split("#", 1)[0]
        if sku not in intent.skus:
            r.append(f"purchased sku {sku!r} is outside intent.skus")
    authorised = {md.supported_methods for md in contents.payment_request.method_data}
    if pmc.payment_response.method_name not in authorised:
        r.append(
            f"payment method {pmc.payment_response.method_name!r} is not in the cart's method_data"
        )

    # -- refundability ----------------------------------------------------- #
    if intent.requires_refundability:
        bad = [i.label for i in details.display_items if i.refund_period <= 0]
        if bad:
            r.append(f"intent requires refundability but non-refundable items present: {bad}")

    # -- cross-mandate binding --------------------------------------------- #
    if pmc.payment_details_id != details.id:
        r.append("payment.payment_details_id does not reference the cart's payment request")
    if pmc.payment_response.request_id != details.id:
        r.append("payment_response.request_id does not reference the cart's payment request")
    if (
        pmc.payment_details_total.amount.value != total.value
        or pmc.payment_details_total.amount.currency != total.currency
    ):
        r.append("payment total does not match the cart total")
    if pmc.merchant_agent != contents.merchant_name:
        r.append("payment merchant_agent does not match the cart merchant")

    # -- hash chain -------------------------------------------------------- #
    if cart.merchant_authorization:
        ok, claims = crypto.verify(cart.merchant_authorization)
        if not ok:
            r.append("merchant_authorization signature does not verify")
        elif claims.get("cart_hash") != crypto.sha256_hex(contents):
            r.append("merchant_authorization cart_hash does not match the cart contents")
    if payment.user_authorization:
        ok, claims = crypto.verify(payment.user_authorization)
        if not ok:
            r.append("user_authorization signature does not verify")
        else:
            chain = claims.get("transaction_data")
            expected = [crypto.sha256_hex(cart), crypto.sha256_hex(pmc)]
            if not isinstance(chain, list) or len(chain) != 2:
                r.append("user_authorization transaction_data is not a 2-element hash chain")
            elif list(chain) != expected:
                which = "cart" if chain[0] != expected[0] else "payment contents"
                r.append(f"user_authorization hash chain is broken over the {which}")

    # -- TTL / state ------------------------------------------------------- #
    intent_expiry = _iso(intent.intent_expiry)
    cart_expiry = _iso(contents.cart_expiry)
    if intent_expiry is None:
        r.append("intent.intent_expiry is not a valid ISO 8601 timestamp")
    elif now >= intent_expiry:
        r.append("intent mandate has expired")
    if cart_expiry is None:
        r.append("cart.cart_expiry is not a valid ISO 8601 timestamp")
    elif now >= cart_expiry:
        r.append("cart mandate has expired")
    if intent.user_cart_confirmation_required and not contents.user_cart_confirmation_required:
        r.append("intent requires user cart confirmation but the cart does not")

    return ValidationResult(valid=not r, reasons=r)
