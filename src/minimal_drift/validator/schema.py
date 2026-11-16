"""Pydantic mirror of the REAL AP2 mandate schema.

Source of truth (cloned & inspected 2026-08-27):
  google-agentic-commerce/AP2 @ e1ea56d
    code/sdk/python/ap2/models/mandate.py
    code/sdk/python/ap2/models/payment_request.py

Field names below are copied verbatim from that repo. Notably the plan's guessed
path `src/ap2/types/mandate.py` no longer exists (AP2 v0.2.0 moved to
`code/sdk/python/ap2/models/`), but every field name the plan assumed is real.
We mirror rather than import so the validator stays a pure, dependency-light
function and so a schema drift in AP2 shows up as a test failure, not a silent
behaviour change.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

CART_MANDATE_DATA_KEY = "ap2.mandates.CartMandate"
INTENT_MANDATE_DATA_KEY = "ap2.mandates.IntentMandate"
PAYMENT_MANDATE_DATA_KEY = "ap2.mandates.PaymentMandate"


# --------------------------------------------------------------------------- #
# W3C PaymentRequest subset used by AP2
# --------------------------------------------------------------------------- #
class PaymentCurrencyAmount(BaseModel):
    currency: str
    value: float


class PaymentItem(BaseModel):
    label: str
    amount: PaymentCurrencyAmount
    pending: bool | None = None
    refund_period: int = 30  # days; 0 == non-refundable


class PaymentMethodData(BaseModel):
    supported_methods: str
    data: dict[str, Any] = Field(default_factory=dict)


class PaymentDetailsInit(BaseModel):
    id: str
    display_items: list[PaymentItem]
    total: PaymentItem
    shipping_options: list[dict[str, Any]] | None = None
    modifiers: list[dict[str, Any]] | None = None


class PaymentRequest(BaseModel):
    method_data: list[PaymentMethodData]
    details: PaymentDetailsInit
    options: dict[str, Any] | None = None
    shipping_address: dict[str, Any] | None = None


class PaymentResponse(BaseModel):
    request_id: str
    method_name: str
    details: dict[str, Any] | None = None
    payer_name: str | None = None
    payer_email: str | None = None


# --------------------------------------------------------------------------- #
# AP2 mandates
# --------------------------------------------------------------------------- #
class IntentMandate(BaseModel):
    """The user's purchase intent (human-present flow)."""

    user_cart_confirmation_required: bool = True
    natural_language_description: str
    merchants: list[str] | None = None
    skus: list[str] | None = None
    requires_refundability: bool | None = False
    intent_expiry: str  # ISO 8601


class CartContents(BaseModel):
    """Merchant-signed cart body."""

    id: str
    user_cart_confirmation_required: bool
    payment_request: PaymentRequest
    cart_expiry: str  # ISO 8601
    merchant_name: str


class CartMandate(BaseModel):
    contents: CartContents
    merchant_authorization: str | None = None  # JWT over sha256(canonical CartContents)


class PaymentMandateContents(BaseModel):
    payment_mandate_id: str
    payment_details_id: str
    payment_details_total: PaymentItem
    payment_response: PaymentResponse
    merchant_agent: str
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class PaymentMandate(BaseModel):
    payment_mandate_contents: PaymentMandateContents
    user_authorization: str | None = None  # VP over [cart_hash, payment_contents_hash]
