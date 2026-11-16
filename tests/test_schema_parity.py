"""Our Pydantic mirror must not drift from the real AP2 schema.

Parity is checked against a committed snapshot of upstream field names
(`tests/fixtures/ap2_schema_snapshot.json`, regenerate with
`AP2_REPO=... python scripts/snapshot_ap2_schema.py`). We mirror rather than
import so the validator stays dependency-light -- this test is the price of that
choice, and it is what turns a silent schema drift into a red build.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from minimal_drift.validator import schema as mirror

SNAPSHOT = json.loads((Path(__file__).parent / "fixtures" / "ap2_schema_snapshot.json").read_text())

# Upstream classes we mirror, and fields we deliberately do not carry.
MIRRORED = (
    "IntentMandate",
    "CartContents",
    "CartMandate",
    "PaymentMandateContents",
    "PaymentMandate",
    "PaymentItem",
    "PaymentCurrencyAmount",
    "PaymentMethodData",
    "PaymentDetailsInit",
    "PaymentRequest",
    "PaymentResponse",
)
# Optional upstream fields not needed by the validator (documented, not accidental).
OMITTED = {
    "PaymentResponse": {"shipping_address", "shipping_option", "payer_phone"},
}


def test_snapshot_records_its_provenance():
    assert SNAPSHOT["commit"] and len(SNAPSHOT["commit"]) == 40
    assert SNAPSHOT["source"] == "github.com/google-agentic-commerce/AP2"


@pytest.mark.parametrize("name", MIRRORED)
def test_mirror_carries_every_upstream_field(name):
    upstream = set(SNAPSHOT["classes"][name])
    ours = set(getattr(mirror, name).model_fields)
    missing = upstream - ours - OMITTED.get(name, set())
    assert not missing, f"{name} is missing real AP2 fields: {sorted(missing)}"


@pytest.mark.parametrize("name", MIRRORED)
def test_mirror_invents_no_fields(name):
    upstream = set(SNAPSHOT["classes"][name])
    extra = set(getattr(mirror, name).model_fields) - upstream
    assert not extra, f"{name} invents fields that are not in AP2: {sorted(extra)}"


def test_plan_assumed_field_names_are_real():
    """The BUILD_PLAN guessed these names before the repo was read. They hold."""
    intent = set(SNAPSHOT["classes"]["IntentMandate"])
    assert {
        "natural_language_description",
        "merchants",
        "skus",
        "requires_refundability",
        "intent_expiry",
    } <= intent
    assert "id" in SNAPSHOT["classes"]["CartContents"]
    assert {"display_items", "total"} <= set(SNAPSHOT["classes"]["PaymentDetailsInit"])
    assert "supported_methods" in SNAPSHOT["classes"]["PaymentMethodData"]
    assert "user_authorization" in SNAPSHOT["classes"]["PaymentMandate"]


def test_intent_mandate_has_no_budget_field():
    """Why constraint_engine parses the budget out of the NL description."""
    assert not {"budget", "budget_cap", "max_amount"} & set(SNAPSHOT["classes"]["IntentMandate"])
