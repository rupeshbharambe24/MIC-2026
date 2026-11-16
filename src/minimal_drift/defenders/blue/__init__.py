"""BLUE TEAM: the intent-faithfulness gate.

The red team's objective is `min C(delta) s.t. ATE >= tau, V = 1, R >= rho` -- the
smallest change to the world that makes the agent sign for the wrong thing while the
payment stays fully valid. The Constraint Engine already proves V = 1. So by
construction every attack that reaches this package is a **cryptographically valid,
in-mandate, correctly-signed payment**. There is no schema violation left to find.

What is left is the gap between what the mandate *authorises* and what the user
*meant*. That is what these layers measure.

Why the existing two defenders miss it (measured, not asserted -- see
`scripts/blue_eval.py`): both read only `AgentView`, i.e. the prose the attacker
wrote. They catch mechanisms that *inject language* (memory 100%, consent 100%,
tool_output 94%) and are near-blind to mechanisms that **assert nothing false**
(metadata 15%, ranking 4%, omission 2%). `omission x retrieval` at |delta*| = 1 --
the one elite that drifted every live Gemini model zero-shot -- is caught 1.7% of
the time.

The fix is not a better keyword list. It is to stop reading the attacker's prose and
start comparing two things the attacker does not jointly control:

  the INTENT as stated        (natural_language_description + AP2 mandate fields)
  the PRODUCT actually bought (cart_id -> SKU -> the merchant's own record)

`cart_id` carries the merchant's SKU binding and, in the red team's own words, "the
attacker cannot forge this binding". Resolving it is exactly the issuer-side
cart-vs-intent consistency check Mastercard markets for Agent Pay.

**Integrity rule, enforced by `tests/test_blue_no_oracle.py`:** these layers may read
the catalog's authoritative `Product` records, and may read the AP2 mandate fields.
They may NOT read `IntentSpec.ground_truth_skus`, `IntentSpec.target_category` or
`IntentSpec.discriminating_attributes` -- that is the experiment's answer key, and a
defender that consults it is measuring the label, not detecting anything.
"""

from __future__ import annotations

from minimal_drift.defenders.blue.entailment import EntailmentVerdict, check_entailment
from minimal_drift.defenders.blue.intent_spec import IntentConstraints, parse_intent

__all__ = [
    "EntailmentVerdict",
    "IntentConstraints",
    "check_entailment",
    "parse_intent",
]
