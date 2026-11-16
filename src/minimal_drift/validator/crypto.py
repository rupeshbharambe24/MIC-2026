"""Deterministic hash-chain primitives for the AP2 mandate chain.

The chain we model, from the AP2 spec:

    merchant_authorization = JWS over {cart_hash: sha256(canonical CartContents)}
    user_authorization     = VP  over {transaction_data:
                                        [sha256(canonical CartMandate),
                                         sha256(canonical PaymentMandateContents)]}

Signatures are mocked (HMAC-SHA256 with a synthetic in-repo key) so the closed
environment exercises the *chain logic* deterministically. `sign_mode="es256"`
swaps in real ES256 when `cryptography` is installed -- see `crypto_es256.py`.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from typing import Any

from pydantic import BaseModel

# ponytail: synthetic, in-repo, never a real secret. Closed environment only.
_MOCK_KEY = b"minimal-drift-closed-env-mock-key"


def canonical_json(obj: Any) -> str:
    """RFC-8785-flavoured canonical JSON: sorted keys, no spaces, UTF-8."""
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json", exclude_none=True)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(obj: Any) -> str:
    """SHA-256 over the canonical JSON of `obj`."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def b64u_encode(payload: dict[str, Any]) -> str:
    raw = canonical_json(payload).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def b64u_decode(token: str) -> dict[str, Any]:
    pad = "=" * (-len(token) % 4)
    raw = base64.urlsafe_b64decode(token + pad)
    out = json.loads(raw)
    if not isinstance(out, dict):
        raise ValueError("token payload is not an object")
    return out


def _mac(payload: dict[str, Any]) -> str:
    return hmac.new(_MOCK_KEY, canonical_json(payload).encode("utf-8"), hashlib.sha256).hexdigest()


def sign(claims: dict[str, Any]) -> str:
    """Sign `claims`. Real ES256 when MD_SIGN_MODE=es256, otherwise the HMAC mock."""
    from minimal_drift.validator import crypto_es256

    if crypto_es256.enabled():
        return crypto_es256.sign(claims)
    body = dict(claims)
    body["sig"] = _mac(claims)
    return b64u_encode(body)


def verify(token: str) -> tuple[bool, dict[str, Any]]:
    """Return (signature_ok, claims). Never raises on malformed input."""
    from minimal_drift.validator import crypto_es256

    if crypto_es256.enabled():
        return crypto_es256.verify(token)
    try:
        body = b64u_decode(token)
    except Exception:
        return False, {}
    sig = body.pop("sig", None)
    if not isinstance(sig, str):
        return False, body
    return hmac.compare_digest(sig, _mac(body)), body


def claims_of(token: str) -> dict[str, Any]:
    """The claims inside `token`, whichever signing mode produced it.

    Callers must not reach for `b64u_decode` directly: the mock packs the MAC into the
    claim object while ES256 emits a three-part compact JWS, and code that assumed one
    layout broke silently the moment the mode changed.
    """
    if token.count(".") == 2:
        return b64u_decode(token.split(".")[1])
    body = b64u_decode(token)
    body.pop("sig", None)
    return body


def reseal(token: str, **updates: Any) -> str:
    """Re-sign the claims of `token` with `updates` applied. Mode-agnostic."""
    claims = claims_of(token)
    claims.update(updates)
    return sign(claims)


def sign_mode() -> str:
    """Which signing path is in force. Recorded with every run."""
    from minimal_drift.validator import crypto_es256

    return "es256" if crypto_es256.enabled() else "hmac-mock"
