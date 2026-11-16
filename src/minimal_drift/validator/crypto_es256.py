"""Real ES256 (ECDSA P-256) signing for the mandate chain.

The default path in `crypto.py` uses an HMAC mock, which exercises the chain LOGIC
deterministically and needs no dependencies. This module swaps in genuine ES256 so the
closed environment performs actual asymmetric signing and verification -- the same
algorithm family AP2 specifies for `merchant_authorization` (JWS) and
`user_authorization` (SD-JWT-VC key binding).

Enable with `MD_SIGN_MODE=es256` (requires `pip install -e ".[crypto]"`). Keys are
generated per process from a fixed seed and never leave memory: this is a closed
synthetic environment, and there is deliberately no way to load a real private key.

Switching mode changes nothing about the validator's behaviour on honest carts and
nothing about which attacks succeed -- the attacker never touches a signature. It is
here so "the chain verifies" is a cryptographic statement rather than a HMAC one.
"""

from __future__ import annotations

import base64
import json
import os
from functools import lru_cache
from typing import Any

from minimal_drift.validator.crypto import b64u_decode, b64u_encode, canonical_json

# ponytail: deterministic in-repo test key material. Closed environment only; there is
# no loader for real keys, on purpose.
_SEED = b"minimal-drift-closed-env-es256-seed-do-not-reuse"


def available() -> bool:
    """True when `cryptography` is importable."""
    try:
        import cryptography  # noqa: F401
    except ImportError:
        return False
    return True


def enabled() -> bool:
    """True when ES256 signing is both requested and possible."""
    return os.getenv("MD_SIGN_MODE", "").lower() == "es256" and available()


@lru_cache(maxsize=1)
def _keypair() -> Any:
    import hashlib

    from cryptography.hazmat.primitives.asymmetric import ec

    # A fixed private scalar derived from the seed, so runs are reproducible.
    # n is the order of P-256; the scalar must land in [1, n-1].
    scalar = int.from_bytes(hashlib.sha256(_SEED).digest(), "big")
    n = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
    private = ec.derive_private_key((scalar % (n - 1)) + 1, ec.SECP256R1())
    return private, private.public_key()


def sign(claims: dict[str, Any]) -> str:
    """Produce a compact JWS (ES256) over `claims`. Returns base64url header.payload.sig."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, utils

    private, _ = _keypair()
    header = b64u_encode({"alg": "ES256", "typ": "JWT", "kid": "closed-env-1"})
    payload = b64u_encode(claims)
    signing_input = f"{header}.{payload}".encode()
    der = private.sign(signing_input, ec.ECDSA(hashes.SHA256()))
    r, s = utils.decode_dss_signature(der)
    raw = r.to_bytes(32, "big") + s.to_bytes(32, "big")  # JOSE fixed-width form
    sig = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return f"{header}.{payload}.{sig}"


def verify(token: str) -> tuple[bool, dict[str, Any]]:
    """Verify a compact ES256 JWS. Returns (ok, claims). Never raises."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, utils

    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
        claims = b64u_decode(payload_b64)
    except Exception:
        return False, {}
    try:
        pad = "=" * (-len(sig_b64) % 4)
        raw = base64.urlsafe_b64decode(sig_b64 + pad)
        if len(raw) != 64:
            return False, claims
        der = utils.encode_dss_signature(
            int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big")
        )
        _, public = _keypair()
        public.verify(der, f"{header_b64}.{payload_b64}".encode(), ec.ECDSA(hashes.SHA256()))
    except (InvalidSignature, ValueError, TypeError):
        return False, claims
    return True, claims


def public_jwk() -> dict[str, str]:
    """The verification key, so a report can state what was verified against."""
    from cryptography.hazmat.primitives.asymmetric import ec

    _, public = _keypair()
    numbers = public.public_numbers()
    assert isinstance(public.curve, ec.SECP256R1)
    return {
        "kty": "EC",
        "crv": "P-256",
        "kid": "closed-env-1",
        "x": base64.urlsafe_b64encode(numbers.x.to_bytes(32, "big")).decode().rstrip("="),
        "y": base64.urlsafe_b64encode(numbers.y.to_bytes(32, "big")).decode().rstrip("="),
    }


def fingerprint() -> str:
    """Short identifier of the signing mode actually in force, for run records."""
    return json.dumps(canonical_json(public_jwk()))[:24] if enabled() else "hmac-mock"
