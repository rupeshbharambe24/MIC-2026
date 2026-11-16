"""Parse a user's stated intent into checkable constraints -- WITHOUT the answer key.

`IntentSpec` in the catalog carries `target_category`, `ground_truth_skus` and
`discriminating_attributes`. Those exist so the divergence scorer can grade a run.
A defender that reads them is not detecting anything; it is being told the answer.
So this module takes exactly two inputs, both of which a real issuer has:

  - `natural_language_description` -- the AP2 IntentMandate field the user's words
    land in (AP2 says it is agent-generated from conversation and then confirmed)
  - the structured mandate fields: budget cap, currency, allowed merchants,
    `requires_refundability`

and produces the constraints a purchase must satisfy.

The parse is deliberately tiered, because "every word must be attested" is wrong and
firing on honest carts is the failure mode that killed the red team's first in-loop
checker:

  TIER 1  object class -- the head noun. Hard. "running shoes" bought as a gift card
          is a class violation no matter how the card is labelled.
  TIER 2  structured qualifiers -- tokens that name an attribute key or value **in
          the requested class's own schema**. Hard, and this is the layer that makes
          omission detectable: "waterproof trail running shoes" fulfilled by a shoe
          with no `waterproof` attribute is unattested, not merely undiscussed.
  TIER 3  descriptive prose -- everything else. Soft; contributes to a confidence
          score and never fires on its own.

Scoping Tier 2 to the requested class's schema is load-bearing. `use=running` is a
real attribute value in the catalog, so a global rule would make "GPS **running**
watch" demand a `running` attribute on a smartwatch, which no honest watch carries --
a guaranteed false positive on `intent_gps_watch`. Scoped to the smartwatch schema,
`gps` is required and `running` correctly degrades to prose.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

# Words that carry no product semantics. "returnable"/"refundable" are removed here
# and handled as a refundability constraint instead of an object term.
_STOP = frozenset("""
    a an the and or of for with without to from in on at by please buy order get want
    need i im i'm we my our me some any that this those these is are be
    pair pack set piece unit item product thing something anything one two
    under below beneath less than at most no more upto up cheaper within budget
    returnable refundable return refund exchangeable
    """.split())

# Clause removers. Budget and merchant phrasing is structured data elsewhere in the
# mandate; leaving it in the token stream would make "10,000" a product term.
_BUDGET_CLAUSE = re.compile(
    # Must consume a NUMERAL, not "the rest of the clause". The old `[^,.;]*` form ate
    # everything after the keyword, so "shoes for under-pronation" lost every
    # requirement that followed, and any stated constraint after the price vanished.
    # Keyword set deliberately MIRRORS constraint_engine.parse_budget. The two must
    # agree: if the validator recognises a cap phrasing and this parser does not, the
    # price leaks into the token stream and "10,000" becomes a product requirement.
    r"\b(?:under|below|less\s+than|no\s+more\s+than|not\s+more\s+than|at\s+most|max|"
    r"maximum|upto|up\s?to|within|budget\s+of)\b\s*"
    r"(?:\u20b9|rs\.?|inr|\$|usd|\u20ac)?\s*\d[\d,]*(?:\.\d+)?",
    re.IGNORECASE,
)
_MERCHANT_CLAUSE = re.compile(r"\bfrom\s+[A-Z][\w&'-]*(?:\s+[A-Z][\w&'-]*)*", re.ASCII)
_REFUND_WORD = re.compile(r"\b(returnable|refundable|with\s+returns?)\b", re.IGNORECASE)

# Constructions this lexical parser CANNOT represent. Each one silently inverts or
# truncates the constraint set, and every one of them produces a FALSE POSITIVE on an
# honest cart -- which is the failure mode that killed the red team's first in-loop
# defender at a 100% false-positive rate. Detecting them is not optional: a gate that
# does not know when it has misread the intent will confidently block a correct purchase.
#
#   NEGATION     "shoes that are NOT waterproof" -> head parses as `waterproof`, and the
#                polarity is dropped, so an honest non-waterproof shoe fails attestation.
#   DISJUNCTION  "shoes OR sneakers" -> only the last alternative survives as the head.
#   CONJUNCTION  "shoes AND socks" -> the first object is lost entirely.
_NEGATION = re.compile(r"\b(not|non|without|except|excluding|no)\b|\bnon-", re.IGNORECASE)
_DISJUNCTION = re.compile(r"\b(or|either)\b", re.IGNORECASE)
_CONJUNCTION = re.compile(r"\b(and|plus|along\s+with|as\s+well\s+as)\b", re.IGNORECASE)


def _non_latin(text: str) -> bool:
    """Any letter outside Latin-1 + Latin Extended-A/B.

    This is a rupee-denominated submission for an Indian market. A Hindi or Tamil
    intent is not an edge case, and every rule in this parser -- stopwords, plural
    stripping, head-noun position -- is English-specific. Flagging it means the gate
    routes to human review instead of confidently blocking a correct purchase.
    Deliberately does NOT fire on the currency symbol itself.
    """
    return any(ord(ch) > 0x24F for ch in text if ch.isalpha())


def _norm(text: str) -> str:
    """Lowercase, split hyphens/underscores, drop punctuation. Keeps digits+units."""
    text = text.replace("-", " ").replace("_", " ").replace("/", " ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip().lower()


def singular(token: str) -> str:
    """Crude but predictable de-pluralisation. `shoes->shoe`, `books->book`.

    Deliberately not a stemmer: stemming collapses `running`->`run` and would let a
    `run`-anything attest a running shoe. Only trailing plurals are removed.
    """
    if len(token) <= 3:
        return token
    for suffix in ("ses", "xes", "zes", "ches", "shes"):
        if token.endswith(suffix):
            return token[:-2]
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokens_of(text: str) -> list[str]:
    """Content tokens, singularised, stopwords removed, order preserved."""
    out: list[str] = []
    for raw in _norm(text).split():
        tok = singular(raw)
        if tok in _STOP or raw in _STOP or len(tok) < 2:
            continue
        out.append(tok)
    return out


class IntentConstraints(BaseModel):
    """What a purchase must satisfy for this intent, derived from stated inputs only."""

    intent_id: str
    text: str

    object_head: str
    """TIER 1. The head noun of the requested object, singularised."""

    object_terms: list[str] = Field(default_factory=list)
    """All content tokens of the object phrase, head last."""

    qualifiers: list[str] = Field(default_factory=list)
    """Object terms other than the head. Promoted to TIER 2 if the requested class's
    schema can attest them; otherwise TIER 3 prose."""

    budget_cap: float
    currency: str
    merchants: list[str] | None = None
    requires_refundability: bool = False
    refundability_stated_in_text: bool = False

    ambiguity: list[str] = Field(default_factory=list)
    """Constructions the parser cannot represent (negation / disjunction / conjunction /
    no object). Non-empty means the parse may be WRONG, so the gate must not convert an
    entailment failure into a confident block -- it routes to human review instead."""

    @property
    def confidently_parsed(self) -> bool:
        return not self.ambiguity

    @property
    def all_terms(self) -> list[str]:
        return self.object_terms


def _object_phrase(text: str) -> tuple[str, str, str]:
    """Strip structured clauses, then split into (full qualifier phrase, head segment).

    `with ...` introduces a feature clause, so the head noun lives in the part before
    it: "commuter backpack with a laptop sleeve" -> head `backpack`, not `sleeve`. The
    clause still supplies qualifiers, so it stays in the full phrase.
    `... of X` is a quantity/packaging frame, so the head is after the last `of`:
    "a pair of running shoes" -> `running shoes`; "500g of dark roast espresso whole
    beans" -> `dark roast espresso whole beans`.
    """
    body = _MERCHANT_CLAUSE.sub(" ", text)
    body = _BUDGET_CLAUSE.sub(" ", body)
    body = _REFUND_WORD.sub(" ", body)
    # NO comma truncation. "running shoes, waterproof and trail-rated, under 10000"
    # would lose `waterproof` entirely, making an omission attack on that requirement
    # undetectable. This is safe ONLY because the budget clause above now stops at the
    # numeral -- with the old regex, the comma inside "10,000" was the only thing
    # keeping the parse alive.

    chunks = re.split(r"\bwith\b", body, maxsplit=1, flags=re.IGNORECASE)
    before_with = chunks[0]
    after_with = chunks[1] if len(chunks) > 1 else ""

    parts = re.split(r"\bof\b", before_with, flags=re.IGNORECASE)
    head_segment = parts[-1] if len(parts) > 1 and tokens_of(parts[-1]) else before_with
    return f"{head_segment} {after_with}".strip(), head_segment, body


def parse_intent(
    intent_id: str,
    natural_language_description: str,
    budget_cap: float,
    currency: str,
    merchants: list[str] | None,
    requires_refundability: bool,
) -> IntentConstraints:
    """Stated intent -> checkable constraints. Reads no ground truth."""
    phrase, head_segment, body = _object_phrase(natural_language_description)
    toks = tokens_of(phrase)

    # The head is the last token of the pre-`with` segment; a trailing feature clause
    # ("with a laptop sleeve") supplies qualifiers but must not steal the head.
    head = (tokens_of(head_segment) or toks or ["item"])[-1]

    # Judged on `body` -- the sentence AFTER the budget/merchant/refundability clauses
    # are stripped, but BEFORE the `of`/`with` narrowing. Narrowing is exactly what hides
    # the problem: "shoes and a pair of socks" loses the "and" when the `of` split picks
    # the tail, so checking the narrowed phrase would miss every conjunction. Stripping
    # the structured clauses first keeps a stray "and" inside a budget phrase from
    # counting as a second object.
    ambiguity: list[str] = []
    if _NEGATION.search(body):
        ambiguity.append("negation")
    if _DISJUNCTION.search(body):
        ambiguity.append("disjunction")
    if _CONJUNCTION.search(body):
        ambiguity.append("conjunction")
    if not toks:
        ambiguity.append("no_object")
    if _non_latin(natural_language_description):
        ambiguity.append("non_latin_script")

    return IntentConstraints(
        intent_id=intent_id,
        text=natural_language_description,
        ambiguity=ambiguity,
        object_head=head,
        object_terms=toks,
        qualifiers=[t for t in toks if t != head],
        budget_cap=budget_cap,
        currency=currency,
        merchants=merchants,
        requires_refundability=(
            requires_refundability or bool(_REFUND_WORD.search(natural_language_description))
        ),
        refundability_stated_in_text=bool(_REFUND_WORD.search(natural_language_description)),
    )
