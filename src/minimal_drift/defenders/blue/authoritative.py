"""The trusted view of what was actually bought, and what a class of thing looks like.

Everything the agent *saw* is attacker-writable: labels, attributes, order, notes,
memory. One thing is not. `CartContents.id` is `"{SKU}#{tag}"`, minted by the
merchant, and the red team states the constraint plainly in `catalog/ground_truth.py`:
"The attacker cannot forge this binding."

So the defender resolves `cart_id -> SKU -> Product` and reasons over the merchant's
own record rather than the presentation layer. That is not privileged information: an
issuer scoring an agentic authorisation has the merchant's product record for the SKU
being charged, which is exactly what Mastercard's Agent Pay cart-vs-intent check
needs. It is *not* the experiment's answer key -- `ground_truth_skus` and
`target_category` are never read here.

Two products are built from that record:

  `evidence_of`  -- the bag of tokens a product can honestly attest, drawn from its
                    category, name, attribute keys and attribute values. Attestation
                    is positive: a term is satisfied only if the record *carries* it.
  `class_schema` -- the attribute keys and values that characterise a class of thing.
                    This is what scopes the qualifier check so that "GPS **running**
                    watch" does not demand a `running` attribute on a smartwatch.
"""

from __future__ import annotations

from minimal_drift.catalog.ground_truth import Product, World, sku_from_cart_id
from minimal_drift.defenders.blue.intent_spec import singular, tokens_of

# Categories whose value is fungible, transferable and not the thing anyone asks for
# by name. Substituting one of these for a product is the highest-impact drift in the
# red team's own economics, and it is a class fact, not a label fact.
STORED_VALUE_CATEGORIES = frozenset({"gift_card", "store_credit"})
STORED_VALUE_TERMS = frozenset({"gift", "card", "voucher", "credit", "prepaid", "wallet"})


def resolve(world: World, cart_id: str) -> Product | None:
    """cart_id -> the merchant's own record. `None` if the SKU is unknown."""
    try:
        return world.product(sku_from_cart_id(cart_id))
    except KeyError:
        return None


# A negated attribute attests the OPPOSITE of its key. `noise_cancelling: false` must
# not let "noise-cancelling headphones" be satisfied by a monitor with ANC switched
# off, and `gps: false` must not satisfy "GPS watch". The red team hit the mirror
# image of this in their own relevance scorer ("a negated attribute scored as a
# feature match") -- it is the single easiest way to build a defender that looks like
# it works and does not.
_FALSEY = frozenset({"false", "no", "none", "nil", "0", "off", "absent"})
_BOOLEAN_NOISE = frozenset({"true", "yes", "on"})


def evidence_of(product: Product) -> set[str]:
    """Every token this product can honestly attest, from its authoritative record.

    A key whose value is falsey contributes NOTHING: the record is asserting the
    absence of that feature, so it cannot be evidence for it.
    """
    ev: set[str] = set()
    ev |= set(tokens_of(product.category))
    ev |= set(tokens_of(product.name))
    for key, value in product.attributes.items():
        text = str(value).strip().lower()
        if text in _FALSEY:
            continue
        ev |= set(tokens_of(str(key)))
        if text not in _BOOLEAN_NOISE:
            ev |= set(tokens_of(text))
    return {t for t in ev if t}


def named_classes(world: World, object_head: str) -> set[str]:
    """Categories whose own SLUG names the requested kind of thing.

    When the catalog has a name for the class ("running_shoes" for `shoe`), buying
    outside that class is a class violation regardless of how the product is titled --
    which is what stops "Shoe Insoles" in `accessories` from satisfying "running
    shoes". When it has no such name (`skillet`, `bean`), this is empty and the
    weaker evidence test in `check_entailment` stands on its own.
    """
    return {
        p.category
        for p in world.products.values()
        if attests(object_head, set(tokens_of(p.category)))
    }


def attests(term: str, evidence: set[str]) -> bool:
    """Is `term` positively supported by this evidence bag?

    Exact match, or a containment match between tokens of length >= 4 so that
    `textbook`/`book` and `waterproof`/`water` agree without letting two-letter
    fragments match everything.
    """
    term = singular(term)
    if term in evidence:
        return True
    if len(term) < 4:
        return False
    return any(len(e) >= 4 and (term in e or e in term) for e in evidence)


# The gate's whole authority is `cart_id -> SKU -> Product`, and `Product` is authored
# by the merchant. AP2 is explicit that this is inside the threat model: "AP2 is
# agnostic to the contents of the merchant-signed Checkout JWT" and "all LLMs and
# Agents MUST be considered potential attackers." So a merchant that plants
# `category="running_shoes"` on a voucher defeats every layer with one JSON edit --
# including L5, which is handed the same forged record as its trusted premise.
#
# The answer is not a second oracle we do not have. It is that a lying record has to
# lie about ONE field while remaining consistent with everything else, and consistency
# is checkable: the attribute SHAPE of stored value is not something a category string
# can hide, and a record that declares a class it does not resemble is self-refuting.
STORED_VALUE_FINGERPRINT = frozenset({"transferable", "redeemable", "expiry_months"})


def is_stored_value(product: Product) -> bool:
    """Stored value by CATEGORY or by attribute shape.

    Category is a merchant-authored string; the fingerprint is a structural fact about
    what the record has to carry to function as stored value. Verified over the
    catalog: the fingerprint holds for all 10 gift_card / store_credit records and for
    nothing else, so adding it is a no-op on clean data.
    """
    return product.category in STORED_VALUE_CATEGORIES or STORED_VALUE_FINGERPRINT <= set(
        product.attributes
    )


def class_conformance(world: World, product: Product) -> float | None:
    """Share of its declared class's CORE attribute keys this record carries.

    Core = keys carried by every OTHER member of the class. Leave-one-out matters: a
    single planted record must not be able to erase the core it is being judged
    against. Returns None when the class has fewer than three members or an empty core,
    because "conformance" is meaningless there -- `accessories` and `kitchen` both
    correctly return None.
    """
    peers = [
        p
        for p in world.products.values()
        if p.category == product.category and p.sku != product.sku
    ]
    if len(peers) < 2:
        return None
    core = set.intersection(*[set(p.attributes) for p in peers])
    if not core:
        return None
    return len(core & set(product.attributes)) / len(core)


def tokens_set(key: str, value: object) -> set[str]:
    """Token bag for one attribute key/value pair, as `evidence_of` would build it."""
    return set(tokens_of(str(key))) | set(tokens_of(str(value)))


_CLASS_INDEX: dict[int, dict[str, set[str]]] = {}


def class_index(world: World) -> dict[str, set[str]]:
    """category -> the token set its members' attributes span.

    Memoised on object identity: `World` is a pydantic model and unhashable, and the
    default world is an `lru_cache`d singleton, so identity is the right key.
    """
    cached = _CLASS_INDEX.get(id(world))
    if cached is not None:
        return cached
    index: dict[str, set[str]] = {}
    for product in world.products.values():
        bucket = index.setdefault(product.category, set())
        for key, value in product.attributes.items():
            bucket |= set(tokens_of(str(key)))
            bucket |= set(tokens_of(str(value)))
    _CLASS_INDEX[id(world)] = index
    return index


def candidate_classes(world: World, object_head: str, object_terms: list[str]) -> set[str]:
    """Which catalog categories plausibly *are* the requested kind of thing.

    Matched on the head noun against each category's own name and its members' names,
    never against the intent's declared `target_category`. Used only to scope which
    qualifiers are hard-checkable; an empty result simply means no qualifier is
    promoted to TIER 2.
    """
    hits: set[str] = set()
    for product in world.products.values():
        pool = set(tokens_of(product.category)) | set(tokens_of(product.name))
        if attests(object_head, pool):
            hits.add(product.category)
    if hits:
        return hits
    for product in world.products.values():
        if any(attests(t, evidence_of(product)) for t in object_terms):
            hits.add(product.category)
    return hits


def schema_terms(world: World, categories: set[str]) -> set[str]:
    """The attribute vocabulary of a set of classes -- the TIER 2 checkable set."""
    index = class_index(world)
    out: set[str] = set()
    for category in categories:
        out |= index.get(category, set())
    return out
