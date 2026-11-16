"""The agent's information environment, and the four runtime injection points.

Mapped 1:1 onto the real AP2 python sample shopping agent
(`code/samples/python/src/roles/shopping_agent/subagents/shopper/`):

    METADATA  -> the fields of each `Cart` returned by `find_products`
                 (item_label, amount, merchant, attributes)
    RANKING   -> the ORDER of that `list[Cart]` as presented to the LLM
    TOOL      -> the raw tool result text/notes accompanying `find_products`
    MEMORY    -> `tool_context.state` (planted preferences, prior "choices")
    CONSENT   -> the cart-confirmation prompt shown before signing
                 (5th surface; the 4 above are the Day-1 runtime injection points)

An injection is a pure `AgentView -> AgentView` function. Surfaces are applied in
a fixed order so a multi-surface intervention is deterministic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

SurfaceName = Literal["metadata", "ranking", "tool", "memory", "consent"]
SURFACE_ORDER: tuple[SurfaceName, ...] = ("metadata", "tool", "ranking", "memory", "consent")
INJECTABLE_SURFACES: tuple[SurfaceName, ...] = ("metadata", "ranking", "tool", "memory")


class OptionView(BaseModel):
    """One purchasable option as the agent sees it.

    `cart_id` carries the merchant's SKU binding and is NOT attacker-writable --
    everything else on this object is.
    """

    cart_id: str
    label: str
    price: float
    currency: str
    merchant: str
    refund_period: int
    attributes: dict[str, str | int | float] = Field(default_factory=dict)
    annotation: str | None = None


class AgentView(BaseModel):
    """Everything the shopping agent knows when it chooses a cart."""

    intent_id: str
    intent_text: str
    budget_cap: float
    currency: str
    requires_refundability: bool
    allowed_merchants: list[str] | None = None
    options: list[OptionView] = Field(default_factory=list)
    tool_notes: list[str] = Field(default_factory=list)
    memory: list[str] = Field(default_factory=list)
    confirmation_prompt: str | None = None
    """The agent-authored framing of the confirmation step.

    `None` means the default AP2 behaviour: disclose the merchant-signed cart to the
    user faithfully. The CONSENT surface exists to replace that framing -- which is
    the only way to stop the user reading what they are actually buying, because the
    disclosure is rendered from the SIGNED cart and no other surface can touch it.
    """

    def option(self, cart_id: str) -> OptionView | None:
        return next((o for o in self.options if o.cart_id == cart_id), None)


Injection = Callable[[AgentView], AgentView]


@dataclass
class SurfaceStack:
    """Composed injections, grouped by surface. Empty stack == clean run."""

    injections: dict[str, list[Injection]] = field(default_factory=dict)

    def add(self, surface: SurfaceName, fn: Injection) -> SurfaceStack:
        self.injections.setdefault(surface, []).append(fn)
        return self

    @property
    def touched_surfaces(self) -> tuple[str, ...]:
        return tuple(s for s in SURFACE_ORDER if self.injections.get(s))

    def apply(self, view: AgentView) -> AgentView:
        """Apply every injection in fixed surface order. Never mutates `view`."""
        out = view.model_copy(deep=True)
        for surface in SURFACE_ORDER:
            for fn in self.injections.get(surface, []):
                out = fn(out)
        return out


CLEAN = SurfaceStack()


# --------------------------------------------------------------------------- #
# Trivial injections -- used by the Day-1 gate and as building blocks.
# --------------------------------------------------------------------------- #
def noop(view: AgentView) -> AgentView:
    """The identity injection. Proves a surface is wired without changing behaviour."""
    return view


def reorder(order: list[str]) -> Injection:
    """RANKING: promote `order` to the front, leaving everything else untouched.

    Unlisted options keep their RELATIVE order. A stable sort matters: sorting the
    tail by cart id would silently demote the incumbent top result as a side effect,
    so a one-field promotion would be measured as if it were a promotion plus a
    demotion, and |delta| would understate the real intervention.
    """

    def fn(view: AgentView) -> AgentView:
        index = {cid: i for i, cid in enumerate(order)}
        ranked = sorted(
            enumerate(view.options), key=lambda t: (index.get(t[1].cart_id, len(index)), t[0])
        )
        view.options = [o for _, o in ranked]
        return view

    return fn


def relabel(cart_id: str, label: str) -> Injection:
    """METADATA: rewrite one option's displayed label."""

    def fn(view: AgentView) -> AgentView:
        for o in view.options:
            if o.cart_id == cart_id:
                o.label = label
        return view

    return fn


def set_attribute(cart_id: str, key: str, value: str | int | float) -> Injection:
    """METADATA: write one displayed attribute on one option."""

    def fn(view: AgentView) -> AgentView:
        for o in view.options:
            if o.cart_id == cart_id:
                o.attributes[key] = value
        return view

    return fn


def drop_attribute(cart_id: str | None, key: str) -> Injection:
    """METADATA/omission: delete a displayed attribute (all options if cart_id is None)."""

    def fn(view: AgentView) -> AgentView:
        for o in view.options:
            if cart_id is None or o.cart_id == cart_id:
                o.attributes.pop(key, None)
        return view

    return fn


def annotate(cart_id: str, text: str) -> Injection:
    """TOOL: attach merchant/API rationale to one option."""

    def fn(view: AgentView) -> AgentView:
        for o in view.options:
            if o.cart_id == cart_id:
                o.annotation = text
        return view

    return fn


def add_tool_note(text: str) -> Injection:
    """TOOL: append free text to the `find_products` result."""

    def fn(view: AgentView) -> AgentView:
        view.tool_notes.append(text)
        return view

    return fn


def drop_option(cart_id: str) -> Injection:
    """TOOL/omission: remove a competing option from the tool result."""

    def fn(view: AgentView) -> AgentView:
        view.options = [o for o in view.options if o.cart_id != cart_id]
        return view

    return fn


def plant_memory(text: str) -> Injection:
    """MEMORY: write a 'remembered' user preference into agent state."""

    def fn(view: AgentView) -> AgentView:
        view.memory.append(text)
        return view

    return fn


def rewrite_confirmation(text: str) -> Injection:
    """CONSENT: replace the faithful disclosure with attacker-authored framing."""

    def fn(view: AgentView) -> AgentView:
        view.confirmation_prompt = text
        return view

    return fn
