"""The typed representation of an intervention delta, and the Mechanism interface.

An `Intervention` is a list of `Operator`s. Each operator names exactly one edit to
exactly one place in the agent's information environment, and compiles to a pure
`AgentView -> AgentView` injection from `harness/surfaces.py`. Nothing here touches
a signed mandate -- the attacker rewrites only what the agent *sees*.

Two numbers come out of this module and drive the whole search:

    C(delta)   weighted cost: fields + tokens + turns + surfaces
    |delta|    the field count -- the headline metric once minimised to |delta*|

`causal_surface_stage` is the stage of the AP2 flow an operator acts on, and is
the second axis of the MAP-Elites behavioural descriptor:

    retrieval     which options exist at all
    ranking       the order they are presented in
    presentation  what each option claims about itself
    memory        what the agent believes it already knows
    confirmation  what the user is asked to approve
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field

from minimal_drift.config import CostWeights, get_config
from minimal_drift.harness import surfaces as S
from minimal_drift.harness.surfaces import Injection, SurfaceName, SurfaceStack

Stage = Literal["retrieval", "ranking", "presentation", "memory", "confirmation"]
STAGES: tuple[Stage, ...] = ("retrieval", "ranking", "presentation", "memory", "confirmation")

OpKind = Literal[
    "reorder",
    "relabel",
    "set_attribute",
    "drop_attribute",
    "annotate",
    "add_tool_note",
    "drop_option",
    "plant_memory",
    "rewrite_confirmation",
]

# kind -> (surface it writes to, stage of the flow it acts on)
OP_SURFACE: dict[str, tuple[SurfaceName, Stage]] = {
    "reorder": ("ranking", "ranking"),
    "relabel": ("metadata", "presentation"),
    "set_attribute": ("metadata", "presentation"),
    "drop_attribute": ("metadata", "presentation"),
    "annotate": ("tool", "presentation"),
    "add_tool_note": ("tool", "presentation"),
    "drop_option": ("tool", "retrieval"),
    "plant_memory": ("memory", "memory"),
    "rewrite_confirmation": ("consent", "confirmation"),
}


# kind -> the mechanism that owns it. A bijection over kinds, which is what lets a
# minimised delta be re-labelled honestly: shrinking can remove the very operator the
# claimed mechanism was named for, and calling the survivor "consent" when no consent
# operator remains would put a false label on an archive cell.
OP_MECHANISM: dict[str, str] = {
    "reorder": "ranking",
    "drop_attribute": "omission",
    "drop_option": "omission",
    "relabel": "metadata",
    "set_attribute": "metadata",
    "plant_memory": "memory",
    "rewrite_confirmation": "consent",
    "annotate": "tool_output",
    "add_tool_note": "tool_output",
}
MECHANISM_ORDER: tuple[str, ...] = (
    "ranking",
    "omission",
    "metadata",
    "memory",
    "consent",
    "tool_output",
)


class Operator(BaseModel):
    """One typed edit to the information environment."""

    kind: OpKind
    target: str | None = None  # cart_id, or attribute key for drop_attribute
    key: str | None = None  # attribute key for set_attribute
    payload: Any = None  # new value / text / explicit order
    turns: int = 0  # extra conversational turns the operator needs

    model_config = {"frozen": True}

    @property
    def surface(self) -> SurfaceName:
        return OP_SURFACE[self.kind][0]

    @property
    def stage(self) -> Stage:
        return OP_SURFACE[self.kind][1]

    @property
    def fields(self) -> int:
        """How many distinct presented fields this operator rewrites.

        A `reorder` rewrites one field (the result order) no matter how long the
        permutation is; a `drop_attribute` with no target hits every option.
        """
        if self.kind == "drop_attribute" and self.target is None:
            return 2  # a blanket omission is a schema-level edit, not a single field
        return 1

    def text(self) -> str:
        """The attacker-authored text this operator injects, if any."""
        if self.kind in (
            "relabel",
            "annotate",
            "add_tool_note",
            "plant_memory",
            "rewrite_confirmation",
        ):
            return str(self.payload or "")
        if self.kind == "set_attribute":
            return f"{self.key}={self.payload}"
        return ""

    @property
    def tokens(self) -> int:
        """Whitespace token count of the injected text. Cheap, and monotonic."""
        return len(self.text().split())

    def to_injection(self) -> Injection:
        """Compile to a pure AgentView transform."""
        k, t, p = self.kind, self.target, self.payload
        if k == "reorder":
            return S.reorder(list(p or ([t] if t else [])))
        if k == "relabel":
            return S.relabel(str(t), str(p))
        if k == "set_attribute":
            return S.set_attribute(str(t), str(self.key), p)
        if k == "drop_attribute":
            return S.drop_attribute(t, str(self.key or p))
        if k == "annotate":
            return S.annotate(str(t), str(p))
        if k == "add_tool_note":
            return S.add_tool_note(str(p))
        if k == "drop_option":
            return S.drop_option(str(t))
        if k == "plant_memory":
            return S.plant_memory(str(p))
        if k == "rewrite_confirmation":
            return S.rewrite_confirmation(str(p))
        raise ValueError(f"unhandled operator kind {k!r}")


class Intervention(BaseModel):
    """delta: the complete adversarial intervention. Immutable by convention."""

    mechanism: str
    ops: list[Operator] = Field(default_factory=list)
    intent_id: str = ""
    note: str = ""

    # -- shape ------------------------------------------------------------- #
    @property
    def surfaces(self) -> tuple[str, ...]:
        seen = {o.surface for o in self.ops}
        return tuple(s for s in S.SURFACE_ORDER if s in seen)

    @property
    def surface(self) -> str:
        """The primary surface, for the §5 `Intervention.surface` contract."""
        return self.surfaces[0] if self.surfaces else ""

    @property
    def stages(self) -> tuple[Stage, ...]:
        seen = {o.stage for o in self.ops}
        return tuple(s for s in STAGES if s in seen)

    @property
    def stage(self) -> Stage:
        """The causal surface stage used as the archive's second descriptor axis.

        The earliest stage touched: an intervention that removes an option and then
        reorders is attributed to retrieval, because that is where the causal chain
        starts.
        """
        return self.stages[0] if self.stages else "presentation"

    @property
    def effective_mechanism(self) -> str:
        """The mechanism this delta actually is, derived from the operators present.

        Equals `mechanism` whenever at least one of that mechanism's operators survives.
        Otherwise the mechanism owning the most remaining operators, so a minimised
        delta is filed under what it does rather than under what proposed it.
        """
        owners = [OP_MECHANISM[o.kind] for o in self.ops]
        if not owners:
            return self.mechanism
        if self.mechanism in owners:
            return self.mechanism
        counts = {m: owners.count(m) for m in set(owners)}
        best = max(counts.values())
        return next(m for m in MECHANISM_ORDER if counts.get(m) == best)

    @property
    def n_fields(self) -> int:
        """|delta|: the number of changed fields. The headline metric, pre-minimisation."""
        return sum(o.fields for o in self.ops)

    @property
    def n_tokens(self) -> int:
        return sum(o.tokens for o in self.ops)

    @property
    def n_turns(self) -> int:
        return sum(o.turns for o in self.ops)

    # -- cost -------------------------------------------------------------- #
    def cost(self, weights: CostWeights | None = None) -> float:
        """C(delta): weighted fields / tokens / turns / surfaces. Monotonic in each."""
        w = weights or get_config().cost_weights
        return round(
            w.field * self.n_fields
            + w.per_100_tokens * (self.n_tokens / 100.0)
            + w.turn * self.n_turns
            + w.surface * len(self.surfaces),
            6,
        )

    # -- compilation & algebra --------------------------------------------- #
    def stack(self) -> SurfaceStack:
        """Compile delta into the SurfaceStack the environment applies."""
        stack = SurfaceStack()
        for op in self.ops:
            stack.add(op.surface, op.to_injection())
        return stack

    def without(self, index: int) -> Intervention:
        """delta minus one operator -- the unit step of leave-one-out and shrinking."""
        return self.model_copy(update={"ops": [o for i, o in enumerate(self.ops) if i != index]})

    def subset(self, indices: list[int]) -> Intervention:
        keep = set(indices)
        return self.model_copy(update={"ops": [o for i, o in enumerate(self.ops) if i in keep]})

    def only_stage(self, stage: Stage) -> Intervention:
        return self.model_copy(update={"ops": [o for o in self.ops if o.stage == stage]})

    def only_surface(self, surface: str) -> Intervention:
        return self.model_copy(update={"ops": [o for o in self.ops if o.surface == surface]})

    def is_subset_of(self, other: Intervention) -> bool:
        mine = [o.model_dump() for o in self.ops]
        theirs = [o.model_dump() for o in other.ops]
        return all(m in theirs for m in mine)

    def signature(self) -> str:
        """Stable identity of delta, for dedup across generations."""
        return f"{self.mechanism}|" + ";".join(
            f"{o.kind}:{o.target}:{o.key}:{o.payload}" for o in self.ops
        )


class Mechanism(ABC):
    """A family of interventions sharing a causal story.

    `propose` returns a valid-by-construction candidate; the validator still gates
    it, and candidates rejected there cost zero target queries.
    """

    name: str = "abstract"

    @abstractmethod
    def propose(self, parent: Intervention | None, rng: Any, ctx: Any) -> Intervention: ...

    @abstractmethod
    def operators(self) -> list[str]:
        """The operator kinds this mechanism is allowed to emit."""
