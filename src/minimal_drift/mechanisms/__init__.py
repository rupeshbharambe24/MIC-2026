"""The six attack mechanisms, and the registry the search selects from."""

from __future__ import annotations

from minimal_drift.mechanisms.base import Intervention, Mechanism, Operator, Stage
from minimal_drift.mechanisms.consent import ConsentMechanism
from minimal_drift.mechanisms.memory import MemoryMechanism
from minimal_drift.mechanisms.metadata import MetadataMechanism
from minimal_drift.mechanisms.omission import OmissionMechanism
from minimal_drift.mechanisms.ranking import RankingMechanism
from minimal_drift.mechanisms.tool_output import ToolOutputMechanism

MECHANISMS: dict[str, Mechanism] = {
    m.name: m
    for m in (
        RankingMechanism(),
        OmissionMechanism(),
        MetadataMechanism(),
        MemoryMechanism(),
        ConsentMechanism(),
        ToolOutputMechanism(),
    )
}
MECHANISM_NAMES: tuple[str, ...] = tuple(MECHANISMS)

__all__ = [
    "MECHANISMS",
    "MECHANISM_NAMES",
    "Intervention",
    "Mechanism",
    "Operator",
    "Stage",
]
