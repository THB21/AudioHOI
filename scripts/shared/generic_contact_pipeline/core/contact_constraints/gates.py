from __future__ import annotations

from dataclasses import replace

from ..measurements.types import SourceRef
from .types import ContactConstraint, ContactState


def apply_contact_state_gate(
    constraint: ContactConstraint,
    *,
    state: ContactState,
    confidence: float | None,
    evidence: SourceRef,
) -> ContactConstraint:
    """Apply a discrete gate while preserving feature and continuous coordinate."""
    return replace(
        constraint,
        state=state,
        confidence=confidence,
        gate_provenance=constraint.gate_provenance + (evidence,),
    )
