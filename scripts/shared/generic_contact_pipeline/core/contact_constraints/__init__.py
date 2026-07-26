"""Typed contact semantics independent of legacy case schemas."""

from .adapters import ContactAdaptationResult, adapt_legacy_contact_rows, detect_legacy_contact_schema
from .shadow import build_contact_constraint_shadow
from .gates import apply_contact_state_gate
from .types import (
    ContactConstraint,
    ContactMode,
    ContactState,
    FrameInterval,
    HumanSite,
    LineS,
    LocalXYZ,
    SurfaceUV,
)
from .timeline import (
    ContactEventConstraint,
    ContactStateSample,
    adapt_contact_event_rows,
    adapt_contact_state_rows,
)

__all__ = [
    "ContactAdaptationResult",
    "ContactConstraint",
    "ContactEventConstraint",
    "ContactMode",
    "ContactState",
    "ContactStateSample",
    "FrameInterval",
    "HumanSite",
    "LineS",
    "LocalXYZ",
    "SurfaceUV",
    "adapt_legacy_contact_rows",
    "adapt_contact_event_rows",
    "adapt_contact_state_rows",
    "apply_contact_state_gate",
    "build_contact_constraint_shadow",
    "detect_legacy_contact_schema",
]
