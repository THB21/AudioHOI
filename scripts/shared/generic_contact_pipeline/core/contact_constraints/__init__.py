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

__all__ = [
    "ContactAdaptationResult",
    "ContactConstraint",
    "ContactMode",
    "ContactState",
    "FrameInterval",
    "HumanSite",
    "LineS",
    "LocalXYZ",
    "SurfaceUV",
    "adapt_legacy_contact_rows",
    "apply_contact_state_gate",
    "build_contact_constraint_shadow",
    "detect_legacy_contact_schema",
]
