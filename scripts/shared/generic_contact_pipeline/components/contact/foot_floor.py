from __future__ import annotations

from ...core.config import CaseProfile
from .ball_contact_state import build as build_ball_contact_state


def build(profile: CaseProfile) -> dict[str, object]:
    return build_ball_contact_state(profile, default_part="foot", source_name="foot_floor_from_anchor_floor_candidates")
