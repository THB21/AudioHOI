"""Contact persistence — a grasp/contact state machine over discrete events.

Adopted and generalized from the teammate's mug grasp-anchor state machine
(`scripts/.../stage2_contact_candidates/build_mug_grasp_anchor_state.py`): only a *confirmed
part-contact* updates the stable anchor; same-entity follow-ups within a gap *keep* the
previous grasp (so a held object stays attached through silent/occluded frames); a support
or no-contact event *releases* it. This adds the sustained-contact dimension that an
event-only (impulsive) view lacks — exactly her object-agnostic contribution.

If her per-frame state CSV is available we consume it directly (true adoption); otherwise we
run this generic version over our own events.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Persistence:
    contact_state: str    # direct_contact | keep_grasp | transient_support | no_contact
    interval_id: int      # >=0 for a sustained grasp interval, else -1
    stable_entity: str


def assign_persistence(events: list[dict], max_gap_frames: int = 15) -> list[Persistence]:
    """events: dicts with frame, target_entity, contact_target, relevant — sorted by frame."""
    open_entity, interval, last_frame = None, -1, None
    out: list[Persistence] = []
    for e in sorted(events, key=lambda r: int(r["frame"])):
        ent, ct = e["target_entity"], e["contact_target"]
        rel = int(e.get("relevant", 1))
        fr = int(e["frame"])
        is_part = rel and ct == "part" and ent not in ("none", "support", "object")
        if is_part:
            if open_entity == ent and last_frame is not None and fr - last_frame <= max_gap_frames:
                state = "keep_grasp"                      # continuation of the same grasp
            else:
                interval += 1
                state = "direct_contact"                  # confirmed anchor (open new interval)
            open_entity, last_frame = ent, fr
            out.append(Persistence(state, interval, ent))
        elif ct == "support":
            out.append(Persistence("transient_support", -1, "support"))
            open_entity, last_frame = None, None          # a support hit does not hold a grasp
        else:
            out.append(Persistence("no_contact", -1, "none"))
            open_entity, last_frame = None, None
    return out


# her frame_mode (mug_grasp_anchor_state.csv) → our generic contact_state
_HER_MODE_MAP = {
    "direct_grasp_anchor": "direct_contact",
    "keep_previous_grasp_anchor": "keep_grasp",
    "rim_contact_keep_previous_grasp_anchor": "keep_grasp",
    "rim_contact_no_hand_anchor": "transient_support",
    "no_attachment": "no_contact",
}


def from_teammate_state(frame_mode: str) -> str:
    return _HER_MODE_MAP.get(frame_mode, "no_contact")
