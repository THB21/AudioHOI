"""Stage 5 — fuse the audio axes with the visual context.

The downstream handle is derived from the two object-agnostic axes, not from any
object-specific name, so it transfers to mug / chair / hammer / drawer / broom:

  interaction_mode → temporal constraint   (impulsive/repetitive→point anchor,
                                            continuous→interval, resonant→point+free-ring)
  contact_quality  → constraint stiffness  (hard→strong, soft→medium, friction→tangential-free,
                                            air→no contact)
  visual context   → which entity          (object↔support vs object↔hand/foot)

Emitted fields: fused_event_type, interaction_mode, contact_quality, contact_constraint
(point|interval|none), allow_slide, contact_target, target_entity, manip_gamma, manip_weight,
promote_anchor, fused_conf.
"""
from __future__ import annotations

from dataclasses import dataclass

from .classify import Label, event_from_axes
from .visual_context import VisualContext

_GAMMA = {"impulsive": 1.0, "repetitive": 1.0, "resonant": 0.3, "continuous": 0.2, "none": 0.8}
_WEIGHT = {"hard": 1.0, "soft": 0.5, "friction": 0.3, "air": 0.0, "na": 0.0}


def _constraint(mode: str, quality: str) -> str:
    if mode == "none" or quality == "air":
        return "none"
    if mode == "continuous":
        return "interval"
    return "point"


@dataclass
class FusedEvent:
    frame: int
    time: float
    source: str            # audio | visual | audio+visual
    detector: str
    audio_score: float
    audio_event_type: str
    audio_conf: float
    fused_event_type: str
    interaction_mode: str
    contact_quality: str
    contact_constraint: str
    allow_slide: bool
    fused_conf: float
    contact_target: str
    target_entity: str
    manip_gamma: float
    manip_weight: float
    promote_anchor: bool


def fuse(frame: int, time: float, detector: str, audio_score: float,
         audio: Label, vc: VisualContext, *, source: str = "audio",
         kind: str = "onset", entity_hint: str = "") -> FusedEvent:
    mode, quality = audio.mode, audio.quality

    # a visual-only seed is a SILENT contact (hand push / soft touch) the mic never heard:
    # don't trust the (near-silent) audio axes — it is a soft impulsive contact by construction.
    if source == "visual":
        mode, quality = "impulsive", "soft"

    near = vc.nearest_part != "none" and (_isnan(vc.part_dist_px) or vc.part_dist_px < 120.0)
    near_hand = near and vc.nearest_part in ("left_hand", "right_hand")
    # an *active* limb (moving fast relative to the object) is the contactor — a kick/hit/dribble;
    # a slow limb that merely happens to be near a rebounding object is not (it's the surface).
    active_part = near and (vc.part_speed >= max(8.0, 0.5 * abs(vc.obj_speed)))
    corroborated = vc.visual_cue != "none"

    # --- attribution priority (object-agnostic) ---
    if source == "visual" and entity_hint:           # silent hand contact seeded from vision
        contact_target, target_entity = "part", entity_hint
    elif quality == "air":
        contact_target, target_entity = "none", "none"
    elif near_hand:                                   # hands are primary manipulators
        contact_target, target_entity = "part", vc.nearest_part
    elif vc.vel_reversal and not active_part:         # momentum reversal off a surface (bounce)
        quality = "hard"
        contact_target, target_entity = "support", "support"
    elif active_part:                                 # a fast foot/limb striking the object (kick)
        contact_target, target_entity = "part", vc.nearest_part
    elif near:                                         # near a slow part, no reversal → soft touch
        contact_target, target_entity = "part", vc.nearest_part
    elif vc.vel_reversal:
        contact_target, target_entity = "support", "support"
    else:
        # contact with an untracked surface or the object itself (drawer slide, etc.)
        contact_target = "either"
        target_entity = "object" if vc.obj_speed > 0 or mode == "continuous" else "none"

    event_type = event_from_axes(mode, quality)
    constraint = _constraint(mode, quality)
    gamma = _GAMMA.get(mode, 0.8)
    weight = _WEIGHT.get(quality, 0.5)
    allow_slide = quality == "friction"

    # promote to a hard point-anchor only for a concrete body part with a rigid hit
    promote = bool(constraint == "point" and quality == "hard"
                   and contact_target == "part"
                   and target_entity not in ("support", "none", "object"))

    fused_conf = float(min(1.0, 0.6 * audio.confidence + (0.4 if corroborated else 0.0)))

    return FusedEvent(
        frame=frame, time=time, source=source, detector=detector, audio_score=audio_score,
        audio_event_type=audio.event_type, audio_conf=audio.confidence,
        fused_event_type=event_type, interaction_mode=mode, contact_quality=quality,
        contact_constraint=constraint, allow_slide=allow_slide, fused_conf=fused_conf,
        contact_target=contact_target, target_entity=target_entity,
        manip_gamma=gamma, manip_weight=weight, promote_anchor=promote,
    )


def _isnan(x: float) -> bool:
    return x != x
