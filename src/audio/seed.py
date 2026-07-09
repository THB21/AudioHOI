"""Event seeding from BOTH modalities.

Audio only fires on loud, rigid contacts (a ball hitting the floor); soft/silent contacts
(a hand pushing the ball, a finger on a mug handle) make no onset. So an audio-only pipeline
misses half the HOI contacts. We therefore seed events from two streams and merge them:

  audio   onset detectors                          → loud contacts, precise timing
  visual  hand↔object proximity minima (+ object   → silent contacts, attribution
          velocity reversals near no limb)

Each merged event carries ``source ∈ {audio, visual, audio+visual}`` so downstream knows
whether the evidence is acoustic, visual, or both.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detect import Onset
from .visual_context import _load_object_uv, _load_vitpose


@dataclass
class EventSeed:
    time: float
    frame: int
    source: str       # audio | visual | audio+visual
    score: float
    detector: str     # audio detector name, or 'visual'
    kind: str         # onset | hand_contact
    entity_hint: str  # '' or the hand side for a visual hand-contact


def detect_visual_contacts(sample_dir, n_frames: int, fps: float = 24.0,
                           max_dist_px: float = 130.0) -> list[EventSeed]:
    """Hand↔object proximity minima = candidate (often silent) hand contacts."""
    from scipy.signal import find_peaks

    obj = _load_object_uv(sample_dir, n_frames)
    kp = _load_vitpose(sample_dir)
    if obj is None or kp is None:
        return []
    rh, lh = kp[:n_frames, 10, :2], kp[:n_frames, 9, :2]
    dr = np.hypot(*(rh - obj).T)
    dl = np.hypot(*(lh - obj).T)
    dmin = np.minimum(dr, dl)
    side = np.where(dr <= dl, "right_hand", "left_hand")
    pk, _ = find_peaks(-dmin, distance=max(1, int(0.15 * fps)), prominence=12)
    seeds = []
    for i in pk:
        if dmin[i] <= max_dist_px:
            f = int(i + 1)
            seeds.append(EventSeed(
                time=(f - 1) / fps, frame=f, source="visual",
                score=float(np.clip(1.0 - dmin[i] / 200.0, 0.0, 1.0)),
                detector="visual", kind="hand_contact", entity_hint=str(side[i]),
            ))
    return seeds


def merge_seeds(audio_onsets: list[Onset], visual_seeds: list[EventSeed],
                n_frames: int, fps: float = 24.0, tol_s: float = 0.12) -> list[EventSeed]:
    seeds = [EventSeed(o.time_s, int(np.clip(round(o.time_s * fps) + 1, 1, n_frames)),
                       "audio", o.score, o.detector, "onset", "") for o in audio_onsets]
    for vs in visual_seeds:
        near = min(seeds, key=lambda s: abs(s.time - vs.time)) if seeds else None
        if near is not None and abs(near.time - vs.time) <= tol_s:
            near.source = "audio+visual"
            if not near.entity_hint:
                near.entity_hint = vs.entity_hint
        else:
            seeds.append(vs)
    seeds.sort(key=lambda s: s.time)
    return seeds
