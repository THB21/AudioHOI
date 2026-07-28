from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite


class VisibilityState(str, Enum):
    VISIBLE = "visible"
    PARTIALLY_VISIBLE = "partially_visible"
    OCCLUDED = "occluded"
    ABSENT = "absent"
    UNKNOWN = "unknown"


class ContactStateAxis(str, Enum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    PERSISTENT = "persistent"
    OCCLUDED_HOLD = "occluded_hold"
    RELEASE = "release"
    INACTIVE = "inactive"


class InteractionContactMode(str, Enum):
    GRASP = "grasp"
    IMPACT = "impact"
    SUPPORT = "support"
    SLIDING = "sliding"
    ROLLING = "rolling"
    RELEASE = "release"
    UNKNOWN = "unknown"


class MotionMode(str, Enum):
    FREE = "free"
    BALLISTIC = "ballistic"
    ATTACHED = "attached"
    SUPPORTED_STATIC = "supported_static"
    SUPPORTED_MOVING = "supported_moving"
    HIGH_SPEED = "high_speed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FrameInteractionState:
    frame: int
    time: float
    target_entity_id: str
    visibility_state: VisibilityState
    contact_state: ContactStateAxis
    contact_mode: InteractionContactMode
    motion_mode: MotionMode
    active_contact_ids: tuple[str, ...]
    support_contact_ids: tuple[str, ...]
    audio_event_ids: tuple[str, ...]
    semantic_relation_ids: tuple[str, ...]
    confidence: float
    provenance: dict[str, object]

    def __post_init__(self) -> None:
        if self.frame < 1 or not isfinite(self.time):
            raise ValueError("interaction state requires positive frame and finite time")
        if not self.target_entity_id:
            raise ValueError("interaction state requires target_entity_id")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("interaction state confidence must be within [0, 1]")


@dataclass(frozen=True)
class InteractionTimeline:
    schema_version: int
    sample_id: str
    target_entity_id: str
    frames: tuple[FrameInteractionState, ...]
    metrics: dict[str, object]


def frame_record(state: FrameInteractionState) -> dict[str, object]:
    record = asdict(state)
    for key in ("visibility_state", "contact_state", "contact_mode", "motion_mode"):
        value = record[key]
        record[key] = value.value if isinstance(value, Enum) else str(value)
    return record
