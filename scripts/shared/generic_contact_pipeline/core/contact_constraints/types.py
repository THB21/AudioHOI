from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from typing import TypeAlias

from ..measurements.types import FeatureRef, SourceRef


class ContactMode(str, Enum):
    UNKNOWN = "unknown"
    GRASP = "grasp"
    SUPPORT = "support"
    IMPACT = "impact"
    SLIDING = "sliding"
    RELEASE = "release"


class ContactState(str, Enum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    OCCLUDED_HOLD = "occluded_hold"
    INACTIVE = "inactive"


@dataclass(frozen=True)
class HumanSite:
    body_part: str
    side: str

    def __post_init__(self) -> None:
        if not self.body_part or self.side not in {"left", "right", "center", "none", "unknown"}:
            raise ValueError("HumanSite requires a body part and normalized side")


@dataclass(frozen=True)
class FrameInterval:
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if self.start_frame < 1 or self.end_frame < self.start_frame:
            raise ValueError("invalid contact frame interval")


@dataclass(frozen=True)
class LocalXYZ:
    x_m: float
    y_m: float
    z_m: float
    kind: str = "local_xyz"

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.x_m, self.y_m, self.z_m)):
            raise ValueError("LocalXYZ values must be finite")


@dataclass(frozen=True)
class LineS:
    s: float
    kind: str = "line_s"

    def __post_init__(self) -> None:
        if not isfinite(self.s):
            raise ValueError("LineS must be finite")


@dataclass(frozen=True)
class SurfaceUV:
    u: float
    v: float
    kind: str = "surface_uv"

    def __post_init__(self) -> None:
        if not isfinite(self.u) or not isfinite(self.v):
            raise ValueError("SurfaceUV values must be finite")


ObjectCoordinate: TypeAlias = LocalXYZ | LineS | SurfaceUV | None


@dataclass(frozen=True)
class ContactConstraint:
    constraint_id: str
    sample_id: str
    interval: FrameInterval
    time_start: float
    time_end: float
    human_site: HumanSite
    object_feature: FeatureRef
    object_coordinate: ObjectCoordinate
    mode: ContactMode
    state: ContactState
    confidence: float | None
    normal_policy: str | None
    source: SourceRef
    gate_provenance: tuple[SourceRef, ...] = ()

    def __post_init__(self) -> None:
        if not self.constraint_id or not self.sample_id or not isfinite(self.time_start) or not isfinite(self.time_end):
            raise ValueError("invalid contact identity/time")
        if self.time_end < self.time_start:
            raise ValueError("contact time interval is reversed")
        if self.confidence is not None and (not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0):
            raise ValueError("contact confidence must be absent or within [0, 1]")
        if self.state == ContactState.OCCLUDED_HOLD and self.object_coordinate is None:
            raise ValueError("occluded hold requires a persistent object coordinate")


def constraint_record(constraint: ContactConstraint) -> dict[str, object]:
    return asdict(constraint)
