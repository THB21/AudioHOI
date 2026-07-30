from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite


class FactorKind(str, Enum):
    POINT_REPROJECTION = "point_reprojection"
    LINE_REPROJECTION = "line_reprojection"
    MASK_SILHOUETTE = "mask_silhouette"
    METRIC_DEPTH = "metric_depth"
    DEPTH_ORDER = "depth_order"
    CONTACT_DISTANCE = "contact_distance"
    CONTACT_RELATIVE_VELOCITY = "contact_relative_velocity"
    CONTACT_TWIST_GAUGE = "contact_twist_gauge"
    SUPPORT_AND_PENETRATION = "support_and_penetration"
    TEMPORAL_VELOCITY = "temporal_velocity"
    TEMPORAL_ACCELERATION = "temporal_acceleration"
    STATIC_FREEZE = "static_freeze"
    AUDIO_EVENT_PRIOR = "audio_event_prior"
    PERIODIC_PHASE_PRIOR = "periodic_phase_prior"
    JOINT_LIMIT = "joint_limit"
    GAUGE_CONSTRAINT = "gauge_constraint"
    POSE_PRIOR = "pose_prior"
    REGULARIZATION = "regularization"


@dataclass(frozen=True)
class FactorInputRef:
    role: str
    source_ir: str
    source_id: str

    def __post_init__(self) -> None:
        if not self.role or not self.source_ir or not self.source_id:
            raise ValueError("FactorInputRef requires role, source_ir, and source_id")


@dataclass(frozen=True)
class FactorSourceRef:
    artifact: str
    fields: tuple[str, ...]
    producer: str

    def __post_init__(self) -> None:
        if not self.artifact or not self.fields or not self.producer:
            raise ValueError("FactorSourceRef requires artifact, fields, and producer")


@dataclass(frozen=True)
class FactorSpec:
    factor_id: str
    kind: FactorKind
    frame_count: int
    input_refs: tuple[FactorInputRef, ...]
    residual_unit: str
    weight_source: str
    gate_source: str | None
    residual_source: FactorSourceRef
    consumed_by_solver: bool = False

    def __post_init__(self) -> None:
        if not self.factor_id or self.frame_count < 0 or not self.residual_unit or not self.weight_source:
            raise ValueError("FactorSpec requires id, frame_count, residual_unit, and weight_source")
        if self.consumed_by_solver:
            raise ValueError("FactorSpec shadows must not be solver-consumed in this branch")


@dataclass(frozen=True)
class FactorGap:
    gap_id: str
    status: str
    reason: str
    source: str

    def __post_init__(self) -> None:
        if self.status not in {"known_gap", "blocked", "deferred"}:
            raise ValueError("invalid factor gap status")
        if not self.gap_id or not self.reason or not self.source:
            raise ValueError("FactorGap requires id, reason, and source")


@dataclass(frozen=True)
class FactorEnergySummary:
    term: str
    kind: FactorKind
    active_frames: int
    total_energy: float

    def __post_init__(self) -> None:
        if not self.term or self.active_frames < 0 or not isfinite(self.total_energy):
            raise ValueError("invalid factor energy summary")


def factor_record(factor: FactorSpec) -> dict[str, object]:
    return asdict(factor)


def gap_record(gap: FactorGap) -> dict[str, object]:
    return asdict(gap)


def energy_record(summary: FactorEnergySummary) -> dict[str, object]:
    return asdict(summary)
