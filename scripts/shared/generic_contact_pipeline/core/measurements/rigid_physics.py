"""Typed, solver-independent evidence for known rigid-object reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any


_VISIBILITY_STATES = {"visible", "partially_visible", "occluded", "absent", "unknown"}


def _finite(*values: float) -> None:
    if not all(isfinite(float(value)) for value in values):
        raise ValueError("rigid physics evidence values must be finite")


def _identity(sample_id: str, frame: int) -> None:
    if not sample_id or int(frame) < 1:
        raise ValueError("rigid physics evidence requires sample_id and frame >= 1")


@dataclass(frozen=True)
class RigidSilhouetteEvidence:
    sample_id: str
    frame: int
    time: float
    visibility: str
    centroid_uv: tuple[float, float]
    body_bbox_xyxy: tuple[float, float, float, float]
    mask_area_px: float
    log_body_width_px: float
    log_body_height_px: float
    log_mask_area_px: float
    log_aspect_ratio: float
    scale_reliable: bool
    source_artifact: str

    def __post_init__(self) -> None:
        _identity(self.sample_id, self.frame)
        if self.visibility not in _VISIBILITY_STATES:
            raise ValueError(f"invalid visibility state {self.visibility!r}")
        if not self.source_artifact:
            raise ValueError("silhouette evidence requires source_artifact")
        _finite(
            self.time,
            *self.centroid_uv,
            *self.body_bbox_xyxy,
            self.mask_area_px,
            self.log_body_width_px,
            self.log_body_height_px,
            self.log_mask_area_px,
            self.log_aspect_ratio,
        )
        x1, y1, x2, y2 = self.body_bbox_xyxy
        if x2 <= x1 or y2 <= y1 or self.mask_area_px <= 0:
            raise ValueError("silhouette evidence requires positive bbox dimensions and mask area")


@dataclass(frozen=True)
class RelativeDepthEvidence:
    sample_id: str
    frame: int
    time: float
    depth_m: float
    confidence: float
    log_depth: float
    source_artifact: str

    def __post_init__(self) -> None:
        _identity(self.sample_id, self.frame)
        _finite(self.time, self.depth_m, self.confidence, self.log_depth)
        if self.depth_m <= 0 or not 0.0 < self.confidence <= 1.0:
            raise ValueError("depth must be positive and confidence within (0, 1]")
        if not self.source_artifact:
            raise ValueError("depth evidence requires source_artifact")


@dataclass(frozen=True)
class RigidFeatureTrackEvidence:
    sample_id: str
    frame: int
    query_id: str
    feature_kind: str
    candidate_feature_ids: tuple[str, ...]
    u: float
    v: float
    tracker_visibility: float
    boundary_distance_px: float
    cross_bank_error_px: float
    anchor_frame: int
    anchor_trusted: bool
    role_compatible: bool
    usable: bool
    rejection_reason: str | None

    def __post_init__(self) -> None:
        _identity(self.sample_id, self.frame)
        _finite(
            self.u,
            self.v,
            self.tracker_visibility,
            self.boundary_distance_px,
            self.cross_bank_error_px,
        )
        if not self.query_id or not self.feature_kind or not self.candidate_feature_ids:
            raise ValueError("feature evidence requires query, role, and candidate identities")
        if self.anchor_frame < 1 or not 0.0 <= self.tracker_visibility <= 1.0:
            raise ValueError("invalid feature anchor or tracker visibility")
        if self.boundary_distance_px < 0 or self.cross_bank_error_px < 0:
            raise ValueError("feature diagnostic distances must be non-negative")
        if self.usable and self.rejection_reason is not None:
            raise ValueError("usable feature evidence cannot have a rejection reason")
        if not self.usable and not self.rejection_reason:
            raise ValueError("rejected feature evidence requires a reason")


@dataclass(frozen=True)
class RigidPoseHypothesisEvidence:
    sample_id: str
    frame: int
    rank: int
    score: float
    mask_iou: float
    translation_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    selected_by_provider: bool
    provider_status: str
    source_artifact: str

    def __post_init__(self) -> None:
        _identity(self.sample_id, self.frame)
        _finite(self.score, self.mask_iou, *self.translation_m, *self.quaternion_xyzw)
        if self.rank < 0 or not 0.0 <= self.mask_iou <= 1.0:
            raise ValueError("invalid pose-hypothesis rank or mask IoU")
        norm = sum(value * value for value in self.quaternion_xyzw) ** 0.5
        if abs(norm - 1.0) > 1e-5:
            raise ValueError("pose-hypothesis quaternion must be normalized")
        if not self.provider_status or not self.source_artifact:
            raise ValueError("pose hypothesis requires provider status and source artifact")


@dataclass(frozen=True)
class RigidPhysicsEvidenceManifest:
    schema_version: int
    sample_id: str
    frame_count: int
    clear_scale_frame_count: int
    relative_depth_frame_count: int
    usable_feature_frame_count: int
    trusted_rail_frame_count: int
    contaminated_track_count: int
    untrusted_anchor_track_count: int
    megapose_ambiguous_frame_count: int
    source_hashes: dict[str, str]
    gates: dict[str, bool]
    ready_for_solver: bool

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not self.sample_id or self.frame_count < 1:
            raise ValueError("invalid rigid physics evidence manifest identity")
        counts = (
            self.clear_scale_frame_count,
            self.relative_depth_frame_count,
            self.usable_feature_frame_count,
            self.trusted_rail_frame_count,
            self.contaminated_track_count,
            self.untrusted_anchor_track_count,
            self.megapose_ambiguous_frame_count,
        )
        if any(value < 0 for value in counts):
            raise ValueError("manifest counts must be non-negative")
        if not self.source_hashes or not self.gates:
            raise ValueError("manifest requires source hashes and gates")
        if self.ready_for_solver != all(bool(value) for value in self.gates.values()):
            raise ValueError("ready_for_solver must equal all evidence gates")


RigidPhysicsEvidence = (
    RigidSilhouetteEvidence
    | RelativeDepthEvidence
    | RigidFeatureTrackEvidence
    | RigidPoseHypothesisEvidence
)


def rigid_physics_record(value: RigidPhysicsEvidence | RigidPhysicsEvidenceManifest) -> dict[str, Any]:
    """Return a JSON-safe record while preserving explicit absent values."""

    return asdict(value)
