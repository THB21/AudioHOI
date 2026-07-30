from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from typing import TypeAlias


class CoordinateFrame(str, Enum):
    IMAGE_PIXELS = "image_pixels"
    CAMERA_METERS = "camera_meters"
    OBJECT_LOCAL_METERS = "object_local_meters"


class Unit(str, Enum):
    PIXEL = "pixel"
    METER = "meter"
    UNITLESS = "unitless"


@dataclass(frozen=True)
class FeatureRef:
    semantic_role: str
    geometry_feature_id: str

    def __post_init__(self) -> None:
        if not self.semantic_role or not self.geometry_feature_id:
            raise ValueError("FeatureRef requires semantic_role and geometry_feature_id")


@dataclass(frozen=True)
class SourceRef:
    artifact: str
    fields: tuple[str, ...]
    producer: str = "legacy_stage1_adapter"

    def __post_init__(self) -> None:
        if not self.artifact or not self.fields:
            raise ValueError("SourceRef requires an artifact and at least one source field")


@dataclass(frozen=True)
class MeasurementMeta:
    measurement_id: str
    sample_id: str
    frame: int
    time: float
    feature: FeatureRef
    coordinate_frame: CoordinateFrame
    unit: Unit
    confidence: float | None
    source: SourceRef

    def __post_init__(self) -> None:
        if not self.measurement_id or not self.sample_id or self.frame < 1 or not isfinite(self.time):
            raise ValueError("invalid measurement identity/frame/time")
        if self.confidence is not None and (not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be absent or within [0, 1]")


def _finite(*values: float) -> None:
    if not all(isfinite(value) for value in values):
        raise ValueError("measurement values must be finite")


@dataclass(frozen=True)
class Point2DMeasurement:
    meta: MeasurementMeta
    u: float
    v: float
    covariance_px2: tuple[tuple[float, float], tuple[float, float]] | None = None
    kind: str = "point2d"

    def __post_init__(self) -> None:
        _finite(self.u, self.v)
        if self.meta.coordinate_frame != CoordinateFrame.IMAGE_PIXELS or self.meta.unit != Unit.PIXEL:
            raise ValueError("Point2D requires image_pixels/pixel metadata")
        if self.covariance_px2 is not None:
            covariance = self.covariance_px2
            if len(covariance) != 2 or any(len(row) != 2 for row in covariance):
                raise ValueError("Point2D covariance must have shape (2, 2)")
            _finite(*covariance[0], *covariance[1])
            if covariance[0][0] < 0 or covariance[1][1] < 0 or abs(covariance[0][1] - covariance[1][0]) > 1e-9:
                raise ValueError("Point2D covariance must be symmetric with non-negative diagonal")


@dataclass(frozen=True)
class Line2DMeasurement:
    meta: MeasurementMeta
    start_uv: tuple[float, float]
    end_uv: tuple[float, float]
    kind: str = "line2d"

    def __post_init__(self) -> None:
        _finite(*self.start_uv, *self.end_uv)
        if self.meta.coordinate_frame != CoordinateFrame.IMAGE_PIXELS or self.meta.unit != Unit.PIXEL:
            raise ValueError("Line2D requires image_pixels/pixel metadata")


@dataclass(frozen=True)
class Mask2DMeasurement:
    meta: MeasurementMeta
    bbox_xyxy: tuple[float, float, float, float]
    area_px: float | None = None
    mask_artifact: str | None = None
    principal_axis_uv: tuple[float, float] | None = None
    principal_variances_px2: tuple[float, float] | None = None
    kind: str = "mask2d"

    def __post_init__(self) -> None:
        _finite(*self.bbox_xyxy)
        x1, y1, x2, y2 = self.bbox_xyxy
        if x2 < x1 or y2 < y1 or (self.area_px is not None and self.area_px < 0):
            raise ValueError("invalid mask bounds/area")
        if self.meta.coordinate_frame != CoordinateFrame.IMAGE_PIXELS or self.meta.unit != Unit.PIXEL:
            raise ValueError("Mask2D requires image_pixels/pixel metadata")
        if self.principal_axis_uv is not None:
            _finite(*self.principal_axis_uv)
            if abs(sum(value * value for value in self.principal_axis_uv) - 1.0) > 1e-6:
                raise ValueError("Mask2D principal axis must be unit length")
        if self.principal_variances_px2 is not None:
            _finite(*self.principal_variances_px2)
            minor, major = self.principal_variances_px2
            if minor < 0.0 or major < minor:
                raise ValueError("Mask2D principal variances must be ordered and non-negative")


@dataclass(frozen=True)
class MetricDepthMeasurement:
    meta: MeasurementMeta
    depth_m: float
    sigma_m: float | None = None
    kind: str = "metric_depth"

    def __post_init__(self) -> None:
        _finite(self.depth_m)
        if self.depth_m <= 0 or (self.sigma_m is not None and self.sigma_m <= 0):
            raise ValueError("metric depth and optional sigma must be positive")
        if self.meta.coordinate_frame != CoordinateFrame.CAMERA_METERS or self.meta.unit != Unit.METER:
            raise ValueError("MetricDepth requires camera_meters/meter metadata")


@dataclass(frozen=True)
class TrackMeasurement:
    meta: MeasurementMeta
    track_id: str
    u: float
    v: float
    kind: str = "track2d"

    def __post_init__(self) -> None:
        _finite(self.u, self.v)
        if not self.track_id:
            raise ValueError("TrackMeasurement requires a track id")
        if self.meta.coordinate_frame != CoordinateFrame.IMAGE_PIXELS or self.meta.unit != Unit.PIXEL:
            raise ValueError("TrackMeasurement requires image_pixels/pixel metadata")


@dataclass(frozen=True)
class VisibilityMeasurement:
    meta: MeasurementMeta
    state: str
    kind: str = "visibility"

    def __post_init__(self) -> None:
        if self.state not in {"visible", "occluded", "absent", "unknown"}:
            raise ValueError(f"invalid visibility state {self.state!r}")
        if self.meta.unit != Unit.UNITLESS:
            raise ValueError("VisibilityMeasurement requires unitless metadata")


Measurement: TypeAlias = (
    Point2DMeasurement | Line2DMeasurement | Mask2DMeasurement | MetricDepthMeasurement | TrackMeasurement | VisibilityMeasurement
)


def measurement_record(measurement: Measurement) -> dict[str, object]:
    """Return a JSON-safe tagged record without dropping absent values."""
    return asdict(measurement)
