"""Typed, solver-independent observation measurements."""

from .adapters import AdaptationResult, adapt_legacy_observation_rows, detect_legacy_observation_schema
from .types import (
    CoordinateFrame,
    FeatureRef,
    Line2DMeasurement,
    Mask2DMeasurement,
    Measurement,
    MeasurementMeta,
    MetricDepthMeasurement,
    Point2DMeasurement,
    SourceRef,
    TrackMeasurement,
    Unit,
    VisibilityMeasurement,
)
from .shadow import build_measurement_shadow
from .configured import ConfiguredMeasurementResult, adapt_configured_supplemental_measurements
from .rigid_physics import (
    RelativeDepthEvidence,
    RigidFeatureTrackEvidence,
    RigidPhysicsEvidence,
    RigidPhysicsEvidenceManifest,
    RigidPoseHypothesisEvidence,
    RigidSilhouetteEvidence,
    rigid_physics_record,
)

__all__ = [
    "AdaptationResult",
    "CoordinateFrame",
    "FeatureRef",
    "Line2DMeasurement",
    "Mask2DMeasurement",
    "Measurement",
    "MeasurementMeta",
    "MetricDepthMeasurement",
    "Point2DMeasurement",
    "SourceRef",
    "TrackMeasurement",
    "Unit",
    "VisibilityMeasurement",
    "RelativeDepthEvidence",
    "RigidFeatureTrackEvidence",
    "RigidPhysicsEvidence",
    "RigidPhysicsEvidenceManifest",
    "RigidPoseHypothesisEvidence",
    "RigidSilhouetteEvidence",
    "rigid_physics_record",
    "adapt_legacy_observation_rows",
    "build_measurement_shadow",
    "ConfiguredMeasurementResult",
    "adapt_configured_supplemental_measurements",
    "detect_legacy_observation_schema",
]
