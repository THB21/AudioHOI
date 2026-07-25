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
    "adapt_legacy_observation_rows",
    "build_measurement_shadow",
    "detect_legacy_observation_schema",
]
