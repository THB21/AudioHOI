"""Profile-configured typed measurement adapters selected by schema capability."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..base.config import CaseProfile
from ..base.io import repo_relative_value
from .types import CoordinateFrame, FeatureRef, Line2DMeasurement, Measurement, MeasurementMeta, SourceRef, Unit


@dataclass(frozen=True)
class ConfiguredMeasurementResult:
    measurements: tuple[Measurement, ...]
    source_paths: tuple[str, ...]


def adapt_configured_supplemental_measurements(
    profile: CaseProfile,
    result_dir: Path,
) -> ConfiguredMeasurementResult:
    measurements: list[Measurement] = []
    sources: list[str] = []
    configured = profile.data.get("supplemental_measurements", ())
    if not isinstance(configured, (list, tuple)):
        raise ValueError("supplemental_measurements must be a sequence")
    for spec in configured:
        if not isinstance(spec, Mapping):
            raise ValueError("supplemental measurement entries must be mappings")
        adapter = str(spec.get("adapter", ""))
        if adapter != "physical_line_endpoints_v1":
            raise ValueError(f"unsupported supplemental measurement adapter: {adapter}")
        path = result_dir / str(spec["artifact"])
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"supplemental measurement artifact is empty: {path}")
        source_path = str(repo_relative_value(path))
        sources.append(source_path)
        fps = float(spec.get("fps", 24.0))
        feature_id = str(spec["feature_id"])
        semantic_role = str(spec.get("semantic_role", "physical_line"))
        fields = ("physical_x1", "physical_y1", "physical_x2", "physical_y2")
        for row in rows:
            frame = int(row["frame"])
            confidence_raw = row.get("endpoint_track_conf", "")
            confidence = float(confidence_raw) if confidence_raw not in {"", None} else None
            if confidence is not None:
                confidence = min(1.0, max(0.0, confidence))
            if row.get("line_observation_trusted", "1") != "1":
                confidence = 0.0
            meta = MeasurementMeta(
                measurement_id=f"{profile.case_name}:{frame}:line2d:{feature_id}",
                sample_id=profile.case_name,
                frame=frame,
                time=(frame - 1) / fps,
                feature=FeatureRef(semantic_role, feature_id),
                coordinate_frame=CoordinateFrame.IMAGE_PIXELS,
                unit=Unit.PIXEL,
                confidence=confidence,
                source=SourceRef(source_path, fields, adapter),
            )
            measurements.append(
                Line2DMeasurement(
                    meta,
                    (float(row["physical_x1"]), float(row["physical_y1"])),
                    (float(row["physical_x2"]), float(row["physical_y2"])),
                )
            )
    return ConfiguredMeasurementResult(tuple(measurements), tuple(sources))
