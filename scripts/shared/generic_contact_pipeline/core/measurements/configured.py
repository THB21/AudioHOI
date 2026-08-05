"""Profile-configured typed measurement adapters selected by schema capability."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Mapping

from ..base.config import CaseProfile
from ..base.io import repo_relative_value
from .adapters import adapt_legacy_observation_rows
from .types import CoordinateFrame, FeatureRef, Line2DMeasurement, Measurement, MeasurementMeta, Point2DMeasurement, SourceRef, Unit


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
        path = result_dir / str(spec["artifact"])
        if adapter == "legacy_observation_v1":
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                raise ValueError(f"supplemental measurement artifact is empty: {path}")
            source_path = str(repo_relative_value(path))
            allowed_roles = {str(value) for value in spec.get("include_roles", ())}
            adapted = adapt_legacy_observation_rows(profile.case_name, rows, source_path)
            measurements.extend(
                measurement
                for measurement in adapted.measurements
                if not allowed_roles or measurement.meta.feature.semantic_role in allowed_roles
            )
            sources.append(source_path)
            continue
        if adapter == "rigid_feature_points_v1":
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                raise ValueError(f"supplemental measurement artifact is empty: {path}")
            source_path = str(repo_relative_value(path))
            sources.append(source_path)
            fps = float(spec.get("fps", 24.0))
            if not isfinite(fps) or fps <= 0.0:
                raise ValueError("rigid feature point measurements require positive finite fps")
            seen: set[tuple[int, str]] = set()
            fields = (
                "u", "v", "track_id", "geometry_feature_id",
                "local_x", "local_y", "local_z",
            )
            for row in rows:
                frame = int(row["frame"])
                track_id = str(row.get("track_id", "")).strip()
                geometry_feature_id = str(row.get("geometry_feature_id", "")).strip()
                semantic_role = str(row.get("semantic_role", "")).strip()
                if not track_id or not geometry_feature_id or not semantic_role:
                    raise ValueError("rigid feature point row is missing typed feature identity")
                identity = (frame, track_id)
                if identity in seen:
                    raise ValueError(f"duplicate rigid feature point observation: {identity}")
                seen.add(identity)
                u, v = float(row["u"]), float(row["v"])
                confidence = float(row["confidence"])
                local = tuple(float(row[f"local_{axis}"]) for axis in "xyz")
                if not all(isfinite(value) for value in (u, v, confidence, *local)):
                    raise ValueError("rigid feature point coordinates and confidence must be finite")
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError("rigid feature point confidence must be within [0, 1]")
                time_raw = row.get("time", "")
                time = float(time_raw) if time_raw not in {"", None} else (frame - 1) / fps
                if not isfinite(time):
                    raise ValueError("rigid feature point time must be finite")
                meta = MeasurementMeta(
                    measurement_id=(
                        f"{profile.case_name}:{frame}:rigid_track:{track_id}:"
                        f"{geometry_feature_id}"
                    ),
                    sample_id=profile.case_name,
                    frame=frame,
                    time=time,
                    feature=FeatureRef(semantic_role, geometry_feature_id),
                    coordinate_frame=CoordinateFrame.IMAGE_PIXELS,
                    unit=Unit.PIXEL,
                    confidence=confidence,
                    source=SourceRef(source_path, fields, adapter),
                )
                measurements.append(Point2DMeasurement(meta, u, v))
            continue
        if adapter != "physical_line_endpoints_v1":
            raise ValueError(f"unsupported supplemental measurement adapter: {adapter}")
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"supplemental measurement artifact is empty: {path}")
        source_path = str(repo_relative_value(path))
        sources.append(source_path)
        fps = float(spec.get("fps", 24.0))
        feature_id = str(spec.get("feature_id", ""))
        feature_id_field = str(spec.get("feature_id_field", ""))
        if not feature_id and not feature_id_field:
            raise ValueError("physical line measurements require feature_id or feature_id_field")
        semantic_role = str(spec.get("semantic_role", "physical_line"))
        semantic_role_field = str(spec.get("semantic_role_field", ""))
        fields = ("physical_x1", "physical_y1", "physical_x2", "physical_y2")
        for row in rows:
            frame = int(row["frame"])
            row_feature_id = str(row.get(feature_id_field, "")) if feature_id_field else feature_id
            row_semantic_role = str(row.get(semantic_role_field, "")) if semantic_role_field else semantic_role
            if not row_feature_id and str(row.get("line_observation_mode", "")) == "unassigned_axis":
                # A single visible rail is intentionally identity-ambiguous.
                # Keep it in the source artifact for uncertainty/VLM evidence,
                # but do not invent a left/right geometry feature identity for
                # a typed reprojection residual.
                if not str(row.get("candidate_feature_ids", "")).strip():
                    raise ValueError("unassigned line row requires candidate feature identities")
                continue
            if not row_feature_id or not row_semantic_role:
                raise ValueError("physical line row is missing configured feature identity")
            confidence_raw = row.get("endpoint_track_conf", "")
            confidence = float(confidence_raw) if confidence_raw not in {"", None} else None
            if confidence is not None:
                confidence = min(1.0, max(0.0, confidence))
            if row.get("line_observation_trusted", "1") != "1":
                confidence = 0.0
            meta = MeasurementMeta(
                measurement_id=f"{profile.case_name}:{frame}:line2d:{row_feature_id}",
                sample_id=profile.case_name,
                frame=frame,
                time=(frame - 1) / fps,
                feature=FeatureRef(row_semantic_role, row_feature_id),
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
