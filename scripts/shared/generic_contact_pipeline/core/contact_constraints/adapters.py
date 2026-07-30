from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..measurements.types import FeatureRef, SourceRef
from .types import ContactConstraint, ContactMode, ContactState, FrameInterval, HumanSite, LineS, LocalXYZ


@dataclass(frozen=True)
class ContactAdaptationResult:
    schema: str
    constraints: tuple[ContactConstraint, ...]
    mapped_fields: tuple[str, ...]
    unmapped_nonempty_fields: tuple[str, ...]


def detect_legacy_contact_schema(fields: set[str]) -> str:
    if {"object_local_s", "stable_object_local_s"} <= fields:
        return "line_s_contact_v1"
    if {"stable_local_x", "stable_local_y", "stable_local_z"} <= fields:
        return "stable_local_xyz_contact_v1"
    if {"object_local_x", "object_local_y", "object_local_z"} <= fields:
        return "local_xyz_contact_v1"
    if {"contact_active", "human_part", "object_part", "object_local_id"} <= fields:
        return "feature_contact_v1"
    raise ValueError(f"unsupported legacy contact schema with fields: {sorted(fields)}")


def _number(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    if value in {"", None}:
        return None
    parsed = float(value)
    return parsed if isfinite(parsed) else None


def _side(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"left", "left_hand"}:
        return "left"
    if normalized in {"right", "right_hand"}:
        return "right"
    if normalized in {"none", ""}:
        return "none"
    if normalized in {"center", "bilateral", "both"}:
        return "center"
    return "unknown"


def _mode(row: dict[str, str]) -> ContactMode:
    human = row.get("human_part", "").lower()
    object_part = row.get("object_part", "").lower()
    if human in {"", "none"} and object_part in {"", "none"}:
        return ContactMode.UNKNOWN
    if human == "floor" or object_part == "floor_support":
        return ContactMode.SUPPORT
    if human == "foot":
        return ContactMode.IMPACT
    return ContactMode.GRASP


def _state(row: dict[str, str], coordinate: LocalXYZ | LineS | None) -> ContactState:
    if row.get("contact_active") != "1":
        return ContactState.INACTIVE
    if row.get("visibility", "").lower() in {"hidden", "occluded"} and coordinate is not None:
        return ContactState.OCCLUDED_HOLD
    return ContactState.ACTIVE


def adapt_legacy_contact_rows(sample_id: str, rows: list[dict[str, str]], artifact: str) -> ContactAdaptationResult:
    if not rows:
        raise ValueError("cannot adapt an empty contact table")
    schema = detect_legacy_contact_schema(set(rows[0]))
    mapped = {"frame", "time", "contact_active", "human_part", "human_side", "object_part", "object_local_id", "geometry_feature_id", "source"}
    constraints: list[ContactConstraint] = []
    persistent_coordinates: dict[tuple[str, str], tuple[str, str, str, LocalXYZ | LineS]] = {}
    for ordinal, row in enumerate(rows):
        frame = int(row["frame"])
        coordinate: LocalXYZ | LineS | None = None
        coordinate_fields: tuple[str, ...] = ()
        if schema == "stable_local_xyz_contact_v1":
            xyz = tuple(_number(row, field) for field in ("stable_local_x", "stable_local_y", "stable_local_z"))
            if all(value is not None for value in xyz):
                coordinate = LocalXYZ(*(float(value) for value in xyz))
                coordinate_fields = ("stable_local_x", "stable_local_y", "stable_local_z")
        elif schema == "local_xyz_contact_v1":
            xyz = tuple(_number(row, field) for field in ("object_local_x", "object_local_y", "object_local_z"))
            if all(value is not None for value in xyz):
                coordinate = LocalXYZ(*(float(value) for value in xyz))
                coordinate_fields = ("object_local_x", "object_local_y", "object_local_z")
        elif schema == "line_s_contact_v1":
            value = _number(row, "stable_object_local_s")
            field = "stable_object_local_s"
            if value is None:
                value, field = _number(row, "object_local_s"), "object_local_s"
            if value is not None:
                coordinate, coordinate_fields = LineS(value), (field,)
        mapped.update(coordinate_fields)
        confidence_field = "contact_conf" if _number(row, "contact_conf") is not None else "anchor_score"
        confidence = _number(row, confidence_field)
        if confidence is not None:
            confidence = min(1.0, max(0.0, confidence))
            mapped.add(confidence_field)
        source_fields = tuple(field for field in ("contact_active", "human_part", "human_side", "object_part", "object_local_id", *coordinate_fields, "visibility", "anchor_update", "keep_previous", confidence_field, "source") if field in row)
        object_part = row.get("object_part", "") or "none"
        local_id = row.get("object_local_id", "") or "none"
        geometry_feature_id = row.get("geometry_feature_id", "") or f"object:{object_part}:{local_id}"
        stream_id = (row.get("human_part", "") or "none", row.get("human_side", "") or "unknown")
        keep_previous = row.get("keep_previous", "0") == "1"
        if keep_previous and stream_id in persistent_coordinates:
            object_part, local_id, geometry_feature_id, coordinate = persistent_coordinates[stream_id]
            mapped.add("keep_previous")
        elif coordinate is not None and row.get("contact_active") == "1":
            persistent_coordinates[stream_id] = (object_part, local_id, geometry_feature_id, coordinate)
        mapped.update(field for field in ("visibility", "anchor_update", "keep_previous") if field in row)
        constraints.append(
            ContactConstraint(
                constraint_id=f"{sample_id}:{frame}:contact:{ordinal}",
                sample_id=sample_id,
                interval=FrameInterval(frame, frame),
                time_start=float(row["time"]),
                time_end=float(row["time"]),
                human_site=HumanSite(row.get("human_part", "") or "none", _side(row.get("human_side", ""))),
                object_feature=FeatureRef(object_part, geometry_feature_id),
                object_coordinate=coordinate,
                mode=_mode(row),
                state=_state(row, coordinate),
                confidence=confidence,
                normal_policy=None,
                source=SourceRef(artifact, source_fields),
            )
        )
    nonempty = {field for field in rows[0] if any(row.get(field, "") not in {"", None} for row in rows)}
    return ContactAdaptationResult(schema, tuple(constraints), tuple(sorted(mapped)), tuple(sorted(nonempty - mapped)))
