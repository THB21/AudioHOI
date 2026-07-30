from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

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
    Unit,
    VisibilityMeasurement,
)


@dataclass(frozen=True)
class AdaptationResult:
    schema: str
    measurements: tuple[Measurement, ...]
    mapped_fields: tuple[str, ...]
    unmapped_nonempty_fields: tuple[str, ...]


def detect_legacy_observation_schema(fields: set[str]) -> str:
    if {"ref_u", "ref_v", "object_ref_depth_m"} <= fields:
        return "proxy_center_depth_v1"
    if {"center_x", "center_y", "handle_visible", "body_bbox_x1"} <= fields:
        return "rigid_body_parts_v1"
    if {"top_rail_left_u", "top_rail_right_u", "front_leg_left_top_u"} <= fields:
        return "semantic_graph_v1"
    raise ValueError(f"unsupported legacy observation schema with fields: {sorted(fields)}")


def _number(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    if value in {"", None}:
        return None
    parsed = float(value)
    return parsed if isfinite(parsed) else None


def _confidence(row: dict[str, str], field: str) -> float | None:
    value = _number(row, field)
    return min(1.0, max(0.0, value)) if value is not None else None


def _meta(
    sample_id: str,
    row: dict[str, str],
    kind: str,
    role: str,
    feature_id: str,
    artifact: str,
    fields: tuple[str, ...],
    confidence: float | None,
    coordinate_frame: CoordinateFrame = CoordinateFrame.IMAGE_PIXELS,
    unit: Unit = Unit.PIXEL,
) -> MeasurementMeta:
    frame = int(row["frame"])
    return MeasurementMeta(
        measurement_id=f"{sample_id}:{frame}:{kind}:{feature_id}",
        sample_id=sample_id,
        frame=frame,
        time=float(row["time"]),
        feature=FeatureRef(role, feature_id),
        coordinate_frame=coordinate_frame,
        unit=unit,
        confidence=confidence,
        source=SourceRef(artifact, fields),
    )


def _point(
    out: list[Measurement], mapped: set[str], sample_id: str, row: dict[str, str], artifact: str,
    role: str, feature_id: str, u_field: str, v_field: str, confidence: float | None,
) -> None:
    u, v = _number(row, u_field), _number(row, v_field)
    if u is None or v is None:
        return
    mapped.update((u_field, v_field))
    out.append(Point2DMeasurement(_meta(sample_id, row, "point2d", role, feature_id, artifact, (u_field, v_field), confidence), u, v))


def _line(
    out: list[Measurement], mapped: set[str], sample_id: str, row: dict[str, str], artifact: str,
    role: str, feature_id: str, fields: tuple[str, str, str, str], confidence: float | None,
) -> None:
    values = tuple(_number(row, field) for field in fields)
    if any(value is None for value in values):
        return
    mapped.update(fields)
    a, b, c, d = (float(value) for value in values)
    out.append(Line2DMeasurement(_meta(sample_id, row, "line2d", role, feature_id, artifact, fields, confidence), (a, b), (c, d)))


def _mask(
    out: list[Measurement], mapped: set[str], sample_id: str, row: dict[str, str], artifact: str,
    role: str, feature_id: str, fields: tuple[str, str, str, str], confidence: float | None, area_field: str | None = None,
) -> None:
    values = tuple(_number(row, field) for field in fields)
    if any(value is None for value in values):
        return
    mapped.update(fields)
    area = _number(row, area_field) if area_field else None
    if area is not None and area_field:
        mapped.add(area_field)
    out.append(Mask2DMeasurement(_meta(sample_id, row, "mask2d", role, feature_id, artifact, fields, confidence), tuple(float(v) for v in values), area))


def adapt_legacy_observation_rows(sample_id: str, rows: list[dict[str, str]], artifact: str) -> AdaptationResult:
    if not rows:
        raise ValueError("cannot adapt an empty observation table")
    schema = detect_legacy_observation_schema(set(rows[0]))
    out: list[Measurement] = []
    mapped: set[str] = {"frame", "time"}
    for row in rows:
        if schema == "proxy_center_depth_v1":
            conf = _confidence(row, "observation_conf")
            _point(out, mapped, sample_id, row, artifact, "object_center", "object:center", "ref_u", "ref_v", conf)
            _point(out, mapped, sample_id, row, artifact, "object_center_smoothed", "object:center", "ref_u_smooth", "ref_v_smooth", conf)
            _point(out, mapped, sample_id, row, artifact, "support_point", "object:support", "support_u", "support_v", _confidence(row, "support_conf"))
            depth = _number(row, "object_ref_depth_m")
            if depth is not None:
                mapped.add("object_ref_depth_m")
                out.append(MetricDepthMeasurement(_meta(sample_id, row, "metric_depth", "object_center_depth", "object:center", artifact, ("object_ref_depth_m",), _confidence(row, "depth_conf"), CoordinateFrame.CAMERA_METERS, Unit.METER), depth))
        elif schema == "rigid_body_parts_v1":
            conf = _confidence(row, "observation_conf")
            _point(out, mapped, sample_id, row, artifact, "object_center", "object:center", "center_x", "center_y", conf)
            _point(out, mapped, sample_id, row, artifact, "lowest_visible_point", "object:lowest_visible", "lowest_visible_x", "lowest_visible_y", conf)
            _mask(out, mapped, sample_id, row, artifact, "object_mask", "object:body", ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"), _confidence(row, "mask_conf"), "mask_area_px")
            _mask(out, mapped, sample_id, row, artifact, "object_body_mask", "object:body", ("body_bbox_x1", "body_bbox_y1", "body_bbox_x2", "body_bbox_y2"), conf)
            semantic_visibility = row.get("vlm_visibility", "").strip().lower()
            visibility_blocks_observation = semantic_visibility in {
                "hidden",
                "occluded",
                "occluded_by_human",
                "absent",
            }
            if not visibility_blocks_observation:
                _point(out, mapped, sample_id, row, artifact, "handle_center", "object:handle", "handle_center_x", "handle_center_y", _confidence(row, "handle_conf"))
            if semantic_visibility:
                state = {
                    "visible": "visible",
                    "partially_visible": "visible",
                    "hidden": "occluded",
                    "occluded": "occluded",
                    "occluded_by_human": "occluded",
                    "absent": "absent",
                    "unknown": "unknown",
                    "unclear": "unknown",
                }.get(semantic_visibility, "unknown")
                mapped.add("vlm_visibility")
                visibility_fields = ("vlm_visibility",)
            else:
                state = {"1": "visible", "0": "occluded"}.get(row.get("handle_visible", ""), "unknown")
                visibility_fields = ("handle_visible",)
            mapped.add("handle_visible")
            out.append(VisibilityMeasurement(_meta(sample_id, row, "visibility", "handle_visibility", "object:handle", artifact, visibility_fields, _confidence(row, "handle_conf"), CoordinateFrame.IMAGE_PIXELS, Unit.UNITLESS), state))
        else:
            conf = _confidence(row, "semantic_conf")
            _mask(out, mapped, sample_id, row, artifact, "object_mask_bbox", "object:body", ("mask_bbox_x1", "mask_bbox_y1", "mask_bbox_x2", "mask_bbox_y2"), conf)
            for role, fid, fields in (
                ("backrest_top_edge", "backrest:top_edge", ("top_rail_left_u", "top_rail_left_v", "top_rail_right_u", "top_rail_right_v")),
                ("seat_front_edge", "seat:front_edge", ("seat_front_left_u", "seat_front_left_v", "seat_front_right_u", "seat_front_right_v")),
                ("front_leg_left", "leg:front_left", ("front_leg_left_top_u", "front_leg_left_top_v", "front_leg_left_bottom_u", "front_leg_left_bottom_v")),
                ("front_leg_right", "leg:front_right", ("front_leg_right_top_u", "front_leg_right_top_v", "front_leg_right_bottom_u", "front_leg_right_bottom_v")),
                ("rear_leg_left", "leg:rear_left", ("rear_leg_left_top_u", "rear_leg_left_top_v", "rear_leg_left_bottom_u", "rear_leg_left_bottom_v")),
                ("rear_leg_right", "leg:rear_right", ("rear_leg_right_top_u", "rear_leg_right_top_v", "rear_leg_right_bottom_u", "rear_leg_right_bottom_v")),
            ):
                _line(out, mapped, sample_id, row, artifact, role, fid, fields, conf)
            for fid, u_field, v_field in (
                ("backrest_top_left", "top_rail_left_u", "top_rail_left_v"),
                ("backrest_top_right", "top_rail_right_u", "top_rail_right_v"),
            ):
                _point(out, mapped, sample_id, row, artifact, "contact_anchor_keypoint", fid, u_field, v_field, conf)
            for fid, u_field, v_field in (
                ("seat_front_left", "seat_front_left_u", "seat_front_left_v"),
                ("seat_front_right", "seat_front_right_u", "seat_front_right_v"),
            ):
                _point(out, mapped, sample_id, row, artifact, "rigid_keypoint_diagnostic", fid, u_field, v_field, conf)
            for fid, u_field, v_field in (
                ("front_leg_left_bottom", "front_leg_left_bottom_u", "front_leg_left_bottom_v"),
                ("front_leg_right_bottom", "front_leg_right_bottom_u", "front_leg_right_bottom_v"),
                ("rear_leg_left_bottom", "rear_leg_left_bottom_u", "rear_leg_left_bottom_v"),
                ("rear_leg_right_bottom", "rear_leg_right_bottom_u", "rear_leg_right_bottom_v"),
            ):
                _point(out, mapped, sample_id, row, artifact, "support_foot", fid, u_field, v_field, conf)

    nonempty = {field for field in rows[0] if any(row.get(field, "") not in {"", None} for row in rows)}
    return AdaptationResult(schema, tuple(out), tuple(sorted(mapped)), tuple(sorted(nonempty - mapped)))
