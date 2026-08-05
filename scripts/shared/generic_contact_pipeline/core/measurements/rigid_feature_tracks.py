"""Bind persistent image tracks to descriptor-declared rigid feature points."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

import numpy as np
from ..state.geometry_provider import FeaturePointGeometryProvider, PinholeCamera


@dataclass(frozen=True)
class RigidTrackAssociation:
    track_id: str
    geometry_feature_id: str
    anchor_frame: int
    anchor_error_px: float
    confidence: float
    source_anchor_frames: tuple[int, ...] = ()
    local_xyz_m: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class RigidFeatureTrackBinding:
    associations: tuple[RigidTrackAssociation, ...]
    measurement_rows: tuple[dict[str, object], ...]
    reliable_anchor_frames: tuple[int, ...]
    rejected_by_reason: Mapping[str, int]


def _semantic_role(feature_id: str) -> str:
    if feature_id.startswith("track_local:"):
        return "rigid_surface_track"
    if ":body_corner_" in feature_id:
        return "rigid_body_corner"
    if ":wheel_" in feature_id:
        return "rigid_wheel_center"
    if ":rail_" in feature_id:
        return "rigid_rail_endpoint"
    if feature_id == "object:handle":
        return "handle_center"
    return "rigid_feature_point"


def _normalized_track_rows(
    track_rows: Sequence[Mapping[str, str]],
) -> tuple[dict[str, object], ...]:
    normalized: list[dict[str, object]] = []
    for row in track_rows:
        try:
            frame = int(row["frame"])
            track_id = str(row["track_id"]).strip()
            u = float(row.get("u", row.get("x", "")))
            v = float(row.get("v", row.get("y", "")))
            visible = float(row.get("visible", "1"))
            confidence = float(row.get("confidence", "1"))
            time = float(row.get("time", frame - 1))
        except (KeyError, TypeError, ValueError):
            continue
        if (
            not track_id
            or not all(isfinite(value) for value in (u, v, visible, confidence, time))
            or visible < 0.0
            or confidence < 0.0
        ):
            continue
        normalized.append(
            {
                "frame": frame,
                "time": time,
                "track_id": track_id,
                "u": u,
                "v": v,
                "visible": min(1.0, visible),
                "confidence": min(1.0, confidence),
            }
        )
    return tuple(normalized)


def _ray_box_local_surface_point(
    *,
    state: Sequence[float],
    camera: PinholeCamera,
    uv: Sequence[float],
    local_bounds: np.ndarray,
) -> np.ndarray | None:
    """Intersect one image ray with a rigid local bounding box.

    The state is translation followed by quaternion ``qw,qx,qy,qz``.  This is
    the same near-surface slab intersection used by the isolated rigid solver,
    but returns the fixed object-local point needed by production factors.
    """

    translation = np.asarray(state[:3], dtype=float)
    qw, qx, qy, qz = np.asarray(state[3:7], dtype=float)
    quaternion_norm = float(np.linalg.norm((qw, qx, qy, qz)))
    if quaternion_norm <= 1e-12:
        raise ValueError("rigid surface binding requires a nonzero quaternion")
    qw, qx, qy, qz = (value / quaternion_norm for value in (qw, qx, qy, qz))
    rotation = np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )
    ray_camera = np.asarray(
        ((float(uv[0]) - camera.cx) / camera.fx, (float(uv[1]) - camera.cy) / camera.fy, 1.0),
        dtype=float,
    )
    ray_origin_local = -rotation.T @ translation
    ray_direction_local = rotation.T @ ray_camera
    safe = np.where(
        np.abs(ray_direction_local) < 1e-10,
        np.copysign(1e-10, ray_direction_local + 1e-12),
        ray_direction_local,
    )
    intersections = (local_bounds - ray_origin_local[None, :]) / safe[None, :]
    near = float(np.max(np.min(intersections, axis=0)))
    far = float(np.min(np.max(intersections, axis=0)))
    if far < max(near, 0.0):
        return None
    return ray_origin_local + near * ray_direction_local


def _spatially_distributed_track_ids(
    rows: Sequence[Mapping[str, object]], maximum_track_count: int
) -> tuple[str, ...]:
    """Deterministically retain a spatially distributed object-only track set."""

    if maximum_track_count <= 0 or len(rows) <= maximum_track_count:
        return tuple(str(row["track_id"]) for row in rows)
    points = np.asarray([[float(row["u"]), float(row["v"])] for row in rows])
    center = np.mean(points, axis=0)
    selected = [int(np.argmin(np.linalg.norm(points - center[None, :], axis=1)))]
    minimum_distance = np.linalg.norm(points - points[selected[0]], axis=1)
    while len(selected) < maximum_track_count:
        index = int(np.argmax(minimum_distance))
        selected.append(index)
        minimum_distance = np.minimum(
            minimum_distance, np.linalg.norm(points - points[index], axis=1)
        )
    return tuple(str(rows[index]["track_id"]) for index in selected)


def bind_rigid_feature_tracks(
    *,
    track_rows: Sequence[Mapping[str, str]],
    states_by_frame: Mapping[int, Sequence[float]],
    cameras: Mapping[int, PinholeCamera],
    geometry_provider: FeaturePointGeometryProvider,
    feature_ids: Sequence[str],
    reliable_anchor_frames: Sequence[int],
    maximum_anchor_error_px: float,
    minimum_track_visibility: float,
    surface_feature_id: str = "object:body",
    maximum_track_count: int = 16,
    frame_stride: int = 3,
) -> RigidFeatureTrackBinding:
    """Bind arbitrary persistent image tracks to their own rigid local points.

    CoTracker points are surface texture observations, not semantic corners.
    A reliable external pose is therefore used once to ray-cast each selected
    point onto the rigid body.  Subsequent rows retain that fixed local point and
    never relabel it as a wheel, rail, or box corner.
    """

    if maximum_anchor_error_px <= 0.0:
        raise ValueError("maximum anchor error must be positive")
    if not 0.0 <= minimum_track_visibility <= 1.0:
        raise ValueError("minimum track visibility must be within [0, 1]")
    if frame_stride < 1:
        raise ValueError("rigid surface track frame stride must be positive")
    provider_points = getattr(geometry_provider, "feature_points_local", None)
    if not isinstance(provider_points, Mapping) or surface_feature_id not in provider_points:
        raise ValueError("rigid surface binding requires descriptor body points")
    surface_points = np.asarray(provider_points[surface_feature_id], dtype=float)
    local_bounds = np.stack((surface_points.min(axis=0), surface_points.max(axis=0)))

    normalized = _normalized_track_rows(track_rows)
    rejected: Counter[str] = Counter()
    if not normalized:
        raise ValueError("rigid point track artifact has no valid rows")
    rows_by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
    rows_by_track: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in normalized:
        rows_by_frame[int(row["frame"])].append(row)
        rows_by_track[str(row["track_id"])].append(row)

    ordered_anchors = tuple(dict.fromkeys(int(value) for value in reliable_anchor_frames))
    anchor_frame = next(
        (frame for frame in ordered_anchors if frame in states_by_frame and frame in cameras),
        None,
    )
    if anchor_frame is None:
        raise ValueError("rigid surface binding has no usable pose anchor")
    visible_rows = sorted(
        (
            row for row in rows_by_frame.get(anchor_frame, ())
            if float(row["visible"]) >= minimum_track_visibility and float(row["confidence"]) > 0.0
        ),
        key=lambda row: str(row["track_id"]),
    )
    local_by_track: dict[str, np.ndarray] = {}
    intersecting_rows: list[Mapping[str, object]] = []
    for row in visible_rows:
        local = _ray_box_local_surface_point(
            state=states_by_frame[anchor_frame],
            camera=cameras[anchor_frame],
            uv=(float(row["u"]), float(row["v"])),
            local_bounds=local_bounds,
        )
        if local is None:
            rejected["anchor_ray_misses_rigid_surface"] += 1
            continue
        local_by_track[str(row["track_id"])] = local
        intersecting_rows.append(row)
    selected_ids = set(
        _spatially_distributed_track_ids(intersecting_rows, maximum_track_count)
    )
    associations: list[RigidTrackAssociation] = []
    for row in intersecting_rows:
        track_id = str(row["track_id"])
        if track_id not in selected_ids:
            rejected["spatial_track_budget"] += 1
            continue
        local = local_by_track[track_id]
        feature_id = f"track_local:{track_id}"
        associations.append(
            RigidTrackAssociation(
                track_id=track_id,
                geometry_feature_id=feature_id,
                anchor_frame=anchor_frame,
                anchor_error_px=0.0,
                confidence=float(row["confidence"]),
                source_anchor_frames=(anchor_frame,),
                local_xyz_m=tuple(float(value) for value in local),
            )
        )

    by_track = {association.track_id: association for association in associations}
    measurement_rows: list[dict[str, object]] = []
    for track_id, association in by_track.items():
        for row in sorted(rows_by_track[track_id], key=lambda item: int(item["frame"])):
            if float(row["visible"]) < minimum_track_visibility:
                continue
            if (int(row["frame"]) - association.anchor_frame) % frame_stride != 0:
                continue
            measurement_rows.append(
                {
                    "frame": int(row["frame"]),
                    "time": float(row["time"]),
                    "u": float(row["u"]),
                    "v": float(row["v"]),
                    "geometry_feature_id": association.geometry_feature_id,
                    "semantic_role": _semantic_role(association.geometry_feature_id),
                    "track_id": track_id,
                    "confidence": float(row["confidence"]) * association.confidence,
                    "local_x": float(association.local_xyz_m[0]),
                    "local_y": float(association.local_xyz_m[1]),
                    "local_z": float(association.local_xyz_m[2]),
                    "source_anchor_frames": ";".join(
                        str(value) for value in association.source_anchor_frames
                    ),
                }
            )

    return RigidFeatureTrackBinding(
        associations=tuple(associations),
        measurement_rows=tuple(
            sorted(measurement_rows, key=lambda row: (int(row["frame"]), str(row["track_id"])))
        ),
        reliable_anchor_frames=(anchor_frame,),
        rejected_by_reason=dict(sorted(rejected.items())),
    )
