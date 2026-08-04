"""Bind persistent image tracks to descriptor-declared rigid feature points."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import exp, isfinite
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..state.geometry_provider import FeaturePointGeometryProvider, PinholeCamera


@dataclass(frozen=True)
class RigidTrackAssociation:
    track_id: str
    geometry_feature_id: str
    anchor_frame: int
    anchor_error_px: float
    confidence: float
    source_anchor_frames: tuple[int, ...] = ()


@dataclass(frozen=True)
class RigidFeatureTrackBinding:
    associations: tuple[RigidTrackAssociation, ...]
    measurement_rows: tuple[dict[str, object], ...]
    reliable_anchor_frames: tuple[int, ...]
    rejected_by_reason: Mapping[str, int]


def _semantic_role(feature_id: str) -> str:
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


def _project_one_point_features(
    *,
    state: Sequence[float],
    camera: PinholeCamera,
    geometry_provider: FeaturePointGeometryProvider,
    feature_ids: Sequence[str],
) -> np.ndarray:
    projected: list[np.ndarray] = []
    for feature_id in feature_ids:
        points = geometry_provider.feature_points_world(state, feature_id)
        if points.shape != (1, 3):
            raise ValueError(
                f"rigid tracked feature must resolve to one local point: {feature_id}"
            )
        projected.append(camera.project(points)[0])
    return np.asarray(projected, dtype=float)


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
) -> RigidFeatureTrackBinding:
    """Associate persistent tracks with distinct rigid feature identities.

    Associations are proposed independently at reliable anchors, reduced to the
    best trustworthy anchor for each track/feature pair, and then made one-to-one
    across the sequence with a global assignment.
    The initializer is used only to establish feature identity; every emitted
    measurement keeps the original tracked image coordinate.
    """

    if maximum_anchor_error_px <= 0.0:
        raise ValueError("maximum anchor error must be positive")
    if not 0.0 <= minimum_track_visibility <= 1.0:
        raise ValueError("minimum track visibility must be within [0, 1]")
    ordered_features = tuple(dict.fromkeys(str(value) for value in feature_ids))
    if not ordered_features or any(not value for value in ordered_features):
        raise ValueError("rigid tracked feature IDs must be nonempty")

    normalized = _normalized_track_rows(track_rows)
    rejected: Counter[str] = Counter()
    if not normalized:
        raise ValueError("rigid point track artifact has no valid rows")
    rows_by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
    rows_by_track: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in normalized:
        rows_by_frame[int(row["frame"])].append(row)
        rows_by_track[str(row["track_id"])].append(row)

    best_by_pair: dict[tuple[str, str], tuple[int, float, float]] = {}
    used_anchors: list[int] = []
    for frame in tuple(dict.fromkeys(int(value) for value in reliable_anchor_frames)):
        if frame not in states_by_frame or frame not in cameras:
            rejected["anchor_missing_state_or_camera"] += 1
            continue
        visible_rows = [
            row
            for row in rows_by_frame.get(frame, ())
            if float(row["visible"]) >= minimum_track_visibility
            and float(row["confidence"]) > 0.0
        ]
        if not visible_rows:
            rejected["anchor_has_no_visible_tracks"] += 1
            continue
        feature_uv = _project_one_point_features(
            state=states_by_frame[frame],
            camera=cameras[frame],
            geometry_provider=geometry_provider,
            feature_ids=ordered_features,
        )
        track_uv = np.asarray(
            [[float(row["u"]), float(row["v"])] for row in visible_rows], dtype=float
        )
        distances = np.linalg.norm(track_uv[:, None, :] - feature_uv[None, :, :], axis=2)
        used_anchors.append(frame)
        for track_index, row in enumerate(visible_rows):
            track_id = str(row["track_id"])
            for feature_index, feature_id in enumerate(ordered_features):
                error = float(distances[track_index, feature_index])
                if error > maximum_anchor_error_px:
                    continue
                confidence = float(row["confidence"]) * exp(
                    -0.5 * (error / maximum_anchor_error_px) ** 2
                )
                key = (track_id, feature_id)
                previous = best_by_pair.get(key)
                if previous is None or error < previous[1]:
                    best_by_pair[key] = (frame, error, confidence)

    associations: list[RigidTrackAssociation] = []
    candidate_tracks = tuple(sorted({key[0] for key in best_by_pair}))
    if candidate_tracks:
        costs = np.full((len(candidate_tracks), len(ordered_features)), np.inf, dtype=float)
        for track_index, track_id in enumerate(candidate_tracks):
            for feature_index, feature_id in enumerate(ordered_features):
                evidence = best_by_pair.get((track_id, feature_id))
                if evidence is not None:
                    costs[track_index, feature_index] = evidence[1]
        finite_costs = np.where(np.isfinite(costs), costs, maximum_anchor_error_px * 1e6)
        track_indices, feature_indices = linear_sum_assignment(finite_costs)
    else:
        track_indices = feature_indices = np.asarray([], dtype=int)
    for track_index, feature_index in zip(track_indices, feature_indices):
        track_id = candidate_tracks[int(track_index)]
        feature_id = ordered_features[int(feature_index)]
        evidence = best_by_pair.get((track_id, feature_id))
        if evidence is None:
            rejected["global_assignment_has_no_in_threshold_pair"] += 1
            continue
        anchor_frame, error, confidence = evidence
        associations.append(
            RigidTrackAssociation(
                track_id=track_id,
                geometry_feature_id=feature_id,
                anchor_frame=anchor_frame,
                anchor_error_px=error,
                confidence=confidence,
                source_anchor_frames=(anchor_frame,),
            )
        )
    rejected["feature_without_unique_in_threshold_track"] += len(ordered_features) - len(associations)

    by_track = {association.track_id: association for association in associations}
    measurement_rows: list[dict[str, object]] = []
    for track_id, association in by_track.items():
        for row in sorted(rows_by_track[track_id], key=lambda item: int(item["frame"])):
            if float(row["visible"]) < minimum_track_visibility:
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
        reliable_anchor_frames=tuple(used_anchors),
        rejected_by_reason=dict(sorted(rejected.items())),
    )
