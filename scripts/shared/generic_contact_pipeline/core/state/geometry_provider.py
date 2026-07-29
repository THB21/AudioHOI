from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from typing import Mapping, Protocol, Sequence

import numpy as np

from .articulated import ArticulatedKinematicProvider


class GeometryProvider(Protocol):
    """Geometry-family boundary used by generic factors."""

    def contact_point_world(
        self,
        state: Sequence[float],
        feature_id: str,
        query_world_m: Sequence[float],
    ) -> tuple[float, float, float]: ...


def _xyz(values: Sequence[float], label: str) -> tuple[float, float, float]:
    if len(values) < 3:
        raise ValueError(f"{label} requires at least three coordinates")
    xyz = (float(values[0]), float(values[1]), float(values[2]))
    if not all(isfinite(value) for value in xyz):
        raise ValueError(f"{label} coordinates must be finite")
    return xyz


def _rotation_from_state(state: Sequence[float]) -> np.ndarray:
    if len(state) < 7:
        raise ValueError("rigid state requires translation followed by quaternion qw,qx,qy,qz")
    qw, qx, qy, qz = (float(state[index]) for index in range(3, 7))
    norm = sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if not isfinite(norm) or norm <= 1e-12:
        raise ValueError("rigid-state quaternion must be finite and nonzero")
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    return np.asarray(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def _point_cloud(points: Sequence[Sequence[float]], label: str) -> np.ndarray:
    cloud = np.asarray(points, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) == 0 or not np.isfinite(cloud).all():
        raise ValueError(f"{label} must be a nonempty finite (N, 3) point cloud")
    return cloud


def _closest(points: np.ndarray, query_world_m: Sequence[float]) -> tuple[float, float, float]:
    query = np.asarray(_xyz(query_world_m, "contact query"), dtype=float)
    index = int(np.argmin(np.sum((points - query[None, :]) ** 2, axis=1)))
    return tuple(float(value) for value in points[index])


def _transform_points(points: np.ndarray, state: Sequence[float], scale_index: int | None) -> np.ndarray:
    translation = np.asarray(_xyz(state, "rigid state"), dtype=float)
    rotation = _rotation_from_state(state)
    scale = float(state[scale_index]) if scale_index is not None else 1.0
    if not isfinite(scale) or scale <= 0.0:
        raise ValueError("rigid feature scale must be finite and positive")
    return points @ (scale * rotation).T + translation


def _rotate_local_points(
    points: np.ndarray,
    axis: Sequence[float],
    origin: Sequence[float],
    angle: float,
) -> np.ndarray:
    axis_vector = np.asarray(_xyz(axis, "periodic feature axis"), dtype=float)
    axis_vector /= np.linalg.norm(axis_vector)
    x, y, z = axis_vector
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)
    rotation = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
    origin_vector = np.asarray(_xyz(origin, "periodic feature origin"), dtype=float)
    return (points - origin_vector) @ rotation.T + origin_vector


@dataclass(frozen=True)
class SphereGeometryProvider:
    radius_m: float

    def __post_init__(self) -> None:
        if not isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError("sphere radius must be finite and positive")

    def contact_point_world(
        self,
        state: Sequence[float],
        feature_id: str,
        query_world_m: Sequence[float],
    ) -> tuple[float, float, float]:
        center = _xyz(state, "sphere state")
        if feature_id == "object:center":
            return center
        if feature_id not in {"object:surface", "object:support"}:
            raise ValueError(f"unsupported sphere feature: {feature_id}")
        query = _xyz(query_world_m, "contact query")
        delta = tuple(query[index] - center[index] for index in range(3))
        distance = sqrt(sum(value * value for value in delta))
        if distance <= 1e-12:
            raise ValueError("sphere contact query cannot coincide with its center")
        return tuple(center[index] + self.radius_m * delta[index] / distance for index in range(3))


@dataclass(frozen=True)
class CapsuleGeometryProvider:
    length_m: float
    radius_m: float
    axis_local: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not isfinite(self.length_m) or self.length_m <= 0.0:
            raise ValueError("capsule length must be finite and positive")
        if not isfinite(self.radius_m) or self.radius_m < 0.0:
            raise ValueError("capsule radius must be finite and non-negative")
        axis = np.asarray(self.axis_local, dtype=float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or float(np.linalg.norm(axis)) <= 1e-12:
            raise ValueError("capsule axis must be a finite nonzero vector")

    def contact_point_world(
        self,
        state: Sequence[float],
        feature_id: str,
        query_world_m: Sequence[float],
    ) -> tuple[float, float, float]:
        center = np.asarray(_xyz(state, "capsule state"), dtype=float)
        axis = _rotation_from_state(state) @ np.asarray(self.axis_local, dtype=float)
        axis /= np.linalg.norm(axis)
        left = center - 0.5 * self.length_m * axis
        right = center + 0.5 * self.length_m * axis
        if feature_id == "object:center":
            return tuple(float(value) for value in center)
        if feature_id == "line:left_endpoint":
            return tuple(float(value) for value in left)
        if feature_id == "line:right_endpoint":
            return tuple(float(value) for value in right)
        if feature_id not in {"line:axis", "object:surface"}:
            raise ValueError(f"unsupported capsule feature: {feature_id}")
        query = np.asarray(_xyz(query_world_m, "contact query"), dtype=float)
        segment = right - left
        alpha = float(np.clip(np.dot(query - left, segment) / np.dot(segment, segment), 0.0, 1.0))
        axis_point = left + alpha * segment
        radial = query - axis_point
        distance = float(np.linalg.norm(radial))
        if self.radius_m > 0.0 and distance > 1e-12:
            axis_point = axis_point + self.radius_m * radial / distance
        return tuple(float(value) for value in axis_point)


@dataclass(frozen=True)
class PeriodicFeatureRule:
    phase_state_index: int
    axis_local: tuple[float, float, float]
    origin_local: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.phase_state_index < 0:
            raise ValueError("periodic feature state index must be non-negative")
        axis = np.asarray(self.axis_local, dtype=float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or float(np.linalg.norm(axis)) <= 1e-12:
            raise ValueError("periodic feature axis must be a finite nonzero vector")
        _xyz(self.origin_local, "periodic feature origin")


@dataclass(frozen=True)
class RigidFeatureGeometryProvider:
    feature_points_local: Mapping[str, Sequence[Sequence[float]]]
    scale_state_index: int | None = None
    periodic_feature_rules: Mapping[str, PeriodicFeatureRule] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.feature_points_local:
            raise ValueError("rigid geometry provider requires semantic feature points")
        for feature_id, points in self.feature_points_local.items():
            if not feature_id:
                raise ValueError("rigid geometry feature ids must not be empty")
            _point_cloud(points, feature_id)

    def feature_points_world(self, state: Sequence[float], feature_id: str) -> np.ndarray:
        points = self.feature_points_local.get(feature_id)
        if points is None:
            raise ValueError(f"unsupported rigid feature: {feature_id}")
        local = _point_cloud(points, feature_id)
        periodic_rule = self.periodic_feature_rules.get(feature_id)
        if periodic_rule is not None:
            if periodic_rule.phase_state_index >= len(state):
                raise ValueError("periodic feature phase index exceeds state width")
            local = _rotate_local_points(
                local,
                periodic_rule.axis_local,
                periodic_rule.origin_local,
                float(state[periodic_rule.phase_state_index]),
            )
        return _transform_points(local, state, self.scale_state_index)

    def contact_point_world(
        self,
        state: Sequence[float],
        feature_id: str,
        query_world_m: Sequence[float],
    ) -> tuple[float, float, float]:
        return _closest(self.feature_points_world(state, feature_id), query_world_m)


@dataclass(frozen=True)
class ArticulatedFeatureGeometryProvider:
    feature_points_local: Mapping[str, Sequence[Sequence[float]]]
    feature_parts: Mapping[str, str]
    kinematic_provider: ArticulatedKinematicProvider
    joint_state_indices: Mapping[str, int]
    scale_state_index: int | None = None

    def __post_init__(self) -> None:
        if set(self.feature_points_local) != set(self.feature_parts):
            raise ValueError("articulated feature points and part mappings must use identical feature ids")
        for feature_id, points in self.feature_points_local.items():
            _point_cloud(points, feature_id)
        if not self.joint_state_indices:
            raise ValueError("articulated geometry provider requires joint state indices")

    def feature_points_world(self, state: Sequence[float], feature_id: str) -> np.ndarray:
        points = self.feature_points_local.get(feature_id)
        part = self.feature_parts.get(feature_id)
        if points is None or part is None:
            raise ValueError(f"unsupported articulated feature: {feature_id}")
        joint_values = {
            joint_id: float(state[index])
            for joint_id, index in self.joint_state_indices.items()
        }
        articulated = self.kinematic_provider.articulate_segment(
            feature_id,
            part,
            _point_cloud(points, feature_id),
            joint_values,
        )
        return _transform_points(articulated, state, self.scale_state_index)

    def contact_point_world(
        self,
        state: Sequence[float],
        feature_id: str,
        query_world_m: Sequence[float],
    ) -> tuple[float, float, float]:
        return _closest(self.feature_points_world(state, feature_id), query_world_m)
