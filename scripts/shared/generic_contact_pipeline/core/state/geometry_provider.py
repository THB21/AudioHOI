from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Protocol, Sequence


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
