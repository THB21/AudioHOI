from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from typing import TypeAlias


BoundValue: TypeAlias = float | tuple[float | None, ...] | None


def _bound_components(value: BoundValue) -> tuple[float | None, ...]:
    if isinstance(value, tuple):
        if not value:
            raise ValueError("component bound must not be empty")
        return value
    return (value,)


class DofKind(str, Enum):
    TRANSLATION = "translation"
    ROTATION_SO3 = "rotation_so3"
    SCALAR = "scalar"
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    PERIODIC = "periodic"


class GeometryKind(str, Enum):
    SPHERE = "sphere"
    LINE_CAPSULE = "line_capsule"
    RIGID_MESH = "rigid_mesh"
    ARTICULATED_URDF = "articulated_urdf"


@dataclass(frozen=True)
class Bound:
    lower: BoundValue
    upper: BoundValue
    unit: str
    source: str
    closed: bool = True

    def __post_init__(self) -> None:
        lower = _bound_components(self.lower)
        upper = _bound_components(self.upper)
        if len(lower) != len(upper):
            raise ValueError("component lower and upper bounds must have equal width")
        if any(value is not None and not isfinite(value) for value in lower):
            raise ValueError("bound lower must be finite or absent")
        if any(value is not None and not isfinite(value) for value in upper):
            raise ValueError("bound upper must be finite or absent")
        if any(
            low is not None and high is not None and high < low
            for low, high in zip(lower, upper)
        ):
            raise ValueError("bound upper must be greater than or equal to lower")
        if not self.unit or not self.source:
            raise ValueError("bound requires unit and source")


@dataclass(frozen=True)
class DofSpec:
    dof_id: str
    kind: DofKind
    dimension: int
    unit: str
    source_fields: tuple[str, ...]
    bound: Bound | None = None
    observable: bool = True

    def __post_init__(self) -> None:
        if not self.dof_id or self.dimension < 1 or not self.unit or not self.source_fields:
            raise ValueError("DofSpec requires id, positive dimension, unit, and source fields")
        if self.kind == DofKind.ROTATION_SO3 and self.dimension != 4:
            raise ValueError("rotation_so3 is represented by a quaternion with dimension 4")
        if self.kind == DofKind.TRANSLATION and self.dimension != 3:
            raise ValueError("translation must have dimension 3")
        if self.bound is not None:
            widths = {
                len(_bound_components(self.bound.lower)),
                len(_bound_components(self.bound.upper)),
            }
            if widths not in ({1}, {self.dimension}):
                raise ValueError("dof component bound width must be one or match dimension")


@dataclass(frozen=True)
class GaugeConstraint:
    gauge_id: str
    target_dofs: tuple[str, ...]
    reason: str
    source: str

    def __post_init__(self) -> None:
        if not self.gauge_id or not self.target_dofs or not self.reason or not self.source:
            raise ValueError("GaugeConstraint requires id, target dofs, reason, and source")


@dataclass(frozen=True)
class StaticParameter:
    parameter_id: str
    value: float | str
    unit: str
    source_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.parameter_id or not self.unit or not self.source_fields:
            raise ValueError("StaticParameter requires id, unit, and source fields")
        if isinstance(self.value, float) and not isfinite(self.value):
            raise ValueError("static parameter value must be finite")


@dataclass(frozen=True)
class StateSpec:
    spec_id: str
    state_model: str
    dofs: tuple[DofSpec, ...]
    static_parameters: tuple[StaticParameter, ...] = ()
    gauge_constraints: tuple[GaugeConstraint, ...] = ()
    consumed_by_solver: bool = False

    def __post_init__(self) -> None:
        if not self.spec_id or not self.state_model or not self.dofs:
            raise ValueError("StateSpec requires id, state model, and dofs")
        ids = [item.dof_id for item in self.dofs]
        if len(ids) != len(set(ids)):
            raise ValueError("StateSpec dof ids must be unique")
        known = set(ids)
        for gauge in self.gauge_constraints:
            missing = set(gauge.target_dofs) - known
            if missing:
                raise ValueError(f"gauge references unknown dofs: {sorted(missing)}")


@dataclass(frozen=True)
class GeometryDescriptor:
    geometry_id: str
    kind: GeometryKind
    feature_ids: tuple[str, ...]
    capabilities: tuple[str, ...]
    resource_path: str | None = None
    resource_sha256: str | None = None
    parameters: tuple[StaticParameter, ...] = ()

    def __post_init__(self) -> None:
        if not self.geometry_id or not self.feature_ids or not self.capabilities:
            raise ValueError("GeometryDescriptor requires id, feature ids, and capabilities")
        if self.resource_path and not self.resource_sha256:
            raise ValueError("resource_path requires resource_sha256")


def state_spec_record(spec: StateSpec) -> dict[str, object]:
    return asdict(spec)


def geometry_record(geometry: GeometryDescriptor) -> dict[str, object]:
    return asdict(geometry)
