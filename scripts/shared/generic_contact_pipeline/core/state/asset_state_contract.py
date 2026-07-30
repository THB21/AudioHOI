"""Build case-agnostic object state contracts from asset descriptors."""
from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .types import Bound, DofKind, DofSpec, GeometryDescriptor, GeometryKind, StateSpec


_DOF_LAYOUTS: dict[str, tuple[DofKind, int, str]] = {
    "translation": (DofKind.TRANSLATION, 3, "meter"),
    "rotation_so3": (DofKind.ROTATION_SO3, 4, "quaternion"),
    "scalar": (DofKind.SCALAR, 1, "unitless"),
    "revolute": (DofKind.REVOLUTE, 1, "radian"),
    "prismatic": (DofKind.PRISMATIC, 1, "meter"),
    "periodic": (DofKind.PERIODIC, 1, "radian"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"asset descriptor {label} must be a mapping")
    return value


def _source_fields(raw: Mapping[str, object], dimension: int) -> tuple[str, ...]:
    values = raw.get("fields")
    if values is None and raw.get("field") is not None:
        values = (raw["field"],)
    if not isinstance(values, (list, tuple)) or len(values) != dimension:
        raise ValueError(f"asset state DOF {raw.get('dof_id')} fields must match dimension {dimension}")
    result = tuple(str(value) for value in values)
    if any(not value for value in result) or len(set(result)) != len(result):
        raise ValueError("asset state DOF fields must be nonempty and unique")
    return result


def _urdf_joint_bounds(root: ET.Element, joint_name: str) -> tuple[float, float]:
    joint = next((node for node in root.findall("joint") if node.attrib.get("name") == joint_name), None)
    if joint is None:
        raise ValueError(f"asset state contract cannot resolve URDF joint: {joint_name}")
    limit = joint.find("limit")
    if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
        raise ValueError(f"asset state contract URDF joint has no finite limits: {joint_name}")
    lower, upper = float(limit.attrib["lower"]), float(limit.attrib["upper"])
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError(f"asset state contract URDF joint limits are invalid: {joint_name}")
    return lower, upper


@dataclass(frozen=True)
class AssetStateContract:
    state_spec: StateSpec
    geometry: GeometryDescriptor
    initializer: Mapping[str, object]
    descriptor_path: str
    descriptor_sha256: str
    resource_path: str
    resource_sha256: str
    default_state_by_dof: Mapping[str, tuple[float, ...]]

    def __post_init__(self) -> None:
        if not self.initializer.get("kind"):
            raise ValueError("asset state contract requires initializer kind")
        known = {dof.dof_id for dof in self.state_spec.dofs}
        if set(self.default_state_by_dof) - known:
            raise ValueError("asset state defaults reference unknown DOFs")

    def state_index(self, dof_id: str) -> int:
        offset = 0
        for dof in self.state_spec.dofs:
            if dof.dof_id == dof_id:
                return offset
            offset += dof.dimension
        raise KeyError(dof_id)


def build_asset_state_contract(
    descriptor_path: Path,
    repository_root: Path,
) -> AssetStateContract:
    descriptor_path = descriptor_path.resolve()
    descriptor = _as_mapping(json.loads(descriptor_path.read_text()), "root")
    if int(descriptor.get("schema_version", 0)) != 1:
        raise ValueError("unsupported asset descriptor schema version")
    geometry_kind = GeometryKind(str(descriptor["geometry_kind"]))
    resource_path = (repository_root / str(descriptor["resource_path"])).resolve()
    if not resource_path.is_file():
        raise FileNotFoundError(f"asset state contract resource is missing: {resource_path}")
    state_contract = _as_mapping(descriptor.get("state_contract"), "state_contract")
    raw_dofs = state_contract.get("dofs")
    if not isinstance(raw_dofs, list) or not raw_dofs:
        raise ValueError("asset state contract requires a nonempty DOF list")
    urdf_root = ET.parse(resource_path).getroot() if geometry_kind == GeometryKind.ARTICULATED_URDF else None
    dofs: list[DofSpec] = []
    defaults: dict[str, tuple[float, ...]] = {}
    for raw_value in raw_dofs:
        raw = _as_mapping(raw_value, "state_contract.dofs[]")
        dof_id = str(raw.get("dof_id", ""))
        kind_name = str(raw.get("kind", ""))
        if kind_name not in _DOF_LAYOUTS or not dof_id:
            raise ValueError(f"unsupported asset state DOF: {dof_id}/{kind_name}")
        kind, dimension, unit = _DOF_LAYOUTS[kind_name]
        fields = _source_fields(raw, dimension)
        bound: Bound | None = None
        if dof_id == "root.translation":
            bound = Bound((None, None, 1e-4), (None, None, None), "meter", "pinhole_camera_positive_depth")
        elif raw.get("bounds_from_urdf") is not None:
            if urdf_root is None:
                raise ValueError("bounds_from_urdf requires articulated_urdf geometry")
            joint_name = str(raw["bounds_from_urdf"])
            lower, upper = _urdf_joint_bounds(urdf_root, joint_name)
            bound = Bound(lower, upper, unit, f"asset_urdf:{joint_name}")
        elif isinstance(raw.get("bounds"), (list, tuple)) and len(raw["bounds"]) == 2:
            lower, upper = (float(value) for value in raw["bounds"])
            bound = Bound(lower, upper, unit, "asset_descriptor")
        dofs.append(DofSpec(dof_id, kind, dimension, unit, fields, bound))
        raw_default = raw.get("default")
        if raw_default is not None:
            values = raw_default if isinstance(raw_default, (list, tuple)) else (raw_default,)
            if len(values) != dimension:
                raise ValueError(f"asset state default width mismatch: {dof_id}")
            defaults[dof_id] = tuple(float(value) for value in values)
    spec = StateSpec(
        spec_id=str(state_contract["spec_id"]),
        state_model=str(state_contract.get("state_model", state_contract["spec_id"])),
        dofs=tuple(dofs),
    )
    feature_ids = tuple(str(value) for value in dict(descriptor.get("feature_segments", {})))
    if not feature_ids:
        feature_ids = tuple(str(value) for value in descriptor.get("semantic_parts", ()))
    if not feature_ids:
        raise ValueError("asset state contract requires declared geometry features")
    capabilities = ["contact_point", "surface_query"]
    if geometry_kind == GeometryKind.ARTICULATED_URDF:
        capabilities.extend(("project_line", "joint_transform"))
    else:
        capabilities.extend(("project_point", "periodic_feature"))
    geometry = GeometryDescriptor(
        geometry_id=str(descriptor.get("geometry_id", f"asset:{geometry_kind.value}")),
        kind=geometry_kind,
        feature_ids=feature_ids,
        capabilities=tuple(capabilities),
        resource_path=str(resource_path),
        resource_sha256=_sha256(resource_path),
    )
    return AssetStateContract(
        state_spec=spec,
        geometry=geometry,
        initializer=_as_mapping(descriptor.get("initializer"), "initializer"),
        descriptor_path=str(descriptor_path),
        descriptor_sha256=_sha256(descriptor_path),
        resource_path=str(resource_path),
        resource_sha256=_sha256(resource_path),
        default_state_by_dof=defaults,
    )
