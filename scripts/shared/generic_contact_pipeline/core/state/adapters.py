from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from ..base.config import CaseProfile
from ..base.io import repo_path
from .types import (
    Bound,
    DofKind,
    DofSpec,
    GaugeConstraint,
    GeometryDescriptor,
    GeometryKind,
    StateSpec,
    StaticParameter,
)


@dataclass(frozen=True)
class StateAdaptationResult:
    schema: str
    state_spec: StateSpec
    geometry: GeometryDescriptor
    mapped_fields: tuple[str, ...]
    unmapped_nonempty_fields: tuple[str, ...]


def detect_legacy_state_schema(fields: set[str]) -> str:
    if {"rear_joint_angle", "seat_joint_angle", "qx", "qy", "qz", "qw"} <= fields:
        return "semantic_graph_6d_v1"
    if {"handle_phase_rad", "scale", "yaw", "pitch", "roll", "qw", "qx", "qy", "qz"} <= fields:
        return "rigid6_plus_periodic_phase_v1"
    if {"line_contact_pose_prior", "object_pixel_len", "qw", "qx", "qy", "qz"} <= fields:
        return "translation3_line_capsule_v1"
    if {"tx", "ty", "tz", "radius_m", "qw", "qx", "qy", "qz"} <= fields:
        return "translation3_sphere_v1"
    raise ValueError(f"unsupported legacy state schema with fields: {sorted(fields)}")


def _nonempty_fields(rows: list[dict[str, str]]) -> set[str]:
    return {field for field in rows[0] if any(row.get(field, "") not in {"", None} for row in rows)}


def _sha256_path(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).hexdigest().encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _relpath(path: Path) -> str:
    try:
        return str(path.relative_to(repo_path(".")))
    except ValueError:
        return str(path)


def _first_float(rows: list[dict[str, str]], field: str) -> float | None:
    for row in rows:
        value = row.get(field, "")
        if value not in {"", None}:
            return float(value)
    return None


def _profile_geometry_resource(profile: CaseProfile) -> Path | None:
    if "articraft_urdf" in profile.data:
        return repo_path(str(profile.data["articraft_urdf"]))
    if "articraft_model_py" in profile.data:
        model_py = repo_path(str(profile.data["articraft_model_py"]))
        urdf = model_py.parents[2] / "model.urdf"
        return urdf if urdf.exists() else model_py
    if profile.data.get("geometry_model") == "articraft_mug_mesh":
        return profile.sample_dir / "articraft/materialized_mug_mesh"
    return None


def _translation_dof(fields: tuple[str, str, str] = ("tx", "ty", "tz")) -> DofSpec:
    return DofSpec("root.translation", DofKind.TRANSLATION, 3, "meter", fields)


def _rotation_dof(fields: tuple[str, str, str, str]) -> DofSpec:
    return DofSpec("root.rotation", DofKind.ROTATION_SO3, 4, "quaternion", fields)


def adapt_legacy_state_rows(profile: CaseProfile, rows: list[dict[str, str]], artifact: str) -> StateAdaptationResult:
    if not rows:
        raise ValueError("cannot adapt an empty state table")
    schema = detect_legacy_state_schema(set(rows[0]))
    mapped: set[str] = {"frame", "time", "source", "se3_schema"}
    geometry_model = profile.component("geometry_model")

    if schema == "translation3_sphere_v1":
        mapped.update(("tx", "ty", "tz", "radius_m", "qw", "qx", "qy", "qz", "coord_frame"))
        spec = StateSpec(
            spec_id="translation3:sphere",
            state_model="translation3",
            dofs=(
                _translation_dof(),
                DofSpec("root.rotation", DofKind.ROTATION_SO3, 4, "quaternion", ("qw", "qx", "qy", "qz"), observable=False),
            ),
            static_parameters=(StaticParameter("sphere.radius", float(_first_float(rows, "radius_m") or 0.0), "meter", ("radius_m",)),),
            gauge_constraints=(GaugeConstraint("sphere.rotation_unobservable", ("root.rotation",), "sphere orientation is not observable from masks/depth", artifact),),
        )
        geometry = GeometryDescriptor(
            geometry_id=geometry_model,
            kind=GeometryKind.SPHERE,
            feature_ids=("object:center", "object:surface", "object:support"),
            capabilities=("project_point", "surface_query", "contact_point"),
            parameters=spec.static_parameters,
        )
    elif schema == "translation3_line_capsule_v1":
        mapped.update(("tx", "ty", "tz", "qw", "qx", "qy", "qz", "radius_m", "object_pixel_len", "z_from_geometry_len_m", "generic_geometry_pose_prior", "line_contact_pose_prior"))
        length_m = float(profile.data.get("line_object", {}).get("length_m", 0.0))
        spec = StateSpec(
            spec_id="translation3:line_capsule",
            state_model="translation3_with_line_orientation_prior",
            dofs=(
                _translation_dof(),
                DofSpec("root.rotation", DofKind.ROTATION_SO3, 4, "quaternion", ("qw", "qx", "qy", "qz"), observable=False),
            ),
            static_parameters=(StaticParameter("line.length", length_m, "meter", ("line_object.length_m",)),),
            gauge_constraints=(GaugeConstraint("line.roll_unobservable", ("root.rotation",), "line roll around its axis is unobservable in the legacy anchor schema", artifact),),
        )
        geometry = GeometryDescriptor(
            geometry_id=geometry_model,
            kind=GeometryKind.LINE_CAPSULE,
            feature_ids=("object:center", "line:axis", "line:left_endpoint", "line:right_endpoint"),
            capabilities=("project_line", "line_parameter", "contact_point"),
            resource_path=_relpath(_profile_geometry_resource(profile)) if _profile_geometry_resource(profile) else None,
            resource_sha256=_sha256_path(_profile_geometry_resource(profile)) if _profile_geometry_resource(profile) else None,
            parameters=spec.static_parameters,
        )
    elif schema == "rigid6_plus_periodic_phase_v1":
        mapped.update(("x", "y", "z", "tx", "ty", "tz", "yaw", "pitch", "roll", "scale", "handle_phase_rad", "qw", "qx", "qy", "qz", "vlm_visibility"))
        resource = _profile_geometry_resource(profile)
        spec = StateSpec(
            spec_id="rigid6_plus_phase:mug",
            state_model="rigid6_plus_phase",
            dofs=(
                _translation_dof(),
                _rotation_dof(("qw", "qx", "qy", "qz")),
                DofSpec("scale", DofKind.SCALAR, 1, "unitless", ("scale",)),
                DofSpec("handle.phase", DofKind.PERIODIC, 1, "radian", ("handle_phase_rad",), Bound(-3.141592653589793, 3.141592653589793, "radian", "periodic_wrap")),
            ),
            gauge_constraints=(GaugeConstraint("mug_yaw_not_forced_zero", ("root.rotation",), "legacy yaw is observed state, not a zero-filled gauge", artifact),),
        )
        geometry = GeometryDescriptor(
            geometry_id=geometry_model,
            kind=GeometryKind.RIGID_MESH,
            feature_ids=("object:body", "object:handle", "object:rim", "object:bottom"),
            capabilities=("project_mesh", "surface_query", "contact_point"),
            resource_path=_relpath(resource) if resource else None,
            resource_sha256=_sha256_path(resource) if resource else None,
        )
    else:
        mapped.update(("tx", "ty", "tz", "qx", "qy", "qz", "qw", "rear_joint_angle", "seat_joint_angle", "fx", "fy", "cx", "cy", "da3_mask_depth_m", "da3_depth_conf", "residual_norm"))
        resource = _profile_geometry_resource(profile)
        spec = StateSpec(
            spec_id="semantic_graph_6d:chair",
            state_model="semantic_graph_6d",
            dofs=(
                _translation_dof(),
                _rotation_dof(("qw", "qx", "qy", "qz")),
                DofSpec("joint.front_to_rear", DofKind.REVOLUTE, 1, "radian", ("rear_joint_angle",), Bound(-0.82, 0.12, "radian", "chair_urdf:front_to_rear")),
                DofSpec("joint.front_to_seat", DofKind.REVOLUTE, 1, "radian", ("seat_joint_angle",), Bound(0.0, 1.35, "radian", "chair_urdf:front_to_seat")),
            ),
        )
        geometry = GeometryDescriptor(
            geometry_id=geometry_model,
            kind=GeometryKind.ARTICULATED_URDF,
            feature_ids=("backrest:top_edge", "seat:front_edge", "leg:front_left", "leg:front_right", "leg:rear_left", "leg:rear_right"),
            capabilities=("project_line", "joint_transform", "contact_point", "surface_query"),
            resource_path=_relpath(resource) if resource else None,
            resource_sha256=_sha256_path(resource) if resource else None,
        )

    nonempty = _nonempty_fields(rows)
    return StateAdaptationResult(schema, spec, geometry, tuple(sorted(mapped)), tuple(sorted(nonempty - mapped)))
