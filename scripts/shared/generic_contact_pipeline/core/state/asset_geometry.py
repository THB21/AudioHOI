"""Build geometry providers from asset descriptors and typed feature artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..contact_constraints import ContactConstraint, LocalXYZ
from .articulated import ArticulatedKinematicProvider, SegmentJointRule
from .geometry_provider import ArticulatedFeatureGeometryProvider, GeometryProvider, RigidFeatureGeometryProvider
from .types import StateSpec


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _vector(text: str | None, label: str) -> np.ndarray:
    values = np.asarray([float(value) for value in str(text or "").split()], dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"{label} must contain three finite values")
    return values


def _state_indices(state_spec: StateSpec) -> dict[str, int]:
    indices: dict[str, int] = {}
    offset = 0
    for dof in state_spec.dofs:
        if dof.dimension == 1:
            indices[dof.dof_id] = offset
        offset += dof.dimension
    return indices


@dataclass(frozen=True)
class AssetGeometryBuildResult:
    provider: GeometryProvider
    descriptor_path: str
    descriptor_sha256: str
    resource_path: str
    resource_sha256: str
    semantic_segments_path: str
    semantic_segments_sha256: str
    feature_ids: tuple[str, ...]
    contact_feature_ids: tuple[str, ...]
    case_dispatch_used: bool
    canonical_sha256: str

    def __post_init__(self) -> None:
        if self.case_dispatch_used:
            raise ValueError("asset geometry construction cannot use case dispatch")
        if not self.feature_ids:
            raise ValueError("asset geometry construction requires semantic features")


def build_rigid_geometry_from_asset_descriptor(
    *,
    descriptor_path: Path,
    repository_root: Path,
    state_spec: StateSpec,
    contact_constraints: Sequence[ContactConstraint] = (),
) -> AssetGeometryBuildResult:
    """Build fixed rigid semantic geometry without interpreting object identity."""

    descriptor = json.loads(descriptor_path.read_text())
    if descriptor.get("schema_version") != 1 or descriptor.get("geometry_kind") != "rigid_mesh":
        raise ValueError("unsupported fixed rigid asset geometry descriptor")
    resource_path = repository_root / str(descriptor["resource_path"])
    if not resource_path.is_file():
        raise FileNotFoundError("fixed rigid asset resource is missing")
    raw_features = descriptor.get("feature_points", {})
    if not isinstance(raw_features, Mapping) or not raw_features:
        raise ValueError("fixed rigid asset requires descriptor-declared feature points")
    feature_points: dict[str, list[list[float]]] = {}
    for feature_id, raw_points in raw_features.items():
        points = np.asarray(raw_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError(f"invalid fixed rigid feature points: {feature_id}")
        feature_points[str(feature_id)] = points.tolist()

    contact_feature_ids: set[str] = set()
    for constraint in contact_constraints:
        coordinate = constraint.object_coordinate
        if not isinstance(coordinate, LocalXYZ):
            continue
        feature_id = constraint.object_feature.geometry_feature_id
        point = [coordinate.x_m, coordinate.y_m, coordinate.z_m]
        values = feature_points.setdefault(feature_id, [])
        if not any(np.allclose(existing, point, atol=1e-9) for existing in values):
            values.append(point)
        contact_feature_ids.add(feature_id)

    scale_state_index = _state_indices(state_spec).get("scale")
    provider = RigidFeatureGeometryProvider(
        feature_points_local=feature_points,
        scale_state_index=scale_state_index,
    )
    payload = {
        "descriptor_sha256": _sha256(descriptor_path),
        "resource_sha256": _sha256(resource_path),
        "state_spec_id": state_spec.spec_id,
        "feature_ids": sorted(feature_points),
        "contact_feature_ids": sorted(contact_feature_ids),
        "case_dispatch_used": False,
    }
    return AssetGeometryBuildResult(
        provider=provider,
        descriptor_path=str(descriptor_path),
        descriptor_sha256=payload["descriptor_sha256"],
        resource_path=str(resource_path),
        resource_sha256=payload["resource_sha256"],
        semantic_segments_path=str(descriptor_path),
        semantic_segments_sha256=payload["descriptor_sha256"],
        feature_ids=tuple(sorted(feature_points)),
        contact_feature_ids=tuple(sorted(contact_feature_ids)),
        case_dispatch_used=False,
        canonical_sha256=_canonical_hash(payload),
    )


def build_articulated_geometry_from_asset_descriptor(
    *,
    descriptor_path: Path,
    repository_root: Path,
    result_dir: Path,
    state_spec: StateSpec,
    contact_constraints: Sequence[ContactConstraint] = (),
) -> AssetGeometryBuildResult:
    descriptor = json.loads(descriptor_path.read_text())
    if descriptor.get("schema_version") != 1 or descriptor.get("geometry_kind") != "articulated_urdf":
        raise ValueError("unsupported asset geometry descriptor")
    resource_path = repository_root / str(descriptor["resource_path"])
    segments_path = result_dir / str(descriptor["semantic_segments_artifact"])
    if not resource_path.is_file() or not segments_path.is_file():
        raise FileNotFoundError("asset descriptor resource or semantic segment artifact is missing")

    segments: dict[str, tuple[str, np.ndarray]] = {}
    with segments_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            segment_id = str(row["segment_id"])
            points = np.asarray(
                [
                    [float(row[f"start_local_{axis}"]) for axis in "xyz"],
                    [float(row[f"end_local_{axis}"]) for axis in "xyz"],
                ],
                dtype=float,
            )
            if segment_id in segments or not np.isfinite(points).all():
                raise ValueError(f"invalid or duplicate semantic segment: {segment_id}")
            segments[segment_id] = (str(row["part"]), points)

    feature_points: dict[str, Sequence[Sequence[float]]] = {}
    feature_parts: dict[str, str] = {}
    for feature_id, segment_id in dict(descriptor["feature_segments"]).items():
        if str(segment_id) not in segments:
            raise ValueError(f"asset feature references missing semantic segment: {segment_id}")
        part, points = segments[str(segment_id)]
        feature_points[str(feature_id)] = points
        feature_parts[str(feature_id)] = part

    for feature_id, raw_point in dict(descriptor.get("point_features", {})).items():
        point_spec = dict(raw_point)
        segment_id = str(point_spec["segment"])
        endpoint = str(point_spec["endpoint"])
        if segment_id not in segments or endpoint not in {"start", "end"}:
            raise ValueError(f"asset point feature cannot resolve endpoint: {feature_id}")
        part, points = segments[segment_id]
        feature_points[str(feature_id)] = [points[0 if endpoint == "start" else 1]]
        feature_parts[str(feature_id)] = part

    contact_feature_ids: set[str] = set()
    for constraint in contact_constraints:
        coordinate = constraint.object_coordinate
        if not isinstance(coordinate, LocalXYZ):
            continue
        feature_id = constraint.object_feature.geometry_feature_id
        point = [[coordinate.x_m, coordinate.y_m, coordinate.z_m]]
        existing = feature_points.get(feature_id)
        if existing is not None and not np.allclose(np.asarray(existing, dtype=float), point, atol=1e-9):
            raise ValueError(f"contact feature has inconsistent LocalXYZ coordinates: {feature_id}")
        feature_points[feature_id] = point
        feature_parts[feature_id] = constraint.object_feature.semantic_role
        contact_feature_ids.add(feature_id)

    root = ET.parse(resource_path).getroot()
    urdf_joints = {str(node.attrib.get("name", "")): node for node in root.findall("joint")}
    state_indices = _state_indices(state_spec)
    fixed_joint_values = {
        str(name): float(value)
        for name, value in dict(descriptor.get("fixed_assembly_joints", {})).items()
    }
    joint_state_indices: dict[str, int] = {}
    rules: list[SegmentJointRule] = []
    for raw_rule in descriptor["articulation_rules"]:
        urdf_joint = str(raw_rule["urdf_joint"])
        state_dof_id = str(raw_rule["state_dof_id"])
        node = urdf_joints.get(urdf_joint)
        if node is None or (state_dof_id not in state_indices and urdf_joint not in fixed_joint_values):
            raise ValueError(f"asset articulation rule cannot resolve joint: {urdf_joint}/{state_dof_id}")
        origin_node = node.find("origin")
        axis_node = node.find("axis")
        origin = _vector(origin_node.attrib.get("xyz") if origin_node is not None else None, "joint origin")
        axis = _vector(axis_node.attrib.get("xyz") if axis_node is not None else None, "joint axis")
        if state_dof_id in state_indices:
            joint_state_indices[state_dof_id] = state_indices[state_dof_id]
        rules.append(
            SegmentJointRule(
                rule_id=str(raw_rule["rule_id"]),
                joint_id=state_dof_id,
                parts=tuple(str(value) for value in raw_rule.get("parts", ())),
                segment_ids=tuple(str(value) for value in raw_rule.get("segment_ids", ())),
                origin=origin,
                axis=axis,
                endpoint_selector=str(raw_rule.get("endpoint_selector", "all")),
            )
        )

    kinematics = ArticulatedKinematicProvider(tuple(rules))
    if joint_state_indices:
        provider: GeometryProvider = ArticulatedFeatureGeometryProvider(
            feature_points_local=feature_points,
            feature_parts=feature_parts,
            kinematic_provider=kinematics,
            joint_state_indices=joint_state_indices,
        )
    else:
        fixed_by_state_dof = {
            str(raw_rule["state_dof_id"]): fixed_joint_values[str(raw_rule["urdf_joint"])]
            for raw_rule in descriptor["articulation_rules"]
        }
        baked_points = {
            feature_id: kinematics.articulate_segment(
                feature_id,
                feature_parts[feature_id],
                np.asarray(points, dtype=float),
                fixed_by_state_dof,
            )
            for feature_id, points in feature_points.items()
        }
        provider = RigidFeatureGeometryProvider(baked_points)
    payload = {
        "descriptor_sha256": _sha256(descriptor_path),
        "resource_sha256": _sha256(resource_path),
        "semantic_segments_sha256": _sha256(segments_path),
        "state_spec_id": state_spec.spec_id,
        "feature_ids": sorted(feature_points),
        "contact_feature_ids": sorted(contact_feature_ids),
        "case_dispatch_used": False,
    }
    return AssetGeometryBuildResult(
        provider=provider,
        descriptor_path=str(descriptor_path),
        descriptor_sha256=payload["descriptor_sha256"],
        resource_path=str(resource_path),
        resource_sha256=payload["resource_sha256"],
        semantic_segments_path=str(segments_path),
        semantic_segments_sha256=payload["semantic_segments_sha256"],
        feature_ids=tuple(sorted(feature_points)),
        contact_feature_ids=tuple(sorted(contact_feature_ids)),
        case_dispatch_used=False,
        canonical_sha256=_canonical_hash(payload),
    )
