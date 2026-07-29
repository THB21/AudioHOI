"""Capability-driven adapters from current artifacts to one generic object problem."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..base.config import CaseProfile
from ..contact_constraints import ContactConstraint, ContactState, LineS, LocalXYZ, adapt_legacy_contact_rows
from ..human_sites import GVHMRSiteExtractionResult, HumanSiteMeasurement, extract_gvhmr_site_measurements
from ..measurements import adapt_legacy_observation_rows
from ..state import (
    Bound,
    CapsuleGeometryProvider,
    DofKind,
    DofSpec,
    GaugeConstraint,
    GeometryDescriptor,
    GeometryKind,
    PeriodicFeatureRule,
    RigidFeatureGeometryProvider,
    StateAdaptationResult,
    StateSpec,
    StaticParameter,
)
from .legacy_production_problem import (
    LegacyObjectProblemPreparation,
    _configured_records,
    _contact_samples,
    _rows,
    _state_scales,
    prepare_legacy_articulated_object_problem,
)
from .problem import build_sequence_problem_shadow
from .problem_factory import SequenceFactorInputs, SequenceProblemFactory, SequenceProblemPreparation
from .residual_inputs import ContactFactorInput, PeriodicPhaseFactorInput, WorldSpaceContactSample


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _quaternion_from_matrix(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Return normalized qw,qx,qy,qz without depending on an object-specific rotation convention."""
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = (0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale)
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            values = ((matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale)
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            values = ((matrix[0, 2] - matrix[2, 0]) / scale, (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale)
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            values = ((matrix[1, 0] - matrix[0, 1]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale)
    vector = np.asarray(values, dtype=float)
    vector /= np.linalg.norm(vector)
    if vector[0] < 0.0:
        vector *= -1.0
    return tuple(float(value) for value in vector)


def _quaternion_from_yxz(yaw: float, pitch: float, roll: float) -> tuple[float, float, float, float]:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(roll), math.sin(roll)
    ry = np.asarray(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)))
    rx = np.asarray(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)))
    rz = np.asarray(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)))
    return _quaternion_from_matrix(ry @ rx @ rz)


def _quaternion_align_x(axis: np.ndarray) -> tuple[float, float, float, float]:
    x_axis = np.asarray((1.0, 0.0, 0.0))
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    cross = np.cross(x_axis, axis)
    dot = float(np.clip(np.dot(x_axis, axis), -1.0, 1.0))
    if dot < -1.0 + 1e-9:
        return (0.0, 0.0, 1.0, 0.0)
    quaternion = np.asarray((1.0 + dot, cross[0], cross[1], cross[2]), dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


def _state_spec(kind: GeometryKind, source: str, *, length_m: float | None = None) -> StateAdaptationResult:
    translation = DofSpec("root.translation", DofKind.TRANSLATION, 3, "meter", ("tx", "ty", "tz"))
    rotation = DofSpec("root.rotation", DofKind.ROTATION_SO3, 4, "quaternion", ("qw", "qx", "qy", "qz"))
    if kind == GeometryKind.RIGID_MESH:
        spec = StateSpec(
            spec_id="rigid6_plus_phase:periodic_rigid",
            state_model="rigid6_plus_phase",
            dofs=(
                translation,
                rotation,
                DofSpec("scale", DofKind.SCALAR, 1, "unitless", ("scale",), Bound(0.25, 3.5, "unitless", "asset_descriptor")),
                DofSpec("assembly.phase", DofKind.PERIODIC, 1, "radian", ("handle_phase_rad",), Bound(-math.pi, math.pi, "radian", "periodic_wrap")),
            ),
            gauge_constraints=(GaugeConstraint("axial_phase_gauge", ("root.rotation", "assembly.phase"), "axially symmetric body orientation is represented by periodic assembly phase", source),),
        )
        geometry = GeometryDescriptor("periodic_rigid_asset", kind, ("object:body", "object:handle"), ("surface_query", "contact_point", "periodic_feature"))
        schema = "observation_periodic_rigid_v1"
    elif kind == GeometryKind.LINE_CAPSULE:
        if length_m is None or length_m <= 0.0:
            raise ValueError("line/capsule StateSpec requires positive length")
        spec = StateSpec(
            spec_id="rigid6:line_capsule",
            state_model="rigid6_line_parameter",
            dofs=(translation, rotation),
            static_parameters=(StaticParameter("line.length", length_m, "meter", ("asset_descriptor.length_m",)),),
            gauge_constraints=(GaugeConstraint("line.roll_unobservable", ("root.rotation",), "roll around the line axis is frozen by the initializer gauge", source),),
        )
        geometry = GeometryDescriptor("line_capsule_asset", kind, ("line:axis", "line:left_endpoint", "line:right_endpoint"), ("project_line", "line_parameter", "contact_point"), parameters=spec.static_parameters)
        schema = "line_s_correspondence_v1"
    else:
        raise ValueError(f"unsupported capability initializer geometry: {kind.value}")
    return StateAdaptationResult(schema, spec, geometry, tuple(field for dof in spec.dofs for field in dof.source_fields), ())


def _gvhmr_sites(profile: CaseProfile, frame_times: Mapping[int, float], body_models_root: Path) -> GVHMRSiteExtractionResult:
    return extract_gvhmr_site_measurements(
        sample_id=profile.case_name,
        result_pkl=profile.sample_dir / "results/gvhmr/result.pkl",
        body_models_root=body_models_root,
        frame_times=frame_times,
    )


def _periodic_seed(result_dir: Path) -> tuple[StateAdaptationResult, dict[int, tuple[float, ...]], list[dict[str, object]], dict[int, float]]:
    body_path = result_dir / "observation_seed/body_pose.csv"
    phase_path = result_dir / "observation_seed/axial_phase.csv"
    bodies = _rows(body_path)
    phases = {int(row["frame"]): row for row in _rows(phase_path)}
    adaptation = _state_spec(GeometryKind.RIGID_MESH, f"{body_path}+{phase_path}")
    states: dict[int, tuple[float, ...]] = {}
    templates: list[dict[str, object]] = []
    times: dict[int, float] = {}
    for row in bodies:
        frame = int(row["frame"])
        phase = phases[frame]
        quaternion = _quaternion_from_yxz(float(row["yaw"]), float(row["pitch"]), float(row["roll"]))
        state = (float(row["x"]), float(row["y"]), float(row["z"]), *quaternion, float(row["scale"]), float(phase["m17_phase_rad"]))
        states[frame] = state
        times[frame] = float(row["time"])
        templates.append({"frame": frame, "time": row["time"], "tx": state[0], "ty": state[1], "tz": state[2], "qw": state[3], "qx": state[4], "qy": state[5], "qz": state[6], "scale": state[7], "handle_phase_rad": state[8], "source": "observation_seed_body_plus_axial_phase"})
    return adaptation, states, templates, times


def _line_seed(contact_rows: Sequence[Mapping[str, str]], length_m: float, source: str) -> tuple[StateAdaptationResult, dict[int, tuple[float, ...]], list[dict[str, object]], dict[int, float]]:
    adaptation = _state_spec(GeometryKind.LINE_CAPSULE, source, length_m=length_m)
    grouped: dict[int, list[Mapping[str, str]]] = {}
    for row in contact_rows:
        if row.get("contact_active") == "1":
            grouped.setdefault(int(row["frame"]), []).append(row)
    states: dict[int, tuple[float, ...]] = {}
    templates: list[dict[str, object]] = []
    times: dict[int, float] = {}
    previous_axis: np.ndarray | None = None
    previous_state: tuple[float, ...] | None = None
    for frame, rows in sorted(grouped.items()):
        resolved = [row for row in rows if row.get("stable_object_local_s") and row.get("palm_x")]
        if len(resolved) < 2:
            if previous_state is None:
                raise ValueError(f"line initializer requires two LineS/site correspondences at first frame {frame}")
            time = float(rows[0]["time"])
            states[frame] = previous_state
            times[frame] = time
            templates.append({"frame": frame, "time": time, "tx": previous_state[0], "ty": previous_state[1], "tz": previous_state[2], "qw": previous_state[3], "qx": previous_state[4], "qy": previous_state[5], "qz": previous_state[6], "source": "line_s_capability_initializer_previous_valid_hold"})
            continue
        first, second = resolved[:2]
        s1, s2 = float(first["stable_object_local_s"]), float(second["stable_object_local_s"])
        p1 = np.asarray([float(first[f"palm_{axis}"]) for axis in "xyz"])
        p2 = np.asarray([float(second[f"palm_{axis}"]) for axis in "xyz"])
        if abs(s1 - s2) <= 1e-6:
            raise ValueError(f"line initializer has degenerate LineS chord at frame {frame}")
        axis = (p1 - p2) / ((s1 - s2) * length_m)
        axis /= np.linalg.norm(axis)
        if previous_axis is not None and float(np.dot(axis, previous_axis)) < 0.0:
            axis *= -1.0
            s1, s2 = 1.0 - s1, 1.0 - s2
        previous_axis = axis
        center = 0.5 * (p1 - (s1 - 0.5) * length_m * axis + p2 - (s2 - 0.5) * length_m * axis)
        quaternion = _quaternion_align_x(axis)
        state = (*[float(value) for value in center], *quaternion)
        states[frame] = state
        previous_state = state
        time = float(first["time"])
        times[frame] = time
        templates.append({"frame": frame, "time": time, "tx": state[0], "ty": state[1], "tz": state[2], "qw": state[3], "qx": state[4], "qy": state[5], "qz": state[6], "source": "line_s_two_site_capability_initializer"})
    return adaptation, states, templates, times


def _contact_samples_with_line_s(constraints: Sequence[ContactConstraint], sites: Sequence[HumanSiteMeasurement]) -> tuple[WorldSpaceContactSample, ...]:
    local_samples = list(_contact_samples(constraints, sites))
    by_key = {(site.frame, "palm" if site.site.body_part == "hand" else site.site.body_part, site.site.side): site for site in sites}
    for constraint in constraints:
        if constraint.state not in {ContactState.ACTIVE, ContactState.OCCLUDED_HOLD} or not isinstance(constraint.object_coordinate, LineS):
            continue
        for frame in range(constraint.interval.start_frame, constraint.interval.end_frame + 1):
            site = by_key.get((frame, constraint.human_site.body_part, constraint.human_site.side))
            if site is not None:
                local_samples.append(WorldSpaceContactSample(frame, site.xyz_m, "line:axis", constraint.object_coordinate.s))
    return tuple(local_samples)


@dataclass(frozen=True)
class CapabilityObjectProblemPreparation:
    preparation: SequenceProblemPreparation
    state_adaptation: StateAdaptationResult
    template_rows: tuple[Mapping[str, object], ...]
    geometry_kind: str
    geometry_descriptor_path: str
    geometry_descriptor_sha256: str
    initializer_kind: str
    initializer_input_sha256: str
    gvhmr_sites: GVHMRSiteExtractionResult
    measurement_count: int
    contact_constraint_count: int
    case_dispatch_used: bool = False
    baseline_pose_read: bool = False
    human_state_optimized: bool = False
    accepted_outputs_written: bool = False


def prepare_capability_object_problem(*, profile: CaseProfile, result_dir: Path, repository_root: Path, body_models_root: Path) -> CapabilityObjectProblemPreparation | LegacyObjectProblemPreparation:
    config = profile.data.get("generic_object_problem")
    if not isinstance(config, Mapping):
        raise ValueError("case profile is missing generic_object_problem capability configuration")
    initializer = str(config["initializer"])
    if initializer == "legacy_state_artifact":
        return prepare_legacy_articulated_object_problem(profile=profile, result_dir=result_dir, repository_root=repository_root, body_models_root=body_models_root)

    descriptor_path = repository_root / str(profile.data["geometry_asset_descriptor"])
    descriptor = json.loads(descriptor_path.read_text())
    contact_path = result_dir / str(config.get("contact_artifact", "object_contact_points.csv"))
    contact_rows = _rows(contact_path)
    constraints = adapt_legacy_contact_rows(profile.case_name, contact_rows, str(contact_path)).constraints
    if initializer == "observation_periodic_rigid":
        adaptation, initial_states, templates, frame_times = _periodic_seed(result_dir)
        feature_points: dict[str, list[list[float]]] = {"object:body": [[0.0, 0.0, 0.0]]}
        periodic_rules: dict[str, PeriodicFeatureRule] = {}
        for constraint in constraints:
            if isinstance(constraint.object_coordinate, LocalXYZ):
                feature_id = constraint.object_feature.geometry_feature_id
                feature_points[feature_id] = [[constraint.object_coordinate.x_m, constraint.object_coordinate.y_m, constraint.object_coordinate.z_m]]
                periodic_rules[feature_id] = PeriodicFeatureRule(8, tuple(descriptor["periodic_axis_local"]), tuple(descriptor.get("periodic_origin_local", (0.0, 0.0, 0.0))))
        provider = RigidFeatureGeometryProvider(feature_points, scale_state_index=7, periodic_feature_rules=periodic_rules)
    elif initializer == "line_s_two_site":
        length_m = float(descriptor["length_m"])
        adaptation, initial_states, templates, frame_times = _line_seed(contact_rows, length_m, str(contact_path))
        provider = CapsuleGeometryProvider(length_m, float(descriptor.get("radius_m", 0.0)), tuple(descriptor.get("axis_local", (1.0, 0.0, 0.0))))
    else:
        raise ValueError(f"unsupported object capability initializer: {initializer}")

    gvhmr_sites = _gvhmr_sites(profile, frame_times, body_models_root)
    observation_path = result_dir / "object_observations.csv"
    measurements = adapt_legacy_observation_rows(profile.case_name, _rows(observation_path), str(observation_path)).measurements
    shadow = build_sequence_problem_shadow(profile, result_dir)
    records = _configured_records(shadow["residual_execution_plan"])
    samples = _contact_samples_with_line_s(constraints, gvhmr_sites.measurements)
    contact_factors: dict[str, ContactFactorInput] = {}
    phase_factors: dict[str, PeriodicPhaseFactorInput] = {}
    for record in records:
        factor_id = str(record["factor_id"])
        residual_ref = str(record["residual_fn_ref"])
        if residual_ref == "shadow_residual::contact_distance":
            contact_factors[factor_id] = ContactFactorInput(provider, samples, None)
        elif residual_ref == "shadow_residual::periodic_phase_prior":
            phase_targets = {frame: initial_states[frame][8] for frame in sorted(initial_states)}
            phase_factors[factor_id] = PeriodicPhaseFactorInput((), (), state_index=8, target_by_frame=phase_targets)
    factor_inputs = SequenceFactorInputs(
        state_scales=_state_scales(records, sum(dof.dimension for dof in adaptation.state_spec.dofs)),
        contact_factors=contact_factors,
        periodic_phase_factors=phase_factors,
    )
    preparation = SequenceProblemFactory().prepare(
        attempt_id=str(shadow["attempt_ledger"]["attempt_id"]),
        sequence_contract_sha256=str(shadow["sequence_problem_contract"]["canonical_sha256"]),
        state_spec=adaptation.state_spec,
        initial_states=initial_states,
        residual_execution_plan=shadow["residual_execution_plan"],
        factor_inputs=factor_inputs,
    )
    initializer_inputs = [result_dir / str(path) for path in config.get("initializer_artifacts", ())]
    return CapabilityObjectProblemPreparation(
        preparation=preparation,
        state_adaptation=adaptation,
        template_rows=tuple(templates),
        geometry_kind=str(descriptor["geometry_kind"]),
        geometry_descriptor_path=str(descriptor_path),
        geometry_descriptor_sha256=_sha256(descriptor_path),
        initializer_kind=initializer,
        initializer_input_sha256=_canonical_hash({str(path): _sha256(path) for path in initializer_inputs}),
        gvhmr_sites=gvhmr_sites,
        measurement_count=len(measurements),
        contact_constraint_count=len(constraints),
    )


def capability_object_problem_preparation_record(prepared: CapabilityObjectProblemPreparation | LegacyObjectProblemPreparation) -> dict[str, object]:
    from .legacy_production_problem import legacy_object_problem_preparation_record
    from .problem_factory import sequence_problem_preparation_record

    if isinstance(prepared, LegacyObjectProblemPreparation):
        record = legacy_object_problem_preparation_record(prepared)
        record["initializer_kind"] = "legacy_state_artifact"
        return record
    return {
        "schema_version": 1,
        "mode": "generic_object_sequence_problem_preparation",
        "problem": sequence_problem_preparation_record(prepared.preparation),
        "state_spec_id": prepared.state_adaptation.state_spec.spec_id,
        "geometry": {"kind": prepared.geometry_kind, "descriptor_path": prepared.geometry_descriptor_path, "descriptor_sha256": prepared.geometry_descriptor_sha256},
        "initializer_kind": prepared.initializer_kind,
        "initializer_input_sha256": prepared.initializer_input_sha256,
        "read_only_human_sites": {"measurement_count": len(prepared.gvhmr_sites.measurements), "source_artifact": prepared.gvhmr_sites.source_artifact, "source_sha256": prepared.gvhmr_sites.source_sha256, "read_only": True},
        "measurement_count": prepared.measurement_count,
        "contact_constraint_count": prepared.contact_constraint_count,
        "case_dispatch_used": False,
        "baseline_pose_read": False,
        "human_state_optimized": False,
        "accepted_outputs_written": False,
    }
