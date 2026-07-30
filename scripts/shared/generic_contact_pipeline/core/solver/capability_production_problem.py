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
from ..contact_constraints import ContactConstraint, ContactMode, ContactState, LineS, LocalXYZ, adapt_contact_event_rows, adapt_contact_state_rows, adapt_legacy_contact_rows
from ..human_sites import GVHMRSiteExtractionResult, HumanSiteMeasurement, extract_gvhmr_site_measurements
from ..factors import FactorArbitrationLedger, build_factor_arbitration_ledger, factor_arbitration_ledger_record
from ..gates import load_factor_arbitration_ledger
from ..measurements import Line2DMeasurement, MetricDepthMeasurement, Point2DMeasurement, adapt_configured_supplemental_measurements, adapt_legacy_observation_rows
from ..state import (
    Bound,
    CapsuleGeometryProvider,
    DofKind,
    DofSpec,
    GaugeConstraint,
    GeometryDescriptor,
    GeometryKind,
    PeriodicFeatureRule,
    PlaneSurface,
    PinholeCamera,
    RigidFeatureGeometryProvider,
    SphereGeometryProvider,
    StateAdaptationResult,
    StateSpec,
    StaticParameter,
    build_articulated_geometry_from_asset_descriptor,
    build_asset_state_contract,
)
from .legacy_production_problem import (
    LegacyObjectProblemPreparation,
    _configured_records,
    _contact_samples,
    _rows,
    _state_scales,
)
from .capability_initializers import InitializationRequest, initialize_from_capabilities
from .problem import build_sequence_problem_shadow
from .problem_factory import SequenceFactorInputs, SequenceProblemFactory, SequenceProblemPreparation
from .residual_inputs import ContactFactorInput, LineReprojectionFactorInput, MetricDepthFactorInput, PeriodicPhaseFactorInput, PointReprojectionFactorInput, SupportPlaneFactorInput, WorldSpaceContactSample


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _support_plane_from_human_foot_sites(
    measurements: Sequence[HumanSiteMeasurement],
    *,
    surface_offset_m: float,
) -> PlaneSurface:
    points = np.asarray(
        [measurement.xyz_m for measurement in measurements if measurement.site.body_part == "foot"],
        dtype=float,
    )
    if points.ndim != 2 or points.shape[0] < 6 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("support-plane fitting requires at least six finite read-only foot sites")
    center = np.median(points, axis=0)
    _u, _singular_values, vectors = np.linalg.svd(points - center, full_matrices=False)
    normal = vectors[-1]
    if normal[1] < 0.0:
        normal = -normal
    normal = normal / np.linalg.norm(normal)
    offset = -float(normal @ center) - float(surface_offset_m)
    return PlaneSurface(tuple(float(value) for value in normal), offset)


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


def _state_spec(
    kind: GeometryKind,
    source: str,
    *,
    length_m: float | None = None,
    radius_m: float | None = None,
) -> StateAdaptationResult:
    camera_depth_margin = 1e-4
    if kind == GeometryKind.LINE_CAPSULE and length_m is not None:
        camera_depth_margin += 0.5 * length_m
    translation = DofSpec(
        "root.translation",
        DofKind.TRANSLATION,
        3,
        "meter",
        ("tx", "ty", "tz"),
        Bound(
            (None, None, camera_depth_margin),
            (None, None, None),
            "meter",
            "pinhole_camera_geometry_envelope_positive_depth",
        ),
    )
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
    elif kind == GeometryKind.SPHERE:
        if radius_m is None or radius_m <= 0.0:
            raise ValueError("sphere StateSpec requires positive radius")
        spec = StateSpec(
            spec_id="translation3:sphere",
            state_model="translation3",
            dofs=(translation,),
            static_parameters=(StaticParameter("sphere.radius", radius_m, "meter", ("asset_profile.radius_m",)),),
        )
        geometry = GeometryDescriptor("sphere_asset", kind, ("object:center", "object:surface", "object:support"), ("project_point", "surface_query", "contact_point"), parameters=spec.static_parameters)
        schema = "point_depth_sphere_v1"
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


def _point_depth_seed(measurements: Sequence[object], camera: PinholeCamera, radius_m: float, source: str) -> tuple[StateAdaptationResult, dict[int, tuple[float, ...]], list[dict[str, object]], dict[int, float]]:
    points = {m.meta.frame: m for m in measurements if isinstance(m, Point2DMeasurement) and m.meta.feature.semantic_role == "object_center"}
    depths = {m.meta.frame: m for m in measurements if isinstance(m, MetricDepthMeasurement) and m.meta.feature.semantic_role == "object_center_depth"}
    frames = sorted(set(points) & set(depths))
    if not frames:
        raise ValueError("point/depth initializer requires frame-aligned object center and metric depth")
    adaptation = _state_spec(GeometryKind.SPHERE, source, radius_m=radius_m)
    states: dict[int, tuple[float, ...]] = {}
    templates: list[dict[str, object]] = []
    times: dict[int, float] = {}
    for frame in frames:
        point, depth = points[frame], depths[frame]
        z = float(depth.depth_m)
        state = ((float(point.u) - camera.cx) * z / camera.fx, (float(point.v) - camera.cy) * z / camera.fy, z)
        states[frame] = state
        times[frame] = float(point.meta.time)
        templates.append({"frame": frame, "time": point.meta.time, "tx": state[0], "ty": state[1], "tz": state[2], "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "radius_m": radius_m, "source": "point_depth_capability_initializer"})
    return adaptation, states, templates, times


def _contact_samples_with_line_s(
    constraints: Sequence[ContactConstraint],
    sites: Sequence[HumanSiteMeasurement],
    *,
    line_contact_projection: str = "line_s",
) -> tuple[WorldSpaceContactSample, ...]:
    local_samples = list(_contact_samples(constraints, sites))
    by_key = {(site.frame, "palm" if site.site.body_part == "hand" else site.site.body_part, site.site.side): site for site in sites}
    for constraint in constraints:
        if constraint.state not in {ContactState.ACTIVE, ContactState.OCCLUDED_HOLD} or not isinstance(constraint.object_coordinate, LineS):
            continue
        for frame in range(constraint.interval.start_frame, constraint.interval.end_frame + 1):
            site = by_key.get((frame, constraint.human_site.body_part, constraint.human_site.side))
            if site is not None:
                if line_contact_projection not in {"line_s", "closest_line_point"}:
                    raise ValueError(f"unsupported line contact projection: {line_contact_projection}")
                line_s = constraint.object_coordinate.s if line_contact_projection == "line_s" else None
                local_samples.append(
                    WorldSpaceContactSample(
                        frame,
                        site.xyz_m,
                        "line:axis",
                        line_s,
                        constraint.confidence,
                    )
                )
    return tuple(local_samples)


def _surface_contact_samples_from_states(
    contact_states: Sequence[object],
    sites: Sequence[HumanSiteMeasurement],
) -> tuple[WorldSpaceContactSample, ...]:
    """Join typed active contact states to fixed human sites by semantic site identity."""
    sites_by_key = {
        (site.frame, site.site.body_part, site.site.side): site
        for site in sites
    }
    samples: list[WorldSpaceContactSample] = []
    for state in contact_states:
        if not state.human_active:
            continue
        site = sites_by_key.get((state.frame, state.human_site.body_part, state.human_site.side))
        if site is None:
            continue
        state_confidence = state.confidence if state.confidence is not None else 1.0
        site_confidence = site.confidence if site.confidence is not None else 1.0
        samples.append(
            WorldSpaceContactSample(
                state.frame,
                site.xyz_m,
                "object:surface",
                None,
                state_confidence * site_confidence,
            )
        )
    return tuple(samples)


def _directional_contact_samples_from_events(
    events: Sequence[object],
    contact_states: Sequence[object],
    sites: Sequence[HumanSiteMeasurement],
    *,
    target_feature_id: str,
    residual_axes: tuple[int, ...],
    maximum_offset_m: float | None,
) -> tuple[WorldSpaceContactSample, ...]:
    """Resolve event-timed directional contact observations without object-family dispatch."""
    states_by_frame = {state.frame: state for state in contact_states}
    sites_by_key = {
        (site.frame, site.site.body_part, site.site.side): site
        for site in sites
    }
    samples: list[WorldSpaceContactSample] = []
    for event in events:
        if event.mode == ContactMode.SUPPORT:
            continue
        state = states_by_frame.get(event.peak_frame)
        site = sites_by_key.get((event.peak_frame, event.human_site.body_part, event.human_site.side))
        if state is None or site is None:
            continue
        offset = [0.0, 0.0, 0.0]
        if 2 in residual_axes:
            raw_offset = float(state.contact_depth_offset_m)
            if maximum_offset_m is None or abs(raw_offset) <= maximum_offset_m:
                offset[2] = -raw_offset
        samples.append(
            WorldSpaceContactSample(
                event.peak_frame,
                site.xyz_m,
                target_feature_id,
                None,
                event.confidence,
                tuple(offset),
            )
        )
    return tuple(samples)


def _directional_anchor_depth_reference(
    frames: Sequence[int],
    samples: Sequence[WorldSpaceContactSample],
) -> dict[int, float]:
    """Interpolate a camera-depth gauge from sparse directional contact anchors."""
    anchors = sorted(
        (sample.frame, sample.source_xyz_m[2] + sample.source_offset_xyz_m[2])
        for sample in samples
    )
    if not anchors:
        raise ValueError("directional contact depth reference requires at least one anchor")
    anchor_frames = np.asarray([frame for frame, _ in anchors], dtype=float)
    anchor_depths = np.asarray([depth for _, depth in anchors], dtype=float)
    target_frames = np.asarray([int(frame) for frame in frames], dtype=float)
    interpolated = np.interp(target_frames, anchor_frames, anchor_depths)
    return {int(frame): float(depth) for frame, depth in zip(frames, interpolated)}


def _apply_translation_depth_reference(
    initial_states: Mapping[int, Sequence[float]],
    template_rows: Sequence[Mapping[str, object]],
    measurements: Sequence[object],
    camera: PinholeCamera,
    depth_by_frame: Mapping[int, float],
) -> tuple[dict[int, tuple[float, ...]], list[dict[str, object]]]:
    centers = {
        measurement.meta.frame: measurement
        for measurement in measurements
        if isinstance(measurement, Point2DMeasurement)
        and measurement.meta.feature.semantic_role == "object_center"
    }
    states = {int(frame): tuple(float(value) for value in state) for frame, state in initial_states.items()}
    for frame, depth in depth_by_frame.items():
        center = centers.get(frame)
        if center is None or frame not in states:
            continue
        state = list(states[frame])
        state[:3] = [
            (float(center.u) - camera.cx) * depth / camera.fx,
            (float(center.v) - camera.cy) * depth / camera.fy,
            depth,
        ]
        states[frame] = tuple(state)
    rows = [dict(row) for row in template_rows]
    for row in rows:
        frame = int(row["frame"])
        if frame not in states:
            continue
        row.update({"tx": states[frame][0], "ty": states[frame][1], "tz": states[frame][2]})
        row["source"] = f"{row.get('source', 'capability_initializer')}+directional_anchor_depth_reference"
    return states, rows


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
    initializer_ledger: Mapping[str, object]
    gvhmr_sites: GVHMRSiteExtractionResult
    measurement_count: int
    contact_constraint_count: int
    factor_arbitration: FactorArbitrationLedger
    case_dispatch_used: bool = False
    baseline_pose_read: bool = False
    human_state_optimized: bool = False
    accepted_outputs_written: bool = False


def prepare_capability_object_problem(
    *,
    profile: CaseProfile,
    result_dir: Path,
    repository_root: Path,
    body_models_root: Path,
    factor_arbitration_mode: str = "auto",
) -> CapabilityObjectProblemPreparation | LegacyObjectProblemPreparation:
    if factor_arbitration_mode not in {"auto", "off", "required"}:
        raise ValueError("factor arbitration mode must be auto, off, or required")
    config = profile.data.get("generic_object_problem")
    if not isinstance(config, Mapping):
        raise ValueError("case profile is missing generic_object_problem capability configuration")
    initializer = str(config["initializer"])

    descriptor_path = repository_root / str(profile.data["geometry_asset_descriptor"])
    descriptor = json.loads(descriptor_path.read_text())
    observation_path = result_dir / "object_observations.csv"
    measurements = list(adapt_legacy_observation_rows(profile.case_name, _rows(observation_path), str(observation_path)).measurements)
    measurements.extend(adapt_configured_supplemental_measurements(profile, result_dir).measurements)
    contact_path = result_dir / str(config.get("contact_artifact", "object_contact_points.csv"))
    contact_rows = _rows(contact_path)
    constraints = adapt_legacy_contact_rows(profile.case_name, contact_rows, str(contact_path)).constraints
    gvhmr_sites: GVHMRSiteExtractionResult | None = None
    initializer_ledger: Mapping[str, object] = {
        "initializer_kind": initializer,
        "case_dispatch_used": False,
        "baseline_pose_read": False,
        "human_state_optimized": False,
    }
    if initializer in {"articulated_correspondence", "fixed_assembly_correspondence"}:
        contract = build_asset_state_contract(descriptor_path, repository_root)
        if str(contract.initializer["kind"]) != initializer:
            raise ValueError("case initializer and asset initializer capability disagree")
        frame_times = {
            measurement.meta.frame: measurement.meta.time
            for measurement in measurements
        }
        if not frame_times:
            raise ValueError("articulated initializer requires typed frame measurements")
        gvhmr_sites = _gvhmr_sites(profile, frame_times, body_models_root)
        geometry_build = build_articulated_geometry_from_asset_descriptor(
            descriptor_path=descriptor_path,
            repository_root=repository_root,
            result_dir=result_dir,
            state_spec=contract.state_spec,
            contact_constraints=constraints,
        )
        cameras = {frame: PinholeCamera(**profile.camera) for frame in sorted(frame_times)}
        initialized = initialize_from_capabilities(
            InitializationRequest(
                state_spec=contract.state_spec,
                geometry_provider=geometry_build.provider,
                measurements=tuple(measurements),
                contact_constraints=tuple(constraints),
                human_sites=tuple(gvhmr_sites.measurements),
                cameras=cameras,
                initializer=contract.initializer,
                default_state_by_dof=contract.default_state_by_dof,
            )
        )
        adaptation = StateAdaptationResult(
            schema="articulated_correspondence_v1",
            state_spec=contract.state_spec,
            geometry=contract.geometry,
            mapped_fields=tuple(field for dof in contract.state_spec.dofs for field in dof.source_fields),
            unmapped_nonempty_fields=(),
        )
        initial_states = dict(initialized.states_by_frame)
        templates = [dict(row) for row in initialized.template_rows]
        provider = geometry_build.provider
        initializer_ledger = initialized.hypothesis_ledger
    elif initializer == "observation_periodic_rigid":
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
    elif initializer == "point_depth":
        radius_m = float(profile.data["sphere"]["radius_m"])
        adaptation, initial_states, templates, frame_times = _point_depth_seed(
            measurements,
            PinholeCamera(**profile.camera),
            radius_m,
            str(observation_path),
        )
        provider = SphereGeometryProvider(radius_m)
    else:
        raise ValueError(f"unsupported object capability initializer: {initializer}")

    if gvhmr_sites is None:
        gvhmr_sites = _gvhmr_sites(profile, frame_times, body_models_root)
    base_shadow = build_sequence_problem_shadow(profile, result_dir)
    base_compiled_records = base_shadow["inputs"]["compiled_factor_shadow"]["records"]
    factor_arbitration = (
        load_factor_arbitration_ledger(
            sample_id=profile.case_name,
            result_dir=result_dir,
            factor_records=base_compiled_records,
        )
        if factor_arbitration_mode != "off"
        else build_factor_arbitration_ledger(sample_id=profile.case_name, status="not_evaluated")
    )
    if factor_arbitration_mode == "required" and factor_arbitration.status != "evaluated":
        raise ValueError("required VLM factor arbitration has not been evaluated")
    shadow = (
        build_sequence_problem_shadow(profile, result_dir, factor_arbitration)
        if factor_arbitration.status == "evaluated"
        else base_shadow
    )
    records = _configured_records(shadow["residual_execution_plan"])
    samples = _contact_samples_with_line_s(
        constraints,
        gvhmr_sites.measurements,
        line_contact_projection=str(descriptor.get("contact_projection", "line_s")),
    )
    contact_state_path = result_dir / str(config.get("contact_state_artifact", "contact_state_frames.csv"))
    if config.get("contact_source") == "interaction_state":
        contact_states = adapt_contact_state_rows(profile.case_name, _rows(contact_state_path), str(contact_state_path))
        samples = _surface_contact_samples_from_states(contact_states, gvhmr_sites.measurements)
    elif config.get("contact_source") == "event_directional":
        contact_states = adapt_contact_state_rows(profile.case_name, _rows(contact_state_path), str(contact_state_path))
        contact_event_path = result_dir / str(config["contact_event_artifact"])
        contact_events = adapt_contact_event_rows(profile.case_name, _rows(contact_event_path), str(contact_event_path))
        residual_axes = tuple(int(axis) for axis in config.get("contact_residual_axes", (0, 1, 2)))
        samples = _directional_contact_samples_from_events(
            contact_events,
            contact_states,
            gvhmr_sites.measurements,
            target_feature_id=str(config.get("contact_target_feature", "object:surface")),
            residual_axes=residual_axes,
            maximum_offset_m=(
                None
                if config.get("maximum_contact_offset_m") is None
                else float(config["maximum_contact_offset_m"])
            ),
        )
    depth_targets: dict[int, float] | None = None
    if config.get("depth_reference") == "directional_contact_interpolation":
        depth_targets = _directional_anchor_depth_reference(sorted(initial_states), samples)
        initial_states, templates = _apply_translation_depth_reference(
            initial_states,
            templates,
            measurements,
            PinholeCamera(**profile.camera),
            depth_targets,
        )
    contact_factors: dict[str, ContactFactorInput] = {}
    phase_factors: dict[str, PeriodicPhaseFactorInput] = {}
    line_factors: dict[str, LineReprojectionFactorInput] = {}
    point_factors: dict[str, PointReprojectionFactorInput] = {}
    depth_factors: dict[str, MetricDepthFactorInput] = {}
    support_factors: dict[str, SupportPlaneFactorInput] = {}
    cameras = {frame: PinholeCamera(**profile.camera) for frame in initial_states}
    line_measurements = tuple(item for item in measurements if isinstance(item, Line2DMeasurement))
    measurement_roles = config.get("measurement_roles", {})
    point_roles = set(measurement_roles.get("point_reprojection", ())) if isinstance(measurement_roles, Mapping) else set()
    depth_roles = set(measurement_roles.get("metric_depth", ())) if isinstance(measurement_roles, Mapping) else set()
    for record in records:
        factor_id = str(record["factor_id"])
        residual_ref = str(record["residual_fn_ref"])
        if residual_ref == "shadow_residual::contact_distance":
            contact_factors[factor_id] = ContactFactorInput(
                provider,
                samples,
                None,
                residual_axes=tuple(int(axis) for axis in config.get("contact_residual_axes", (0, 1, 2))),
            )
        elif residual_ref == "shadow_residual::periodic_phase_prior":
            phase_targets = {frame: initial_states[frame][8] for frame in sorted(initial_states)}
            phase_factors[factor_id] = PeriodicPhaseFactorInput((), (), state_index=8, target_by_frame=phase_targets)
        elif residual_ref == "shadow_residual::line_reprojection":
            line_factors[factor_id] = LineReprojectionFactorInput(
                provider,
                line_measurements,
                cameras,
                allow_endpoint_swap=True,
                constraint_mode=str(descriptor.get("line_reprojection_constraint", "endpoints")),
            )
        elif residual_ref == "shadow_residual::point_reprojection":
            point_factors[factor_id] = PointReprojectionFactorInput(
                provider,
                tuple(
                    item
                    for item in measurements
                    if isinstance(item, Point2DMeasurement)
                    and (not point_roles or item.meta.feature.semantic_role in point_roles)
                ),
                cameras,
            )
        elif residual_ref == "shadow_residual::metric_depth":
            depth_factors[factor_id] = MetricDepthFactorInput(
                tuple(
                    item
                    for item in measurements
                    if isinstance(item, MetricDepthMeasurement)
                    and (not depth_roles or item.meta.feature.semantic_role in depth_roles)
                ),
                target_by_frame=depth_targets,
            )
        elif residual_ref == "shadow_residual::support_and_penetration":
            support_config = config.get("support_plane", {})
            if not isinstance(support_config, Mapping) or support_config.get("source") != "gvhmr_foot_sites":
                raise ValueError("support factor requires a configured generic support-plane source")
            support_feature_ids = tuple(str(value) for value in descriptor.get("support_features", ()))
            if not support_feature_ids:
                raise ValueError("support factor requires asset-declared support features")
            active_frames = tuple(
                frame
                for interval in record.get("activation_intervals", ())
                if isinstance(interval, Mapping) and interval.get("status") == "active"
                for frame in range(int(interval["start_frame"]), int(interval["end_frame"]) + 1)
            )
            runtime = profile.data.get("factor_runtime", {}).get("support_and_penetration", {})
            if not isinstance(runtime, Mapping):
                raise ValueError("support factor requires runtime configuration")
            support_factors[factor_id] = SupportPlaneFactorInput(
                provider,
                support_feature_ids,
                active_frames,
                _support_plane_from_human_foot_sites(
                    gvhmr_sites.measurements,
                    surface_offset_m=float(support_config.get("human_site_surface_offset_m", 0.0)),
                ),
                support_weight=float(runtime.get("weight", 1.0)),
                penetration_weight=float(support_config.get("penetration_weight", runtime.get("weight", 1.0))),
                sigma_m=float(runtime.get("sigma", 1.0)),
            )
    factor_inputs = SequenceFactorInputs(
        state_scales=_state_scales(records, sum(dof.dimension for dof in adaptation.state_spec.dofs)),
        contact_factors=contact_factors,
        periodic_phase_factors=phase_factors,
        line_reprojection_factors=line_factors,
        point_reprojection_factors=point_factors,
        metric_depth_factors=depth_factors,
        support_plane_factors=support_factors,
    )
    preparation = SequenceProblemFactory().prepare(
        attempt_id=str(shadow["attempt_ledger"]["attempt_id"]),
        sequence_contract_sha256=str(shadow["sequence_problem_contract"]["canonical_sha256"]),
        state_spec=adaptation.state_spec,
        initial_states=initial_states,
        residual_execution_plan=shadow["residual_execution_plan"],
        factor_inputs=factor_inputs,
        contact_constraints=constraints,
        human_sites=gvhmr_sites.measurements,
        contact_initialization_mode="seed",
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
        initializer_ledger=initializer_ledger,
        gvhmr_sites=gvhmr_sites,
        measurement_count=len(measurements),
        contact_constraint_count=len(constraints),
        factor_arbitration=factor_arbitration,
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
        "initializer_ledger": prepared.initializer_ledger,
        "read_only_human_sites": {"measurement_count": len(prepared.gvhmr_sites.measurements), "source_artifact": prepared.gvhmr_sites.source_artifact, "source_sha256": prepared.gvhmr_sites.source_sha256, "read_only": True},
        "measurement_count": prepared.measurement_count,
        "contact_constraint_count": prepared.contact_constraint_count,
        "vlm_factor_arbitration": factor_arbitration_ledger_record(prepared.factor_arbitration),
        "case_dispatch_used": False,
        "baseline_pose_read": False,
        "human_state_optimized": False,
        "accepted_outputs_written": False,
    }
