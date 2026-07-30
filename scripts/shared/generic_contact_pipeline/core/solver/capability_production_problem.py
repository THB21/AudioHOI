"""Capability-driven adapters from current artifacts to one generic object problem."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..base.config import CaseProfile
from ..base.schema import resolve_contact_artifact
from ..contact_constraints import ContactConstraint, ContactMode, ContactState, LineS, LocalXYZ, adapt_contact_event_rows, adapt_contact_state_rows, adapt_legacy_contact_rows
from ..human_sites import GVHMRSiteExtractionResult, HumanSiteMeasurement, extract_gvhmr_site_measurements
from ..factors import FactorArbitrationLedger, build_factor_arbitration_ledger, factor_arbitration_ledger_record
from ..gates import load_factor_arbitration_ledger
from ..measurements import Line2DMeasurement, Mask2DMeasurement, MetricDepthMeasurement, Point2DMeasurement, VisibilityMeasurement, adapt_configured_supplemental_measurements, adapt_legacy_observation_rows
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
    build_rigid_geometry_from_asset_descriptor,
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
from .residual_inputs import ContactFactorInput, LineReprojectionFactorInput, MaskSilhouetteFactorInput, MetricDepthFactorInput, PeriodicPhaseFactorInput, PointReprojectionFactorInput, SupportPlaneFactorInput, WorldSpaceContactSample


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _gate_feature_points_by_human_site(
    measurements: Sequence[object],
    human_sites: Sequence[HumanSiteMeasurement],
    cameras: Mapping[int, PinholeCamera],
    raw_gates: Mapping[str, object],
) -> tuple[list[object], dict[str, object]]:
    """Reject part detections inconsistent with a read-only interacting site."""

    masks_by_role_frame = {
        (measurement.meta.feature.semantic_role, measurement.meta.frame): measurement
        for measurement in measurements
        if isinstance(measurement, Mask2DMeasurement)
    }
    sites_by_frame: dict[int, list[HumanSiteMeasurement]] = {}
    for site in human_sites:
        sites_by_frame.setdefault(site.frame, []).append(site)
    kept: list[object] = []
    rejected_ids: list[str] = []
    for measurement in measurements:
        if not isinstance(measurement, Point2DMeasurement):
            kept.append(measurement)
            continue
        raw_gate = raw_gates.get(measurement.meta.feature.semantic_role)
        if not isinstance(raw_gate, Mapping):
            kept.append(measurement)
            continue
        frame = measurement.meta.frame
        mask = masks_by_role_frame.get((str(raw_gate["body_mask_role"]), frame))
        camera = cameras.get(frame)
        candidates = [
            site for site in sites_by_frame.get(frame, ())
            if site.site.body_part == str(raw_gate["human_body_part"])
        ]
        if mask is None or camera is None or not candidates:
            kept.append(measurement)
            continue
        projected_sites = camera.project([site.xyz_m for site in candidates])
        distance_px = float(np.min(np.linalg.norm(projected_sites - np.asarray((measurement.u, measurement.v)), axis=1)))
        x1, y1, x2, y2 = mask.bbox_xyxy
        extent_px = min(float(x2 - x1), float(y2 - y1))
        limit_px = float(raw_gate["max_distance_min_bbox_extent_ratio"]) * extent_px
        if extent_px > 0.0 and distance_px > limit_px:
            rejected_ids.append(measurement.meta.measurement_id)
            continue
        kept.append(measurement)
    return kept, {
        "gate": "human_site_proximity_to_feature_measurement",
        "input_point_count": sum(isinstance(item, Point2DMeasurement) for item in measurements),
        "rejected_point_count": len(rejected_ids),
        "rejected_measurement_ids": rejected_ids,
        "human_state_optimized": False,
        "case_dispatch_used": False,
    }


def _enrich_mask_shape_measurements(
    measurements: Sequence[object],
    *,
    sample_dir: Path,
    raw_config: Mapping[str, object],
) -> tuple[list[object], dict[str, object]]:
    """Attach orientation statistics derived from the declared binary mask artifact."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - production runtime dependency
        raise RuntimeError("mask shape measurements require OpenCV in the solver runtime") from exc
    semantic_role = str(raw_config["semantic_role"])
    artifact_pattern = str(raw_config["artifact_pattern"])
    minimum_pixels = int(raw_config.get("minimum_pixels", 16))
    enriched: list[object] = []
    artifacts: list[str] = []
    for item in measurements:
        if not isinstance(item, Mask2DMeasurement) or item.meta.feature.semantic_role != semantic_role:
            enriched.append(item)
            continue
        relative = Path(artifact_pattern.format(frame=item.meta.frame))
        path = relative if relative.is_absolute() else sample_dir / relative
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"missing configured mask shape artifact: {path}")
        rows, cols = np.where(image > 0)
        if len(rows) < minimum_pixels:
            raise ValueError(f"mask shape artifact has too few foreground pixels: {path}")
        centered = np.column_stack((cols - np.mean(cols), rows - np.mean(rows))).astype(float)
        covariance = centered.T @ centered / len(centered)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)
        axis = eigenvectors[:, int(order[-1])]
        axis /= np.linalg.norm(axis)
        enriched.append(
            replace(
                item,
                mask_artifact=str(path),
                principal_axis_uv=(float(axis[0]), float(axis[1])),
                principal_variances_px2=(
                    float(eigenvalues[int(order[0])]),
                    float(eigenvalues[int(order[-1])]),
                ),
            )
        )
        artifacts.append(str(path))
    return enriched, {
        "adapter": "binary_mask_principal_axis",
        "semantic_role": semantic_role,
        "enriched_measurement_count": len(artifacts),
        "artifact_pattern": artifact_pattern,
        "case_dispatch_used": False,
    }
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
    # Camera-meter coordinates follow image axes, so +Y points downward.
    # The support-plane normal must point into free space (upward), otherwise
    # carried objects are classified as penetrating the floor.
    if normal[1] > 0.0:
        normal = -normal
    normal = normal / np.linalg.norm(normal)
    # Read-only foot joints sit above the physical floor by the configured
    # surface offset, hence they have positive signed distance.
    offset = -float(normal @ center) + float(surface_offset_m)
    return PlaneSurface(tuple(float(value) for value in normal), offset)


def _gravity_plane_through_initial_support(
    geometry_provider: object,
    object_states: Mapping[int, Sequence[float]],
    support_feature_ids: Sequence[str],
    active_frames: Sequence[int],
    normal: Sequence[float],
) -> PlaneSurface:
    """Anchor a gravity-aligned plane at the first observed support contact.

    The plane normal is scene calibration, while its offset comes from the
    observed support transition and asset-declared support geometry.  Using a
    ring/patch instead of one point also constrains a resting rigid object's
    support face to be parallel to the plane.
    """

    if not active_frames:
        raise ValueError("gravity support-plane fitting requires active support frames")
    frame = int(active_frames[0])
    state = object_states.get(frame)
    if state is None:
        raise ValueError("gravity support-plane fitting is missing the first active object state")
    normal_array = np.asarray(normal, dtype=float)
    if normal_array.shape != (3,) or not np.isfinite(normal_array).all() or np.linalg.norm(normal_array) <= 1e-12:
        raise ValueError("gravity support-plane normal must be a finite nonzero three-vector")
    normal_array /= np.linalg.norm(normal_array)
    points = np.concatenate(
        [geometry_provider.feature_points_world(state, feature_id) for feature_id in support_feature_ids],
        axis=0,
    )
    center = np.mean(points, axis=0)
    return PlaneSurface(
        tuple(float(value) for value in normal_array),
        -float(normal_array @ center),
    )


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
                        contact_track_id=(
                            f"human:{constraint.human_site.body_part}:{constraint.human_site.side}"
                            "->object:line:axis"
                        ),
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
                contact_track_id=(
                    f"human:{state.human_site.body_part}:{state.human_site.side}->object:surface"
                ),
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
                f"human:{event.human_site.body_part}:{event.human_site.side}->object:{target_feature_id}",
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
    mask_shape_ledger: dict[str, object] | None = None
    raw_mask_shape = config.get("mask_shape_observations", {})
    if raw_mask_shape:
        if not isinstance(raw_mask_shape, Mapping):
            raise ValueError("mask_shape_observations must be a mapping")
        measurements, mask_shape_ledger = _enrich_mask_shape_measurements(
            measurements,
            sample_dir=profile.sample_dir,
            raw_config=raw_mask_shape,
        )
    contact_path = resolve_contact_artifact(profile, result_dir)
    contact_rows = _rows(contact_path)
    contact_feature_overrides = descriptor.get("contact_feature_overrides", {})
    if contact_feature_overrides:
        if not isinstance(contact_feature_overrides, Mapping):
            raise ValueError("asset contact_feature_overrides must be a mapping")
        descriptor_points = descriptor.get("feature_points", {})
        overridden_rows: list[dict[str, str]] = []
        for source_row in contact_rows:
            row = dict(source_row)
            raw_override = contact_feature_overrides.get(row.get("object_part", ""))
            if raw_override is not None:
                override = ({"geometry_feature_id": raw_override, "coordinate_feature_id": raw_override} if isinstance(raw_override, str) else dict(raw_override))
                feature_id = str(override["geometry_feature_id"])
                coordinate_feature_id = str(override["coordinate_feature_id"])
                points = np.asarray(dict(descriptor_points).get(coordinate_feature_id, ()), dtype=float)
                if points.shape != (1, 3):
                    raise ValueError("contact feature override must resolve to one descriptor-declared fixed point")
                row["geometry_feature_id"] = feature_id
                row["stable_local_x"], row["stable_local_y"], row["stable_local_z"] = (
                    f"{float(value):.9f}" for value in points[0]
                )
                row["source"] = (
                    row.get("source", "")
                    + f"+asset_contact_feature_override:{feature_id}"
                ).lstrip("+")
            overridden_rows.append(row)
        contact_rows = overridden_rows
    visibility_by_feature_frame = {
        (measurement.meta.feature.geometry_feature_id, measurement.meta.frame): measurement.state
        for measurement in measurements
        if isinstance(measurement, VisibilityMeasurement)
    }
    contact_visibility_features = descriptor.get("contact_visibility_features", {})
    if contact_visibility_features:
        if not isinstance(contact_visibility_features, Mapping):
            raise ValueError("asset contact_visibility_features must be a mapping")
        gated_contact_rows: list[dict[str, str]] = []
        for source_row in contact_rows:
            row = dict(source_row)
            visibility_feature = contact_visibility_features.get(row.get("object_part", ""))
            visibility = visibility_by_feature_frame.get(
                (str(visibility_feature), int(row["frame"]))
            )
            if visibility in {"occluded", "absent"} and row.get("contact_active") == "1":
                row["visibility"] = "hidden"
                row["anchor_update"] = "0"
                row["keep_previous"] = "1"
                row["source"] = (
                    row.get("source", "")
                    + f"+semantic_visibility_gate:{visibility_feature}"
                ).lstrip("+")
            gated_contact_rows.append(row)
        contact_rows = gated_contact_rows
    constraints = adapt_legacy_contact_rows(profile.case_name, contact_rows, str(contact_path)).constraints
    gvhmr_sites: GVHMRSiteExtractionResult | None = None
    initializer_ledger: Mapping[str, object] = {
        "initializer_kind": initializer,
        "case_dispatch_used": False,
        "baseline_pose_read": False,
        "human_state_optimized": False,
    }
    if initializer in {"articulated_correspondence", "fixed_assembly_correspondence", "axial_rigid_feature_correspondence"}:
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
        geometry_build = (
            build_rigid_geometry_from_asset_descriptor(
                descriptor_path=descriptor_path,
                repository_root=repository_root,
                state_spec=contract.state_spec,
                contact_constraints=constraints,
            )
            if initializer == "axial_rigid_feature_correspondence"
            else build_articulated_geometry_from_asset_descriptor(
                descriptor_path=descriptor_path,
                repository_root=repository_root,
                result_dir=result_dir,
                state_spec=contract.state_spec,
                contact_constraints=constraints,
            )
        )
        cameras = {frame: PinholeCamera(**profile.camera) for frame in sorted(frame_times)}
        measurement_gate_ledger: dict[str, object] | None = None
        raw_measurement_gates = config.get("measurement_human_site_gates", {})
        if raw_measurement_gates:
            if not isinstance(raw_measurement_gates, Mapping):
                raise ValueError("measurement_human_site_gates must be a mapping")
            measurements, measurement_gate_ledger = _gate_feature_points_by_human_site(
                measurements,
                gvhmr_sites.measurements,
                cameras,
                raw_measurement_gates,
            )
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
            schema=f"{initializer}_v1",
            state_spec=contract.state_spec,
            geometry=contract.geometry,
            mapped_fields=tuple(field for dof in contract.state_spec.dofs for field in dof.source_fields),
            unmapped_nonempty_fields=(),
        )
        initial_states = dict(initialized.states_by_frame)
        templates = [dict(row) for row in initialized.template_rows]
        provider = geometry_build.provider
        initializer_ledger = dict(initialized.hypothesis_ledger)
        if mask_shape_ledger is not None:
            initializer_ledger["mask_shape_observations"] = mask_shape_ledger
        if measurement_gate_ledger is not None:
            initializer_ledger["measurement_human_site_gate"] = measurement_gate_ledger
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
    contact_relative_velocity_factors: dict[str, ContactFactorInput] = {}
    contact_twist_gauge_factors: dict[str, ContactFactorInput] = {}
    phase_factors: dict[str, PeriodicPhaseFactorInput] = {}
    line_factors: dict[str, LineReprojectionFactorInput] = {}
    point_factors: dict[str, PointReprojectionFactorInput] = {}
    mask_factors: dict[str, MaskSilhouetteFactorInput] = {}
    depth_factors: dict[str, MetricDepthFactorInput] = {}
    support_factors: dict[str, SupportPlaneFactorInput] = {}
    cameras = {frame: PinholeCamera(**profile.camera) for frame in initial_states}
    line_measurements = tuple(item for item in measurements if isinstance(item, Line2DMeasurement))
    measurement_roles = config.get("measurement_roles", {})
    point_roles = set(measurement_roles.get("point_reprojection", ())) if isinstance(measurement_roles, Mapping) else set()
    mask_roles = set(measurement_roles.get("mask_silhouette", ())) if isinstance(measurement_roles, Mapping) else set()
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
        elif residual_ref == "shadow_residual::contact_relative_velocity":
            contact_relative_velocity_factors[factor_id] = ContactFactorInput(
                provider,
                samples,
                None,
                residual_axes=tuple(int(axis) for axis in config.get("contact_residual_axes", (0, 1, 2))),
            )
        elif residual_ref == "shadow_residual::contact_twist_gauge":
            contact_twist_gauge_factors[factor_id] = ContactFactorInput(provider, samples, None)
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
        elif residual_ref == "shadow_residual::mask_silhouette":
            mask_shape_config = config.get("mask_shape_observations", {})
            mask_factors[factor_id] = MaskSilhouetteFactorInput(
                provider,
                tuple(
                    item
                    for item in measurements
                    if isinstance(item, Mask2DMeasurement)
                    and (not mask_roles or item.meta.feature.semantic_role in mask_roles)
                ),
                cameras,
                (
                    float(mask_shape_config["principal_axis_sigma_rad"])
                    if isinstance(mask_shape_config, Mapping)
                    and mask_shape_config.get("principal_axis_sigma_rad") is not None
                    else None
                ),
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
            if not isinstance(support_config, Mapping) or support_config.get("source") not in {
                "gvhmr_foot_sites",
                "gravity_plane_through_initial_support",
            }:
                raise ValueError("support factor requires a configured generic support-plane source")
            support_feature_ids = tuple(str(value) for value in descriptor.get("support_features", ()))
            if not support_feature_ids:
                raise ValueError("support factor requires asset-declared support features")
            runtime = profile.data.get("factor_runtime", {}).get("support_and_penetration", {})
            if not isinstance(runtime, Mapping):
                raise ValueError("support factor requires runtime configuration")
            raw_tiers = runtime.get(
                "activation_weight_tiers",
                {"active": 1.0, "downweighted": 1.0, "inactive": 0.0},
            )
            if not isinstance(raw_tiers, Mapping):
                raise ValueError("support factor activation tiers must be a mapping")
            status_by_frame: dict[int, str] = {}
            weight_by_frame: dict[int, float] = {}
            for interval in record.get("activation_intervals", ()):
                if not isinstance(interval, Mapping):
                    continue
                status = str(interval["status"])
                tier_weight = float(raw_tiers[status])
                for frame in range(int(interval["start_frame"]), int(interval["end_frame"]) + 1):
                    status_by_frame[frame] = status
                    weight_by_frame[frame] = tier_weight
            active_frames = tuple(frame for frame in sorted(status_by_frame) if weight_by_frame[frame] > 0.0)
            plane = (
                _support_plane_from_human_foot_sites(
                    gvhmr_sites.measurements,
                    surface_offset_m=float(support_config.get("human_site_surface_offset_m", 0.0)),
                )
                if support_config.get("source") == "gvhmr_foot_sites"
                else _gravity_plane_through_initial_support(
                    provider,
                    initial_states,
                    support_feature_ids,
                    tuple(frame for frame in active_frames if status_by_frame.get(frame) == "active"),
                    tuple(float(value) for value in support_config.get("normal_camera", (0.0, -1.0, 0.0))),
                )
            )
            proximity_gate_m = (
                None
                if support_config.get("proximity_gate_m") is None
                else float(support_config["proximity_gate_m"])
            )
            support_factors[factor_id] = SupportPlaneFactorInput(
                provider,
                support_feature_ids,
                active_frames,
                plane,
                support_weight=float(runtime.get("weight", 1.0)),
                penetration_weight=float(support_config.get("penetration_weight", runtime.get("weight", 1.0))),
                sigma_m=float(runtime.get("sigma", 1.0)),
                activation_status_by_frame=status_by_frame,
                activation_weight_by_frame=weight_by_frame,
                proximity_gate_m=proximity_gate_m,
                tangent_gauge_weight=float(support_config.get("tangent_gauge_weight", 0.0)),
                tangent_gauge_sigma_rad=float(support_config.get("tangent_gauge_sigma_rad", 1.0)),
            )
    factor_inputs = SequenceFactorInputs(
        state_scales=_state_scales(records, sum(dof.dimension for dof in adaptation.state_spec.dofs)),
        contact_factors=contact_factors,
        contact_relative_velocity_factors=contact_relative_velocity_factors,
        contact_twist_gauge_factors=contact_twist_gauge_factors,
        periodic_phase_factors=phase_factors,
        line_reprojection_factors=line_factors,
        point_reprojection_factors=point_factors,
        mask_silhouette_factors=mask_factors,
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
    configured_contact_name = str(config.get("contact_artifact", "object_contact_points.csv"))
    initializer_inputs = [
        (
            contact_path
            if str(path) == configured_contact_name
            else result_dir / str(path)
        )
        for path in config.get("initializer_artifacts", ())
    ]
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
