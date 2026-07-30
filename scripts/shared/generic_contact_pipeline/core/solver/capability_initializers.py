"""Observation-derived initializers selected only by declared capabilities."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..contact_constraints import ContactConstraint
from ..human_sites import HumanSiteMeasurement
from ..measurements import Line2DMeasurement, Mask2DMeasurement, Measurement, MetricDepthMeasurement, Point2DMeasurement
from ..state import DofKind, PinholeCamera, StateSpec


@dataclass(frozen=True)
class InitializationRequest:
    state_spec: StateSpec
    geometry_provider: object
    measurements: tuple[Measurement, ...]
    contact_constraints: tuple[ContactConstraint, ...]
    human_sites: tuple[HumanSiteMeasurement, ...]
    cameras: Mapping[int, PinholeCamera]
    initializer: Mapping[str, object]
    default_state_by_dof: Mapping[str, tuple[float, ...]]


@dataclass(frozen=True)
class InitializationResult:
    states_by_frame: Mapping[int, tuple[float, ...]]
    template_rows: tuple[Mapping[str, object], ...]
    hypothesis_ledger: Mapping[str, object]
    input_artifact_ids: tuple[str, ...]


def _layout(state_spec: StateSpec) -> dict[str, slice]:
    result: dict[str, slice] = {}
    offset = 0
    for dof in state_spec.dofs:
        result[dof.dof_id] = slice(offset, offset + dof.dimension)
        offset += dof.dimension
    return result


def _base_state(request: InitializationRequest) -> np.ndarray:
    width = sum(dof.dimension for dof in request.state_spec.dofs)
    state = np.zeros(width, dtype=float)
    layout = _layout(request.state_spec)
    translation = layout.get("root.translation")
    rotation = layout.get("root.rotation")
    if translation is None or rotation is None:
        raise ValueError("capability initializer requires root translation and rotation DOFs")
    state[translation] = (0.0, 0.0, 1.0)
    state[rotation] = (1.0, 0.0, 0.0, 0.0)
    for dof_id, values in request.default_state_by_dof.items():
        target = layout.get(dof_id)
        if target is None or target.stop - target.start != len(values):
            raise ValueError(f"initializer default does not match StateSpec: {dof_id}")
        state[target] = values
    return state


def _joint_hypotheses(request: InitializationRequest, base: np.ndarray) -> tuple[np.ndarray, ...]:
    layout = _layout(request.state_spec)
    candidates: list[tuple[float, ...]] = []
    joint_ids: list[str] = []
    for dof in request.state_spec.dofs:
        if dof.kind not in {DofKind.REVOLUTE, DofKind.PRISMATIC}:
            continue
        joint_ids.append(dof.dof_id)
        default = float(base[layout[dof.dof_id]][0])
        values = [default]
        if (
            dof.observable
            and dof.bound is not None
            and isinstance(dof.bound.lower, (float, int))
            and isinstance(dof.bound.upper, (float, int))
        ):
            lower, upper = float(dof.bound.lower), float(dof.bound.upper)
            values.extend((lower, 0.5 * (lower + upper), upper))
        candidates.append(tuple(dict.fromkeys(round(value, 12) for value in values)))
    if not joint_ids:
        return (base.copy(),)
    states: list[np.ndarray] = []
    for values in itertools.product(*candidates):
        state = base.copy()
        for dof_id, value in zip(joint_ids, values):
            state[layout[dof_id]] = value
        states.append(state)
    return tuple(states)


def _camera_matrix(camera: PinholeCamera) -> np.ndarray:
    return np.asarray(
        [[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _state_from_pnp(
    request: InitializationRequest,
    hypothesis: np.ndarray,
    lines: Sequence[Line2DMeasurement],
    camera: PinholeCamera,
    point_measurements: Sequence[Point2DMeasurement] = (),
) -> tuple[np.ndarray, float] | None:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - production runtime dependency
        raise RuntimeError("articulated_correspondence requires OpenCV in the solver runtime") from exc
    try:
        from scipy.spatial.transform import Rotation
    except ImportError as exc:  # pragma: no cover - production runtime dependency
        raise RuntimeError("articulated_correspondence requires SciPy in the solver runtime") from exc
    local_points: list[np.ndarray] = []
    image_points: list[tuple[float, float]] = []
    local_state = hypothesis.copy()
    local_state[_layout(request.state_spec)["root.translation"]] = 0.0
    local_state[_layout(request.state_spec)["root.rotation"]] = (1.0, 0.0, 0.0, 0.0)
    feature_points_world = getattr(request.geometry_provider, "feature_points_world", None)
    if feature_points_world is None:
        raise ValueError("articulated_correspondence requires feature_points_world geometry capability")
    for measurement in lines:
        feature_points = np.asarray(
            feature_points_world(local_state, measurement.meta.feature.geometry_feature_id),
            dtype=np.float64,
        )
        if feature_points.shape[0] < 2:
            continue
        local_points.extend((feature_points[0], feature_points[-1]))
        image_points.extend((measurement.start_uv, measurement.end_uv))
    if len(local_points) < 6:
        return None
    object_array = np.asarray(local_points, dtype=np.float64)
    image_array = np.asarray(image_points, dtype=np.float64)
    success, rvec, tvec = cv2.solvePnP(
        object_array,
        image_array,
        _camera_matrix(camera),
        None,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success:
        return None
    success, rvec, tvec = cv2.solvePnP(
        object_array,
        image_array,
        _camera_matrix(camera),
        None,
        rvec,
        tvec,
        True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
        return None
    if str(request.initializer.get("line_constraint_mode", "endpoints")) == "axis_line":
        from scipy.optimize import least_squares

        line_local: list[np.ndarray] = []
        line_targets: list[tuple[np.ndarray, np.ndarray]] = []
        for measurement in lines:
            feature = np.asarray(
                feature_points_world(local_state, measurement.meta.feature.geometry_feature_id),
                dtype=np.float64,
            )
            if feature.shape[0] < 2:
                continue
            line_local.extend((feature[0], feature[-1]))
            line_targets.append(
                (
                    np.asarray(measurement.start_uv, dtype=float),
                    np.asarray(measurement.end_uv, dtype=float),
                )
            )
        point_local: list[np.ndarray] = []
        point_targets: list[np.ndarray] = []
        for measurement in point_measurements:
            feature = np.asarray(
                feature_points_world(local_state, measurement.meta.feature.geometry_feature_id),
                dtype=np.float64,
            )
            if feature.shape != (1, 3):
                continue
            point_local.append(feature[0])
            point_targets.append(np.asarray((measurement.u, measurement.v), dtype=float))

        def axis_line_residual(parameters: np.ndarray) -> np.ndarray:
            trial_rvec = parameters[:3].reshape(3, 1)
            trial_tvec = parameters[3:].reshape(3, 1)
            residuals: list[float] = []
            cursor = 0
            for start_uv, end_uv in line_targets:
                local_pair = np.asarray(line_local[cursor:cursor + 2], dtype=np.float64)
                cursor += 2
                projected_pair, _ = cv2.projectPoints(
                    local_pair, trial_rvec, trial_tvec, _camera_matrix(camera), None
                )
                direction = end_uv - start_uv
                norm = float(np.linalg.norm(direction))
                if norm <= 1e-6:
                    residuals.extend((1e4, 1e4))
                    continue
                for projected_uv in projected_pair.reshape(-1, 2):
                    residuals.append(float(np.cross(direction, projected_uv - start_uv) / norm))
            if point_local:
                projected_points, _ = cv2.projectPoints(
                    np.asarray(point_local, dtype=np.float64),
                    trial_rvec,
                    trial_tvec,
                    _camera_matrix(camera),
                    None,
                )
                for projected_uv, target_uv in zip(projected_points.reshape(-1, 2), point_targets):
                    residuals.extend((float(projected_uv[0] - target_uv[0]), float(projected_uv[1] - target_uv[1])))
            return np.asarray(residuals, dtype=float)

        refined = least_squares(
            axis_line_residual,
            np.concatenate((rvec.reshape(3), tvec.reshape(3))),
            method="trf",
            loss="soft_l1",
            f_scale=7.5,
            max_nfev=30,
        )
        rvec = refined.x[:3].reshape(3, 1)
        tvec = refined.x[3:].reshape(3, 1)
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    camera_points = object_array @ rotation_matrix.T + tvec.reshape(1, 3)
    if float(np.min(camera_points[:, 2])) <= 1e-4:
        return None
    projected, _ = cv2.projectPoints(object_array, rvec, tvec, _camera_matrix(camera), None)
    if str(request.initializer.get("line_constraint_mode", "endpoints")) == "axis_line":
        residual_values = axis_line_residual(np.concatenate((rvec.reshape(3), tvec.reshape(3))))
        error = float(np.sqrt(np.mean(residual_values ** 2)))
    else:
        error = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image_array) ** 2, axis=1))))
    qx, qy, qz, qw = Rotation.from_matrix(rotation_matrix).as_quat()
    state = hypothesis.copy()
    layout = _layout(request.state_spec)
    state[layout["root.translation"]] = tvec.reshape(3)
    state[layout["root.rotation"]] = (qw, qx, qy, qz)
    return state, error


def _interpolate_states(
    frames: tuple[int, ...],
    solved: Mapping[int, np.ndarray],
    state_spec: StateSpec,
) -> dict[int, np.ndarray]:
    try:
        from scipy.spatial.transform import Rotation, Slerp
    except ImportError as exc:  # pragma: no cover - production runtime dependency
        raise RuntimeError("capability state interpolation requires SciPy in the solver runtime") from exc
    if not solved:
        raise ValueError("articulated_correspondence has insufficient typed evidence")
    known = np.asarray(sorted(solved), dtype=float)
    layout = _layout(state_spec)
    rotation_slice = layout["root.rotation"]
    rotations = Rotation.from_quat(
        np.asarray(
            [
                [solved[int(frame)][rotation_slice][1], solved[int(frame)][rotation_slice][2], solved[int(frame)][rotation_slice][3], solved[int(frame)][rotation_slice][0]]
                for frame in known
            ]
        )
    )
    slerp = Slerp(known, rotations) if len(known) > 1 else None
    result: dict[int, np.ndarray] = {}
    for frame in frames:
        if frame in solved:
            result[frame] = solved[frame].copy()
            continue
        query = float(np.clip(frame, known[0], known[-1]))
        vector = np.empty_like(next(iter(solved.values())))
        for index in range(len(vector)):
            if rotation_slice.start <= index < rotation_slice.stop:
                continue
            vector[index] = float(np.interp(query, known, [solved[int(item)][index] for item in known]))
        if slerp is None:
            quaternion = next(iter(solved.values()))[rotation_slice]
        else:
            qx, qy, qz, qw = slerp([query]).as_quat()[0]
            quaternion = np.asarray((qw, qx, qy, qz), dtype=float)
        vector[rotation_slice] = quaternion
        result[frame] = vector
    return result


def _axis_rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 1e-12:
        raise ValueError("axial rigid initializer requires a finite nonzero local axis")
    vector /= np.linalg.norm(vector)
    x, y, z = vector
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=float)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _axial_body_points(axis: Sequence[float], radius_m: float, height_m: float, samples: int = 72) -> np.ndarray:
    axis_vector = np.asarray(axis, dtype=float)
    axis_vector /= np.linalg.norm(axis_vector)
    helper = np.asarray((1.0, 0.0, 0.0) if abs(axis_vector[0]) < 0.8 else (0.0, 0.0, 1.0))
    first = np.cross(axis_vector, helper)
    first /= np.linalg.norm(first)
    second = np.cross(axis_vector, first)
    angles = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    ring = radius_m * (np.cos(angles)[:, None] * first + np.sin(angles)[:, None] * second)
    return np.vstack(
        (
            ring - 0.5 * height_m * axis_vector,
            ring + 0.5 * height_m * axis_vector,
            np.zeros((1, 3), dtype=float),
        )
    )


def _project_rigid_points(
    camera: PinholeCamera,
    local_points: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
    scale: float,
) -> np.ndarray:
    world = local_points @ (float(scale) * rotation).T + translation
    if np.any(world[:, 2] <= 1e-6):
        return np.full((len(world), 2), 1e6, dtype=float)
    return camera.project(world)


def _wrap_pi(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _axial_rigid_feature_correspondence(request: InitializationRequest) -> InitializationResult:
    try:
        from scipy.ndimage import gaussian_filter1d
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
    except ImportError as exc:  # pragma: no cover - production runtime dependency
        raise RuntimeError("axial rigid initialization requires SciPy in the solver runtime") from exc

    initializer = request.initializer
    center_role = str(initializer["body_center_role"])
    mask_role = str(initializer["body_mask_role"])
    depth_role = str(initializer["depth_role"])
    feature_role = str(initializer["off_axis_feature_role"])
    axis = np.asarray(initializer["axis_local"], dtype=float)
    radius_m = float(initializer["body_radius_m"])
    height_m = float(initializer["body_height_m"])
    if radius_m <= 0.0 or height_m <= 0.0:
        raise ValueError("axial rigid body proxy dimensions must be positive")
    body_points = _axial_body_points(axis, radius_m, height_m)
    centers = {
        item.meta.frame: item
        for item in request.measurements
        if isinstance(item, Point2DMeasurement) and item.meta.feature.semantic_role == center_role
    }
    masks = {
        item.meta.frame: item
        for item in request.measurements
        if isinstance(item, Mask2DMeasurement) and item.meta.feature.semantic_role == mask_role
    }
    depths = {
        item.meta.frame: item
        for item in request.measurements
        if isinstance(item, MetricDepthMeasurement) and item.meta.feature.semantic_role == depth_role
    }
    feature_measurements = {
        item.meta.frame: item
        for item in request.measurements
        if isinstance(item, Point2DMeasurement) and item.meta.feature.semantic_role == feature_role
    }
    frames = tuple(sorted(request.cameras))
    if not frames or any(frame not in centers or frame not in masks or frame not in depths for frame in frames):
        raise ValueError("axial rigid initializer requires frame-complete center, mask, and depth evidence")
    provider_points = getattr(request.geometry_provider, "feature_points_local", None)
    if not isinstance(provider_points, Mapping):
        raise ValueError("axial rigid initializer requires descriptor-backed rigid feature points")
    feature_id = next(iter(feature_measurements.values())).meta.feature.geometry_feature_id if feature_measurements else ""
    local_feature = np.asarray(provider_points.get(feature_id), dtype=float)
    if local_feature.shape != (1, 3):
        raise ValueError("axial rigid off-axis feature must resolve to one fixed local point")

    scale_dof = next((dof for dof in request.state_spec.dofs if dof.dof_id == "scale"), None)
    if (
        scale_dof is None
        or scale_dof.bound is None
        or not isinstance(scale_dof.bound.lower, (float, int))
        or not isinstance(scale_dof.bound.upper, (float, int))
    ):
        raise ValueError("axial rigid initializer requires finite asset-declared scale bounds")
    scale_lower = float(scale_dof.bound.lower)
    scale_upper = float(scale_dof.bound.upper)
    raw_scale_by_frame = {
        frame: ((masks[frame].bbox_xyxy[3] - masks[frame].bbox_xyxy[1]) * float(depths[frame].depth_m))
        / (request.cameras[frame].fy * height_m)
        for frame in frames
    }
    sequence_scale = float(
        np.clip(np.median(tuple(raw_scale_by_frame.values())), scale_lower, scale_upper)
    )
    scale_prior_sigma = float(initializer.get("scale_prior_sigma", 0.15))
    if scale_prior_sigma <= 0.0:
        raise ValueError("axial rigid initializer scale prior sigma must be positive")
    fit_scale_lower = scale_lower if scale_dof.observable else sequence_scale - 1e-9
    fit_scale_upper = scale_upper if scale_dof.observable else sequence_scale + 1e-9

    body_fits: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    body_errors: dict[int, float] = {}
    artifacts: set[str] = set()
    times: dict[int, float] = {}
    for measurement in request.measurements:
        if measurement.meta.frame in request.cameras:
            artifacts.add(measurement.meta.source.artifact)
            times.setdefault(measurement.meta.frame, measurement.meta.time)
    for frame in frames:
        camera = request.cameras[frame]
        center = centers[frame]
        mask = masks[frame]
        depth = depths[frame]
        z0 = float(depth.depth_m)
        translation0 = np.asarray(
            (
                (center.u - camera.cx) * z0 / camera.fx,
                (center.v - camera.cy) * z0 / camera.fy,
                z0,
            ),
            dtype=float,
        )
        x1, y1, x2, y2 = mask.bbox_xyxy
        scale0 = float(np.clip(raw_scale_by_frame[frame], scale_lower, scale_upper))
        initial = np.asarray(
            (*translation0, 0.0, 0.0, scale0 if scale_dof.observable else sequence_scale),
            dtype=float,
        )

        def body_residual(values: np.ndarray) -> np.ndarray:
            translation = values[:3]
            tilt = _axis_rotation((1.0, 0.0, 0.0), float(values[3])) @ _axis_rotation((0.0, 0.0, 1.0), float(values[4]))
            projected = _project_rigid_points(camera, body_points, translation, tilt, float(values[5]))
            center_uv = _project_rigid_points(camera, np.zeros((1, 3)), translation, tilt, float(values[5]))[0]
            predicted_bbox = (
                float(np.min(projected[:, 0])),
                float(np.min(projected[:, 1])),
                float(np.max(projected[:, 0])),
                float(np.max(projected[:, 1])),
            )
            residuals = ((center_uv - np.asarray((center.u, center.v))) / 7.0).tolist()
            residuals.extend((np.asarray(predicted_bbox) - np.asarray(mask.bbox_xyxy)) / 10.0)
            residuals.append((float(translation[2]) - z0) / float(depth.sigma_m or 0.55))
            residuals.append((float(values[5]) - sequence_scale) / scale_prior_sigma)
            return np.asarray(residuals, dtype=float)

        fit = least_squares(
            body_residual,
            initial,
            bounds=(
                np.asarray((translation0[0] - 0.45, translation0[1] - 0.45, max(0.2, z0 - 0.9), math.radians(-85.0), math.radians(-85.0), fit_scale_lower)),
                np.asarray((translation0[0] + 0.45, translation0[1] + 0.45, z0 + 0.9, math.radians(80.0), math.radians(85.0), fit_scale_upper)),
            ),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=int(initializer.get("body_max_function_evaluations", 80)),
        )
        tilt = _axis_rotation((1.0, 0.0, 0.0), float(fit.x[3])) @ _axis_rotation((0.0, 0.0, 1.0), float(fit.x[4]))
        body_fits[frame] = (fit.x[:3].copy(), tilt, float(fit.x[5]))
        values = body_residual(fit.x)
        body_errors[frame] = float(np.sqrt(np.mean(values * values)))

    grid = np.linspace(-math.pi, math.pi, int(initializer.get("phase_grid_samples", 721)))
    candidate_count = int(initializer.get("phase_candidate_count", 8))
    sigma_px = float(initializer.get("off_axis_feature_sigma_px", 5.0))
    visible: list[tuple[int, list[tuple[float, float]], float]] = []
    for frame, measurement in sorted(feature_measurements.items()):
        if frame not in body_fits:
            continue
        translation, tilt, scale = body_fits[frame]
        predictions = np.asarray(
            [
                _project_rigid_points(
                    request.cameras[frame],
                    local_feature,
                    translation,
                    tilt @ _axis_rotation(axis, float(angle)),
                    scale,
                )[0]
                for angle in grid
            ]
        )
        target = np.asarray((measurement.u, measurement.v), dtype=float)
        distances = np.linalg.norm(predictions - target[None, :], axis=1)
        minima = [index for index in range(1, len(grid) - 1) if distances[index] <= distances[index - 1] and distances[index] <= distances[index + 1]]
        minima.extend((0, len(grid) - 1, int(np.argmin(distances))))
        selected = sorted(set(minima), key=lambda index: (float(distances[index]), index))[:candidate_count]
        candidates = [(float(grid[index]), float(distances[index])) for index in selected]
        confidence = float(measurement.meta.confidence if measurement.meta.confidence is not None else 1.0)
        visible.append((frame, candidates, max(0.05, confidence)))
    if not visible:
        raise ValueError("axial rigid orientation is unobservable without an off-axis feature measurement")

    costs: list[np.ndarray] = []
    parents: list[np.ndarray] = []
    transition_rate = float(initializer.get("temporal_phase_sigma_rad_per_frame", 0.25))
    for item_index, (frame, candidates, confidence) in enumerate(visible):
        emission = np.asarray([confidence * (distance / sigma_px) ** 2 for _angle, distance in candidates])
        if item_index == 0:
            costs.append(emission)
            parents.append(np.full(len(candidates), -1, dtype=int))
            continue
        previous_frame, previous_candidates, _previous_confidence = visible[item_index - 1]
        transition_sigma = transition_rate * max(1, frame - previous_frame)
        current_cost = np.full(len(candidates), np.inf)
        current_parent = np.full(len(candidates), -1, dtype=int)
        for current_index, (angle, _distance) in enumerate(candidates):
            transitions = np.asarray(
                [
                    costs[-1][previous_index] + (_wrap_pi(angle - previous_angle) / transition_sigma) ** 2
                    for previous_index, (previous_angle, _previous_distance) in enumerate(previous_candidates)
                ]
            )
            parent = int(np.argmin(transitions))
            current_cost[current_index] = emission[current_index] + transitions[parent]
            current_parent[current_index] = parent
        costs.append(current_cost)
        parents.append(current_parent)
    selected_indices = [0] * len(visible)
    selected_indices[-1] = int(np.argmin(costs[-1]))
    for index in range(len(visible) - 1, 0, -1):
        selected_indices[index - 1] = int(parents[index][selected_indices[index]])
    visible_frames = np.asarray([frame for frame, _candidates, _confidence in visible], dtype=float)
    visible_angles = np.unwrap(
        np.asarray([visible[index][1][selected_indices[index]][0] for index in range(len(visible))])
    )
    interpolated = np.interp(np.asarray(frames, dtype=float), visible_frames, visible_angles)
    angles = gaussian_filter1d(
        interpolated,
        sigma=float(initializer.get("phase_smooth_sigma_frames", 1.5)),
        mode="nearest",
    )

    layout = _layout(request.state_spec)
    scale_slice = layout.get("scale")
    if scale_slice is None or scale_slice.stop - scale_slice.start != 1:
        raise ValueError("axial rigid StateSpec requires one scalar scale DOF")
    states: dict[int, tuple[float, ...]] = {}
    templates: list[dict[str, object]] = []
    fields = tuple(field for dof in request.state_spec.dofs for field in dof.source_fields)
    previous_quaternion: np.ndarray | None = None
    for frame, angle in zip(frames, angles):
        translation, tilt, scale = body_fits[frame]
        rotation = tilt @ _axis_rotation(axis, float(angle))
        qx, qy, qz, qw = Rotation.from_matrix(rotation).as_quat()
        quaternion = np.asarray((qw, qx, qy, qz), dtype=float)
        if previous_quaternion is not None and float(previous_quaternion @ quaternion) < 0.0:
            quaternion *= -1.0
        previous_quaternion = quaternion
        state = _base_state(request)
        state[layout["root.translation"]] = translation
        state[layout["root.rotation"]] = quaternion
        state[scale_slice] = scale
        states[frame] = tuple(float(value) for value in state)
        row: dict[str, object] = {
            "frame": frame,
            "time": times.get(frame, (frame - 1) / 24.0),
            "source": "axial_rigid_feature_correspondence_initializer",
        }
        row.update({field: float(value) for field, value in zip(fields, state)})
        templates.append(row)
    feature_errors = [visible[index][1][selected_indices[index]][1] for index in range(len(visible))]
    return InitializationResult(
        states_by_frame=states,
        template_rows=tuple(templates),
        hypothesis_ledger={
            "initializer_kind": "axial_rigid_feature_correspondence",
            "frame_count": len(frames),
            "visible_feature_frame_count": len(visible),
            "hidden_feature_frame_count": len(frames) - len(visible),
            "fabricated_feature_measurement_count": 0,
            "body_residual_rms_median": float(np.median(tuple(body_errors.values()))),
            "body_residual_rms_p90": float(np.quantile(tuple(body_errors.values()), 0.90)),
            "asset_scale_bounds": [scale_lower, scale_upper],
            "sequence_scale_prior": sequence_scale,
            "scale_prior_sigma": scale_prior_sigma,
            "scale_optimized_by_sequence_solver": scale_dof.observable,
            "feature_reprojection_median_px": float(np.median(feature_errors)),
            "feature_reprojection_p90_px": float(np.quantile(feature_errors, 0.90)),
            "baseline_pose_read": False,
            "historical_phase_read": False,
            "case_dispatch_used": False,
            "human_state_optimized": False,
        },
        input_artifact_ids=tuple(sorted(artifacts)),
    )


def _articulated_correspondence(request: InitializationRequest) -> InitializationResult:
    initializer_kind = str(request.initializer.get("kind", "articulated_correspondence"))
    allowed = {str(value) for value in request.initializer.get("line_feature_roles", ())}
    minimum = int(request.initializer.get("minimum_line_count", 2))
    by_frame: dict[int, list[Line2DMeasurement]] = {}
    points_by_frame: dict[int, list[Point2DMeasurement]] = {}
    allowed_point_roles = {str(value) for value in request.initializer.get("point_semantic_roles", ())}
    times: dict[int, float] = {}
    artifacts: set[str] = set()
    for measurement in request.measurements:
        if isinstance(measurement, Point2DMeasurement):
            if not allowed_point_roles or measurement.meta.feature.semantic_role in allowed_point_roles:
                points_by_frame.setdefault(measurement.meta.frame, []).append(measurement)
                artifacts.add(measurement.meta.source.artifact)
            continue
        if not isinstance(measurement, Line2DMeasurement):
            continue
        feature_id = measurement.meta.feature.geometry_feature_id
        if allowed and feature_id not in allowed:
            continue
        by_frame.setdefault(measurement.meta.frame, []).append(measurement)
        times[measurement.meta.frame] = measurement.meta.time
        artifacts.add(measurement.meta.source.artifact)
    frames = tuple(sorted(request.cameras))
    base = _base_state(request)
    hypotheses = _joint_hypotheses(request, base)
    solved: dict[int, np.ndarray] = {}
    selections: list[dict[str, object]] = []
    candidates_by_frame: dict[int, list[tuple[float, np.ndarray, int]]] = {}
    for frame in frames:
        lines = by_frame.get(frame, ())
        if len(lines) < minimum:
            continue
        candidates: list[tuple[float, np.ndarray, int]] = []
        for index, hypothesis in enumerate(hypotheses):
            result = _state_from_pnp(
                request,
                hypothesis,
                lines,
                request.cameras[frame],
                points_by_frame.get(frame, ()),
            )
            if result is not None and math.isfinite(result[1]):
                candidates.append((result[1], result[0], index))
        if not candidates:
            continue
        candidates_by_frame[frame] = candidates
    sequence_constant_joints = any(
        dof.kind in {DofKind.REVOLUTE, DofKind.PRISMATIC} and not dof.observable
        for dof in request.state_spec.dofs
    )
    selected_hypothesis: int | None = None
    if sequence_constant_joints and candidates_by_frame:
        aggregate: list[tuple[int, float, int]] = []
        for hypothesis_index in range(len(hypotheses)):
            scores = [
                score
                for candidates in candidates_by_frame.values()
                for score, _state, index in candidates
                if index == hypothesis_index
            ]
            if scores:
                aggregate.append((-len(scores), float(np.median(scores)), hypothesis_index))
        if aggregate:
            selected_hypothesis = min(aggregate)[2]
    for frame, candidates in sorted(candidates_by_frame.items()):
        eligible = (
            [candidate for candidate in candidates if candidate[2] == selected_hypothesis]
            if selected_hypothesis is not None
            else candidates
        )
        if not eligible:
            continue
        score, state, selected_index = min(eligible, key=lambda value: value[0])
        solved[frame] = state
        selections.append(
            {
                "frame": frame,
                "selected_hypothesis_id": f"joint-grid-{selected_index:03d}",
                "line_measurement_count": len(by_frame.get(frame, ())),
                "reprojection_rmse_px": score,
                "rejected_hypothesis_count": len(candidates) - 1,
                "joint_selection_scope": "sequence_constant" if selected_hypothesis is not None else "per_frame",
            }
        )
    states = _interpolate_states(frames, solved, request.state_spec)
    fields = tuple(field for dof in request.state_spec.dofs for field in dof.source_fields)
    templates: list[dict[str, object]] = []
    for frame in frames:
        state = states[frame]
        row: dict[str, object] = {
            "frame": frame,
            "time": times.get(frame, (frame - 1) / 24.0),
            "source": f"{initializer_kind}_initializer",
        }
        row.update({field: float(value) for field, value in zip(fields, state)})
        templates.append(row)
    return InitializationResult(
        states_by_frame={frame: tuple(float(value) for value in states[frame]) for frame in frames},
        template_rows=tuple(templates),
        hypothesis_ledger={
            "initializer_kind": initializer_kind,
            "frame_count": len(frames),
            "directly_solved_frame_count": len(solved),
            "joint_hypothesis_count": len(hypotheses),
            "selected_sequence_joint_hypothesis_id": (
                None if selected_hypothesis is None else f"joint-grid-{selected_hypothesis:03d}"
            ),
            "selections": selections,
            "case_dispatch_used": False,
            "baseline_pose_read": False,
            "human_state_optimized": False,
        },
        input_artifact_ids=tuple(sorted(artifacts)),
    )


def initialize_from_capabilities(request: InitializationRequest) -> InitializationResult:
    kind = str(request.initializer.get("kind", ""))
    if kind in {"articulated_correspondence", "fixed_assembly_correspondence"}:
        return _articulated_correspondence(request)
    if kind == "axial_rigid_feature_correspondence":
        return _axial_rigid_feature_correspondence(request)
    raise ValueError(f"unsupported capability initializer: {kind}")
