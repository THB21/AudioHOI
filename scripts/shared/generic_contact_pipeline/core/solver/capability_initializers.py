"""Observation-derived initializers selected only by declared capabilities."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..contact_constraints import ContactConstraint
from ..human_sites import HumanSiteMeasurement
from ..measurements import Line2DMeasurement, Measurement
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
        if dof.bound is not None and isinstance(dof.bound.lower, (float, int)) and isinstance(dof.bound.upper, (float, int)):
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
        points = np.asarray(
            feature_points_world(local_state, measurement.meta.feature.geometry_feature_id),
            dtype=np.float64,
        )
        if points.shape[0] < 2:
            continue
        local_points.extend((points[0], points[-1]))
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
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    camera_points = object_array @ rotation_matrix.T + tvec.reshape(1, 3)
    if float(np.min(camera_points[:, 2])) <= 1e-4:
        return None
    projected, _ = cv2.projectPoints(object_array, rvec, tvec, _camera_matrix(camera), None)
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


def _articulated_correspondence(request: InitializationRequest) -> InitializationResult:
    allowed = {str(value) for value in request.initializer.get("line_feature_roles", ())}
    minimum = int(request.initializer.get("minimum_line_count", 2))
    by_frame: dict[int, list[Line2DMeasurement]] = {}
    times: dict[int, float] = {}
    artifacts: set[str] = set()
    for measurement in request.measurements:
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
    for frame in frames:
        lines = by_frame.get(frame, ())
        if len(lines) < minimum:
            continue
        candidates: list[tuple[float, np.ndarray, int]] = []
        for index, hypothesis in enumerate(hypotheses):
            result = _state_from_pnp(request, hypothesis, lines, request.cameras[frame])
            if result is not None and math.isfinite(result[1]):
                candidates.append((result[1], result[0], index))
        if not candidates:
            continue
        score, state, selected_index = min(candidates, key=lambda value: value[0])
        solved[frame] = state
        selections.append(
            {
                "frame": frame,
                "selected_hypothesis_id": f"joint-grid-{selected_index:03d}",
                "line_measurement_count": len(lines),
                "reprojection_rmse_px": score,
                "rejected_hypothesis_count": len(candidates) - 1,
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
            "source": "articulated_correspondence_initializer",
        }
        row.update({field: float(value) for field, value in zip(fields, state)})
        templates.append(row)
    return InitializationResult(
        states_by_frame={frame: tuple(float(value) for value in states[frame]) for frame in frames},
        template_rows=tuple(templates),
        hypothesis_ledger={
            "initializer_kind": "articulated_correspondence",
            "frame_count": len(frames),
            "directly_solved_frame_count": len(solved),
            "joint_hypothesis_count": len(hypotheses),
            "selections": selections,
            "case_dispatch_used": False,
            "baseline_pose_read": False,
            "human_state_optimized": False,
        },
        input_artifact_ids=tuple(sorted(artifacts)),
    )


def initialize_from_capabilities(request: InitializationRequest) -> InitializationResult:
    kind = str(request.initializer.get("kind", ""))
    if kind == "articulated_correspondence":
        return _articulated_correspondence(request)
    raise ValueError(f"unsupported capability initializer: {kind}")
