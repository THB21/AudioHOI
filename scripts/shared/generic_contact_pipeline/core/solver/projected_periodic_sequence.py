from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ProjectedPeriodicObservation:
    frame: int
    time: float
    body_center_uv: tuple[float, float]
    body_bbox_xyxy: tuple[float, float, float, float]
    body_extent_uv: tuple[float, float]
    metric_depth_m: float
    periodic_feature_uv: tuple[float, float] | None
    periodic_feature_visible: bool


@dataclass(frozen=True)
class ProjectedPeriodicParameters:
    center_sigma_px: float = 7.0
    bbox_sigma_px: float = 10.0
    depth_sigma_m: float = 0.55
    root_regularization_sigma: float = 0.45
    periodic_feature_sigma_px: float = 5.0
    temporal_phase_sigma_rad_per_frame: float = 0.25
    phase_smooth_sigma_frames: float = 1.5
    phase_grid_samples: int = 721
    phase_candidate_count: int = 8
    robust_loss: str = "soft_l1"
    robust_scale: float = 1.0
    max_function_evaluations: int = 80


@dataclass(frozen=True)
class PeriodicKinematicContract:
    root_node: str
    periodic_feature_node: str
    axial_gauge_dof: str
    periodic_feature_dof: str
    physical_joint: bool
    relative_motion_allowed: bool
    observable_combination: str
    gauge_constraint: str
    gauge_transform: str


class ProjectedPeriodicGeometryProvider(Protocol):
    provider_id: str
    axial_gauge: str
    kinematic_contract: PeriodicKinematicContract

    def initial_root_state(self, observation: ProjectedPeriodicObservation, camera: np.ndarray) -> np.ndarray: ...

    def optimization_vector(self, initial_root: np.ndarray) -> np.ndarray: ...

    def root_state(self, values: np.ndarray) -> np.ndarray: ...

    def optimization_bounds(self, initial_root: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...

    def project_body(self, root_state: np.ndarray, camera: np.ndarray) -> np.ndarray: ...

    def project_origin(self, root_state: np.ndarray, camera: np.ndarray) -> np.ndarray: ...

    def project_periodic_feature(
        self,
        root_state: np.ndarray,
        phase_rad: float,
        camera: np.ndarray,
    ) -> np.ndarray: ...

    def root_regularization(self, root_state: np.ndarray, initial_root: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class ProjectedPeriodicSolution:
    root_states: dict[int, np.ndarray]
    periodic_phase_rad: dict[int, float]
    root_fit_info: dict[int, dict[str, object]]
    report: dict[str, object]


def _bbox_residual(projected: np.ndarray, target: tuple[float, float, float, float], sigma: float) -> list[float]:
    predicted = (
        float(np.min(projected[:, 0])),
        float(np.min(projected[:, 1])),
        float(np.max(projected[:, 0])),
        float(np.max(projected[:, 1])),
    )
    return [(predicted[index] - target[index]) / sigma for index in range(4)]


def _wrap_pi(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _fit_root_frame(
    observation: ProjectedPeriodicObservation,
    camera: np.ndarray,
    geometry: ProjectedPeriodicGeometryProvider,
    parameters: ProjectedPeriodicParameters,
) -> tuple[np.ndarray, dict[str, object]]:
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:  # pragma: no cover - exercised in the audiohoi runtime
        raise RuntimeError("projected periodic solve requires scipy; run with the audiohoi environment") from exc

    initial = geometry.initial_root_state(observation, camera)
    x0 = geometry.optimization_vector(initial)

    def residual(values: np.ndarray) -> np.ndarray:
        root = geometry.root_state(values)
        projected_body = geometry.project_body(root, camera)
        projected_center = geometry.project_origin(root, camera)
        out = (
            (projected_center - np.asarray(observation.body_center_uv, dtype=float))
            / parameters.center_sigma_px
        ).tolist()
        out.extend(_bbox_residual(projected_body, observation.body_bbox_xyxy, parameters.bbox_sigma_px))
        if np.isfinite(observation.metric_depth_m):
            out.append((float(root[2]) - observation.metric_depth_m) / parameters.depth_sigma_m)
        out.extend((geometry.root_regularization(root, initial) / parameters.root_regularization_sigma).tolist())
        return np.asarray(out, dtype=float)

    lower, upper = geometry.optimization_bounds(initial)
    fit = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        loss=parameters.robust_loss,
        f_scale=parameters.robust_scale,
        max_nfev=parameters.max_function_evaluations,
    )
    root = geometry.root_state(fit.x)
    final_residual = residual(fit.x)
    return root, {
        "success": bool(fit.success),
        "cost": float(fit.cost),
        "nfev": int(fit.nfev),
        "body_residual_rms": float(np.sqrt(np.mean(final_residual * final_residual))),
    }


def _phase_candidates(
    root_state: np.ndarray,
    camera: np.ndarray,
    target: tuple[float, float],
    geometry: ProjectedPeriodicGeometryProvider,
    parameters: ProjectedPeriodicParameters,
) -> list[tuple[float, float]]:
    grid = np.linspace(-math.pi, math.pi, parameters.phase_grid_samples)
    predictions = np.asarray(
        [geometry.project_periodic_feature(root_state, float(angle), camera) for angle in grid]
    )
    distances = np.linalg.norm(predictions - np.asarray(target, dtype=float)[None, :], axis=1)
    minima = [
        index
        for index in range(1, len(grid) - 1)
        if distances[index] <= distances[index - 1] and distances[index] <= distances[index + 1]
    ]
    minima.extend([0, len(grid) - 1, int(np.argmin(distances))])
    selected = sorted(set(minima), key=lambda index: (float(distances[index]), index))[
        : parameters.phase_candidate_count
    ]
    return [(float(grid[index]), float(distances[index])) for index in selected]


def _fit_periodic_track(
    observations: list[ProjectedPeriodicObservation],
    root_states: dict[int, np.ndarray],
    cameras: dict[int, np.ndarray],
    geometry: ProjectedPeriodicGeometryProvider,
    parameters: ProjectedPeriodicParameters,
) -> tuple[dict[int, float], dict[str, object]]:
    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError as exc:  # pragma: no cover - exercised in the audiohoi runtime
        raise RuntimeError("projected periodic solve requires scipy; run with the audiohoi environment") from exc

    visible: list[tuple[int, list[tuple[float, float]]]] = []
    for observation in observations:
        if not observation.periodic_feature_visible or observation.periodic_feature_uv is None:
            continue
        visible.append(
            (
                observation.frame,
                _phase_candidates(
                    root_states[observation.frame],
                    cameras[observation.frame],
                    observation.periodic_feature_uv,
                    geometry,
                    parameters,
                ),
            )
        )
    if not visible:
        raise RuntimeError("periodic feature is unobservable: no visible feature-center frames")

    costs: list[np.ndarray] = []
    parents: list[np.ndarray] = []
    for item_index, (frame, candidates) in enumerate(visible):
        emission = np.asarray(
            [(distance / parameters.periodic_feature_sigma_px) ** 2 for _angle, distance in candidates]
        )
        if item_index == 0:
            costs.append(emission)
            parents.append(np.full(len(candidates), -1, dtype=int))
            continue
        previous_frame, previous_candidates = visible[item_index - 1]
        gap = max(1, frame - previous_frame)
        transition_sigma = parameters.temporal_phase_sigma_rad_per_frame * gap
        current_cost = np.full(len(candidates), np.inf)
        current_parent = np.full(len(candidates), -1, dtype=int)
        for current_index, (angle, _distance) in enumerate(candidates):
            transitions = np.asarray(
                [
                    costs[-1][previous_index]
                    + (_wrap_pi(angle - previous_angle) / transition_sigma) ** 2
                    for previous_index, (previous_angle, _previous_distance) in enumerate(previous_candidates)
                ]
            )
            best = int(np.argmin(transitions))
            current_cost[current_index] = emission[current_index] + transitions[best]
            current_parent[current_index] = best
        costs.append(current_cost)
        parents.append(current_parent)

    selected = [0] * len(visible)
    selected[-1] = int(np.argmin(costs[-1]))
    for index in range(len(visible) - 1, 0, -1):
        selected[index - 1] = int(parents[index][selected[index]])
    visible_frames = np.asarray([frame for frame, _candidates in visible], dtype=float)
    visible_angles = np.unwrap(
        np.asarray([visible[index][1][selected[index]][0] for index in range(len(visible))], dtype=float)
    )
    visible_errors = np.asarray(
        [visible[index][1][selected[index]][1] for index in range(len(visible))], dtype=float
    )
    frames = np.asarray([observation.frame for observation in observations], dtype=float)
    interpolated = np.interp(frames, visible_frames, visible_angles)
    smoothed = gaussian_filter1d(
        interpolated,
        sigma=parameters.phase_smooth_sigma_frames,
        mode="nearest",
    )
    phase = {
        observation.frame: float(_wrap_pi(angle))
        for observation, angle in zip(observations, smoothed)
    }
    return phase, {
        "visible_frames": len(visible),
        "hidden_or_unobserved_frames": len(observations) - len(visible),
        "handle_reprojection_median_px": float(np.median(visible_errors)),
        "handle_reprojection_p90_px": float(np.percentile(visible_errors, 90)),
        "handle_reprojection_max_px": float(np.max(visible_errors)),
        "phase_gauge": geometry.axial_gauge,
        "phase_grid_step_deg": 360.0 / (parameters.phase_grid_samples - 1),
        "phase_smooth_sigma_frames": parameters.phase_smooth_sigma_frames,
    }


def solve_projected_periodic_sequence(
    observations: list[ProjectedPeriodicObservation],
    cameras: dict[int, np.ndarray],
    geometry: ProjectedPeriodicGeometryProvider,
    parameters: ProjectedPeriodicParameters = ProjectedPeriodicParameters(),
) -> ProjectedPeriodicSolution:
    if not observations:
        raise ValueError("projected periodic sequence requires observations")
    frames = [observation.frame for observation in observations]
    if frames != sorted(frames) or len(frames) != len(set(frames)):
        raise ValueError("projected periodic observations must have unique sorted frames")
    missing_cameras = sorted(set(frames) - set(cameras))
    if missing_cameras:
        raise ValueError(f"missing cameras for frames: {missing_cameras[:10]}")

    root_states: dict[int, np.ndarray] = {}
    fit_info: dict[int, dict[str, object]] = {}
    for observation in observations:
        root_states[observation.frame], fit_info[observation.frame] = _fit_root_frame(
            observation,
            cameras[observation.frame],
            geometry,
            parameters,
        )
    phase, phase_info = _fit_periodic_track(observations, root_states, cameras, geometry, parameters)
    body_rms = np.asarray([float(item["body_residual_rms"]) for item in fit_info.values()])
    report = {
        "rows": len(observations),
        "body_fit_success_frames": sum(bool(item["success"]) for item in fit_info.values()),
        "body_residual_rms_median": float(np.median(body_rms)),
        "body_residual_rms_p90": float(np.percentile(body_rms, 90)),
        "phase": phase_info,
        "state_spec": "root_se3_scale_plus_periodic_feature",
        "kinematic_contract": asdict(geometry.kinematic_contract),
        "geometry_provider": geometry.provider_id,
        "baseline_pose_read": False,
    }
    return ProjectedPeriodicSolution(root_states, phase, fit_info, report)
