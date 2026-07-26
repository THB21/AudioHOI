from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...core.solver.projected_periodic_sequence import (
    PeriodicKinematicContract,
    ProjectedPeriodicObservation,
)


BODY_RADIUS_M = 0.040
BODY_HEIGHT_M = 0.096
BODY_DEPTH_RADIUS_M = BODY_RADIUS_M


def load_camera_matrices(sample_dir: Path) -> np.ndarray:
    with (sample_dir / "results/gvhmr/result.pkl").open("rb") as handle:
        data = pickle.load(handle)
    return np.asarray(data["K_fullimg"], dtype=float)


def _cylinder_points(samples: int = 96) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    top = np.stack(
        [
            BODY_RADIUS_M * np.cos(theta),
            np.full_like(theta, -BODY_HEIGHT_M / 2),
            BODY_DEPTH_RADIUS_M * np.sin(theta),
        ],
        axis=1,
    )
    bottom = np.stack(
        [
            BODY_RADIUS_M * np.cos(theta),
            np.full_like(theta, BODY_HEIGHT_M / 2),
            BODY_DEPTH_RADIUS_M * np.sin(theta),
        ],
        axis=1,
    )
    side = np.asarray(
        [
            [-BODY_RADIUS_M, -BODY_HEIGHT_M / 2, 0.0],
            [BODY_RADIUS_M, -BODY_HEIGHT_M / 2, 0.0],
            [-BODY_RADIUS_M, BODY_HEIGHT_M / 2, 0.0],
            [BODY_RADIUS_M, BODY_HEIGHT_M / 2, 0.0],
            [0.0, -BODY_HEIGHT_M / 2, 0.0],
            [0.0, BODY_HEIGHT_M / 2, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    return np.vstack([top, bottom, side])


def _load_handle_center(mesh_root: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    with (mesh_root / "assets/meshes/handle_loop.obj").open() as handle:
        for line in handle:
            if line.startswith("v "):
                x, y, z = [float(value) for value in line.split()[1:4]]
                vertices.append([-x, BODY_HEIGHT_M * 0.5 - z, y])
    if not vertices:
        raise ValueError(f"handle mesh contains no vertices: {mesh_root}")
    return np.median(np.asarray(vertices, dtype=float), axis=0)[None, :]


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]], dtype=float)


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]], dtype=float)


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _project_points(root_state: np.ndarray, points: np.ndarray, camera: np.ndarray) -> np.ndarray:
    x, y, z, yaw, pitch, roll, scale = root_state
    rotation = _rotation_y(float(yaw)) @ _rotation_x(float(pitch)) @ _rotation_z(float(roll))
    transformed = points @ (float(scale) * rotation).T + np.asarray([x, y, z], dtype=float)
    projected = []
    for point in transformed:
        depth = max(float(point[2]), 1e-6)
        projected.append(
            [
                camera[0, 0] * point[0] / depth + camera[0, 2],
                camera[1, 1] * point[1] / depth + camera[1, 2],
            ]
        )
    return np.asarray(projected, dtype=float)


class MugPeriodicGeometryProvider:
    provider_id = "articraft_mug_body_handle_periodic_v1"
    axial_gauge = "body_yaw_zero_observable_axial_angle_in_phase"
    kinematic_contract = PeriodicKinematicContract(
        root_node="body",
        periodic_feature_node="handle",
        axial_gauge_dof="body.symmetry_phase",
        periodic_feature_dof="assembly.axial_phase",
        physical_joint=False,
        relative_motion_allowed=False,
        observable_combination="body.symmetry_phase + assembly.axial_phase",
        gauge_constraint="body.symmetry_phase = 0",
        gauge_transform="body.symmetry_phase += delta; assembly.axial_phase -= delta",
    )

    def __init__(self, mesh_root: Path):
        self.mesh_root = mesh_root
        self.body_points = _cylinder_points()
        self.handle_center = _load_handle_center(mesh_root)

    def initial_root_state(self, observation: ProjectedPeriodicObservation, camera: np.ndarray) -> np.ndarray:
        z = observation.metric_depth_m
        if not np.isfinite(z) or z < 0.2:
            z = 3.8
        u, v = observation.body_center_uv
        xyz = np.asarray(
            [
                (u - camera[0, 2]) * z / camera[0, 0],
                (v - camera[1, 2]) * z / camera[1, 1],
                z,
            ],
            dtype=float,
        )
        body_height_px = observation.body_extent_uv[1]
        scale = float(
            np.clip(
                (body_height_px * z) / max(camera[1, 1] * BODY_HEIGHT_M, 1e-6),
                0.30,
                3.5,
            )
        )
        return np.asarray([xyz[0], xyz[1], xyz[2], 0.0, 0.0, 0.0, scale], dtype=float)

    def optimization_vector(self, initial_root: np.ndarray) -> np.ndarray:
        return initial_root[[0, 1, 2, 4, 5, 6]]

    def root_state(self, values: np.ndarray) -> np.ndarray:
        return np.asarray([values[0], values[1], values[2], 0.0, values[3], values[4], values[5]])

    def optimization_bounds(self, initial_root: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lower = np.asarray(
            [
                initial_root[0] - 0.45,
                initial_root[1] - 0.45,
                max(0.8, initial_root[2] - 0.9),
                math.radians(-85),
                math.radians(-85),
                0.25,
            ]
        )
        upper = np.asarray(
            [
                initial_root[0] + 0.45,
                initial_root[1] + 0.45,
                initial_root[2] + 0.9,
                math.radians(80),
                math.radians(85),
                3.5,
            ]
        )
        return lower, upper

    def project_body(self, root_state: np.ndarray, camera: np.ndarray) -> np.ndarray:
        return _project_points(root_state, self.body_points, camera)

    def project_origin(self, root_state: np.ndarray, camera: np.ndarray) -> np.ndarray:
        return _project_points(root_state, np.zeros((1, 3)), camera)[0]

    def project_periodic_feature(
        self,
        root_state: np.ndarray,
        phase_rad: float,
        camera: np.ndarray,
    ) -> np.ndarray:
        rotated = self.handle_center @ _rotation_y(float(phase_rad)).T
        return _project_points(root_state, rotated, camera)[0]

    def root_regularization(self, root_state: np.ndarray, initial_root: np.ndarray) -> np.ndarray:
        return np.asarray([root_state[6] - initial_root[6]], dtype=float)


def _number(row: Mapping[str, str], primary: str, fallback: str, default: float = math.nan) -> float:
    raw = row.get(primary, "")
    if raw in {"", None}:
        raw = row.get(fallback, "")
    try:
        return float(raw) if raw not in {"", None} else default
    except (TypeError, ValueError):
        return default


def adapt_mug_periodic_observations(
    observation_rows: Sequence[Mapping[str, str]],
    proxy_rows: Mapping[int, Mapping[str, str]],
) -> list[ProjectedPeriodicObservation]:
    """Map mug detector rows to the object-agnostic projected-periodic contract."""
    observations: list[ProjectedPeriodicObservation] = []
    for row in sorted(observation_rows, key=lambda item: int(float(item["frame"]))):
        frame = int(float(row["frame"]))
        proxy = proxy_rows.get(frame)
        if proxy is None:
            continue
        bbox = (
            _number(row, "body_bbox_x1", "bbox_x1"),
            _number(row, "body_bbox_y1", "bbox_y1"),
            _number(row, "body_bbox_x2", "bbox_x2"),
            _number(row, "body_bbox_y2", "bbox_y2"),
        )
        extent = (
            _number(row, "body_bbox_w_px", "bbox_w_px", bbox[2] - bbox[0]),
            _number(row, "body_bbox_h_px", "bbox_h_px", bbox[3] - bbox[1]),
        )
        handle_u = _number(row, "handle_center_x", "handle_center_x")
        handle_v = _number(row, "handle_center_y", "handle_center_y")
        visible = row.get("handle_visible", "") == "1" and np.isfinite(handle_u) and np.isfinite(handle_v)
        observations.append(
            ProjectedPeriodicObservation(
                frame=frame,
                time=float(row.get("time") or (frame - 1) / 24.0),
                body_center_uv=(
                    _number(row, "body_center_x", "center_x"),
                    _number(row, "body_center_y", "center_y"),
                ),
                body_bbox_xyxy=bbox,
                body_extent_uv=extent,
                metric_depth_m=_number(proxy, "object_depth_smooth", "da3_depth_smooth", 3.8),
                periodic_feature_uv=(handle_u, handle_v) if visible else None,
                periodic_feature_visible=visible,
            )
        )
    return observations
