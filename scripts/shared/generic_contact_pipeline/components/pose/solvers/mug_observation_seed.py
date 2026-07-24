#!/usr/bin/env python3
"""Build a mug body pose and axial phase from declared image/depth observations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares


REPO = Path(__file__).resolve().parents[6]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.components.pose.solvers import fit_mug_articraft_keyframe_pose as base  # noqa: E402
from scripts.shared.generic_contact_pipeline.components.pose.solvers import fit_mug_body_only_cylinder_pose as bodyfit  # noqa: E402
from scripts.shared.generic_contact_pipeline.components.render.scenes import render_mug_articraft_rigid_mesh_vlm as rigid  # noqa: E402


PHASE_GRID = np.linspace(-math.pi, math.pi, 721)
HANDLE_SIGMA_PX = 5.0
TEMPORAL_PHASE_SIGMA_RAD = 0.25
PHASE_SMOOTH_SIGMA = 1.5


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value != "" else default
    except (TypeError, ValueError):
        return default


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bbox(row: dict[str, str]) -> list[float]:
    return [
        _float(row, "body_bbox_x1", _float(row, "bbox_x1")),
        _float(row, "body_bbox_y1", _float(row, "bbox_y1")),
        _float(row, "body_bbox_x2", _float(row, "bbox_x2")),
        _float(row, "body_bbox_y2", _float(row, "bbox_y2")),
    ]


def _fit_body_frame(
    observation: dict[str, str],
    proxy: dict[str, str],
    K: np.ndarray,
    body_points: np.ndarray,
    *,
    max_nfev: int,
) -> tuple[np.ndarray, dict[str, object]]:
    initial = bodyfit.params_from_observation(observation, proxy, K)
    center = np.array(
        [
            _float(observation, "body_center_x", _float(observation, "center_x")),
            _float(observation, "body_center_y", _float(observation, "center_y")),
        ],
        dtype=float,
    )
    bbox = _bbox(observation)
    depth = _float(proxy, "object_depth_smooth", _float(proxy, "da3_depth_smooth", initial[2]))
    # Axial orientation is represented by phase. Keep body yaw fixed to zero.
    x0 = initial[[0, 1, 2, 4, 5, 6]]

    def params(values: np.ndarray) -> np.ndarray:
        return np.array([values[0], values[1], values[2], 0.0, values[3], values[4], values[5]])

    def residual(values: np.ndarray) -> np.ndarray:
        pose = params(values)
        uv = bodyfit.project_pts(pose, body_points, K)
        projected_center = bodyfit.project_pts(pose, np.zeros((1, 3)), K)[0]
        out = ((projected_center - center) / 7.0).tolist()
        out.extend(base.bbox_residual(uv, bbox, sigma=10.0))
        if np.isfinite(depth):
            out.append((pose[2] - depth) / 0.55)
        out.append((pose[6] - initial[6]) / 0.45)
        return np.asarray(out, dtype=float)

    lower = np.array(
        [initial[0] - 0.45, initial[1] - 0.45, max(0.8, initial[2] - 0.9), math.radians(-85), math.radians(-85), 0.25]
    )
    upper = np.array(
        [initial[0] + 0.45, initial[1] + 0.45, initial[2] + 0.9, math.radians(80), math.radians(85), 3.5]
    )
    fit = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=max_nfev,
    )
    pose = params(fit.x)
    final_residual = residual(fit.x)
    return pose, {
        "success": bool(fit.success),
        "cost": float(fit.cost),
        "nfev": int(fit.nfev),
        "body_residual_rms": float(np.sqrt(np.mean(final_residual * final_residual))),
    }


def _phase_candidates(
    pose: np.ndarray,
    K: np.ndarray,
    target: np.ndarray,
    handle_center: np.ndarray,
    *,
    count: int = 8,
) -> list[tuple[float, float]]:
    predictions = np.asarray(
        [bodyfit.project_pts(pose, handle_center @ rigid.rot_y(float(angle)).T, K)[0] for angle in PHASE_GRID]
    )
    distances = np.linalg.norm(predictions - target[None, :], axis=1)
    minima = [
        index
        for index in range(1, len(PHASE_GRID) - 1)
        if distances[index] <= distances[index - 1] and distances[index] <= distances[index + 1]
    ]
    minima.extend([0, len(PHASE_GRID) - 1, int(np.argmin(distances))])
    selected = sorted(set(minima), key=lambda index: (float(distances[index]), index))[:count]
    return [(float(PHASE_GRID[index]), float(distances[index])) for index in selected]


def _fit_phase_track(
    frames: list[int],
    poses: dict[int, np.ndarray],
    observations: dict[int, dict[str, str]],
    K_all: np.ndarray,
    handle_center: np.ndarray,
) -> tuple[dict[int, float], dict[str, object]]:
    visible: list[tuple[int, list[tuple[float, float]]]] = []
    for frame in frames:
        row = observations[frame]
        u = _float(row, "handle_center_x")
        v = _float(row, "handle_center_y")
        if str(row.get("handle_visible", "")) != "1" or not (np.isfinite(u) and np.isfinite(v)):
            continue
        visible.append(
            (
                frame,
                _phase_candidates(poses[frame], K_all[frame - 1], np.array([u, v]), handle_center),
            )
        )
    if not visible:
        raise RuntimeError("Mug axial phase is unobservable: no visible handle-center frames")

    costs: list[np.ndarray] = []
    parents: list[np.ndarray] = []
    for item_index, (frame, candidates) in enumerate(visible):
        emission = np.array([(distance / HANDLE_SIGMA_PX) ** 2 for _angle, distance in candidates])
        if item_index == 0:
            costs.append(emission)
            parents.append(np.full(len(candidates), -1, dtype=int))
            continue
        previous_frame, previous_candidates = visible[item_index - 1]
        gap = max(1, frame - previous_frame)
        transition_sigma = TEMPORAL_PHASE_SIGMA_RAD * gap
        current_cost = np.full(len(candidates), np.inf)
        current_parent = np.full(len(candidates), -1, dtype=int)
        for current_index, (angle, _distance) in enumerate(candidates):
            transitions = np.array(
                [
                    costs[-1][previous_index]
                    + (float(base.wrap(angle - previous_angle)) / transition_sigma) ** 2
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
    visible_frames = np.array([frame for frame, _candidates in visible], dtype=float)
    visible_angles = np.unwrap(
        np.array([visible[index][1][selected[index]][0] for index in range(len(visible))], dtype=float)
    )
    visible_errors = np.array(
        [visible[index][1][selected[index]][1] for index in range(len(visible))], dtype=float
    )
    interpolated = np.interp(np.asarray(frames, dtype=float), visible_frames, visible_angles)
    smoothed = gaussian_filter1d(interpolated, sigma=PHASE_SMOOTH_SIGMA, mode="nearest")
    phase = {frame: float(base.wrap(angle)) for frame, angle in zip(frames, smoothed)}
    return phase, {
        "visible_frames": len(visible),
        "hidden_or_unobserved_frames": len(frames) - len(visible),
        "handle_reprojection_median_px": float(np.median(visible_errors)),
        "handle_reprojection_p90_px": float(np.percentile(visible_errors, 90)),
        "handle_reprojection_max_px": float(np.max(visible_errors)),
        "phase_gauge": "body_yaw_zero_observable_axial_angle_in_phase",
        "phase_grid_step_deg": 0.5,
        "phase_smooth_sigma_frames": PHASE_SMOOTH_SIGMA,
    }


def build(
    *,
    sample_dir: Path,
    observations_csv: Path,
    proxy_csv: Path,
    out_dir: Path,
    max_nfev: int = 80,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    observation_rows = _read_csv(observations_csv)
    proxy_rows = {int(float(row["frame"])): row for row in _read_csv(proxy_csv)}
    observations = {int(float(row["frame"])): row for row in observation_rows}
    frames = sorted(set(observations) & set(proxy_rows))
    if not frames:
        raise RuntimeError("Mug observation seed has no common observation/depth frames")
    K_all = base.load_K(sample_dir)
    body_points = bodyfit.cylinder_points()[2]
    poses: dict[int, np.ndarray] = {}
    fit_info: dict[int, dict[str, object]] = {}
    for frame in frames:
        poses[frame], fit_info[frame] = _fit_body_frame(
            observations[frame], proxy_rows[frame], K_all[frame - 1], body_points, max_nfev=max_nfev
        )

    mesh_root = sample_dir / "articraft/materialized_mug_mesh"
    handle_vertices = rigid.load_articraft_meshes(mesh_root)["handle_loop"][0]
    handle_center = np.median(handle_vertices, axis=0)[None, :]
    phases, phase_info = _fit_phase_track(frames, poses, observations, K_all, handle_center)

    pose_path = out_dir / "body_pose.csv"
    pose_fields = ["frame", "time", "x", "y", "z", "yaw", "yaw_deg", "pitch", "pitch_deg", "roll", "roll_deg", "scale", "source"]
    with pose_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pose_fields)
        writer.writeheader()
        for frame in frames:
            pose = poses[frame]
            writer.writerow(
                {
                    "frame": frame,
                    "time": observations[frame].get("time", f"{(frame - 1) / 24.0:.6f}"),
                    "x": f"{pose[0]:.9f}", "y": f"{pose[1]:.9f}", "z": f"{pose[2]:.9f}",
                    "yaw": "0.000000000", "yaw_deg": "0.000000",
                    "pitch": f"{pose[4]:.9f}", "pitch_deg": f"{math.degrees(pose[4]):.6f}",
                    "roll": f"{pose[5]:.9f}", "roll_deg": f"{math.degrees(pose[5]):.6f}",
                    "scale": f"{pose[6]:.9f}",
                    "source": "observation_bbox_da3_fit_axial_gauge_yaw_zero",
                }
            )

    phase_path = out_dir / "axial_phase.csv"
    phase_fields = ["frame", "time", "m17_phase_rad", "m17_phase_deg", "m43_phase_rad", "m43_phase_deg", "vlm_visibility", "source"]
    with phase_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=phase_fields)
        writer.writeheader()
        for frame in frames:
            phase = phases[frame]
            visible = str(observations[frame].get("handle_visible", "")) == "1"
            writer.writerow(
                {
                    "frame": frame,
                    "time": observations[frame].get("time", f"{(frame - 1) / 24.0:.6f}"),
                    "m17_phase_rad": f"{phase:.9f}", "m17_phase_deg": f"{math.degrees(phase):.6f}",
                    "m43_phase_rad": f"{phase:.9f}", "m43_phase_deg": f"{math.degrees(phase):.6f}",
                    "vlm_visibility": "visible" if visible else "hidden",
                    "source": "observed_handle_center" if visible else "interpolated_hidden_handle_span",
                }
            )

    body_rms = np.array([float(item["body_residual_rms"]) for item in fit_info.values()])
    report = {
        "schema_version": 1,
        "policy": "observation_derived_body_pose_and_axial_phase",
        "body_pose_csv": str(pose_path),
        "phase_source": str(phase_path),
        "observations_csv": str(observations_csv),
        "proxy_depth_csv": str(proxy_csv),
        "mesh_root": str(mesh_root),
        "rows": len(frames),
        "body_fit_success_frames": sum(bool(item["success"]) for item in fit_info.values()),
        "body_residual_rms_median": float(np.median(body_rms)),
        "body_residual_rms_p90": float(np.percentile(body_rms, 90)),
        "phase": phase_info,
        "inputs": {
            "observations_sha256": _sha256(observations_csv),
            "proxy_depth_sha256": _sha256(proxy_csv),
        },
        "historical_solved_seed_used": False,
    }
    report_path = out_dir / "observation_seed_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {**report, "report": str(report_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--observations-csv", type=Path, required=True)
    parser.add_argument("--proxy-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-nfev", type=int, default=80)
    args = parser.parse_args()
    report = build(
        sample_dir=args.sample_dir.resolve(),
        observations_csv=args.observations_csv.resolve(),
        proxy_csv=args.proxy_csv.resolve(),
        out_dir=args.out_dir.resolve(),
        max_nfev=args.max_nfev,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
