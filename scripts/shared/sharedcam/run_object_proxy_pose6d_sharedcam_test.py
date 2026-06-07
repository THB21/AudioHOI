#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_basketball_pose6d_sharedcam import (  # noqa: E402
    build_segments,
    fit_segment_lines,
    plot_outputs,
    read_alignment_frames,
    read_contact_frames_from_candidates,
    read_gvhmr_camera,
    unpack_state,
    write_csv,
)
from support_geometry import estimate_support_geometry_from_contacts, write_support_geometry_json  # noqa: E402


@dataclass
class CameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    floor_v: float


class ObjectProxyShape:
    def project_ref(self, translation: np.ndarray, camera: CameraModel) -> dict[str, float]:
        x, y, z = translation.tolist()
        z = max(float(z), 1e-6)
        u = camera.cx + camera.fx * x / z
        v = camera.cy + camera.fy * y / z
        return {"u": float(u), "v": float(v)}


def parse_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    return float(value)


def read_object_proxy_observations(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref_u = parse_float(row, "ref_u")
            ref_v = parse_float(row, "ref_v")
            depth = parse_float(row, "object_ref_depth_m")
            if not np.isfinite(ref_u) or not np.isfinite(ref_v) or not np.isfinite(depth):
                continue
            support_v = parse_float(row, "support_v", default=ref_v)
            support_dv = parse_float(row, "support_dv", default=support_v - ref_v)
            rows.append(
                {
                    "frame": int(row["frame"]),
                    "time": float(row["time"]),
                    "u": ref_u,
                    "v": ref_v,
                    "support_v": support_v,
                    "support_dv": support_dv,
                    "object_ref_depth_m": depth,
                    "depth_conf": parse_float(row, "depth_conf", 0.0),
                    "observation_conf": parse_float(row, "observation_conf", 1.0),
                    "contact_conf": parse_float(row, "contact_conf", 0.0),
                }
            )
    if not rows:
        raise RuntimeError(f"No valid object proxy observations in {path}")
    return rows


def build_init_translations_from_depth(obs_rows: list[dict[str, float]], camera: CameraModel) -> np.ndarray:
    translations = []
    for row in obs_rows:
        z = max(float(row["object_ref_depth_m"]), 0.20)
        x = (row["u"] - camera.cx) * z / camera.fx
        y = (row["v"] - camera.cy) * z / camera.fy
        translations.append(np.array([x, y, z], dtype=np.float64))
    return np.stack(translations, axis=0)


def pose_residuals_object_proxy(
    flat_state: np.ndarray,
    obs_rows: list[dict[str, float]],
    camera: CameraModel,
    shape: ObjectProxyShape,
    segments: list[dict[str, np.ndarray | int]],
    contact_indices: set[int],
    weak_contact_indices: set[int],
    center_weight: float,
    depth_weight: float,
    support_weight: float,
    temp_weight: float,
    z_temp_weight: float,
    z_boundary_weight: float,
    z_slope_weight: float,
) -> np.ndarray:
    t, ab = unpack_state(flat_state, len(obs_rows), segments)
    residuals: list[float] = []
    for idx, row in enumerate(obs_rows):
        pred = shape.project_ref(t[idx], camera)
        obs_conf = float(row.get("observation_conf", 1.0))
        depth_conf = float(row.get("depth_conf", 1.0))
        residuals.append(obs_conf * center_weight * (pred["u"] - row["u"]))
        residuals.append(obs_conf * center_weight * (pred["v"] - row["v"]))
        residuals.append(depth_conf * depth_weight * (t[idx, 2] - row["object_ref_depth_m"]))
        if idx in contact_indices or idx in weak_contact_indices:
            weight = support_weight if idx in contact_indices else 0.5 * support_weight
            pred_support_v = pred["v"] + row.get("support_dv", 0.0)
            residuals.append(weight * ((pred_support_v - camera.floor_v) / 20.0))
        residuals.append(0.30 * max(0.0, 0.20 - t[idx, 2]))

    for idx in range(1, len(t) - 1):
        accel_xy = t[idx + 1, :2] - 2.0 * t[idx, :2] + t[idx - 1, :2]
        residuals.extend((temp_weight * accel_xy).tolist())

    for seg in segments:
        indices = seg["indices"]
        if len(indices) >= 3:
            zvals = t[indices, 2]
            accel_z = zvals[2:] - 2.0 * zvals[1:-1] + zvals[:-2]
            residuals.extend((z_temp_weight * accel_z).tolist())

    for seg_idx in range(len(segments) - 1):
        cur_seg = segments[seg_idx]
        next_seg = segments[seg_idx + 1]
        end_idx = int(cur_seg["end"])
        next_start_idx = int(next_seg["start"])
        residuals.append(z_boundary_weight * (t[next_start_idx, 2] - t[end_idx, 2]))
        residuals.append(z_slope_weight * (ab[seg_idx + 1, 0] - ab[seg_idx, 0]))

    return np.asarray(residuals, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental radius-free object-proxy sharedcam baseline.")
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--object-proxy-observation-csv", type=Path, default=None)
    parser.add_argument("--out-subdir", type=str, default="pose6d_object_proxy_test")
    parser.add_argument("--center-weight", type=float, default=0.04)
    parser.add_argument("--depth-weight", type=float, default=1.0)
    parser.add_argument("--support-weight", type=float, default=10.0)
    parser.add_argument("--temp-weight", type=float, default=0.08)
    parser.add_argument("--z-temp-weight", type=float, default=0.22)
    parser.add_argument("--z-boundary-weight", type=float, default=0.10)
    parser.add_argument("--z-slope-weight", type=float, default=0.05)
    args = parser.parse_args()

    sample_dir = args.sample_dir
    results_dir = sample_dir / "results"
    obs_csv = args.object_proxy_observation_csv or (results_dir / "object_proxy_observations_test" / "object_proxy_observations.csv")
    out_dir = results_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_rows = read_object_proxy_observations(obs_csv)
    frames = np.asarray([int(r["frame"]) for r in obs_rows], dtype=np.int32)
    times = np.asarray([r["time"] for r in obs_rows], dtype=np.float64)

    contact_candidate_csv = results_dir / "contact_candidates" / "contact_candidates_labeled.csv"
    contact_frames = read_contact_frames_from_candidates(contact_candidate_csv, contact_type="floor_contact_event")
    audio_frames = read_alignment_frames(results_dir / "events" / "audio_visual_alignment.csv")
    frame_to_idx = {frame: idx for idx, frame in enumerate(frames.tolist())}
    contact_indices = {frame_to_idx[f] for f in contact_frames if f in frame_to_idx}
    weak_contact_indices = {frame_to_idx[f] for f in audio_frames if f in frame_to_idx}

    gvhmr_cam = read_gvhmr_camera(results_dir / "gvhmr" / "result.pkl")
    camera = CameraModel(
        fx=float(gvhmr_cam.fx),
        fy=float(gvhmr_cam.fy),
        cx=float(gvhmr_cam.cx),
        cy=float(gvhmr_cam.cy),
        floor_v=float(gvhmr_cam.floor_v),
    )
    obs_supports = np.asarray([row["support_v"] for row in obs_rows], dtype=np.float64)
    support = estimate_support_geometry_from_contacts(frames, obs_supports, contact_frames)
    camera.floor_v = float(support.floor_v)

    init_t = build_init_translations_from_depth(obs_rows, camera)
    segments = build_segments(len(obs_rows), contact_indices, times)
    init_ab = fit_segment_lines(init_t, segments)
    init_state = np.concatenate([init_t[:, :2].reshape(-1), init_ab.reshape(-1)], axis=0)
    shape = ObjectProxyShape()
    result = least_squares(
        pose_residuals_object_proxy,
        init_state,
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=250,
        args=(
            obs_rows,
            camera,
            shape,
            segments,
            contact_indices,
            weak_contact_indices,
            args.center_weight,
            args.depth_weight,
            args.support_weight,
            args.temp_weight,
            args.z_temp_weight,
            args.z_boundary_weight,
            args.z_slope_weight,
        ),
    )
    opt_t, _ = unpack_state(result.x, len(obs_rows), segments)
    residual_px: list[float] = []
    out_rows: list[dict[str, object]] = []
    reproj_rows: list[dict[str, object]] = []
    for idx, row in enumerate(obs_rows):
        pred = shape.project_ref(opt_t[idx], camera)
        pred_support_v = pred["v"] + row.get("support_dv", 0.0)
        err_u = pred["u"] - row["u"]
        err_v = pred["v"] - row["v"]
        err_px = float(np.hypot(err_u, err_v))
        residual_px.append(err_px)
        out_rows.append(
            {
                "frame": int(row["frame"]),
                "time": f"{row['time']:.6f}",
                "tx": f"{opt_t[idx,0]:.6f}",
                "ty": f"{opt_t[idx,1]:.6f}",
                "tz": f"{opt_t[idx,2]:.6f}",
                "qw": "1.000000",
                "qx": "0.000000",
                "qy": "0.000000",
                "qz": "0.000000",
                "radius_m": "",
                "coord_frame": "gvhmr_incam",
                "u_obs": f"{row['u']:.3f}",
                "v_obs": f"{row['v']:.3f}",
                "radius_obs_px": "",
                "u_proj": f"{pred['u']:.3f}",
                "v_proj": f"{pred['v']:.3f}",
                "radius_proj_px": "",
                "bottom_proj_v": f"{pred_support_v:.3f}",
                "floor_v": f"{camera.floor_v:.3f}",
                "u_ref_obs": f"{row['u']:.3f}",
                "v_ref_obs": f"{row['v']:.3f}",
                "u_ref_proj": f"{pred['u']:.3f}",
                "v_ref_proj": f"{pred['v']:.3f}",
                "support_v_obs": f"{row['support_v']:.3f}",
                "support_proj_v": f"{pred_support_v:.3f}",
                "residual_px": f"{err_px:.6f}",
                "contact_frame": int(idx in contact_indices),
                "audio_contact_frame": int(idx in weak_contact_indices),
                "depth_prior_m": f"{row['object_ref_depth_m']:.6f}",
                "depth_prior_gap_m": f"{(opt_t[idx,2] - row['object_ref_depth_m']):.6f}",
            }
        )
        reproj_rows.append(
            {
                "frame": int(row["frame"]),
                "u_obs": f"{row['u']:.3f}",
                "v_obs": f"{row['v']:.3f}",
                "u_reproj": f"{pred['u']:.3f}",
                "v_reproj": f"{pred['v']:.3f}",
                "error_u": f"{err_u:.6f}",
                "error_v": f"{err_v:.6f}",
                "error_px": f"{err_px:.6f}",
            }
        )

    write_csv(
        out_dir / "object_pose6d_sharedcam_trajectory.csv",
        out_rows,
        [
            "frame","time","tx","ty","tz","qw","qx","qy","qz","radius_m","coord_frame",
            "u_obs","v_obs","radius_obs_px","u_proj","v_proj","radius_proj_px","bottom_proj_v",
            "floor_v","u_ref_obs","v_ref_obs","u_ref_proj","v_ref_proj",
            "support_v_obs","support_proj_v","residual_px","contact_frame","audio_contact_frame",
            "depth_prior_m","depth_prior_gap_m",
        ],
    )
    write_csv(
        out_dir / "object_pose6d_sharedcam_reprojection_comparison.csv",
        reproj_rows,
        ["frame","u_obs","v_obs","u_reproj","v_reproj","error_u","error_v","error_px"],
    )
    plot_outputs(out_dir, times, init_t, opt_t, np.asarray(residual_px, dtype=np.float64))
    write_support_geometry_json(out_dir / "support_geometry.json", support)
    print(f"object_proxy_csv: {out_dir / 'object_pose6d_sharedcam_trajectory.csv'}")
    print(f"object_proxy_reproj_csv: {out_dir / 'object_pose6d_sharedcam_reprojection_comparison.csv'}")
    print("known_ball_radius_used: 0")


if __name__ == "__main__":
    main()
