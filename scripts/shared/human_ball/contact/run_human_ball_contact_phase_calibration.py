#!/usr/bin/env python3
"""Phase-aware contact calibration for the shared-camera basketball branch.

This branch refines only the ball depth trajectory while keeping the observed
2D ball center fixed. It separates contact constraints into two phases:

1. Exact contact-center frames:
   - strong palm-depth equality used to estimate a global Z shift
   - strong floor consistency
2. Nearby support window:
   - weaker floor consistency
   - smooth temporal transition back to the baseline

This is intended to use palm depth at contact as a global depth-alignment reference while keeping the trajectory smooth.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import torch
import smplx


BALL_RADIUS_M = 0.12


def read_ball_pose(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "frame": int(row["frame"]),
                    "time": float(row["time"]),
                    "tx": float(row["tx"]),
                    "ty": float(row["ty"]),
                    "tz": float(row["tz"]),
                    "qw": float(row["qw"]),
                    "qx": float(row["qx"]),
                    "qy": float(row["qy"]),
                    "qz": float(row["qz"]),
                    "radius_m": float(row["radius_m"]),
                    "coord_frame": row["coord_frame"],
                    "u_obs": float(row["u_obs"]),
                    "v_obs": float(row["v_obs"]),
                    "radius_obs_px": float(row["radius_obs_px"]),
                    "u_proj": float(row["u_proj"]),
                    "v_proj": float(row["v_proj"]),
                    "radius_proj_px": float(row["radius_proj_px"]),
                    "bottom_proj_v": float(row["bottom_proj_v"]),
                    "floor_v": float(row["floor_v"]),
                    "residual_px": float(row["residual_px"]),
                    "contact_frame": int(row.get("contact_frame", 0) or 0),
                    "audio_contact_frame": int(row.get("audio_contact_frame", 0) or 0),
                }
            )
    if not rows:
        raise RuntimeError(f"No ball rows found in {path}")
    return rows


def read_human_result(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as f:
        data = pickle.load(f)
    params = data["smpl_params_incam"]
    return {
        "body_pose": np.asarray(params["body_pose"], dtype=np.float32),
        "betas": np.asarray(params["betas"], dtype=np.float32),
        "global_orient": np.asarray(params["global_orient"], dtype=np.float32),
        "transl": np.asarray(params["transl"], dtype=np.float32),
        "K_fullimg": np.asarray(data["K_fullimg"], dtype=np.float32),
    }


def build_body_joints(body_models_root: Path, human_params: dict[str, np.ndarray]) -> np.ndarray:
    model = smplx.create(
        str(body_models_root),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        batch_size=human_params["transl"].shape[0],
    )
    with torch.inference_mode():
        output = model(
            body_pose=torch.from_numpy(human_params["body_pose"]),
            betas=torch.from_numpy(human_params["betas"]),
            global_orient=torch.from_numpy(human_params["global_orient"]),
            transl=torch.from_numpy(human_params["transl"]),
            return_verts=False,
        )
    return output.joints.detach().cpu().numpy().astype(np.float64)


def reconstruct_xyz_from_uvz(u_obs: np.ndarray, v_obs: np.ndarray, z: np.ndarray, K: np.ndarray) -> np.ndarray:
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]
    x = (u_obs - cx) * z / fx
    y = (v_obs - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def project_ball(ball_xyz: np.ndarray, K: np.ndarray, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    z = np.clip(ball_xyz[:, 2], 1e-6, None)
    r = K[:, 0, 0] * (radius_m / z)
    bottom_v = K[:, 1, 1] * (ball_xyz[:, 1] / z) + K[:, 1, 2] + r
    return r, bottom_v


def build_palm_centers(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_ids = [20, 25, 28, 31, 34]
    right_ids = [21, 40, 43, 46, 49]
    left_palm = joints[:, left_ids, :].mean(axis=1)
    right_palm = joints[:, right_ids, :].mean(axis=1)
    return left_palm, right_palm


def choose_active_hand_relation(joints: np.ndarray, ball_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, right = build_palm_centers(joints)
    left_dist = np.linalg.norm(left - ball_xyz, axis=1)
    right_dist = np.linalg.norm(right - ball_xyz, axis=1)
    use_left = left_dist <= right_dist
    hand_y = np.where(use_left, left[:, 1], right[:, 1])
    hand_z = np.where(use_left, left[:, 2], right[:, 2])
    hand_name = np.where(use_left, "left", "right")
    return hand_y, hand_z, hand_name


def build_window_weights(contact_indices: np.ndarray, num_frames: int, radius: int, sigma: float) -> np.ndarray:
    weights = np.zeros(num_frames, dtype=np.float64)
    if len(contact_indices) == 0:
        return weights
    for ci in contact_indices:
        lo = max(0, ci - radius)
        hi = min(num_frames, ci + radius + 1)
        idx = np.arange(lo, hi, dtype=np.float64)
        local = np.exp(-0.5 * ((idx - float(ci)) / sigma) ** 2)
        weights[lo:hi] = np.maximum(weights[lo:hi], local)
    if np.max(weights) > 0:
        weights /= np.max(weights)
    return weights


def z_objective(
    z: np.ndarray,
    z_ref: np.ndarray,
    u_obs: np.ndarray,
    v_obs: np.ndarray,
    r_obs: np.ndarray,
    K: np.ndarray,
    floor_v: np.ndarray,
    floor_window_w: np.ndarray,
    contact_mask: np.ndarray,
    hand_z: np.ndarray,
    w_size: float,
    w_floor_window: float,
    w_floor_contact: float,
    w_hand_contact: float,
    w_z_anchor: float,
    w_temp: float,
) -> np.ndarray:
    z = np.maximum(z, 0.20)
    z = z.copy()
    z[contact_mask] = hand_z[contact_mask]
    ball_xyz = reconstruct_xyz_from_uvz(u_obs, v_obs, z, K)
    r_proj, bottom_v = project_ball(ball_xyz, K, BALL_RADIUS_M)

    residuals: list[np.ndarray] = []
    residuals.append(w_size * (r_proj - r_obs))

    active_floor = floor_window_w > 1e-6
    if np.any(active_floor):
        residuals.append(
            w_floor_window * floor_window_w[active_floor] * ((bottom_v[active_floor] - floor_v[active_floor]) / 20.0)
        )

    if np.any(contact_mask):
        residuals.append(w_floor_contact * ((bottom_v[contact_mask] - floor_v[contact_mask]) / 20.0))

    anchor_delta = z - z_ref
    anchor_delta = anchor_delta.copy()
    anchor_delta[contact_mask] = 0.0
    residuals.append(w_z_anchor * anchor_delta)

    if len(z) >= 3:
        second = z[2:] - 2.0 * z[1:-1] + z[:-2]
        residuals.append(w_temp * second)

    return np.concatenate([r.ravel() for r in residuals]).astype(np.float64)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-aware contact calibration with global palm-Z alignment.")
    parser.add_argument("--sample-dir", type=Path, default=Path("samples/basketball_01"))
    parser.add_argument("--body-model-root", type=Path, default=Path("third-party/GVHMR/inputs/checkpoints/body_models"))
    parser.add_argument("--out-subdir", type=str, default="pose6d_sharedcam_contactphase")
    parser.add_argument("--floor-window-radius", type=int, default=3)
    parser.add_argument("--floor-window-sigma", type=float, default=1.35)
    parser.add_argument("--w-size", type=float, default=0.12)
    parser.add_argument("--w-floor-window", type=float, default=0.45)
    parser.add_argument("--w-floor-contact", type=float, default=1.35)
    parser.add_argument("--w-hand-contact", type=float, default=1.75)
    parser.add_argument("--w-z-anchor", type=float, default=0.70)
    parser.add_argument("--w-temp", type=float, default=1.35)
    args = parser.parse_args()

    sample_dir = args.sample_dir
    results_dir = sample_dir / "results"
    out_dir = results_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    ball_rows = read_ball_pose(results_dir / "pose6d_sharedcam" / "ball_pose6d_sharedcam_trajectory.csv")
    human = read_human_result(results_dir / "gvhmr" / "result.pkl")
    joints = build_body_joints(args.body_model_root, human)

    if len(ball_rows) != len(joints):
        raise RuntimeError("Frame count mismatch between sharedcam ball and GVHMR human")

    K = np.asarray(human["K_fullimg"], dtype=np.float64)
    u_obs = np.asarray([r["u_obs"] for r in ball_rows], dtype=np.float64)
    v_obs = np.asarray([r["v_obs"] for r in ball_rows], dtype=np.float64)
    r_obs = np.asarray([r["radius_obs_px"] for r in ball_rows], dtype=np.float64)
    floor_v = np.asarray([r["floor_v"] for r in ball_rows], dtype=np.float64)
    z_init = np.asarray([r["tz"] for r in ball_rows], dtype=np.float64)
    contact_mask = np.asarray([int(r["contact_frame"]) == 1 for r in ball_rows], dtype=bool)
    contact_indices = np.where(contact_mask)[0]

    xyz_init = np.asarray([[r["tx"], r["ty"], r["tz"]] for r in ball_rows], dtype=np.float64)
    hand_y, hand_z, hand_name = choose_active_hand_relation(joints, xyz_init)
    floor_window_w = build_window_weights(contact_indices, len(ball_rows), args.floor_window_radius, args.floor_window_sigma)

    if np.any(contact_mask):
        global_z_shift = float(np.median(hand_z[contact_mask] - z_init[contact_mask]))
    else:
        global_z_shift = 0.0
    z_ref = z_init + global_z_shift

    result = least_squares(
        z_objective,
        x0=z_ref.copy(),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=400,
        args=(
            z_ref,
            u_obs,
            v_obs,
            r_obs,
            K,
            floor_v,
            floor_window_w,
            contact_mask,
            hand_z,
            args.w_size,
            args.w_floor_window,
            args.w_floor_contact,
            args.w_hand_contact,
            args.w_z_anchor,
            args.w_temp,
        ),
    )

    z_opt = np.maximum(result.x.astype(np.float64), 0.20)
    z_opt[contact_mask] = hand_z[contact_mask]
    xyz_opt = reconstruct_xyz_from_uvz(u_obs, v_obs, z_opt, K)
    r_proj, bottom_proj = project_ball(xyz_opt, K, BALL_RADIUS_M)
    residual_px = np.zeros(len(ball_rows), dtype=np.float64)

    out_rows: list[dict[str, object]] = []
    reproj_rows: list[dict[str, object]] = []
    for idx, row in enumerate(ball_rows):
        out_rows.append(
            {
                "frame": row["frame"],
                "time": f"{row['time']:.6f}",
                "tx": f"{xyz_opt[idx,0]:.6f}",
                "ty": f"{xyz_opt[idx,1]:.6f}",
                "tz": f"{xyz_opt[idx,2]:.6f}",
                "qw": f"{row['qw']:.6f}",
                "qx": f"{row['qx']:.6f}",
                "qy": f"{row['qy']:.6f}",
                "qz": f"{row['qz']:.6f}",
                "radius_m": f"{row['radius_m']:.6f}",
                "coord_frame": row["coord_frame"],
                "u_obs": f"{row['u_obs']:.3f}",
                "v_obs": f"{row['v_obs']:.3f}",
                "radius_obs_px": f"{row['radius_obs_px']:.3f}",
                "u_proj": f"{row['u_obs']:.3f}",
                "v_proj": f"{row['v_obs']:.3f}",
                "radius_proj_px": f"{r_proj[idx]:.3f}",
                "bottom_proj_v": f"{bottom_proj[idx]:.3f}",
                "floor_v": f"{row['floor_v']:.3f}",
                "residual_px": f"{residual_px[idx]:.6f}",
                "contact_frame": row["contact_frame"],
                "audio_contact_frame": row["audio_contact_frame"],
                "active_hand": hand_name[idx],
                "active_hand_y": f"{hand_y[idx]:.6f}",
                "active_hand_z": f"{hand_z[idx]:.6f}",
                "floor_window_weight": f"{floor_window_w[idx]:.6f}",
                "global_z_ref": f"{z_ref[idx]:.6f}",
                "contact_depth_gap": f"{(xyz_opt[idx,2] - hand_z[idx]):.6f}",
            }
        )
        reproj_rows.append(
            {
                "frame": row["frame"],
                "u_obs": f"{row['u_obs']:.3f}",
                "v_obs": f"{row['v_obs']:.3f}",
                "u_reproj": f"{row['u_obs']:.3f}",
                "v_reproj": f"{row['v_obs']:.3f}",
                "error_u": "0.000000",
                "error_v": "0.000000",
                "error_px": "0.000000",
            }
        )

    out_csv = out_dir / "ball_pose6d_sharedcam_contactphase_trajectory.csv"
    reproj_csv = out_dir / "ball_pose6d_sharedcam_contactphase_reprojection_comparison.csv"
    summary_txt = out_dir / "ball_pose6d_sharedcam_contactphase_summary.txt"

    write_csv(
        out_csv,
        out_rows,
        [
            "frame","time","tx","ty","tz","qw","qx","qy","qz","radius_m","coord_frame",
            "u_obs","v_obs","radius_obs_px","u_proj","v_proj","radius_proj_px","bottom_proj_v",
            "floor_v","residual_px","contact_frame","audio_contact_frame",
            "active_hand","active_hand_y","active_hand_z","floor_window_weight","global_z_ref","contact_depth_gap",
        ],
    )
    write_csv(
        reproj_csv,
        reproj_rows,
        ["frame", "u_obs", "v_obs", "u_reproj", "v_reproj", "error_u", "error_v", "error_px"],
    )

    baseline_contact_gap = z_init[contact_mask] - hand_z[contact_mask] if np.any(contact_mask) else np.array([], dtype=float)
    aligned_contact_gap = z_opt[contact_mask] - hand_z[contact_mask] if np.any(contact_mask) else np.array([], dtype=float)
    baseline_bottom_gap = (np.asarray([r["bottom_proj_v"] for r in ball_rows], dtype=float)[contact_mask] - floor_v[contact_mask]) if np.any(contact_mask) else np.array([], dtype=float)
    aligned_bottom_gap = (bottom_proj[contact_mask] - floor_v[contact_mask]) if np.any(contact_mask) else np.array([], dtype=float)
    second_init = z_init[2:] - 2.0 * z_init[1:-1] + z_init[:-2]
    second_opt = z_opt[2:] - 2.0 * z_opt[1:-1] + z_opt[:-2]

    summary_lines = [
        "Contact-phase calibration: palm depth at exact contact frames is enforced as a hard Z constraint, with local floor support in a window.",
        f"floor_window_radius_frames: {args.floor_window_radius}",
        f"floor_window_sigma: {args.floor_window_sigma:.3f}",
        f"global_z_shift_from_contacts_m: {global_z_shift:.6f}",
        f"num_frames: {len(ball_rows)}",
        f"num_contact_frames: {len(contact_indices)}",
        f"num_floor_window_frames: {int(np.count_nonzero(floor_window_w > 1e-6))}",
        f"mean_abs_delta_z_m: {float(np.mean(np.abs(z_opt - z_init))):.6f}",
        f"max_abs_delta_z_m: {float(np.max(np.abs(z_opt - z_init))):.6f}",
        f"baseline_z_second_diff_mean: {float(np.mean(np.abs(second_init))):.6f}",
        f"aligned_z_second_diff_mean: {float(np.mean(np.abs(second_opt))):.6f}",
        f"contact_frames_head: {', '.join(str(int(ball_rows[i]['frame'])) for i in contact_indices[:12])}",
    ]
    if np.any(contact_mask):
        summary_lines.extend(
            [
                f"baseline_contact_depth_gap_mean_m: {float(np.mean(np.abs(baseline_contact_gap))):.6f}",
                f"aligned_contact_depth_gap_mean_m: {float(np.mean(np.abs(aligned_contact_gap))):.6f}",
                f"baseline_contact_bottom_gap_px: {float(np.mean(np.abs(baseline_bottom_gap))):.6f}",
                f"aligned_contact_bottom_gap_px: {float(np.mean(np.abs(aligned_bottom_gap))):.6f}",
            ]
        )
    summary_lines.append(f"optimization_cost: {float(result.cost):.6f}")
    summary_txt.write_text("\n".join(summary_lines) + "\n")

    print(f"contactphase_csv: {out_csv}")
    print(f"contactphase_reproj_csv: {reproj_csv}")
    print(f"contactphase_summary: {summary_txt}")


if __name__ == "__main__":
    main()
