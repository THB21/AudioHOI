#!/usr/bin/env python3
"""Anchor-only z refinement with hard hand-event constraints and global smoothing.

This branch explicitly separates:
- anchor values: hand-contact event depths
- trajectory shape: smooth interpolation between anchors

No local hand windows, no event-centered dragging. Hand events only set exact
depth values; everything between them is determined by a weak global reference
and a strong second-difference smoothness prior.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import smplx
import torch



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


def read_contact_state_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No contact state rows found in {path}")
    return rows


def read_contact_event_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No contact event rows found in {path}")
    return rows


    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            radius_str = row.get("estimated_radius_m", "")
            if radius_str:
                try:
                    radius_by_frame[frame] = float(radius_str)
                except ValueError:
                    pass
    return radius_by_frame


def read_object_observations(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            cx = row.get("center_x", "")
            cy = row.get("center_y", "")
            radius = row.get("enclosing_radius_px", "")
            if not cx or not cy or not radius:
                continue
            frame = int(row["frame"])
            rows[frame] = {
                "u_obs": float(cx),
                "v_obs": float(cy),
                "radius_obs_px": float(radius),
            }
    return rows


def read_support_geometry(path: Path) -> dict[str, float | str]:
    with path.open() as f:
        payload = json.load(f)
    return {
        "support_type": str(payload.get("support_type", "floor")),
        "floor_v": float(payload["floor_v"]),
        "source": str(payload.get("source", "unknown")),
        "confidence": float(payload.get("confidence", 0.0)),
    }


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


def choose_right_hand_relation(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, right = build_palm_centers(joints)
    hand_y = right[:, 1]
    hand_z = right[:, 2]
    hand_name = np.full(len(right), "right", dtype=object)
    return hand_y, hand_z, hand_name


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def solve_anchor_interpolation(
    z_ref: np.ndarray,
    anchor_mask: np.ndarray,
    anchor_values: np.ndarray,
    w_ref: float,
    w_temp: float,
) -> np.ndarray:
    n = len(z_ref)
    H = np.zeros((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)

    # Weak global reference to shifted baseline.
    for i in range(n):
        w2 = w_ref * w_ref
        H[i, i] += w2
        b[i] += w2 * z_ref[i]

    # Strong global smoothness on second differences.
    coeff = np.asarray([1.0, -2.0, 1.0], dtype=np.float64)
    for t in range(1, n - 1):
        idx = np.asarray([t - 1, t, t + 1], dtype=int)
        local = (w_temp * coeff).reshape(3, 1)
        H[np.ix_(idx, idx)] += local @ local.T

    free_idx = np.flatnonzero(~anchor_mask)
    anchor_idx = np.flatnonzero(anchor_mask)
    if len(anchor_idx) == 0:
        raise RuntimeError("No anchors available for anchor interpolation")

    H_ff = H[np.ix_(free_idx, free_idx)]
    H_fa = H[np.ix_(free_idx, anchor_idx)]
    b_f = b[free_idx] - H_fa @ anchor_values[anchor_idx]
    z = np.zeros(n, dtype=np.float64)
    z[anchor_idx] = anchor_values[anchor_idx]
    z[free_idx] = np.linalg.solve(H_ff, b_f)
    return np.maximum(z, 0.20)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hard hand-event anchors plus smooth interpolation between anchors.")
    parser.add_argument("--sample-dir", type=Path, default=Path("samples/basketball_01"))
    parser.add_argument("--body-model-root", type=Path, default=Path("third-party/GVHMR/inputs/checkpoints/body_models"))
    parser.add_argument("--out-subdir", type=str, default="pose6d_sharedcam_contactphase_anchorinterp")
    parser.add_argument("--contact-state-csv", type=Path, default=None)
    parser.add_argument("--contact-event-csv", type=Path, default=None)
    parser.add_argument(
        "--object-observation-csv",
        type=Path,
        default=None,
        help="Optional generic object observation table. If present, use its center/radius observations to keep refinement inputs aligned with the shared observation layer.",
    )
    parser.add_argument(
        "--support-geometry-json",
        type=Path,
        default=None,
        help="Optional sharedcam support geometry json. If present, use it as the explicit scene-level support definition instead of implicitly trusting only per-row floor_v fields.",
    )
    parser.add_argument("--delta-stat", type=str, choices=["median", "mean"], default="median")
    parser.add_argument("--w-ref", type=float, default=0.7)
    parser.add_argument("--w-temp", type=float, default=5.0)
    args = parser.parse_args()

    sample_dir = args.sample_dir
    results_dir = sample_dir / "results"
    out_dir = results_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    state_csv = args.contact_state_csv or (results_dir / "contact_candidates" / "contact_state_frames.csv")
    event_csv = args.contact_event_csv or (results_dir / "contact_candidates" / "contact_candidates_labeled.csv")
    object_obs_csv = args.object_observation_csv or (results_dir / "object_observations" / "object_observations.csv")
    support_json = args.support_geometry_json or (results_dir / "pose6d_sharedcam" / "support_geometry.json")

    ball_rows = read_ball_pose(results_dir / "pose6d_sharedcam" / "ball_pose6d_sharedcam_trajectory.csv")
    if object_obs_csv.exists():
        object_obs = read_object_observations(object_obs_csv)
        for row in ball_rows:
            obs = object_obs.get(int(row["frame"]))
            if obs is None:
                continue
            row["u_obs"] = obs["u_obs"]
            row["v_obs"] = obs["v_obs"]
            row["radius_obs_px"] = obs["radius_obs_px"]

    support = None
    if support_json.exists():
        support = read_support_geometry(support_json)
        for row in ball_rows:
            row["floor_v"] = float(support["floor_v"])

    state_rows = read_contact_state_rows(state_csv)
    event_rows = read_contact_event_rows(event_csv)
    human = read_human_result(results_dir / "gvhmr" / "result.pkl")
    joints = build_body_joints(args.body_model_root, human)

    if not (len(ball_rows) == len(joints) == len(state_rows)):
        raise RuntimeError("Frame count mismatch among sharedcam ball, GVHMR human, and contact states")

    ball_frames = [int(r["frame"]) for r in ball_rows]
    state_frames = [int(r["frame"]) for r in state_rows]
    if ball_frames != state_frames:
        raise RuntimeError("Frame mismatch between sharedcam ball rows and contact state rows")

    hand_event_frames = {int(r["frame"]) for r in event_rows if r["contact_type"] == "hand_contact_event"}
    floor_event_frames = {int(r["frame"]) for r in event_rows if r["contact_type"] == "floor_contact_event"}

    K = np.asarray(human["K_fullimg"], dtype=np.float64)
    u_obs = np.asarray([r["u_obs"] for r in ball_rows], dtype=np.float64)
    v_obs = np.asarray([r["v_obs"] for r in ball_rows], dtype=np.float64)
    z_init = np.asarray([r["tz"] for r in ball_rows], dtype=np.float64)

    hand_event_mask = np.asarray([f in hand_event_frames for f in ball_frames], dtype=bool)
    floor_event_mask = np.asarray([f in floor_event_frames for f in ball_frames], dtype=bool)
    floor_state_mask = np.asarray([int(r["floor_contact_state"]) == 1 for r in state_rows], dtype=bool)
    hand_state_mask = np.asarray([int(r["hand_contact_state"]) == 1 for r in state_rows], dtype=bool)

    hand_y, hand_z, hand_name = choose_right_hand_relation(joints)
    if not np.any(hand_event_mask):
        raise RuntimeError("No hand contact events found; cannot compute global Delta-Z")

    deltas = hand_z[hand_event_mask] - z_init[hand_event_mask]
    global_z_shift = float(np.median(deltas) if args.delta_stat == "median" else np.mean(deltas))
    
    z_ref = np.maximum(z_init + global_z_shift, 0.20)

    z_final = solve_anchor_interpolation(
        z_ref=z_ref,
        anchor_mask=hand_event_mask,
        anchor_values=hand_z,
        w_ref=args.w_ref,
        w_temp=args.w_temp,
    )
    xyz_final = reconstruct_xyz_from_uvz(u_obs, v_obs, z_final, K)
    radius_m = float(ball_rows[0]["radius_m"])
    r_proj, bottom_proj = project_ball(xyz_final, K, radius_m)

    out_rows: list[dict[str, object]] = []
    reproj_rows: list[dict[str, object]] = []
    for idx, row in enumerate(ball_rows):
        out_rows.append(
            {
                "frame": row["frame"],
                "time": f"{row['time']:.6f}",
                "tx": f"{xyz_final[idx,0]:.6f}",
                "ty": f"{xyz_final[idx,1]:.6f}",
                "tz": f"{xyz_final[idx,2]:.6f}",
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
                "support_type": support["support_type"] if support is not None else "floor",
                "support_source": support["source"] if support is not None else "sharedcam_csv",
                "support_confidence": f"{float(support['confidence']):.6f}" if support is not None else "",
                "residual_px": "0.000000",
                "contact_frame": int(hand_event_mask[idx]),
                "audio_contact_frame": row["audio_contact_frame"],
                "hand_contact_event": int(hand_event_mask[idx]),
                "floor_contact_event": int(floor_event_mask[idx]),
                "hand_contact_state": int(hand_state_mask[idx]),
                "floor_contact_state": int(floor_state_mask[idx]),
                "active_hand": hand_name[idx],
                "active_hand_y": f"{hand_y[idx]:.6f}",
                "active_hand_z": f"{hand_z[idx]:.6f}",
                "global_z_ref": f"{z_ref[idx]:.6f}",
                "contact_depth_gap": f"{(xyz_final[idx,2] - hand_z[idx]):.6f}",
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
            "floor_v","support_type","support_source","support_confidence","residual_px","contact_frame","audio_contact_frame",
            "hand_contact_event","floor_contact_event","hand_contact_state","floor_contact_state",
            "active_hand","active_hand_y","active_hand_z","global_z_ref","contact_depth_gap",
        ],
    )
    write_csv(
        reproj_csv,
        reproj_rows,
        ["frame", "u_obs", "v_obs", "u_reproj", "v_reproj", "error_u", "error_v", "error_px"],
    )

    baseline_gap = hand_z[hand_event_mask] - z_init[hand_event_mask]
    aligned_gap = hand_z[hand_event_mask] - z_final[hand_event_mask]
    with summary_txt.open("w") as f:
        f.write("Anchor interpolation z refinement.\n")
        f.write("Hand events provide exact z anchors; all between-anchor shape comes from global reference plus second-difference smoothness.\n")
        f.write(f"delta_stat: {args.delta_stat}\n")
        f.write(f"w_ref: {args.w_ref:.6f}\n")
        f.write(f"w_temp: {args.w_temp:.6f}\n")
        f.write(f"global_z_shift_from_hand_events_m: {global_z_shift:.6f}\n")
        if support is not None:
            f.write(f"support_type: {support['support_type']}\n")
            f.write(f"support_floor_v: {float(support['floor_v']):.6f}\n")
            f.write(f"support_source: {support['source']}\n")
            f.write(f"support_confidence: {float(support['confidence']):.6f}\n")
        f.write(f"num_frames: {len(ball_rows)}\n")
        f.write(f"num_hand_event_frames: {int(hand_event_mask.sum())}\n")
        f.write(f"mean_abs_delta_z_m: {float(np.mean(np.abs(z_final - z_init))):.6f}\n")
        f.write(f"baseline_hand_event_depth_gap_mean_m: {float(np.mean(np.abs(baseline_gap))):.6f}\n")
        f.write(f"aligned_hand_event_depth_gap_mean_m: {float(np.mean(np.abs(aligned_gap))):.6f}\n")

    print(f"contactphase_csv: {out_csv}")
    print(f"contactphase_reproj_csv: {reproj_csv}")
    print(f"contactphase_summary: {summary_txt}")


if __name__ == "__main__":
    main()
