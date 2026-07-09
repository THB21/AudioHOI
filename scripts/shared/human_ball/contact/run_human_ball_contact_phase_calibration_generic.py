#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import smplx
import torch

from contact_part_utils import (
    build_contact_identity,
    choose_active_contact_relation,
    event_frames_by_type,
    human_event_frames_generic,
    infer_default_part,
    normalize_contact_label,
    resolve_human_state_key,
)

BALL_RADIUS_M = 0.12


def read_ball_pose(path: Path) -> list[dict[str, float | int | str]]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
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
            })
    if not rows:
        raise RuntimeError(f"No ball rows found in {path}")
    return rows


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
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


def z_objective(
    z: np.ndarray,
    z_ref: np.ndarray,
    u_obs: np.ndarray,
    v_obs: np.ndarray,
    r_obs: np.ndarray,
    K: np.ndarray,
    floor_v: np.ndarray,
    floor_state_mask: np.ndarray,
    floor_event_mask: np.ndarray,
    human_event_mask: np.ndarray,
    human_z: np.ndarray,
    w_size: float,
    w_floor_window: float,
    w_floor_contact: float,
    w_human_contact: float,
    w_z_anchor: float,
    w_temp: float,
) -> np.ndarray:
    z = np.maximum(z, 0.20)
    z = z.copy()
    z[human_event_mask] = human_z[human_event_mask]
    ball_xyz = reconstruct_xyz_from_uvz(u_obs, v_obs, z, K)
    r_proj, bottom_v = project_ball(ball_xyz, K, BALL_RADIUS_M)

    residuals = [w_size * (r_proj - r_obs)]
    if np.any(floor_state_mask):
        residuals.append(w_floor_window * ((bottom_v[floor_state_mask] - floor_v[floor_state_mask]) / 20.0))
    if np.any(floor_event_mask):
        residuals.append(w_floor_contact * ((bottom_v[floor_event_mask] - floor_v[floor_event_mask]) / 20.0))
    if np.any(human_event_mask):
        residuals.append(w_human_contact * (z[human_event_mask] - human_z[human_event_mask]))

    anchor_delta = z - z_ref
    anchor_delta = anchor_delta.copy()
    anchor_delta[human_event_mask] = 0.0
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generic phase-aware human-ball contact calibration.")
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--body-model-root", type=Path, default=Path("third-party/GVHMR/inputs/checkpoints/body_models"))
    parser.add_argument("--out-subdir", type=str, default="pose6d_sharedcam_contactphase_generic")
    parser.add_argument("--contact-state-csv", type=Path, default=None)
    parser.add_argument("--contact-event-csv", type=Path, default=None)
    parser.add_argument("--default-part", type=str, choices=["hand", "foot"], default=None)
    parser.add_argument("--w-size", type=float, default=0.12)
    parser.add_argument("--w-floor-window", type=float, default=0.45)
    parser.add_argument("--w-floor-contact", type=float, default=1.35)
    parser.add_argument("--w-human-contact", type=float, default=1.75)
    parser.add_argument("--w-z-anchor", type=float, default=0.70)
    parser.add_argument("--w-temp", type=float, default=1.35)
    args = parser.parse_args(argv)

    sample_dir = args.sample_dir
    results_dir = sample_dir / "results"
    out_dir = results_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    state_csv = args.contact_state_csv or (results_dir / "contact_candidates" / "contact_state_frames.csv")
    event_csv = args.contact_event_csv or (results_dir / "contact_candidates" / "contact_candidates_labeled.csv")

    ball_rows = read_ball_pose(results_dir / "pose6d_sharedcam" / "ball_pose6d_sharedcam_trajectory.csv")
    state_rows = read_rows(state_csv)
    event_rows = read_rows(event_csv)
    human = read_human_result(results_dir / "gvhmr" / "result.pkl")
    joints = build_body_joints(args.body_model_root, human)

    if not (len(ball_rows) == len(joints) == len(state_rows)):
        raise RuntimeError("Frame count mismatch among sharedcam ball, GVHMR human, and contact states")

    ball_frames = [int(r["frame"]) for r in ball_rows]
    state_frames = [int(r["frame"]) for r in state_rows]
    if ball_frames != state_frames:
        raise RuntimeError("Frame mismatch between sharedcam ball rows and contact state rows")

    default_part = args.default_part or infer_default_part([*state_rows, *event_rows], fallback="hand")
    human_event_frames = human_event_frames_generic(event_rows)
    floor_event_frames = event_frames_by_type(event_rows, {"floor_contact_event"})

    K = np.asarray(human["K_fullimg"], dtype=np.float64)
    u_obs = np.asarray([r["u_obs"] for r in ball_rows], dtype=np.float64)
    v_obs = np.asarray([r["v_obs"] for r in ball_rows], dtype=np.float64)
    r_obs = np.asarray([r["radius_obs_px"] for r in ball_rows], dtype=np.float64)
    floor_v = np.asarray([r["floor_v"] for r in ball_rows], dtype=np.float64)
    z_init = np.asarray([r["tz"] for r in ball_rows], dtype=np.float64)

    human_event_mask = np.asarray([f in human_event_frames for f in ball_frames], dtype=bool)
    floor_event_mask = np.asarray([f in floor_event_frames for f in ball_frames], dtype=bool)
    floor_state_mask = np.asarray([int(r["floor_contact_state"]) == 1 for r in state_rows], dtype=bool)
    state_key = resolve_human_state_key(state_rows[0])
    if state_key is None:
        raise RuntimeError("No generic human contact state field found")
    human_state_mask = np.asarray([int(r[state_key]) == 1 for r in state_rows], dtype=bool)

    contact_labels = [normalize_contact_label(r, default_part=default_part, fallback_side="right") for r in state_rows]
    part_y, part_z, part_name = choose_active_contact_relation(joints, contact_labels, fallback_label=f"right_{default_part}")

    if np.any(human_event_mask):
        global_z_shift = float(np.median(part_z[human_event_mask] - z_init[human_event_mask]))
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
            z_ref, u_obs, v_obs, r_obs, K, floor_v, floor_state_mask, floor_event_mask, human_event_mask, part_z,
            args.w_size, args.w_floor_window, args.w_floor_contact, args.w_human_contact, args.w_z_anchor, args.w_temp,
        ),
    )

    z_opt = np.maximum(result.x.astype(np.float64), 0.20)
    z_opt[human_event_mask] = part_z[human_event_mask]
    xyz_opt = reconstruct_xyz_from_uvz(u_obs, v_obs, z_opt, K)
    r_proj, bottom_proj = project_ball(xyz_opt, K, BALL_RADIUS_M)

    out_rows = []
    reproj_rows = []
    for idx, row in enumerate(ball_rows):
        contact_part, contact_side, contact_label = build_contact_identity(
            active_label=str(part_name[idx]),
            event_on=bool(human_event_mask[idx]),
            floor_event_on=bool(floor_event_mask[idx]),
            default_part=default_part,
        )
        out_rows.append({
            "frame": row["frame"], "time": f"{row['time']:.6f}", "tx": f"{xyz_opt[idx,0]:.6f}", "ty": f"{xyz_opt[idx,1]:.6f}", "tz": f"{xyz_opt[idx,2]:.6f}",
            "qw": f"{row['qw']:.6f}", "qx": f"{row['qx']:.6f}", "qy": f"{row['qy']:.6f}", "qz": f"{row['qz']:.6f}",
            "radius_m": f"{row['radius_m']:.6f}", "coord_frame": row["coord_frame"],
            "u_obs": f"{row['u_obs']:.3f}", "v_obs": f"{row['v_obs']:.3f}", "radius_obs_px": f"{row['radius_obs_px']:.3f}",
            "u_proj": f"{row['u_obs']:.3f}", "v_proj": f"{row['v_obs']:.3f}", "radius_proj_px": f"{r_proj[idx]:.3f}", "bottom_proj_v": f"{bottom_proj[idx]:.3f}",
            "floor_v": f"{row['floor_v']:.3f}", "residual_px": "0.000000", "contact_frame": int(human_event_mask[idx]),
            "audio_contact_frame": row["audio_contact_frame"],
            "human_contact_event": int(human_event_mask[idx]), "floor_contact_event": int(floor_event_mask[idx]),
            "human_contact_state": int(human_state_mask[idx]), "floor_contact_state": int(floor_state_mask[idx]),
            "contact_part": contact_part, "contact_side": contact_side, "contact_label": contact_label,
            "active_part": str(part_name[idx]), "active_part_y": f"{part_y[idx]:.6f}", "active_part_z": f"{part_z[idx]:.6f}",
            "global_z_ref": f"{z_ref[idx]:.6f}", "contact_depth_gap": f"{(xyz_opt[idx,2] - part_z[idx]):.6f}",
        })
        reproj_rows.append({
            "frame": row["frame"], "u_obs": f"{row['u_obs']:.3f}", "v_obs": f"{row['v_obs']:.3f}",
            "u_reproj": f"{row['u_obs']:.3f}", "v_reproj": f"{row['v_obs']:.3f}", "error_u": "0.000000", "error_v": "0.000000", "error_px": "0.000000",
        })

    out_csv = out_dir / "ball_pose6d_sharedcam_contactphase_trajectory.csv"
    reproj_csv = out_dir / "ball_pose6d_sharedcam_contactphase_reprojection_comparison.csv"
    summary_txt = out_dir / "ball_pose6d_sharedcam_contactphase_summary.txt"

    write_csv(out_csv, out_rows, [
        "frame","time","tx","ty","tz","qw","qx","qy","qz","radius_m","coord_frame",
        "u_obs","v_obs","radius_obs_px","u_proj","v_proj","radius_proj_px","bottom_proj_v",
        "floor_v","residual_px","contact_frame","audio_contact_frame",
        "human_contact_event","floor_contact_event","human_contact_state","floor_contact_state",
        "contact_part","contact_side","contact_label","active_part","active_part_y","active_part_z",
        "global_z_ref","contact_depth_gap",
    ])
    write_csv(reproj_csv, reproj_rows, ["frame", "u_obs", "v_obs", "u_reproj", "v_reproj", "error_u", "error_v", "error_px"])

    with summary_txt.open("w") as f:
        f.write("Generic phase-aware human-ball contact calibration.\n")
        f.write(f"default_part: {default_part}\n")
        f.write(f"global_z_shift_from_human_events_m: {global_z_shift:.6f}\n")
        f.write(f"num_frames: {len(ball_rows)}\n")
        f.write(f"num_human_event_frames: {int(np.count_nonzero(human_event_mask))}\n")
        f.write(f"num_floor_event_frames: {int(np.count_nonzero(floor_event_mask))}\n")

    print(f"contactphase_csv: {out_csv}")
    print(f"contactphase_reproj_csv: {reproj_csv}")
    print(f"contactphase_summary: {summary_txt}")


if __name__ == "__main__":
    main()
