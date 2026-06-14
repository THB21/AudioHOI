#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import numpy as np
from scipy.optimize import least_squares
import smplx
import torch

from contact_part_utils import (
    build_contact_identity,
    choose_active_contact_relation,
    normalize_contact_label,
    resolve_human_state_key,
)


def parse_float(row: dict[str, str], key: str, default: float = np.nan) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    return float(value)


def read_object_proxy_pose(path: Path) -> list[dict[str, float | int | str]]:
    rows = []
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
                    "qw": float(row.get("qw", 1.0) or 1.0),
                    "qx": float(row.get("qx", 0.0) or 0.0),
                    "qy": float(row.get("qy", 0.0) or 0.0),
                    "qz": float(row.get("qz", 0.0) or 0.0),
                    "radius_m": float(row.get("radius_m", 0.12) or 0.12),
                    "coord_frame": row.get("coord_frame", "gvhmr_incam"),
                    "u_obs": float(row.get("u_ref_obs", row.get("u_obs", 0.0)) or 0.0),
                    "v_obs": float(row.get("v_ref_obs", row.get("v_obs", 0.0)) or 0.0),
                    "floor_v": float(row.get("floor_v", 0.0) or 0.0),
                    "contact_frame": int(row.get("contact_frame", 0) or 0),
                    "audio_contact_frame": int(row.get("audio_contact_frame", 0) or 0),
                }
            )
    if not rows:
        raise RuntimeError(f"No object proxy pose rows found in {path}")
    return rows


def read_contact_offsets(path: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            offset = parse_float(row, "contact_depth_offset_m", 0.0)
            conf = parse_float(row, "contact_conf", 0.0)
            out[frame] = {
                "contact_depth_offset_m": offset,
                "contact_conf": conf,
                "contact_u": parse_float(row, "contact_u"),
                "contact_v": parse_float(row, "contact_v"),
            }
    return out


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
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
    model = smplx.create(str(body_models_root), model_type="smplx", gender="neutral", ext="npz", use_pca=False, flat_hand_mean=True, num_betas=10, batch_size=human_params["transl"].shape[0])
    with torch.inference_mode():
        output = model(body_pose=torch.from_numpy(human_params["body_pose"]), betas=torch.from_numpy(human_params["betas"]), global_orient=torch.from_numpy(human_params["global_orient"]), transl=torch.from_numpy(human_params["transl"]), return_verts=False)
    return output.joints.detach().cpu().numpy().astype(np.float64)


def reconstruct_xyz_from_uvz(u_obs: np.ndarray, v_obs: np.ndarray, z: np.ndarray, K: np.ndarray) -> np.ndarray:
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]
    x = (u_obs - cx) * z / fx
    y = (v_obs - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_floor_like_event(row: dict[str, str]) -> bool:
    ctype = str(row.get("contact_type", "") or "")
    target = str(row.get("target", "") or "")
    return ctype in {"floor_contact_event", "plane_support_contact_event"} or target in {"floor", "unknown_plane"}


def human_event_window_bounds(
    event_rows: list[dict[str, str]],
    frame_to_idx: dict[int, int],
) -> tuple[int | None, int | None]:
    if not frame_to_idx:
        return None, None
    min_frame = min(frame_to_idx)
    max_frame = max(frame_to_idx)
    starts: list[int] = []
    ends: list[int] = []
    for row in event_rows:
        if is_floor_like_event(row):
            continue
        if not str(row.get("contact_type", "") or "").endswith("_contact_event"):
            continue
        frame = int(row["frame"])
        start = int(row.get("window_start", frame) or frame)
        end = int(row.get("window_end", frame) or frame)
        start = min(max(start, min_frame), max_frame)
        end = min(max(end, min_frame), max_frame)
        if start in frame_to_idx:
            starts.append(frame_to_idx[start])
        if end in frame_to_idx:
            ends.append(frame_to_idx[end])
    if not starts or not ends:
        return None, None
    return min(starts), max(ends)


def infer_row_default_part(row: dict[str, str], fallback: str = "hand") -> str:
    for key in ("contact_part", "anchor_type"):
        value = str(row.get(key, "") or "").strip()
        if value in {"hand", "foot"}:
            return value
    target = str(row.get("target", "") or "").strip()
    if target.endswith("_hand"):
        return "hand"
    if target.endswith("_foot"):
        return "foot"
    return fallback


def build_state_contact_label(
    row: dict[str, str],
    floor_state_on: bool,
    human_state_on: bool,
    fallback_part: str,
) -> str:
    if floor_state_on and not human_state_on:
        return "floor"
    row_default_part = infer_row_default_part(row, fallback=fallback_part)
    return normalize_contact_label(row, default_part=row_default_part, fallback_side="right")


def build_anchor_segment_reference(
    z_init: np.ndarray,
    anchor_mask: np.ndarray,
    anchor_values: np.ndarray,
) -> np.ndarray:
    anchor_idx = np.flatnonzero(anchor_mask)
    if len(anchor_idx) == 0:
        raise RuntimeError("No anchors available for segmented reference")
    z_ref = np.asarray(z_init, dtype=np.float64).copy()
    first = int(anchor_idx[0])
    last = int(anchor_idx[-1])
    z_ref[: first + 1] = float(anchor_values[first])
    for left, right in zip(anchor_idx[:-1], anchor_idx[1:]):
        left = int(left)
        right = int(right)
        if right <= left:
            continue
        alpha = np.linspace(0.0, 1.0, right - left + 1, dtype=np.float64)
        z_ref[left : right + 1] = (1.0 - alpha) * float(anchor_values[left]) + alpha * float(anchor_values[right])
    z_ref[last:] = float(anchor_values[last])
    return np.maximum(z_ref, 0.20)


def solve_anchor_interpolation(
    z_ref: np.ndarray,
    anchor_mask: np.ndarray,
    anchor_values: np.ndarray,
    u_obs: np.ndarray,
    v_obs: np.ndarray,
    K: np.ndarray,
    times: np.ndarray,
    flight_mask: np.ndarray,
    w_ref: float,
    w_temp: float,
    w_phys_xz: float,
    w_phys_y: float,
    gravity_mps2: float,
) -> np.ndarray:
    free_idx = np.flatnonzero(~anchor_mask)
    anchor_idx = np.flatnonzero(anchor_mask)
    if len(anchor_idx) == 0:
        raise RuntimeError("No anchors available for anchor interpolation")
    dt = np.diff(times)
    dt_mean = float(np.mean(dt)) if len(dt) else (1.0 / 30.0)
    g_dt2 = gravity_mps2 * (dt_mean ** 2)
    flight_triplet = np.zeros(len(z_ref), dtype=bool)
    if len(z_ref) >= 3:
        flight_triplet[1:-1] = flight_mask[:-2] & flight_mask[1:-1] & flight_mask[2:]

    def unpack_z(free_values: np.ndarray) -> np.ndarray:
        z = np.asarray(z_ref, dtype=np.float64).copy()
        z[anchor_idx] = anchor_values[anchor_idx]
        z[free_idx] = free_values
        return np.maximum(z, 0.20)

    def residuals(free_values: np.ndarray) -> np.ndarray:
        z = unpack_z(free_values)
        xyz = reconstruct_xyz_from_uvz(u_obs, v_obs, z, K)
        x = xyz[:, 0]
        y = xyz[:, 1]
        residual_list = [w_ref * (z[free_idx] - z_ref[free_idx])]
        if len(z) >= 3:
            second_z = z[2:] - 2.0 * z[1:-1] + z[:-2]
            smooth_mask = ~anchor_mask[1:-1]
            if np.any(smooth_mask):
                residual_list.append(w_temp * second_z[smooth_mask])
            if np.any(flight_triplet[1:-1]):
                phys_mask = flight_triplet[1:-1]
                second_x = x[2:] - 2.0 * x[1:-1] + x[:-2]
                second_y = y[2:] - 2.0 * y[1:-1] + y[:-2]
                if w_phys_xz > 0.0:
                    residual_list.append(w_phys_xz * second_x[phys_mask])
                    residual_list.append(w_phys_xz * second_z[phys_mask])
                if w_phys_y > 0.0:
                    residual_list.append(w_phys_y * (second_y[phys_mask] - g_dt2))
        return np.concatenate([np.ravel(r) for r in residual_list]).astype(np.float64)

    x0 = z_ref[free_idx].copy()
    result = least_squares(residuals, x0=x0, method="trf", loss="soft_l1", f_scale=1.0, max_nfev=400)
    return unpack_z(result.x)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Experimental radius-free object-contact anchor interpolation.")
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--body-model-root", type=Path, default=Path("third-party/GVHMR/inputs/checkpoints/body_models"))
    parser.add_argument("--out-subdir", type=str, default="pose6d_object_proxy_contactphase_test")
    parser.add_argument("--contact-state-csv", type=Path, default=None)
    parser.add_argument("--contact-event-csv", type=Path, default=None)
    parser.add_argument("--object-proxy-observation-csv", type=Path, default=None)
    parser.add_argument("--object-pose-csv", type=Path, default=None)
    parser.add_argument("--support-geometry-json", type=Path, default=None)
    parser.add_argument("--default-part", type=str, choices=["hand", "foot"], default=None)
    parser.add_argument("--outside-window-mode", type=str, choices=["global_ref", "boundary_constant"], default="boundary_constant")
    parser.add_argument("--z-ref-mode", type=str, choices=["global_shift", "anchor_segment"], default="anchor_segment")
    parser.add_argument("--delta-stat", type=str, choices=["median", "mean"], default="median")
    parser.add_argument("--w-ref", type=float, default=0.7)
    parser.add_argument("--w-temp", type=float, default=5.0)
    parser.add_argument("--w-phys-xz", type=float, default=1.25)
    parser.add_argument("--w-phys-y", type=float, default=1.5)
    parser.add_argument("--gravity-mps2", type=float, default=9.81)
    args = parser.parse_args(argv)

    sample_dir = args.sample_dir
    results_dir = sample_dir / "results"
    out_dir = results_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    state_csv = args.contact_state_csv or (results_dir / "contact_candidates" / "contact_state_frames.csv")
    event_csv = args.contact_event_csv or (results_dir / "contact_candidates" / "contact_candidates_labeled.csv")
    proxy_obs_csv = args.object_proxy_observation_csv or (results_dir / "object_proxy_observations_test" / "object_proxy_observations.csv")
    support_json = args.support_geometry_json or (results_dir / "pose6d_object_proxy_test" / "support_geometry.json")
    pose_csv = args.object_pose_csv or (results_dir / "pose6d_object_proxy_test" / "object_pose6d_sharedcam_trajectory.csv")

    pose_rows = read_object_proxy_pose(pose_csv)
    proxy_offsets = read_contact_offsets(proxy_obs_csv)
    support = read_support_geometry(support_json) if support_json.exists() else None
    if support is not None:
        for row in pose_rows:
            row["floor_v"] = float(support["floor_v"])

    state_rows = read_rows(state_csv)
    event_rows = read_rows(event_csv)
    human = read_human_result(results_dir / "gvhmr" / "result.pkl")
    joints = build_body_joints(args.body_model_root, human)
    K = np.asarray(human["K_fullimg"], dtype=np.float64)
    if len(joints) < len(pose_rows):
        raise RuntimeError(f"GVHMR has fewer frames than object pose rows: {len(joints)} < {len(pose_rows)}")
    joints = joints[: len(pose_rows)]
    K = K[: len(pose_rows)]

    state_by_frame = {int(r["frame"]): r for r in state_rows}
    pose_rows = [r for r in pose_rows if int(r["frame"]) in state_by_frame]
    joints = joints[: len(pose_rows)]
    K = K[: len(pose_rows)]
    state_rows = [state_by_frame[int(r["frame"])] for r in pose_rows]
    object_frames = [int(r["frame"]) for r in pose_rows]
    frame_to_idx = {frame: idx for idx, frame in enumerate(object_frames)}

    explicit_parts = [
        infer_row_default_part(row, fallback="")
        for row in [*event_rows, *state_rows]
    ]
    explicit_parts = [part for part in explicit_parts if part in {"hand", "foot"}]
    fallback_part = args.default_part or (explicit_parts[0] if explicit_parts else "hand")
    human_event_frames = {
        int(row["frame"])
        for row in event_rows
        if not is_floor_like_event(row) and str(row.get("contact_type", "") or "").endswith("_contact_event")
    }
    floor_event_frames = {
        int(row["frame"])
        for row in event_rows
        if is_floor_like_event(row)
    }

    u_obs = np.asarray([r["u_obs"] for r in pose_rows], dtype=np.float64)
    v_obs = np.asarray([r["v_obs"] for r in pose_rows], dtype=np.float64)
    times = np.asarray([r["time"] for r in pose_rows], dtype=np.float64)
    z_init = np.asarray([r["tz"] for r in pose_rows], dtype=np.float64)

    human_event_mask = np.asarray([f in human_event_frames for f in object_frames], dtype=bool)
    floor_event_mask = np.asarray([f in floor_event_frames for f in object_frames], dtype=bool)
    floor_state_mask = np.asarray(
        [
            int(r.get("floor_contact_state", 0) or 0) == 1
            or int(r.get("plane_support_state", 0) or 0) == 1
            for r in state_rows
        ],
        dtype=bool,
    )
    state_key = resolve_human_state_key(state_rows[0])
    if state_key is None:
        raise RuntimeError("No generic human contact state field found")
    human_state_mask = np.asarray([int(r[state_key]) == 1 for r in state_rows], dtype=bool)
    flight_mask = ~(human_state_mask | floor_state_mask)

    contact_labels = [
        build_state_contact_label(row, bool(floor_state_mask[idx]), bool(human_state_mask[idx]), fallback_part)
        for idx, row in enumerate(state_rows)
    ]
    non_floor_labels = [label for label in contact_labels if label != "floor"]
    fallback_label = non_floor_labels[0] if non_floor_labels else f"right_{fallback_part}"
    part_y, part_z, part_name = choose_active_contact_relation(joints, contact_labels, fallback_label=fallback_label)
    if not np.any(human_event_mask):
        raise RuntimeError("No human contact events found; cannot compute global Delta-Z")

    contact_offset = np.asarray(
        [proxy_offsets.get(frame, {}).get("contact_depth_offset_m", 0.0) for frame in object_frames],
        dtype=np.float64,
    )
    anchor_values = part_z - contact_offset
    deltas = anchor_values[human_event_mask] - z_init[human_event_mask]
    global_z_shift = float(np.median(deltas) if args.delta_stat == "median" else np.mean(deltas))
    z_ref_global = np.maximum(z_init + global_z_shift, 0.20)
    z_ref_segment = build_anchor_segment_reference(z_init, human_event_mask, anchor_values)
    z_ref = z_ref_segment if args.z_ref_mode == "anchor_segment" else z_ref_global

    z_final = solve_anchor_interpolation(
        z_ref=z_ref,
        anchor_mask=human_event_mask,
        anchor_values=anchor_values,
        u_obs=u_obs,
        v_obs=v_obs,
        K=K,
        times=times,
        flight_mask=flight_mask,
        w_ref=args.w_ref,
        w_temp=args.w_temp,
        w_phys_xz=args.w_phys_xz,
        w_phys_y=args.w_phys_y,
        gravity_mps2=args.gravity_mps2,
    )
    if args.outside_window_mode == "boundary_constant":
        anchor_idx = np.flatnonzero(human_event_mask)
        interp_start = int(anchor_idx[0])
        interp_end = int(anchor_idx[-1])
        if interp_start > 0:
            z_final[:interp_start] = max(0.20, float(z_final[interp_start]))
        if interp_end + 1 < len(z_final):
            z_final[interp_end + 1 :] = max(0.20, float(z_final[interp_end]))
    else:
        interp_start = interp_end = -1

    xyz_final = reconstruct_xyz_from_uvz(u_obs, v_obs, z_final, K)
    z_contact_final = z_final + contact_offset
    out_rows = []
    reproj_rows = []
    for idx, row in enumerate(pose_rows):
        contact_part, contact_side, contact_label = build_contact_identity(
            active_label=str(part_name[idx]),
            event_on=bool(human_event_mask[idx]),
            floor_event_on=bool(floor_event_mask[idx]),
            default_part=fallback_part,
        )
        meta = proxy_offsets.get(int(row["frame"]), {})
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
                "radius_obs_px": "",
                "u_proj": f"{row['u_obs']:.3f}",
                "v_proj": f"{row['v_obs']:.3f}",
                "radius_proj_px": "",
                "bottom_proj_v": "",
                "floor_v": f"{row['floor_v']:.3f}",
                "support_type": support["support_type"] if support is not None else "floor",
                "support_source": support["source"] if support is not None else "object_proxy_csv",
                "support_confidence": f"{float(support['confidence']):.6f}" if support is not None else "",
                "residual_px": "0.000000",
                "contact_frame": int(human_event_mask[idx]),
                "audio_contact_frame": row["audio_contact_frame"],
                "human_contact_event": int(human_event_mask[idx]),
                "floor_contact_event": int(floor_event_mask[idx]),
                "human_contact_state": int(human_state_mask[idx]),
                "floor_contact_state": int(floor_state_mask[idx]),
                "contact_part": contact_part,
                "contact_side": contact_side,
                "contact_label": contact_label,
                "active_part": str(part_name[idx]),
                "active_part_y": f"{part_y[idx]:.6f}",
                "active_part_z": f"{part_z[idx]:.6f}",
                "u_ref_obs": f"{row['u_obs']:.3f}",
                "v_ref_obs": f"{row['v_obs']:.3f}",
                "contact_u": f"{meta.get('contact_u', np.nan):.3f}" if np.isfinite(meta.get("contact_u", np.nan)) else "",
                "contact_v": f"{meta.get('contact_v', np.nan):.3f}" if np.isfinite(meta.get("contact_v", np.nan)) else "",
                "contact_depth_offset_m": f"{contact_offset[idx]:.6f}",
                "z_contact_final": f"{z_contact_final[idx]:.6f}",
                "global_z_ref": f"{z_ref[idx]:.6f}",
                "z_ref_global_shift": f"{z_ref_global[idx]:.6f}",
                "z_ref_anchor_segment": f"{z_ref_segment[idx]:.6f}",
                "contact_depth_gap": f"{(z_contact_final[idx] - part_z[idx]):.6f}",
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

    out_csv = out_dir / "object_pose6d_sharedcam_contactphase_trajectory.csv"
    reproj_csv = out_dir / "object_pose6d_sharedcam_contactphase_reprojection_comparison.csv"
    summary_txt = out_dir / "object_pose6d_sharedcam_contactphase_summary.txt"
    write_csv(
        out_csv,
        out_rows,
        [
            "frame","time","tx","ty","tz","qw","qx","qy","qz","radius_m","coord_frame",
            "u_obs","v_obs","radius_obs_px","u_proj","v_proj","radius_proj_px","bottom_proj_v",
            "floor_v","support_type","support_source","support_confidence","residual_px","contact_frame","audio_contact_frame",
            "human_contact_event","floor_contact_event","human_contact_state","floor_contact_state",
            "contact_part","contact_side","contact_label","active_part","active_part_y","active_part_z",
            "u_ref_obs","v_ref_obs","contact_u","contact_v","contact_depth_offset_m","z_contact_final","global_z_ref","z_ref_global_shift","z_ref_anchor_segment","contact_depth_gap",
        ],
    )
    write_csv(reproj_csv, reproj_rows, ["frame","u_obs","v_obs","u_reproj","v_reproj","error_u","error_v","error_px"])
    with summary_txt.open("w") as f:
        f.write("Experimental radius-free object-contact anchor interpolation.\n")
        f.write(f"default_part: {fallback_part}\n")
        f.write("contact_part_mode: auto_inferred_from_state_event\n")
        f.write(f"outside_window_mode: {args.outside_window_mode}\n")
        f.write(f"z_ref_mode: {args.z_ref_mode}\n")
        f.write(f"w_phys_xz: {args.w_phys_xz:.6f}\n")
        f.write(f"w_phys_y: {args.w_phys_y:.6f}\n")
        f.write(f"gravity_mps2: {args.gravity_mps2:.6f}\n")
        f.write(f"global_z_shift_from_human_events_m: {global_z_shift:.6f}\n")
        f.write("delta_contact_mode: object_proxy_offset\n")
        if interp_start >= 0:
            f.write(f"interp_start_frame: {pose_rows[interp_start]['frame']}\n")
            f.write(f"interp_end_frame: {pose_rows[interp_end]['frame']}\n")
        f.write(f"num_frames: {len(pose_rows)}\n")
        f.write(f"num_human_event_frames: {int(np.count_nonzero(human_event_mask))}\n")
        f.write(f"num_floor_event_frames: {int(np.count_nonzero(floor_event_mask))}\n")
    print(f"object_contact_csv: {out_csv}")
    print(f"object_contact_reproj_csv: {reproj_csv}")
    print(f"object_contact_summary: {summary_txt}")


if __name__ == "__main__":
    main()
