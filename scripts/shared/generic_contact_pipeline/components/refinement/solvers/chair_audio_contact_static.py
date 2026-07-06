#!/usr/bin/env python3
"""Apply chair audio/contact phase constraints after semantic 6D initialization."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


POSE_KEYS = ["rx_delta", "ry_delta", "rz_delta", "tx", "ty", "tz", "rear_joint_angle", "seat_joint_angle"]
REPO = Path(__file__).resolve().parents[6]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows


def ff(row: dict[str, str], key: str, default: float = math.nan) -> float:
    val = row.get(key, "")
    if val is None or val == "":
        return default
    return float(val)


def pick_place_frame(audio_rows: list[dict[str, str]], n_frames: int) -> int:
    scored = []
    for row in audio_rows:
        fr = int(round(ff(row, "audio_frame")))
        score = ff(row, "audio_score", 0.0)
        if fr >= max(20, int(n_frames * 0.45)):
            scored.append((score, fr))
    if not scored:
        return n_frames
    return int(max(scored)[1])


def pick_contact_start_frame(audio_rows: list[dict[str, str]], place_frame: int, n_frames: int) -> tuple[int, str]:
    min_frame = max(8, int(n_frames * 0.08))
    candidates: list[tuple[int, float]] = []
    for row in audio_rows:
        fr = int(round(ff(row, "audio_frame")))
        score = ff(row, "audio_score", 0.0)
        if min_frame <= fr < place_frame and score >= 0.15:
            candidates.append((fr, score))
    if not candidates:
        return 1, "default_first_frame_no_audio_contact_candidate"
    fr, _ = min(candidates, key=lambda item: item[0])
    return int(fr), "early_audio_contact_candidate"


def support_v(row: dict[str, str]) -> float:
    vals = [ff(row, k) for k in ("support_center_v", "front_left_foot_v", "front_right_foot_v")]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.median(vals)) if vals else math.nan


def semantic_center(row: dict[str, str], prefix: str) -> np.ndarray:
    u1, v1 = ff(row, f"{prefix}_left_u"), ff(row, f"{prefix}_left_v")
    u2, v2 = ff(row, f"{prefix}_right_u"), ff(row, f"{prefix}_right_v")
    if not all(np.isfinite(v) for v in (u1, v1, u2, v2)):
        return np.array([math.nan, math.nan], dtype=float)
    return np.array([(u1 + u2) * 0.5, (v1 + v2) * 0.5], dtype=float)


def pick_lift_frame(
    obs_rows: list[dict[str, str]],
    contact_start: int,
    place_frame: int,
    support_drop_px: float = 18.0,
    top_motion_px: float = 6.0,
) -> tuple[int, str]:
    pre_rows = obs_rows[: min(8, len(obs_rows))]
    floor_vals = [support_v(r) for r in pre_rows]
    floor_vals = [v for v in floor_vals if np.isfinite(v)]
    floor_ref = float(np.median(floor_vals)) if floor_vals else math.nan
    top_base_vals = [semantic_center(r, "top_rail") for r in pre_rows]
    top_base_vals = [v for v in top_base_vals if np.all(np.isfinite(v))]
    top_ref = np.median(np.stack(top_base_vals, axis=0), axis=0) if top_base_vals else np.array([math.nan, math.nan])
    for i, row in enumerate(obs_rows):
        fr = int(row["frame"])
        if fr < 8 or fr >= place_frame:
            continue
        top = semantic_center(row, "top_rail")
        if np.all(np.isfinite(top)) and np.all(np.isfinite(top_ref)):
            top_motion = float(np.linalg.norm(top - top_ref))
            if top_motion >= top_motion_px:
                return fr, "semantic_top_rail_motion"
        if not np.isfinite(floor_ref):
            continue
        win = obs_rows[i : min(len(obs_rows), i + 3)]
        vals = [support_v(r) for r in win]
        vals = [v for v in vals if np.isfinite(v)]
        if len(vals) >= 2 and float(np.median(vals)) < floor_ref - support_drop_px:
            return fr, "support_points_airborne_after_audio_contact"
    return contact_start, "contact_audio_no_airborne_support_drop"


def pose_array(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([[ff(r, k, 0.0) for k in POSE_KEYS] for r in rows], dtype=float)


def detect_static_start(rows: list[dict[str, str]], place_frame: int) -> int:
    P = pose_array(rows)
    d = np.zeros((len(P), 3), dtype=float)
    if len(P) > 1:
        diff = np.diff(P, axis=0)
        d[1:, 0] = np.linalg.norm(diff[:, :3], axis=1)
        d[1:, 1] = np.linalg.norm(diff[:, 3:6], axis=1)
        d[1:, 2] = np.linalg.norm(diff[:, 6:8], axis=1)
    start_idx = max(0, min(len(rows) - 1, place_frame + 10 - 1))
    for i in range(start_idx, len(rows) - 8):
        win = d[i : i + 8]
        if float(np.median(win[:, 0])) < 0.010 and float(np.median(win[:, 1])) < 0.018:
            return int(rows[i]["frame"])
    return min(len(rows), place_frame + 16)


def resolve_static_start(rows: list[dict[str, str]], place_frame: int, use_object_motion_gate: bool = False) -> tuple[int, str]:
    """Choose release/static start for chair.

    Unlike the mug sequence, the chair pipeline does not yet have a reliable
    object motion gate.  In that case the strongest late audio impact is the
    release/static event; pose velocity is too noisy and can delay static by
    many frames.
    """
    if not use_object_motion_gate:
        return place_frame, "audio_place_frame_no_object_motion_gate"
    return detect_static_start(rows, place_frame), "audio_events_plus_pose_velocity"


def freeze_pose(rows: list[dict[str, str]], static_start: int) -> list[dict[str, str]]:
    out = [dict(r) for r in rows]
    idx = next((i for i, r in enumerate(out) if int(r["frame"]) >= static_start), len(out) - 1)
    end = min(len(out), idx + 8)
    P = pose_array(out[idx:end])
    frozen = np.median(P, axis=0)
    for r in out[idx:]:
        for k, v in zip(POSE_KEYS, frozen):
            r[k] = f"{float(v):.9f}"
        r["source"] = "chair_metric_articulated_6d_audio_static"
    return out


def freeze_pre_lift_pose(rows: list[dict[str, str]], lift_frame: int) -> list[dict[str, str]]:
    out = [dict(r) for r in rows]
    pre_idx = [i for i, r in enumerate(out) if int(r["frame"]) < lift_frame]
    if not pre_idx:
        return out
    boundary_idx = next((i for i, r in enumerate(out) if int(r["frame"]) >= lift_frame), pre_idx[-1])
    start = max(0, boundary_idx - 1)
    end = min(len(out), boundary_idx + 2)
    P = pose_array(out[start:end])
    frozen = np.median(P, axis=0)
    for i in pre_idx:
        for k, v in zip(POSE_KEYS, frozen):
            out[i][k] = f"{float(v):.9f}"
        out[i]["source"] = "chair_metric_articulated_6d_pre_lift_static"
    return out


def smooth_series(values: np.ndarray, window: int, order: int = 2) -> np.ndarray:
    if len(values) < 5:
        return values.copy()
    win = min(window, len(values) if len(values) % 2 else len(values) - 1)
    if win < 5:
        return values.copy()
    return savgol_filter(values, win, min(order, win - 1), axis=0, mode="interp")


def depth_only_after_stage3_pose(rows: list[dict[str, str]], static_start: int) -> list[dict[str, str]]:
    """Preserve the 2D-fitted stage3 pose and only regularize camera depth.

    Use this when the 2D overlay is already the authority. Contact/anchor stages
    may refine z for 3D consistency, but must not move image-plane tx/ty,
    orientation, or articulation.
    """
    out = [dict(r) for r in rows]
    P = pose_array(out)
    frames = np.asarray([int(r["frame"]) for r in out], dtype=int)
    tz = smooth_series(P[:, 5:6], 21, 2)[:, 0]
    static_idx = int(np.searchsorted(frames, static_start, side="left"))
    if 0 <= static_idx < len(tz):
        ref = float(np.median(tz[static_idx : min(len(tz), static_idx + 8)]))
        tz[static_idx:] = ref
    for row, z in zip(out, tz):
        row["tz"] = f"{float(max(z, 0.4)):.9f}"
        row["source"] = "chair_metric_stage3_2d_locked_depth_only"
    return out


def contact_stabilize_pose(rows: list[dict[str, str]], contact_end_frame: int, static_start: int) -> list[dict[str, str]]:
    """Low-pass pose while contact is active, then freeze after static start.

    Stage3 pose follows per-frame 2D semantic observations, which are noisy even
    when the human is holding the same top rail. This pass keeps the low
    frequency carry/place trajectory but removes frame-to-frame jitter.
    """
    out = [dict(r) for r in rows]
    P = pose_array(out)
    frames = np.asarray([int(r["frame"]) for r in out], dtype=int)
    end = int(min(max(contact_end_frame, static_start), int(frames[-1])))
    mask = frames <= end
    idx = np.where(mask)[0]
    if len(idx) >= 5:
        seg = P[idx].copy()
        # Orientation and articulation should be steadier under the same grasp.
        seg[:, :3] = smooth_series(seg[:, :3], 25, 2)
        # Translation can move with the carried chair, so use a moderate window.
        seg[:, 3:6] = smooth_series(seg[:, 3:6], 21, 2)
        seg[:, 6:8] = smooth_series(seg[:, 6:8], 31, 2)
        for row_i, vals in zip(idx, seg):
            for k, v in zip(POSE_KEYS, vals):
                out[row_i][k] = f"{float(v):.9f}"
            out[row_i]["source"] = "chair_metric_articulated_6d_contact_stabilized"
    return freeze_pose(out, static_start)


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def update_pose_quat(row: dict[str, str], vals: np.ndarray, source: str) -> dict[str, str]:
    fit = load_pose_render_module()
    out = dict(row)
    for k, v in zip(POSE_KEYS, vals):
        out[k] = f"{float(v):.9f}"
    r_total = Rotation.from_matrix(Rotation.from_rotvec(vals[3:6] if False else vals[:3]).as_matrix() @ fit.BASE_LOCAL_TO_CAM)
    qx, qy, qz, qw = r_total.as_quat()
    out.update({"qx": f"{qx:.9f}", "qy": f"{qy:.9f}", "qz": f"{qz:.9f}", "qw": f"{qw:.9f}", "source": source})
    return out


def grasp_anchor_entries(grasp: dict[str, str] | None) -> list[tuple[np.ndarray, np.ndarray, float, str]]:
    if not grasp:
        return []
    entries: list[tuple[np.ndarray, np.ndarray, float, str]] = []
    if grasp.get("contact_active") == "1":
        local = np.array([ff(grasp, k) for k in ("stable_grasp_local_x", "stable_grasp_local_y", "stable_grasp_local_z")], dtype=float)
        target = np.array([ff(grasp, "contact_target_u"), ff(grasp, "contact_target_v")], dtype=float)
        if np.all(np.isfinite(local)) and np.all(np.isfinite(target)):
            entries.append((local, target, float(np.clip(ff(grasp, "stable_grasp_conf", 0.5), 0.15, 1.0)), grasp.get("frame_mode", "")))
    if grasp.get("secondary_contact_active") == "1":
        local = np.array([ff(grasp, k) for k in ("secondary_stable_grasp_local_x", "secondary_stable_grasp_local_y", "secondary_stable_grasp_local_z")], dtype=float)
        target = np.array([ff(grasp, "secondary_contact_target_u"), ff(grasp, "secondary_contact_target_v")], dtype=float)
        if np.all(np.isfinite(local)) and np.all(np.isfinite(target)):
            entries.append((local, target, float(np.clip(ff(grasp, "secondary_stable_grasp_conf", 0.5), 0.15, 1.0)), "direct_grasp_anchor" if "direct_grasp_anchor" in grasp.get("frame_mode", "") else "keep_previous_grasp_anchor"))
    return entries


def anchor_refine_pose(rows: list[dict[str, str]], grasp_by_frame: dict[int, dict[str, str]], contact_start: int, static_start: int) -> list[dict[str, str]]:
    """Mug-style trajectory refinement with continuous stable object-local anchors."""
    fit = load_pose_render_module()
    frames = np.asarray([int(r["frame"]) for r in rows], dtype=int)
    P0 = pose_array(rows)
    n = len(rows)
    if n == 0:
        return rows
    K_by_i = [(ff(r, "fx"), ff(r, "fy"), ff(r, "cx"), ff(r, "cy")) for r in rows]
    anchors = {
        i: (grasp_anchor_entries(grasp_by_frame.get(int(fr))) if contact_start <= int(fr) < static_start else [])
        for i, fr in enumerate(frames)
    }
    static_idx = int(np.searchsorted(frames, static_start, side="left"))
    scale = np.array([0.060, 0.060, 0.075, 0.065, 0.065, 0.095, 0.085, 0.085], dtype=float)

    tx_anchors: list[tuple[int, float, float]] = []
    ty_anchors: list[tuple[int, float, float]] = []
    for i, entries in anchors.items():
        for local, target, conf, mode in entries:
            fx, fy, cx, cy = K_by_i[i]
            cam = fit.transform(P0[i], local[None, :])[0]
            offset = cam - P0[i, 3:6]
            anchor_z = max(P0[i, 5] + offset[2], 0.4)
            target_tx = (target[0] - cx) * anchor_z / fx - offset[0]
            target_ty = (target[1] - cy) * anchor_z / fy - offset[1]
            weight = (4.0 if "direct_grasp_anchor" in mode else 2.4) * conf
            tx_anchors.append((i, float(target_tx), float(weight)))
            ty_anchors.append((i, float(target_ty), float(weight)))

    def optimize_axis(init: np.ndarray, axis_scale: float, anchor_items: list[tuple[int, float, float]], prior_w: float, smooth_w: float, accel_w: float, static_w: float) -> np.ndarray:
        def residual_axis(x: np.ndarray) -> np.ndarray:
            out: list[float] = []
            out.extend(((x - init) / axis_scale * prior_w).tolist())
            for idx, target, weight in anchor_items:
                out.append((x[idx] - target) / axis_scale * weight)
            if len(x) > 1:
                out.extend((np.diff(x) / axis_scale * smooth_w).tolist())
            if len(x) > 2:
                out.extend((np.diff(x, n=2) / axis_scale * accel_w).tolist())
            if 0 <= static_idx < len(x):
                ref = x[static_idx]
                out.extend(((x[static_idx:] - ref) / axis_scale * static_w).tolist())
            return np.asarray(out, dtype=float)

        res = least_squares(residual_axis, init.copy(), loss="soft_l1", f_scale=0.04, max_nfev=120)
        return res.x

    P = P0.copy()
    P[:, 3] = optimize_axis(P0[:, 3], scale[3], tx_anchors, 0.22, 1.8, 7.5, 18.0)
    P[:, 4] = optimize_axis(P0[:, 4], scale[4], ty_anchors, 0.22, 1.8, 7.5, 18.0)
    # Depth and orientation/articulation get mug-style temporal/static refinement,
    # while staying close to the SAM2/CoTracker semantic fit.
    P[:, 5] = np.maximum(optimize_axis(P0[:, 5], scale[5], [], 0.35, 1.2, 5.0, 16.0), 0.4)
    for j in (0, 1, 2, 6, 7):
        P[:, j] = optimize_axis(P0[:, j], scale[j], [], 0.32, 1.3, 5.5, 14.0)
    return [update_pose_quat(row, vals, "chair_metric_articulated_6d_stable_contact_anchor_refined") for row, vals in zip(rows, P)]


def contact_end_from_qwen(qwen_rows: list[dict[str, str]], static_start: int) -> int:
    frames = [
        int(r["frame"])
        for r in qwen_rows
        if qwen_bool(r.get("contact_visible", "")) and int(r["frame"]) < static_start
    ]
    return max(frames) if frames else max(1, static_start - 1)


def nearest_local_point(local_rows: list[dict[str, str]], part_filter: set[str]) -> dict[str, str]:
    candidates = [r for r in local_rows if r.get("part") in part_filter]
    if not candidates:
        candidates = local_rows
    # Prefer explicit top/backrest points for hand contact.
    candidates.sort(key=lambda r: (0 if r.get("role") in {"top_edge", "board_center", "round_hole_center"} else 1, r.get("point_id", "")))
    return candidates[0]


def local_point_map(local_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {r["point_id"]: r for r in local_rows}


def top_rail_local_from_u(row: dict[str, str]) -> dict[str, str]:
    u = ff(row, "contact_u")
    ul = ff(row, "top_rail_left_u")
    ur = ff(row, "top_rail_right_u")
    t = 0.5
    if np.isfinite(u) and np.isfinite(ul) and np.isfinite(ur) and abs(ur - ul) > 1e-4:
        t = float(np.clip((u - ul) / (ur - ul), 0.0, 1.0))
    x = -0.18 * (1.0 - t) + 0.18 * t
    return {
        "point_id": "qwen_top_rail_interp",
        "part": "backrest_board",
        "role": "top_edge_interpolated",
        "local_x": f"{x:.6f}",
        "local_y": "0.130000",
        "local_z": "0.751000",
    }


def contact_rows_from_observations(
    obs_rows: list[dict[str, str]],
    local_rows: list[dict[str, str]],
    place_frame: int,
    static_start: int,
) -> list[dict[str, str]]:
    local = nearest_local_point(local_rows, {"backrest_board"})
    rows: list[dict[str, str]] = []
    for obs in obs_rows:
        fr = int(obs["frame"])
        phase = "carry_contact" if fr < place_frame else ("place_release" if fr < static_start else "static_no_contact")
        contact_active = fr < static_start
        endpoints = [
            ("left_hand", ff(obs, "top_rail_left_u"), ff(obs, "top_rail_left_v")),
            ("right_hand", ff(obs, "top_rail_right_u"), ff(obs, "top_rail_right_v")),
        ]
        if not contact_active:
            endpoints = [("none", math.nan, math.nan)]
        for hand_side, u, v in endpoints:
            rows.append(
                {
                    "frame": str(fr),
                    "time": obs.get("time", ""),
                    "phase": phase,
                    "contact_active": "1" if contact_active else "0",
                    "hand_side": hand_side if contact_active else "none",
                    "contact_object_part": "top_rail" if contact_active else "none",
                    "contact_point_2d_u": f"{u:.3f}" if np.isfinite(u) else "",
                    "contact_point_2d_v": f"{v:.3f}" if np.isfinite(v) else "",
                    "chair_local_point_id": local.get("point_id", "") if contact_active else "",
                    "chair_local_part": local.get("part", "") if contact_active else "",
                    "chair_local_role": local.get("role", "") if contact_active else "",
                    "chair_local_x": local.get("local_x", "") if contact_active else "",
                    "chair_local_y": local.get("local_y", "") if contact_active else "",
                    "chair_local_z": local.get("local_z", "") if contact_active else "",
                    "source": "semantic_top_rail_contact_proxy_pending_qwen",
                }
            )
    return rows


def contact_rows_from_grasp_state(obs_rows: list[dict[str, str]], grasp_by_frame: dict[int, dict[str, str]], contact_start: int, static_start: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for obs in obs_rows:
        fr = int(obs["frame"])
        in_contact = contact_start <= fr < static_start
        phase = "stable_grasp_contact" if in_contact else "static_no_contact"
        grasp = grasp_by_frame.get(fr, {})
        entries = []
        if in_contact and grasp.get("contact_active") == "1":
            entries.append(
                {
                    "hand_side": grasp.get("hand_side", "right_hand"),
                    "contact_object_part": grasp.get("contact_object_part", "top_rail"),
                    "contact_fingers": grasp.get("contact_fingers", ""),
                    "primary_contact_finger": grasp.get("primary_contact_finger", ""),
                    "grasp_type": grasp.get("grasp_type", ""),
                    "contact_point_2d_u": grasp.get("contact_target_u", ""),
                    "contact_point_2d_v": grasp.get("contact_target_v", ""),
                    "chair_local_point_id": "backrest_top_left",
                    "chair_local_part": "backrest_board",
                    "chair_local_role": "stable_top_rail_endpoint_left",
                    "chair_local_x": grasp.get("stable_grasp_local_x", ""),
                    "chair_local_y": grasp.get("stable_grasp_local_y", ""),
                    "chair_local_z": grasp.get("stable_grasp_local_z", ""),
                }
            )
        if in_contact and grasp.get("secondary_contact_active") == "1":
            entries.append(
                {
                    "hand_side": grasp.get("secondary_hand_side", "left_hand"),
                    "contact_object_part": "top_rail",
                    "contact_fingers": grasp.get("secondary_contact_fingers", ""),
                    "primary_contact_finger": grasp.get("secondary_primary_contact_finger", ""),
                    "grasp_type": grasp.get("secondary_grasp_type", ""),
                    "contact_point_2d_u": grasp.get("secondary_contact_target_u", ""),
                    "contact_point_2d_v": grasp.get("secondary_contact_target_v", ""),
                    "chair_local_point_id": "backrest_top_right",
                    "chair_local_part": "backrest_board",
                    "chair_local_role": "stable_top_rail_endpoint_right",
                    "chair_local_x": grasp.get("secondary_stable_grasp_local_x", ""),
                    "chair_local_y": grasp.get("secondary_stable_grasp_local_y", ""),
                    "chair_local_z": grasp.get("secondary_stable_grasp_local_z", ""),
                }
            )
        if not entries:
            entries = [
                {
                    "hand_side": "none",
                    "contact_object_part": "none",
                    "contact_fingers": "",
                    "primary_contact_finger": "",
                    "grasp_type": "",
                    "contact_point_2d_u": "",
                    "contact_point_2d_v": "",
                    "chair_local_point_id": "",
                    "chair_local_part": "",
                    "chair_local_role": "",
                    "chair_local_x": "",
                    "chair_local_y": "",
                    "chair_local_z": "",
                }
            ]
        for entry in entries:
            active = "1" if entry["hand_side"] != "none" and in_contact else "0"
            rows.append(
                {
                    "frame": str(fr),
                    "time": obs.get("time", ""),
                    "phase": phase,
                    "contact_active": active,
                    **entry,
                    "source": "stable_chair_grasp_anchor_state",
                }
            )
    return rows


def qwen_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_qwen_contact_rows(sample: Path) -> list[dict[str, str]]:
    path = sample / "annotations/vlm_chair_semantic_candidates/chair_semantic_candidate_annotations.csv"
    if not path.exists():
        return []
    return read_rows(path)


def load_chair_grasp_state(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    return {int(r["frame"]): r for r in read_rows(path)}


def apply_qwen_contact_keyframes(
    contacts: list[dict[str, str]],
    qwen_rows: list[dict[str, str]],
    grasp_by_frame: dict[int, dict[str, str]] | None = None,
    contact_start: int | None = None,
    static_start: int | None = None,
) -> list[dict[str, str]]:
    if not qwen_rows:
        return contacts
    out = [dict(r) for r in contacts]
    by_key: dict[tuple[int, str], dict[str, str]] = {
        (int(r["frame"]), r.get("hand_side", "")): r for r in out
    }
    for row in qwen_rows:
        fr = int(row["frame"])
        if contact_start is not None and fr < contact_start:
            continue
        if static_start is not None and fr >= static_start:
            continue
        visible = qwen_bool(row.get("contact_visible", ""))
        grasp = (grasp_by_frame or {}).get(fr, {})
        hands = [row.get("hand_side", "unknown")]
        if visible and row.get("hand_side") == "both_hands":
            hands = ["left_hand", "right_hand"]
        for hand in hands:
            existing = by_key.get((fr, hand))
            if existing is not None and existing.get("contact_active") == "1":
                existing["phase"] = "qwen_verified_contact"
                existing["contact_object_part"] = row.get("contact_object_part", existing.get("contact_object_part", "top_rail"))
                existing["contact_fingers"] = row.get("contact_fingers", existing.get("contact_fingers", ""))
                existing["primary_contact_finger"] = row.get("primary_contact_finger", existing.get("primary_contact_finger", ""))
                existing["grasp_type"] = row.get("grasp_type", existing.get("grasp_type", ""))
                existing["source"] = existing.get("source", "stable_chair_grasp_anchor_state") + "+qwen_semantics"
                continue
            if visible and hand == "left_hand" and grasp.get("secondary_contact_active") == "1":
                local = {
                    "point_id": "secondary_stable_chair_grasp_local",
                    "part": "backrest_board",
                    "role": "stable_top_rail_anchor_left",
                    "local_x": grasp.get("secondary_stable_grasp_local_x", ""),
                    "local_y": grasp.get("secondary_stable_grasp_local_y", ""),
                    "local_z": grasp.get("secondary_stable_grasp_local_z", ""),
                }
                u = row.get("top_rail_left_u", row.get("contact_u", ""))
                v = row.get("top_rail_left_v", row.get("contact_v", ""))
            elif visible and grasp.get("contact_active") == "1":
                local = {
                    "point_id": "stable_chair_grasp_local",
                    "part": "backrest_board",
                    "role": "stable_top_rail_anchor_right",
                    "local_x": grasp.get("stable_grasp_local_x", ""),
                    "local_y": grasp.get("stable_grasp_local_y", ""),
                    "local_z": grasp.get("stable_grasp_local_z", ""),
                }
                if row.get("hand_side") == "both_hands":
                    u = row.get("top_rail_right_u", row.get("contact_u", ""))
                    v = row.get("top_rail_right_v", row.get("contact_v", ""))
                else:
                    u = row.get("contact_u", "")
                    v = row.get("contact_v", "")
            else:
                local = top_rail_local_from_u(row) if visible and row.get("contact_object_part") in {"top_rail", "backrest_board"} else {}
                u = row.get("contact_u", "")
                v = row.get("contact_v", "")
            phase = "qwen_contact" if visible else "qwen_no_contact"
            out.append(
                {
                    "frame": str(fr),
                    "time": row.get("time", f"{(fr - 1) / 24.0:.6f}"),
                    "phase": phase,
                    "contact_active": "1" if visible else "0",
                    "hand_side": hand if visible else "none",
                    "contact_object_part": row.get("contact_object_part", "unknown") if visible else "none",
                    "contact_fingers": row.get("contact_fingers", "") if visible else "",
                    "primary_contact_finger": row.get("primary_contact_finger", "") if visible else "",
                    "grasp_type": row.get("grasp_type", "") if visible else "",
                    "contact_point_2d_u": u if visible else "",
                    "contact_point_2d_v": v if visible else "",
                    "chair_local_point_id": local.get("point_id", ""),
                    "chair_local_part": local.get("part", ""),
                    "chair_local_role": local.get("role", ""),
                    "chair_local_x": local.get("local_x", ""),
                    "chair_local_y": local.get("local_y", ""),
                    "chair_local_z": local.get("local_z", ""),
                    "source": "qwen_candidate_keyframe_contact_no_gvhmr_base",
                }
            )
    hand_rank = {"right_hand": 0, "left_hand": 1, "none": 2}
    out.sort(key=lambda r: (int(r["frame"]), hand_rank.get(r.get("hand_side", ""), 9)))
    return out


def phase_rows(
    obs_rows: list[dict[str, str]],
    audio_rows: list[dict[str, str]],
    contact_start: int,
    lift_frame: int,
    place_frame: int,
    static_start: int,
    static_start_source: str,
) -> list[dict[str, str]]:
    event_by_frame = {int(round(ff(r, "audio_frame"))): r for r in audio_rows}
    out = []
    for obs in obs_rows:
        fr = int(obs["frame"])
        if fr < contact_start:
            phase = "pre_contact_static"
        elif fr < lift_frame:
            phase = "grounded_grasp_static"
        elif fr < place_frame:
            phase = "carry"
        elif fr < static_start:
            phase = "place_settle"
        else:
            phase = "static"
        ev = event_by_frame.get(fr)
        out.append(
            {
                "frame": str(fr),
                "time": obs.get("time", ""),
                "phase": phase,
                "contact_start_frame": str(contact_start),
                "lift_frame": str(lift_frame),
                "place_frame": str(place_frame),
                "static_start_frame": str(static_start),
                "audio_event": ev.get("event", "") if ev else "",
                "audio_score": ev.get("audio_score", "") if ev else "",
                "source": static_start_source,
            }
        )
    return out


def draw_contact_overlay(img: np.ndarray, contact_rows: list[dict[str, str]], frame: int) -> np.ndarray:
    out = img.copy()
    rows = [r for r in contact_rows if int(r["frame"]) == frame and r["contact_active"] == "1"]
    for row in rows:
        u = ff(row, "contact_point_2d_u")
        v = ff(row, "contact_point_2d_v")
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        p = (int(round(u)), int(round(v)))
        cv2.circle(out, p, 8, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, p, 11, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(out, row["hand_side"], (p[0] + 10, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(out, row["hand_side"], (p[0] + 10, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def finalize_video(tmp: Path, out: Path) -> None:
    ffmpeg = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    try:
        subprocess.run([ffmpeg, "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        subprocess.run([ffmpeg, "-y", "-i", str(tmp), "-c:v", "libopenh264", "-pix_fmt", "yuv420p", str(out)], check=True, capture_output=True, text=True)
    tmp.unlink(missing_ok=True)


def render_contacts(sample: Path, contact_rows: list[dict[str, str]], phase_by_frame: dict[int, str], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(sample / "frames/00001.png"))
    if first is None:
        raise RuntimeError("Missing frames/00001.png")
    h, w = first.shape[:2]
    tmp = out_dir / "contact_overlay.tmp.mp4"
    out = out_dir / "contact_overlay.mp4"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (w, h))
    n = len(list((sample / "frames").glob("*.png")))
    for fr in range(1, n + 1):
        img = cv2.imread(str(sample / "frames" / f"{fr:05d}.png"))
        if img is None:
            continue
        over = draw_contact_overlay(img, contact_rows, fr)
        label = f"frame {fr:03d} chair contact/audio phase: {phase_by_frame.get(fr, '')}"
        cv2.putText(over, label, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(over, label, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (40, 240, 40), 1, cv2.LINE_AA)
        writer.write(over)
    writer.release()
    finalize_video(tmp, out)
    for fr in [1, 50, 100, 140, 145, 160, 180, 192]:
        img = cv2.imread(str(sample / "frames" / f"{fr:05d}.png"))
        if img is not None:
            cv2.imwrite(str(out_dir / f"contact_overlay_{fr:03d}.png"), draw_contact_overlay(img, contact_rows, fr))
    return out


def load_pose_render_module():
    fit_path = REPO / "scripts/shared/generic_contact_pipeline/components/pose/solvers/fit_chair_metric_6d_pose.py"
    spec = importlib.util.spec_from_file_location("chair_metric_pose_render", fit_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pose renderer from {fit_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def render_stable_pose_overlay(
    sample: Path,
    pose_rows: list[dict[str, str]],
    out_dir: Path,
    video_name: str = "overlay_contact_stable.mp4",
    label_suffix: str = "stable contact-anchor articulated 6D",
    still_prefix: str = "overlay_contact_stable",
    show_observations: bool = True,
    observations_csv: Path | None = None,
    leg_candidates_csv: Path | None = None,
    local_segments_csv: Path | None = None,
) -> Path:
    fit = load_pose_render_module()
    segs = fit.load_segments(local_segments_csv or (sample / "results/chair_semantic_local_points/chair_semantic_local_segments.csv"))
    obs_rows = fit.read_rows(observations_csv or (sample / "results/chair_semantic_observations/chair_semantic_observations.csv"))
    cands = fit.load_candidates(leg_candidates_csv or (sample / "results/chair_semantic_observations/chair_leg_candidates.csv"), 8)
    first = cv2.imread(str(sample / "frames/00001.png"))
    if first is None:
        raise RuntimeError("Missing frames/00001.png")
    h, w = first.shape[:2]
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"{Path(video_name).stem}.tmp.mp4"
    out = out_dir / video_name
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (w, h))
    obs_by_frame = {int(r["frame"]): r for r in obs_rows}
    for row in pose_rows:
        fr = int(row["frame"])
        img = cv2.imread(str(sample / "frames" / f"{fr:05d}.png"))
        if img is None:
            continue
        params = np.asarray([ff(row, k, 0.0) for k in POSE_KEYS], dtype=float)
        K = (ff(row, "fx"), ff(row, "fy"), ff(row, "cx"), ff(row, "cy"))
        obs = obs_by_frame.get(fr) if show_observations else None
        over = fit.draw_overlay(img, params, segs, K, obs, cands.get(fr, {}) if show_observations else {})
        label = f"frame {fr:03d} {label_suffix}"
        cv2.putText(over, label, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(over, label, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (40, 240, 40), 1, cv2.LINE_AA)
        writer.write(over)
    writer.release()
    finalize_video(tmp, out)
    for fr in [1, 50, 100, 140, 145, 155, 160, 180, 192]:
        row = next((r for r in pose_rows if int(r["frame"]) == fr), None)
        img = cv2.imread(str(sample / "frames" / f"{fr:05d}.png"))
        if row is None or img is None:
            continue
        params = np.asarray([ff(row, k, 0.0) for k in POSE_KEYS], dtype=float)
        K = (ff(row, "fx"), ff(row, "fy"), ff(row, "cx"), ff(row, "cy"))
        obs = obs_by_frame.get(fr) if show_observations else None
        over = fit.draw_overlay(img, params, segs, K, obs, cands.get(fr, {}) if show_observations else {})
        cv2.imwrite(str(out_dir / f"{still_prefix}_{fr:03d}.png"), over)
    return out


def write_pending_vlm_contact_annotations(sample: Path, contact_rows: list[dict[str, str]], keyframes: list[int]) -> tuple[Path, Path]:
    out_dir = sample / "annotations" / "vlm_chair_contact_keyframes"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame",
        "time",
        "crop_path",
        "candidate_json",
        "vlm_status",
        "contact_visible",
        "hand_side",
        "contact_object_part",
        "contact_fingers",
        "primary_contact_finger",
        "grasp_type",
        "contact_point_2d_u",
        "contact_point_2d_v",
        "source",
    ]
    by_frame: dict[int, list[dict[str, str]]] = {}
    for row in contact_rows:
        by_frame.setdefault(int(row["frame"]), []).append(row)
    rows = []
    frames_json = []
    for fr in keyframes:
        active = [r for r in by_frame.get(fr, []) if r.get("contact_active") == "1"]
        if not active:
            active = [
                {
                    "frame": str(fr),
                    "time": f"{(fr - 1) / 24.0:.6f}",
                    "contact_active": "0",
                    "hand_side": "none",
                    "contact_object_part": "none",
                    "contact_point_2d_u": "",
                    "contact_point_2d_v": "",
                }
            ]
        for row in active:
            crop = sample / "annotations/vlm_chair_semantic_candidates/crops" / f"{fr:05d}_candidate_crop.png"
            cand = sample / "annotations/vlm_chair_semantic_candidates/candidates" / f"{fr:05d}_candidates.json"
            rec = {
                "frame": str(fr),
                "time": row.get("time", f"{(fr - 1) / 24.0:.6f}"),
                "crop_path": str(crop) if crop.exists() else "",
                "candidate_json": str(cand) if cand.exists() else "",
                "vlm_status": "qwen_oom_pending",
                "contact_visible": row.get("contact_active", "0"),
                "hand_side": row.get("hand_side", "none"),
                "contact_object_part": row.get("contact_object_part", "none"),
                "contact_fingers": row.get("contact_fingers", ""),
                "primary_contact_finger": row.get("primary_contact_finger", ""),
                "grasp_type": row.get("grasp_type", ""),
                "contact_point_2d_u": row.get("contact_point_2d_u", ""),
                "contact_point_2d_v": row.get("contact_point_2d_v", ""),
                "source": "stage4_proxy_written_for_vlm_keyframe_rerun",
            }
            rows.append(rec)
            frames_json.append(rec)
    csv_path = out_dir / "chair_contact_keyframe_annotations.csv"
    json_path = out_dir / "chair_contact_keyframe_annotations.json"
    write_csv(csv_path, rows, fields)
    json_path.write_text(json.dumps({"frames": frames_json, "note": "Qwen run was attempted but blocked by CUDA OOM in the current audiohoi environment; rows are proxy placeholders for rerun."}, indent=2) + "\n")
    return csv_path, json_path


def write_qwen_vlm_contact_annotations(sample: Path, qwen_rows: list[dict[str, str]]) -> tuple[Path, Path]:
    out_dir = sample / "annotations" / "vlm_chair_contact_keyframes"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame",
        "time",
        "crop_path",
        "candidate_json",
        "vlm_status",
        "contact_visible",
        "hand_side",
        "contact_object_part",
        "contact_fingers",
        "primary_contact_finger",
        "grasp_type",
        "contact_point_2d_u",
        "contact_point_2d_v",
        "confidence",
        "reason",
        "source",
    ]
    rows = []
    for row in qwen_rows:
        visible = qwen_bool(row.get("contact_visible", ""))
        hands = [row.get("hand_side", "unknown")]
        if visible and row.get("hand_side") == "both_hands":
            hands = ["left_hand", "right_hand"]
        for hand in hands:
            if hand == "left_hand" and row.get("hand_side") == "both_hands":
                u, v = row.get("top_rail_left_u", row.get("contact_u", "")), row.get("top_rail_left_v", row.get("contact_v", ""))
            elif hand == "right_hand" and row.get("hand_side") == "both_hands":
                u, v = row.get("top_rail_right_u", row.get("contact_u", "")), row.get("top_rail_right_v", row.get("contact_v", ""))
            else:
                u, v = row.get("contact_u", ""), row.get("contact_v", "")
            rows.append(
                {
                    "frame": row.get("frame", ""),
                    "time": row.get("time", ""),
                    "crop_path": row.get("crop_path", ""),
                    "candidate_json": row.get("candidate_json", ""),
                    "vlm_status": "qwen_ok",
                    "contact_visible": "1" if visible else "0",
                    "hand_side": hand if visible else "none",
                    "contact_object_part": row.get("contact_object_part", "unknown") if visible else "none",
                    "contact_fingers": row.get("contact_fingers", "") if visible else "",
                    "primary_contact_finger": row.get("primary_contact_finger", "") if visible else "",
                    "grasp_type": row.get("grasp_type", "") if visible else "",
                    "contact_point_2d_u": u if visible else "",
                    "contact_point_2d_v": v if visible else "",
                    "confidence": row.get("confidence", ""),
                    "reason": row.get("reason", ""),
                    "source": "qwen_vl_candidate_keyframe",
                }
            )
    csv_path = out_dir / "chair_contact_keyframe_annotations.csv"
    json_path = out_dir / "chair_contact_keyframe_annotations.json"
    write_csv(csv_path, rows, fields)
    json_path.write_text(json.dumps({"frames": rows, "note": "Qwen candidate-id contact annotations from qwen-vl environment."}, indent=2) + "\n")
    return csv_path, json_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-dir", type=Path, default=Path("samples_known_object/05_chair"))
    ap.add_argument("--result-dir", type=Path, default=None, help="Directory containing stage3 pose CSVs and receiving final CSV outputs. Defaults to results/final_result.")
    ap.add_argument("--render-dir", type=Path, default=None, help="Directory receiving overlay/contact videos. Defaults to results/renders/final_result.")
    ap.add_argument("--pose-csv", type=Path, default=None, help="Stage3 pose seed CSV. Defaults to result-dir/object_pose_stage3_raw.csv or object_pose.csv.")
    ap.add_argument("--observations-csv", type=Path, default=None, help="Semantic observation CSV. Defaults to results/chair_semantic_observations/chair_semantic_observations.csv.")
    ap.add_argument("--leg-candidates-csv", type=Path, default=None, help="Leg candidate CSV. Defaults to results/chair_semantic_observations/chair_leg_candidates.csv.")
    ap.add_argument("--grasp-state-csv", type=Path, default=None, help="Generic chair grasp anchor state CSV.")
    ap.add_argument("--local-points-csv", type=Path, default=None, help="Generic chair local semantic points CSV.")
    ap.add_argument("--local-segments-csv", type=Path, default=None, help="Generic chair local semantic segments CSV.")
    ap.add_argument("--depth-only-after-stage3", action="store_true", help="Keep stage3 2D pose locked; only smooth/freeze tz after contact/audio.")
    ap.add_argument("--no-render", action="store_true", help="Only write CSV/JSON outputs; skip diagnostic overlay videos and PNG frames.")
    args = ap.parse_args()

    sample = args.sample_dir
    out_dir = args.result_dir or (sample / "results/final_result")
    raw_pose_path = out_dir / "object_pose_stage3_raw.csv"
    pose_path = args.pose_csv or (raw_pose_path if raw_pose_path.exists() else out_dir / "object_pose.csv")
    obs_path = args.observations_csv or (sample / "results/chair_semantic_observations/chair_semantic_observations.csv")
    audio_path = sample / "results/events/audio_events.csv"
    local_path = args.local_points_csv or (sample / "results/chair_semantic_local_points/chair_semantic_local_points.csv")
    grasp_state_path = args.grasp_state_csv or (sample / "results/chair_grasp_anchor_state/chair_grasp_anchor_state.csv")
    pose_rows = read_rows(pose_path)
    obs_rows = read_rows(obs_path)
    audio_rows = read_rows(audio_path)
    local_rows = read_rows(local_path)

    place_frame = pick_place_frame(audio_rows, len(pose_rows))
    audio_contact_start, audio_contact_start_source = pick_contact_start_frame(audio_rows, place_frame, len(pose_rows))
    lift_frame, lift_frame_source = pick_lift_frame(obs_rows, audio_contact_start, place_frame)
    contact_start = min(audio_contact_start, lift_frame)
    contact_start_source = (
        audio_contact_start_source
        if contact_start == audio_contact_start
        else "semantic_motion_before_audio_contact_candidate"
    )
    static_start, static_start_source = resolve_static_start(pose_rows, place_frame, use_object_motion_gate=False)

    qwen_rows = load_qwen_contact_rows(sample)
    grasp_by_frame = load_chair_grasp_state(grasp_state_path)
    phase = phase_rows(obs_rows, audio_rows, contact_start, lift_frame, place_frame, static_start, static_start_source)
    contacts = contact_rows_from_grasp_state(obs_rows, grasp_by_frame, contact_start, static_start)
    contacts = apply_qwen_contact_keyframes(contacts, qwen_rows, grasp_by_frame, contact_start, static_start)
    contact_end = contact_end_from_qwen(qwen_rows, static_start)
    if not raw_pose_path.exists():
        write_csv(raw_pose_path, pose_rows, list(pose_rows[0].keys()))
    if args.depth_only_after_stage3:
        refined_pose_rows = depth_only_after_stage3_pose(pose_rows, static_start)
        static_pose_rows = refined_pose_rows
        debug_smoothed_pose_rows = refined_pose_rows
    else:
        refined_pose_rows = anchor_refine_pose(pose_rows, grasp_by_frame, contact_start, static_start)
        refined_pose_rows = freeze_pre_lift_pose(refined_pose_rows, lift_frame)
        static_pose_rows = freeze_pose(refined_pose_rows, static_start)
        debug_smoothed_pose_rows = contact_stabilize_pose(pose_rows, contact_end, static_start)
        debug_smoothed_pose_rows = freeze_pre_lift_pose(debug_smoothed_pose_rows, lift_frame)

    write_csv(out_dir / "object_pose_anchor_refined.csv", refined_pose_rows, list(refined_pose_rows[0].keys()))
    write_csv(out_dir / "object_pose.csv", static_pose_rows, list(static_pose_rows[0].keys()))
    write_csv(out_dir / "object_pose_audio_static.csv", static_pose_rows, list(static_pose_rows[0].keys()))
    write_csv(out_dir / "object_pose_contact_stable_debug_smooth.csv", debug_smoothed_pose_rows, list(pose_rows[0].keys()))
    phase_fields = ["frame", "time", "phase", "contact_start_frame", "lift_frame", "place_frame", "static_start_frame", "audio_event", "audio_score", "source"]
    write_csv(out_dir / "object_phase.csv", phase, phase_fields)
    contact_fields = [
        "frame",
        "time",
        "phase",
        "contact_active",
        "hand_side",
        "contact_object_part",
        "contact_fingers",
        "primary_contact_finger",
        "grasp_type",
        "contact_point_2d_u",
        "contact_point_2d_v",
        "chair_local_point_id",
        "chair_local_part",
        "chair_local_role",
        "chair_local_x",
        "chair_local_y",
        "chair_local_z",
        "source",
    ]
    write_csv(out_dir / "object_contact_points.csv", contacts, contact_fields)

    render_dir = args.render_dir or (sample / "results/renders/final_result")
    if args.no_render:
        video = ""
        overlay_video = ""
        stable_video = ""
    else:
        video = render_contacts(sample, contacts, {int(r["frame"]): r["phase"] for r in phase}, render_dir)
        overlay_video = render_stable_pose_overlay(sample, static_pose_rows, render_dir, "overlay.mp4", "stable contact-anchor articulated 6D", "overlay", show_observations=False, observations_csv=obs_path, leg_candidates_csv=args.leg_candidates_csv, local_segments_csv=args.local_segments_csv)
        stable_video = render_stable_pose_overlay(sample, debug_smoothed_pose_rows, render_dir, "overlay_contact_stable.mp4", "debug low-pass articulated 6D", "overlay_contact_stable", show_observations=True, observations_csv=obs_path, leg_candidates_csv=args.leg_candidates_csv, local_segments_csv=args.local_segments_csv)
    keyframes = sorted({50, 100, 140, place_frame, min(place_frame + 5, len(pose_rows)), static_start, min(static_start + 5, len(pose_rows))})
    if qwen_rows:
        vlm_csv, vlm_json = write_qwen_vlm_contact_annotations(sample, qwen_rows)
        contact_note = "Contact rows come from stable chair-local grasp anchors. GVHMR hand projections set per-frame 2D targets; Qwen keyframes only update semantic finger/grasp labels."
    else:
        vlm_csv, vlm_json = write_pending_vlm_contact_annotations(sample, contacts, keyframes)
        contact_note = "Contact points are semantic top-rail proxies pending Qwen/VLM hand contact verification."

    summary = {
        "place_frame": place_frame,
        "contact_start_frame": contact_start,
        "contact_start_source": contact_start_source,
        "lift_frame": lift_frame,
        "lift_frame_source": lift_frame_source,
        "static_start_frame": static_start,
        "static_start_source": static_start_source,
        "contact_stabilized_until_frame": contact_end,
        "depth_only_after_stage3": bool(args.depth_only_after_stage3),
        "render_skipped": bool(args.no_render),
        "object_pose_stage3_raw_csv": str(raw_pose_path),
        "pose_seed_csv": str(pose_path),
        "grasp_state_csv": str(grasp_state_path),
        "object_pose_anchor_refined_csv": str(out_dir / "object_pose_anchor_refined.csv"),
        "object_pose_csv": str(out_dir / "object_pose.csv"),
        "object_pose_audio_static_csv": str(out_dir / "object_pose_audio_static.csv"),
        "object_pose_contact_stable_debug_smooth_csv": str(out_dir / "object_pose_contact_stable_debug_smooth.csv"),
        "object_phase_csv": str(out_dir / "object_phase.csv"),
        "object_contact_points_csv": str(out_dir / "object_contact_points.csv"),
        "vlm_contact_keyframe_csv": str(vlm_csv),
        "vlm_contact_keyframe_json": str(vlm_json),
        "contact_overlay_video": str(video) if video else "",
        "overlay_video": str(overlay_video) if overlay_video else "",
        "debug_smooth_overlay_video": str(stable_video) if stable_video else "",
        "qwen_keyframes_used": len(qwen_rows),
        "contact_note": contact_note,
    }
    (out_dir / "audio_contact_static_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
