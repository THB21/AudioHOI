#!/usr/bin/env python3
"""Fit metric articulated 6D chair pose from semantic 2D anchors and DA3 depth."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


BASE_LOCAL_TO_CAM = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)


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


def load_segments(path: Path) -> dict[str, tuple[str, str, np.ndarray, np.ndarray]]:
    out = {}
    for row in read_rows(path):
        a = np.array([ff(row, "start_local_x"), ff(row, "start_local_y"), ff(row, "start_local_z")], dtype=float)
        b = np.array([ff(row, "end_local_x"), ff(row, "end_local_y"), ff(row, "end_local_z")], dtype=float)
        out[row["segment_id"]] = (row["part"], row["role"], a, b)
    return out


def load_points(path: Path) -> dict[str, np.ndarray]:
    out = {}
    for row in read_rows(path):
        out[row["point_id"]] = np.array([ff(row, "local_x"), ff(row, "local_y"), ff(row, "local_z")], dtype=float)
    return out


def load_candidates(path: Path, max_per_family: int) -> dict[int, dict[str, list[tuple[np.ndarray, np.ndarray]]]]:
    grouped: dict[tuple[int, str], list[dict[str, str]]] = {}
    for row in read_rows(path):
        grouped.setdefault((int(row["frame"]), row["family"]), []).append(row)
    out: dict[int, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}
    for (frame, family), group in grouped.items():
        group.sort(key=lambda r: ff(r, "length_px", 0.0), reverse=True)
        segs = []
        for row in group[:max_per_family]:
            a = np.array([ff(row, "top_u"), ff(row, "top_v")], dtype=float)
            b = np.array([ff(row, "bottom_u"), ff(row, "bottom_v")], dtype=float)
            if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
                segs.append((a, b))
        out.setdefault(frame, {})[family] = segs
    return out


def load_grasp_state(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    return {int(row["frame"]): row for row in read_rows(path)}


def load_front_frame_tracks(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    return {int(row["frame"]): row for row in read_rows(path)}


def read_depth_index(path: Path) -> dict[int, Path]:
    depth_dir = path.parent
    out = {}
    for row in read_rows(path):
        out[int(row["frame"])] = depth_dir / row["file"]
    return out


def mask_depth(sample: Path, frame: int, depth_path: Path) -> tuple[float, float]:
    img = cv2.imread(str(sample / "frames" / f"{frame:05d}.png"))
    mask = cv2.imread(str(sample / "results/segmentation/masks" / f"{frame:05d}_mask.png"), cv2.IMREAD_GRAYSCALE)
    depth = np.load(depth_path)
    if img is None or mask is None:
        return math.nan, 0.0
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return math.nan, 0.0
    sx = depth.shape[1] / img.shape[1]
    sy = depth.shape[0] / img.shape[0]
    dx = np.clip((xs * sx).astype(int), 0, depth.shape[1] - 1)
    dy = np.clip((ys * sy).astype(int), 0, depth.shape[0] - 1)
    vals = depth[dy, dx]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return math.nan, 0.0
    med = float(np.median(vals))
    iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
    conf = float(1.0 / (1.0 + iqr / max(med, 1e-6)))
    return med, conf


def estimate_camera(sample: Path, obs_rows: list[dict[str, str]], depth_by_frame: dict[int, Path]) -> tuple[float, float, float, float, dict[int, tuple[float, float]]]:
    first = cv2.imread(str(sample / "frames/00001.png"))
    if first is None:
        raise RuntimeError("Missing first frame")
    h, w = first.shape[:2]
    depth_priors = {}
    f_est = []
    for row in obs_rows:
        fr = int(row["frame"])
        if fr not in depth_by_frame:
            continue
        z, conf = mask_depth(sample, fr, depth_by_frame[fr])
        depth_priors[fr] = (z, conf)
        bh = ff(row, "mask_bbox_y2") - ff(row, "mask_bbox_y1")
        if np.isfinite(z) and z > 0 and bh > 30:
            f_est.append(bh * z / 0.78)
    focal = float(np.median(f_est)) if f_est else 0.85 * w
    return focal, focal, w * 0.5, h * 0.5, depth_priors


def transform(params: np.ndarray, pts_local: np.ndarray) -> np.ndarray:
    r_delta = Rotation.from_rotvec(params[:3]).as_matrix()
    r = r_delta @ BASE_LOCAL_TO_CAM
    t = params[3:6]
    return pts_local @ r.T + t


def rotate_x(points: np.ndarray, origin: np.ndarray, angle: float) -> np.ndarray:
    r = Rotation.from_rotvec(np.array([angle, 0.0, 0.0])).as_matrix()
    return (points - origin) @ r.T + origin


def articulate_points(segment_id: str, part: str, params: np.ndarray, pts_local: np.ndarray) -> np.ndarray:
    """Apply chair URDF articulation in local/front_frame coordinates."""
    rear_angle = float(params[6]) if len(params) > 6 else 0.0
    seat_angle = float(params[7]) if len(params) > 7 else 0.0
    rear_origin = np.array([0.0, 0.040, 0.548], dtype=float)
    seat_origin = np.array([0.0, 0.090, 0.440], dtype=float)
    if part == "rear_leg" or segment_id == "rear_feet_line":
        return rotate_x(pts_local, rear_origin, rear_angle)
    if part in {"seat", "seat_slat"} or segment_id == "seat_hinge_edge":
        # URDF axis is (-1, 0, 0), so positive joint rotates by -x.
        return rotate_x(pts_local, seat_origin, -seat_angle)
    return pts_local


def project(params: np.ndarray, pts_local: np.ndarray, K: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = K
    cam = transform(params, pts_local)
    z = np.maximum(cam[:, 2], 1e-4)
    uv = np.column_stack([cx + fx * cam[:, 0] / z, cy + fy * cam[:, 1] / z])
    return uv, cam


def project_segment(params: np.ndarray, sid: str, segs: dict, K: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    part, _, a, b = segs[sid]
    pts = articulate_points(sid, part, params, np.vstack([a, b]))
    uv, cam = project(params, pts, K)
    return uv, cam, pts


def segment_obs(row: dict[str, str], prefix: str) -> tuple[np.ndarray, np.ndarray] | None:
    a = np.array([ff(row, f"{prefix}_left_u"), ff(row, f"{prefix}_left_v")])
    b = np.array([ff(row, f"{prefix}_right_u"), ff(row, f"{prefix}_right_v")])
    if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
        return a, b
    return None


def leg_obs(row: dict[str, str], prefix: str) -> tuple[np.ndarray, np.ndarray] | None:
    a = np.array([ff(row, f"{prefix}_top_u"), ff(row, f"{prefix}_top_v")])
    b = np.array([ff(row, f"{prefix}_bottom_u"), ff(row, f"{prefix}_bottom_v")])
    if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
        return a, b
    return None


def valid_leg_obs(row: dict[str, str], prefix: str) -> tuple[np.ndarray, np.ndarray] | None:
    obs = leg_obs(row, prefix)
    if obs is None:
        return None
    a, b = obs
    length = float(np.linalg.norm(b - a))
    if prefix.startswith("rear_leg") and length < 70.0:
        return None
    return obs


def point_obs(row: dict[str, str], prefix: str) -> np.ndarray | None:
    p = np.array([ff(row, f"{prefix}_u"), ff(row, f"{prefix}_v")])
    if np.all(np.isfinite(p)):
        return p
    return None


def endpoint_res(pred: tuple[np.ndarray, np.ndarray], obs: tuple[np.ndarray, np.ndarray], sigma: float, weight: float = 1.0) -> list[float]:
    pa, pb = pred
    oa, ob = obs
    r1 = np.concatenate([pa - oa, pb - ob])
    r2 = np.concatenate([pa - ob, pb - oa])
    r = r1 if np.linalg.norm(r1) <= np.linalg.norm(r2) else r2
    return (math.sqrt(weight) * r / sigma).tolist()


def point_line_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-6:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def line_candidate_res(pred: tuple[np.ndarray, np.ndarray], candidates: list[tuple[np.ndarray, np.ndarray]], sigma: float, weight: float) -> list[float]:
    if not candidates:
        return []
    pa, pb = pred
    best = None
    best_score = float("inf")
    for ca, cb in candidates:
        vals = np.array([point_line_distance(pa, ca, cb), point_line_distance(pb, ca, cb)], dtype=float)
        score = float(vals.mean())
        if score < best_score:
            best = vals
            best_score = score
    return (math.sqrt(weight) * np.clip(best, 0.0, 80.0) / sigma).tolist()


def candidate_pair(candidates: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
    picked: list[tuple[np.ndarray, np.ndarray]] = []
    for cand in candidates:
        x = float(cand[0][0])
        if all(abs(x - float(old[0][0])) > 26.0 for old in picked):
            picked.append(cand)
        if len(picked) >= 2:
            break
    picked.sort(key=lambda ab: float(ab[0][0]))
    return picked


def leg_pair_res(params: np.ndarray, sids: tuple[str, str], obs_lines: list[tuple[np.ndarray, np.ndarray]], segs: dict, K, sigma: float, weight: float) -> list[float]:
    if len(obs_lines) < 2:
        return []
    uv0, _, _ = project_segment(params, sids[0], segs, K)
    uv1, _, _ = project_segment(params, sids[1], segs, K)
    r_a = endpoint_res((uv0[0], uv0[1]), obs_lines[0], sigma, weight) + endpoint_res((uv1[0], uv1[1]), obs_lines[1], sigma, weight)
    r_b = endpoint_res((uv0[0], uv0[1]), obs_lines[1], sigma, weight) + endpoint_res((uv1[0], uv1[1]), obs_lines[0], sigma, weight)
    return r_a if np.linalg.norm(r_a) <= np.linalg.norm(r_b) else r_b


def directed_endpoint_res(pred: tuple[np.ndarray, np.ndarray], obs: tuple[np.ndarray, np.ndarray], sigma: float, weight: float = 1.0) -> list[float]:
    pa, pb = pred
    oa, ob = obs
    r = np.concatenate([pa - oa, pb - ob])
    return (math.sqrt(weight) * r / sigma).tolist()


def front_frame_obs(
    row: dict[str, str],
    tracked: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None:
    if tracked is not None:
        try:
            conf = ff(tracked, "front_frame_track_conf", 0.0)
            lt = np.array([ff(tracked, "front_frame_lt_u"), ff(tracked, "front_frame_lt_v")], dtype=float)
            lb = np.array([ff(tracked, "front_frame_lb_u"), ff(tracked, "front_frame_lb_v")], dtype=float)
            rt = np.array([ff(tracked, "front_frame_rt_u"), ff(tracked, "front_frame_rt_v")], dtype=float)
            rb = np.array([ff(tracked, "front_frame_rb_u"), ff(tracked, "front_frame_rb_v")], dtype=float)
        except Exception:
            conf = 0.0
        else:
            if conf >= 0.25 and all(np.all(np.isfinite(p)) for p in (lt, lb, rt, rb)):
                return lt, lb, rt, rb, float(np.clip(conf, 0.25, 1.0))
    if row.get("front_leg_pair_valid", "0") != "1":
        return None
    left = valid_leg_obs(row, "front_leg_left")
    right = valid_leg_obs(row, "front_leg_right")
    if left is None or right is None:
        return None
    lt, lb = left
    rt, rb = right
    return lt, lb, rt, rb, 0.55


def front_frame_parallelogram_res(
    params: np.ndarray,
    row: dict[str, str],
    tracked: dict[str, str] | None,
    segs: dict,
    K: tuple[float, float, float, float],
    sigma: float = 10.0,
    weight: float = 1.2,
) -> list[float]:
    """Use the CoTracker-snapped front-leg pair as one front-frame constraint.

    The four tracked endpoints form a single front-frame quad; top/bottom edges
    and the two side directions move together. Pair-level on purpose, not yet
    another independent line-detector residual.
    """
    obs = front_frame_obs(row, tracked)
    if obs is None:
        return []
    olt, olb, ort, orb, conf = obs
    luv, _, _ = project_segment(params, "front_leg_left", segs, K)
    ruv, _, _ = project_segment(params, "front_leg_right", segs, K)
    plt, plb = luv[0], luv[1]
    prt, prb = ruv[0], ruv[1]

    pred_top = prt - plt
    pred_bottom = prb - plb
    obs_top = ort - olt
    obs_bottom = orb - olb
    pred_left = plb - plt
    pred_right = prb - prt
    obs_left = olb - olt
    obs_right = orb - ort
    pred_center = (plt + plb + prt + prb) * 0.25
    obs_center = (olt + olb + ort + orb) * 0.25

    vals = np.concatenate(
        [
            pred_top - obs_top,
            pred_bottom - obs_bottom,
            (pred_top - pred_bottom) - (obs_top - obs_bottom),
            (pred_left - pred_right) - (obs_left - obs_right),
            pred_center - obs_center,
        ]
    )
    return (math.sqrt(weight * conf) * vals / sigma).tolist()


def feet_res(params: np.ndarray, row: dict[str, str], segs: dict, K: tuple[float, float, float, float], sigma: float, weight: float) -> list[float]:
    left = point_obs(row, "front_left_foot")
    right = point_obs(row, "front_right_foot")
    if left is None or right is None:
        return []
    uv, _, _ = project_segment(params, "front_feet_line", segs, K)
    r1 = np.concatenate([uv[0] - left, uv[1] - right])
    r2 = np.concatenate([uv[0] - right, uv[1] - left])
    r = r1 if np.linalg.norm(r1) <= np.linalg.norm(r2) else r2
    return (math.sqrt(weight) * r / sigma).tolist()


def grasp_anchor_res(params: np.ndarray, grasp: dict[str, str] | None, K: tuple[float, float, float, float]) -> list[float]:
    if not grasp or grasp.get("contact_active") != "1":
        primary = []
    else:
        primary = [(
            ("stable_grasp_local_x", "stable_grasp_local_y", "stable_grasp_local_z"),
            ("contact_target_u", "contact_target_v"),
            "stable_grasp_conf",
            grasp.get("frame_mode", ""),
        )]
    secondary = []
    if grasp and grasp.get("secondary_contact_active") == "1":
        secondary.append(
            (
                ("secondary_stable_grasp_local_x", "secondary_stable_grasp_local_y", "secondary_stable_grasp_local_z"),
                ("secondary_contact_target_u", "secondary_contact_target_v"),
                "secondary_stable_grasp_conf",
                "keep_previous_grasp_anchor",
            )
        )
    out: list[float] = []
    for local_keys, target_keys, conf_key, mode in primary + secondary:
        local = np.array([ff(grasp, k) for k in local_keys], dtype=float)
        target = np.array([ff(grasp, k) for k in target_keys], dtype=float)
        if not (np.all(np.isfinite(local)) and np.all(np.isfinite(target))):
            continue
        uv, _ = project(params, local[None, :], K)
        conf = float(np.clip(ff(grasp, conf_key, 0.5), 0.15, 1.0))
        weight = (3.0 if "direct_grasp_anchor" in mode else 1.6) * conf
        out.extend((math.sqrt(weight) * (uv[0] - target) / 7.0).tolist())
    return out


def bbox_res(params: np.ndarray, segs: dict, row: dict[str, str], K: tuple[float, float, float, float]) -> list[float]:
    uvs = []
    for sid in segs:
        uv, _, _ = project_segment(params, sid, segs, K)
        uvs.append(uv)
    uv = np.vstack(uvs)
    mn = uv.min(axis=0)
    mx = uv.max(axis=0)
    obs_mn = np.array([ff(row, "mask_bbox_x1"), ff(row, "mask_bbox_y1")])
    obs_mx = np.array([ff(row, "mask_bbox_x2"), ff(row, "mask_bbox_y2")])
    if not (np.all(np.isfinite(obs_mn)) and np.all(np.isfinite(obs_mx))):
        return []
    return np.concatenate([((mn + mx) * 0.5 - (obs_mn + obs_mx) * 0.5) / 45.0, ((mx - mn) - (obs_mx - obs_mn)) / 80.0]).tolist()


def initial_params(row: dict[str, str], z: float, K: tuple[float, float, float, float]) -> np.ndarray:
    fx, fy, cx, cy = K
    x1, y1, x2, y2 = ff(row, "mask_bbox_x1"), ff(row, "mask_bbox_y1"), ff(row, "mask_bbox_x2"), ff(row, "mask_bbox_y2")
    u = (x1 + x2) * 0.5
    v = (y1 + y2) * 0.5
    z = z if np.isfinite(z) and z > 0 else 3.7
    tx = (u - cx) * z / fx
    # local center z is roughly 0.39m up; BASE maps that to negative cam y.
    ty = (v - cy) * z / fy + 0.39
    return np.array([0.0, 0.0, 0.0, tx, ty, z], dtype=float)


def residual(params, row, segs, points, cands, depth_prior, K, prev=None, grasp=None, front_track=None):
    res: list[float] = []
    top = segment_obs(row, "top_rail")
    if top is not None:
        uv, _, _ = project_segment(params, "backrest_top_edge", segs, K)
        res.extend(endpoint_res((uv[0], uv[1]), top, 7.0, 2.0))
    hole = point_obs(row, "backrest_hole")
    hole_local = points.get("backrest_hole_center") if points else None
    if hole is not None and hole_local is not None:
        uv, _ = project(params, hole_local[None, :], K)
        res.extend((math.sqrt(1.6) * (uv[0] - hole) / 6.0).tolist())
    seat = segment_obs(row, "seat_front")
    if seat is not None:
        uv, _, _ = project_segment(params, "seat_front_edge", segs, K)
        res.extend(endpoint_res((uv[0], uv[1]), seat, 8.0, 1.8))
    front_c = cands.get("front_leg_candidate", [])
    for sid in ("front_leg_left", "front_leg_right"):
        obs = valid_leg_obs(row, sid)
        if obs is None:
            continue
        uv, _, _ = project_segment(params, sid, segs, K)
        res.extend(directed_endpoint_res((uv[0], uv[1]), obs, 8.0, 2.4))
    # Rear legs are useful for yaw/pitch, but less reliable under hand/body
    # occlusion. Use them only after stage1 semantic pair-gating, and keep them
    # weaker than the front A-frame anchors.
    if row.get("rear_leg_pair_valid", "0") == "1":
        for sid in ("rear_leg_left", "rear_leg_right"):
            obs = valid_leg_obs(row, sid)
            if obs is None:
                continue
            uv, _, _ = project_segment(params, sid, segs, K)
            res.extend(directed_endpoint_res((uv[0], uv[1]), obs, 12.0, 0.8))
    res.extend(feet_res(params, row, segs, K, 10.0, 1.6))
    res.extend(grasp_anchor_res(params, grasp, K))
    res.extend(front_frame_parallelogram_res(params, row, front_track, segs, K, 10.0, 1.6))
    use_hough_leg_candidates = "tracked_" not in row.get("source", "")
    if use_hough_leg_candidates and row.get("front_leg_pair_valid", "0") == "1":
        res.extend(leg_pair_res(params, ("front_leg_left", "front_leg_right"), candidate_pair(front_c), segs, K, 8.0, 1.8))
    res.extend([0.4 * x for x in bbox_res(params, segs, row, K)])
    z_ref, z_conf = depth_prior
    if np.isfinite(z_ref):
        # chair local origin lies near the hinge; center depth is represented by t_z.
        res.append(math.sqrt(max(z_conf, 0.1)) * (params[5] - z_ref) / 0.28)
    if prev is not None:
        d = params - prev
        res.extend((0.35 * d[:3] / np.array([0.08, 0.08, 0.08])).tolist())
        res.extend((0.35 * d[3:6] / np.array([0.10, 0.10, 0.18])).tolist())
        res.extend((0.25 * d[6:8] / np.array([0.12, 0.12])).tolist())
    # Stay within physically plausible open-chair joint range unless data is strong.
    res.append(0.08 * params[6] / 0.6)
    res.append(0.08 * params[7] / 0.8)
    return np.asarray(res, dtype=float)


def smooth(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    if len(out) >= 7:
        win = min(9, len(out) if len(out) % 2 else len(out) - 1)
        for j in range(out.shape[1]):
            out[:, j] = savgol_filter(out[:, j], win, 2, mode="interp")
    return out


def draw_obs_segment(canvas: np.ndarray, obs: tuple[np.ndarray, np.ndarray] | None, color: tuple[int, int, int], label: str = "") -> None:
    if obs is None:
        return
    p0 = tuple(np.round(obs[0]).astype(int))
    p1 = tuple(np.round(obs[1]).astype(int))
    cv2.line(canvas, p0, p1, (20, 20, 20), 5, cv2.LINE_AA)
    cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)
    cv2.circle(canvas, p0, 4, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, p1, 4, color, -1, cv2.LINE_AA)
    if label:
        mid = ((np.asarray(p0) + np.asarray(p1)) * 0.5).astype(int)
        cv2.putText(canvas, label, tuple(mid + np.array([4, -5])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(canvas, label, tuple(mid + np.array([4, -5])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_observation_targets(img: np.ndarray, row: dict[str, str], cands: dict[str, list[tuple[np.ndarray, np.ndarray]]]) -> np.ndarray:
    layer = img.copy()
    x1, y1, x2, y2 = [ff(row, k) for k in ("mask_bbox_x1", "mask_bbox_y1", "mask_bbox_x2", "mask_bbox_y2")]
    if all(np.isfinite(v) for v in (x1, y1, x2, y2)):
        cv2.rectangle(layer, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (60, 220, 60), 2, cv2.LINE_AA)

    for family, color in (("front_leg_candidate", (0, 190, 255)), ("rear_leg_candidate", (255, 170, 20))):
        for a, b in candidate_pair(cands.get(family, [])):
            cv2.line(layer, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), color, 2, cv2.LINE_AA)

    for prefix, color in (
        ("front_leg_left", (0, 190, 255)),
        ("front_leg_right", (0, 190, 255)),
        ("rear_leg_left", (255, 170, 20)),
        ("rear_leg_right", (255, 170, 20)),
    ):
        draw_obs_segment(layer, valid_leg_obs(row, prefix), color)
    draw_obs_segment(layer, segment_obs(row, "top_rail"), (0, 0, 255), "2D top")
    hole = point_obs(row, "backrest_hole")
    if hole is not None:
        p = tuple(np.round(hole).astype(int))
        cv2.circle(layer, p, 6, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(layer, p, 9, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(layer, "2D hole", (p[0] + 5, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(layer, "2D hole", (p[0] + 5, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    draw_obs_segment(layer, segment_obs(row, "seat_front"), (255, 0, 255), "2D seat")
    return cv2.addWeighted(layer, 0.48, img, 0.52, 0.0)


def draw_overlay(
    img: np.ndarray,
    params: np.ndarray,
    segs: dict,
    K: tuple[float, float, float, float],
    row: dict[str, str] | None = None,
    cands: dict[str, list[tuple[np.ndarray, np.ndarray]]] | None = None,
) -> np.ndarray:
    if row is not None:
        canvas = draw_observation_targets(img, row, cands or {})
    else:
        canvas = img.copy()
    colors = {
        "orange_front_leg": (0, 165, 255),
        "blue_rear_leg": (255, 160, 0),
        "top_rail_observation": (0, 0, 255),
        "seat_front_observation": (255, 0, 255),
        "seat_depth_line": (180, 120, 255),
        "board_bottom_edge": (0, 80, 255),
    }
    for sid, (_, role, a, b) in segs.items():
        if role not in colors:
            continue
        uv, _, _ = project_segment(params, sid, segs, K)
        p0 = tuple(np.round(uv[0]).astype(int))
        p1 = tuple(np.round(uv[1]).astype(int))
        cv2.line(canvas, p0, p1, (20, 20, 20), 6, cv2.LINE_AA)
        cv2.line(canvas, p0, p1, colors[role], 3, cv2.LINE_AA)
        cv2.circle(canvas, p0, 4, colors[role], -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 4, colors[role], -1, cv2.LINE_AA)
    return canvas


def finalize_video(tmp: Path, out: Path) -> None:
    ffmpeg = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    try:
        subprocess.run([ffmpeg, "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        subprocess.run([ffmpeg, "-y", "-i", str(tmp), "-c:v", "libopenh264", "-pix_fmt", "yuv420p", str(out)], check=True, capture_output=True, text=True)
    tmp.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-dir", type=Path, default=Path("samples_known_object/05_chair"))
    ap.add_argument("--max-candidates", type=int, default=8)
    ap.add_argument("--observations-csv", type=Path, default=None, help="Chair semantic observation CSV. Defaults to results/chair_semantic_observations/chair_semantic_observations.csv.")
    ap.add_argument("--leg-candidates-csv", type=Path, default=None, help="Chair leg candidates CSV. Defaults to results/chair_semantic_observations/chair_leg_candidates.csv.")
    ap.add_argument("--front-frame-tracks-csv", type=Path, default=None, help="Front-frame CoTracker observation CSV. Defaults to results/chair_semantic_observations/chair_front_frame_cotracker_observations.csv.")
    ap.add_argument("--grasp-state-csv", type=Path, default=None, help="Chair grasp anchor state CSV. Defaults to results/chair_grasp_anchor_state/chair_grasp_anchor_state.csv.")
    ap.add_argument("--local-points-csv", type=Path, default=None, help="Chair local semantic points CSV. Defaults to results/chair_semantic_local_points/chair_semantic_local_points.csv.")
    ap.add_argument("--local-segments-csv", type=Path, default=None, help="Chair local semantic segments CSV. Defaults to results/chair_semantic_local_points/chair_semantic_local_segments.csv.")
    ap.add_argument("--result-dir", type=Path, default=None, help="Directory for pose CSV outputs. Defaults to results/final_result.")
    ap.add_argument("--render-dir", type=Path, default=None, help="Directory for overlay render outputs. Defaults to results/renders/final_result.")
    ap.add_argument("--fx", type=float, default=None, help="Override camera fx. Use GVHMR K_fullimg/profile K when available.")
    ap.add_argument("--fy", type=float, default=None, help="Override camera fy. Use GVHMR K_fullimg/profile K when available.")
    ap.add_argument("--cx", type=float, default=None, help="Override camera cx. Use GVHMR K_fullimg/profile K when available.")
    ap.add_argument("--cy", type=float, default=None, help="Override camera cy. Use GVHMR K_fullimg/profile K when available.")
    ap.add_argument("--write-debug-png", action="store_true", help="Also write selected overlay_*.png debug stills.")
    ap.add_argument("--no-render", action="store_true", help="Write pose CSV/summary only; skip diagnostic overlay video/stills.")
    args = ap.parse_args()
    sample = args.sample_dir
    obs_csv = args.observations_csv or (sample / "results/chair_semantic_observations/chair_semantic_observations.csv")
    leg_candidates_csv = args.leg_candidates_csv or (sample / "results/chair_semantic_observations/chair_leg_candidates.csv")
    front_frame_csv = args.front_frame_tracks_csv or (sample / "results/chair_semantic_observations/chair_front_frame_cotracker_observations.csv")
    grasp_state_csv = args.grasp_state_csv or (sample / "results/chair_grasp_anchor_state/chair_grasp_anchor_state.csv")
    local_points_csv = args.local_points_csv or (sample / "results/chair_semantic_local_points/chair_semantic_local_points.csv")
    local_segments_csv = args.local_segments_csv or (sample / "results/chair_semantic_local_points/chair_semantic_local_segments.csv")
    obs_rows = read_rows(obs_csv)
    segs = load_segments(local_segments_csv)
    points = load_points(local_points_csv)
    cands = load_candidates(leg_candidates_csv, args.max_candidates)
    grasp_state = load_grasp_state(grasp_state_csv)
    front_frame_tracks = load_front_frame_tracks(front_frame_csv)
    depth_index = read_depth_index(sample / "results/da3/scene_depth/index.csv")
    fx, fy, cx, cy, depth_priors = estimate_camera(sample, obs_rows, depth_index)
    if args.fx is not None:
        fx = float(args.fx)
    if args.fy is not None:
        fy = float(args.fy)
    if args.cx is not None:
        cx = float(args.cx)
    if args.cy is not None:
        cy = float(args.cy)
    K = (fx, fy, cx, cy)

    fitted = []
    rows_out = []
    prev = None
    for row in obs_rows:
        fr = int(row["frame"])
        z_ref, z_conf = depth_priors.get(fr, (math.nan, 0.0))
        base = initial_params(row, z_ref, K)
        base = np.concatenate([base, np.array([0.0, 0.0])])
        starts = [base]
        if prev is not None:
            starts.insert(0, prev.copy())
        best = None
        best_cost = float("inf")
        grasp = grasp_state.get(fr)
        front_track = front_frame_tracks.get(fr)
        for st in starts:
            res = least_squares(
                lambda p: residual(p, row, segs, points, cands.get(fr, {}), (z_ref, z_conf), K, prev, grasp, front_track),
                st,
                bounds=(
                    np.array([-math.pi, -math.pi, -math.pi, -np.inf, -np.inf, 0.4, -0.95, -1.25]),
                    np.array([math.pi, math.pi, math.pi, np.inf, np.inf, 8.0, 0.55, 1.45]),
                ),
                loss="soft_l1",
                f_scale=1.0,
                max_nfev=80,
            )
            vals = residual(res.x, row, segs, points, cands.get(fr, {}), (z_ref, z_conf), K, None, grasp, front_track)
            cost = float(np.sqrt(np.mean(vals * vals))) if len(vals) else 0.0
            if cost < best_cost:
                best = res.x
                best_cost = cost
        assert best is not None
        prev = best
        fitted.append(best)
        if fr == 1 or fr % 24 == 0:
            print(f"fit frame {fr}/{int(obs_rows[-1]['frame'])} cost={best_cost:.3f}", flush=True)
        r_total = Rotation.from_matrix(Rotation.from_rotvec(best[:3]).as_matrix() @ BASE_LOCAL_TO_CAM)
        qx, qy, qz, qw = r_total.as_quat()
        rows_out.append(
            {
                "frame": fr,
                "time": row.get("time", "0"),
                "tx": best[3],
                "ty": best[4],
                "tz": best[5],
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "qw": qw,
                "rx_delta": best[0],
                "ry_delta": best[1],
                "rz_delta": best[2],
                "rear_joint_angle": best[6],
                "seat_joint_angle": best[7],
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "da3_mask_depth_m": z_ref,
                "da3_depth_conf": z_conf,
                "residual_norm": best_cost,
                "source": "chair_metric_articulated_6d_sam2_cotracker_da3",
            }
        )

    P = smooth(np.vstack(fitted))
    for i, best in enumerate(P):
        r_total = Rotation.from_matrix(Rotation.from_rotvec(best[:3]).as_matrix() @ BASE_LOCAL_TO_CAM)
        qx, qy, qz, qw = r_total.as_quat()
        rows_out[i].update({"tx": best[3], "ty": best[4], "tz": best[5], "qx": qx, "qy": qy, "qz": qz, "qw": qw, "rx_delta": best[0], "ry_delta": best[1], "rz_delta": best[2], "rear_joint_angle": best[6], "seat_joint_angle": best[7]})

    out_dir = args.result_dir or (sample / "results/final_result")
    out_dir.mkdir(parents=True, exist_ok=True)
    pose_csv = out_dir / "object_pose.csv"
    raw_pose_csv = out_dir / "object_pose_stage3_raw.csv"
    fields = list(rows_out[0].keys())
    for csv_path in (pose_csv, raw_pose_csv):
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows_out:
                writer.writerow({k: (f"{v:.9f}" if isinstance(v, float) else v) for k, v in row.items()})

    render_dir = args.render_dir or (sample / "results/renders/final_result")
    out_video = render_dir / "overlay.mp4"
    if not args.no_render:
        render_dir.mkdir(parents=True, exist_ok=True)
        first = cv2.imread(str(sample / "frames/00001.png"))
        h, w = first.shape[:2]
        tmp_video = render_dir / "overlay.tmp.mp4"
        writer = cv2.VideoWriter(str(tmp_video), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (w, h))
        for frame in range(1, len(P) + 1):
            img = cv2.imread(str(sample / "frames" / f"{frame:05d}.png"))
            if img is None:
                continue
            row = obs_rows[frame - 1]
            over = draw_overlay(img, P[frame - 1], segs, K, row, cands.get(frame, {}))
            cv2.putText(over, f"frame {frame:03d} 2D target + articulated 6D projection", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(over, f"frame {frame:03d} 2D target + articulated 6D projection", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (40, 240, 40), 1, cv2.LINE_AA)
            writer.write(over)
        writer.release()
        finalize_video(tmp_video, out_video)
        if args.write_debug_png:
            for frame in [1, 50, 100, 140, 180, 192]:
                img = cv2.imread(str(sample / "frames" / f"{frame:05d}.png"))
                if img is None:
                    continue
                params = P[frame - 1]
                row = obs_rows[frame - 1]
                over = draw_overlay(img, params, segs, K, row, cands.get(frame, {}))
                cv2.putText(over, f"frame {frame:03d} 2D target + articulated 6D projection", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 3, cv2.LINE_AA)
                cv2.putText(over, f"frame {frame:03d} 2D target + articulated 6D projection", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (40, 240, 40), 1, cv2.LINE_AA)
                cv2.imwrite(str(render_dir / f"overlay_{frame:03d}.png"), over)

    summary = {
        "object_pose_csv": str(pose_csv),
        "object_pose_stage3_raw_csv": str(raw_pose_csv),
        "observations_csv": str(obs_csv),
        "leg_candidates_csv": str(leg_candidates_csv),
        "front_frame_tracks_csv": str(front_frame_csv),
        "grasp_state_csv": str(grasp_state_csv),
        "local_points_csv": str(local_points_csv),
        "local_segments_csv": str(local_segments_csv),
        "render_dir": "" if args.no_render else str(render_dir),
        "overlay_video": "" if args.no_render else str(out_video),
        "render_skipped": bool(args.no_render),
        "frames": len(rows_out),
        "fx_fy": [fx, fy],
        "cx_cy": [cx, cy],
        "median_residual_norm": float(np.median([float(r["residual_norm"]) for r in rows_out])),
        "da3_depth_frames": len(depth_priors),
        "grasp_anchor_frames": len(grasp_state),
        "front_frame_cotracker_frames": len(front_frame_tracks),
        "note": "Metric articulated 6D initialization using DA3 depth, semantic 2D targets, chunk-local CoTracker front-frame parallelogram constraints, and mug-style stable chair-local grasp anchor reprojection residuals. Stage4 handles audio place/static and contact CSV rendering.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
