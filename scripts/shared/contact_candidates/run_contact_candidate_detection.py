#!/usr/bin/env python3
"""Build first-pass 2D contact candidates and intervals for shared human-object scenes."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import torch
import smplx


LEFT_PALM_IDS = [20, 25, 28, 31, 34]
RIGHT_PALM_IDS = [21, 40, 43, 46, 49]
LEFT_FOOT_IDS = [7, 10, 57, 58, 59]
RIGHT_FOOT_IDS = [8, 11, 60, 61, 62]
SUPPORT_IDS = [7, 8, 10, 11]


def read_ball_track(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "frame": int(row["frame"]),
                    "time": float(row["time"]),
                    "ball_center_x": float(row["ball_center_x"]),
                    "ball_center_y": float(row["ball_center_y"]),
                    "radius": float(row["radius"]),
                    "mask_area": float(row["mask_area"]),
                    "source": row["source"],
                }
            )
    if not rows:
        raise RuntimeError(f"No ball rows found in {path}")
    return rows


def read_audio_events(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("audio_frame"):
                continue
            rows.append(
                {
                    "audio_frame": int(float(row["audio_frame"])),
                    "audio_time": float(row.get("audio_time", 0.0) or 0.0),
                    "audio_score": float(row.get("audio_score", 0.0) or 0.0),
                }
            )
    return rows


def build_audio_support(frames: np.ndarray, audio_rows: list[dict[str, float | int | str]], radius: int = 2) -> np.ndarray:
    support = np.zeros(len(frames), dtype=np.float64)
    frame_to_idx = {int(fr): i for i, fr in enumerate(frames.tolist())}
    for row in audio_rows:
        center = int(row["audio_frame"])
        score = float(row["audio_score"])
        for fr in range(center - radius, center + radius + 1):
            idx = frame_to_idx.get(fr)
            if idx is None:
                continue
            dist = abs(fr - center)
            weight = max(0.0, 1.0 - 0.25 * dist)
            support[idx] = max(support[idx], score * weight)
    return support


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


def project_points(points_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.clip(points_cam[:, 2], 1e-6, None)
    u = K[:, 0, 0] * (points_cam[:, 0] / z) + K[:, 0, 2]
    v = K[:, 1, 1] * (points_cam[:, 1] / z) + K[:, 1, 2]
    valid = points_cam[:, 2] > 1e-6
    return np.stack([u, v], axis=1), valid


def build_palm_centers(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = joints[:, LEFT_PALM_IDS, :].mean(axis=1)
    right = joints[:, RIGHT_PALM_IDS, :].mean(axis=1)
    return left, right


def build_foot_points(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed per-foot GVHMR foot points for contact reasoning.

    Use the explicit GVHMR/SMPL-X body joints with stable semantics:
      10 = left foot, 11 = right foot.
    We intentionally avoid synthetic toe/edge probes here.
    """
    left = joints[:, 10, :]
    right = joints[:, 11, :]
    return left, right


def compute_feet_support_v(joints: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = joints[:, SUPPORT_IDS, :]
    B = pts.shape[0]
    uv, valid = project_points(pts.reshape(-1, 3), np.repeat(K, len(SUPPORT_IDS), axis=0))
    uv = uv.reshape(B, len(SUPPORT_IDS), 2)
    valid = valid.reshape(B, len(SUPPORT_IDS))
    left_v = np.full(B, np.nan, dtype=np.float64)
    right_v = np.full(B, np.nan, dtype=np.float64)
    support_v = np.full(B, np.nan, dtype=np.float64)
    for i in range(B):
        lv = []
        rv = []
        if valid[i, 0]:
            lv.append(float(uv[i, 0, 1]))
        if valid[i, 2]:
            lv.append(float(uv[i, 2, 1]))
        if valid[i, 1]:
            rv.append(float(uv[i, 1, 1]))
        if valid[i, 3]:
            rv.append(float(uv[i, 3, 1]))
        if lv:
            left_v[i] = max(lv)
        if rv:
            right_v[i] = max(rv)
        vals = [v for v in (left_v[i], right_v[i]) if not np.isnan(v)]
        if vals:
            support_v[i] = max(vals)
    return support_v, left_v, right_v


def gaussian_score(x: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)


def local_min_mask(values: np.ndarray, radius: int) -> np.ndarray:
    mask = np.zeros(len(values), dtype=bool)
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        if values[i] <= np.min(values[lo:hi]):
            mask[i] = True
    return mask


def local_max_mask(values: np.ndarray, radius: int) -> np.ndarray:
    mask = np.zeros(len(values), dtype=bool)
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        if values[i] >= np.max(values[lo:hi]):
            mask[i] = True
    return mask


def estimate_floor_v_from_ball(ball_v: np.ndarray, ball_r: np.ndarray, local_radius: int) -> float:
    ball_bottom_v = ball_v + ball_r
    peaks = local_max_mask(ball_bottom_v, local_radius)
    peak_values = ball_bottom_v[peaks]
    if len(peak_values) == 0:
        return float(np.percentile(ball_bottom_v, 95))
    peak_values = np.sort(peak_values)
    keep = max(5, int(round(0.2 * len(peak_values))))
    top_values = peak_values[-keep:]
    return float(np.median(top_values))


def bridge_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    if max_gap <= 0:
        return mask.copy()
    out = mask.copy()
    n = len(mask)
    i = 0
    while i < n:
        if out[i]:
            i += 1
            continue
        start = i
        while i < n and not out[i]:
            i += 1
        end = i
        left_on = start - 1 >= 0 and out[start - 1]
        right_on = end < n and out[end]
        if left_on and right_on and (end - start) <= max_gap:
            out[start:end] = True
    return out


def build_transition_state(anchor_state: np.ndarray, floor_state: np.ndarray, max_gap: int) -> np.ndarray:
    if max_gap <= 0:
        return np.zeros(len(anchor_state), dtype=bool)
    transition = np.zeros(len(anchor_state), dtype=bool)
    occupied = anchor_state | floor_state
    n = len(occupied)
    i = 0
    while i < n:
        if occupied[i]:
            i += 1
            continue
        start = i
        while i < n and not occupied[i]:
            i += 1
        end = i
        gap = end - start
        if gap <= 0 or gap > max_gap:
            continue
        left_idx = start - 1
        right_idx = end if end < n else -1
        if left_idx < 0 or right_idx < 0:
            continue
        left_anchor = bool(anchor_state[left_idx])
        left_floor = bool(floor_state[left_idx])
        right_anchor = bool(anchor_state[right_idx])
        right_floor = bool(floor_state[right_idx])
        if (left_anchor and right_floor) or (left_floor and right_anchor):
            transition[start:end] = True
    return transition


def detect_anchor_contact(*, left_contact_uv: np.ndarray, right_contact_uv: np.ndarray, left_valid: np.ndarray, right_valid: np.ndarray, ball_u: np.ndarray, ball_v: np.ndarray, ball_r: np.ndarray, dist_thresh_px: float, score_sigma_px: float, local_radius: int, state_dist_thresh_px: float, state_score_thresh: float, gap_bridge: int, anchor_region_radius_px: float, overlap_sigma_px: float, occlusion_weight: float, audio_support: np.ndarray | None = None, left_probe_names: list[str] | None = None, right_probe_names: list[str] | None = None) -> dict[str, np.ndarray | list[str]]:
    ball_uv = np.stack([ball_u, ball_v], axis=1)
    if left_contact_uv.ndim == 3:
        left_probe_names = left_probe_names or [f"probe_{i}" for i in range(left_contact_uv.shape[1])]
        right_probe_names = right_probe_names or [f"probe_{i}" for i in range(right_contact_uv.shape[1])]
        left_dists_all = np.linalg.norm(left_contact_uv - ball_uv[:, None, :], axis=2)
        right_dists_all = np.linalg.norm(right_contact_uv - ball_uv[:, None, :], axis=2)
        left_dists_all = np.where(left_valid, left_dists_all, np.inf)
        right_dists_all = np.where(right_valid, right_dists_all, np.inf)
        left_best_idx = np.argmin(left_dists_all, axis=1)
        right_best_idx = np.argmin(right_dists_all, axis=1)
        left_dist = left_dists_all[np.arange(len(ball_u)), left_best_idx]
        right_dist = right_dists_all[np.arange(len(ball_u)), right_best_idx]
        left_best_probe = [left_probe_names[i] for i in left_best_idx.tolist()]
        right_best_probe = [right_probe_names[i] for i in right_best_idx.tolist()]
        left_best_uv = left_contact_uv[np.arange(len(ball_u)), left_best_idx, :]
        right_best_uv = right_contact_uv[np.arange(len(ball_u)), right_best_idx, :]
        left_valid_any = np.any(left_valid, axis=1)
        right_valid_any = np.any(right_valid, axis=1)
    else:
        left_dist = np.linalg.norm(ball_uv - left_contact_uv, axis=1)
        right_dist = np.linalg.norm(ball_uv - right_contact_uv, axis=1)
        left_dist = np.where(left_valid, left_dist, np.inf)
        right_dist = np.where(right_valid, right_dist, np.inf)
        left_best_probe = ["center"] * len(ball_u)
        right_best_probe = ["center"] * len(ball_u)
        left_best_uv = left_contact_uv
        right_best_uv = right_contact_uv
        left_valid_any = left_valid
        right_valid_any = right_valid
    active_is_left = left_dist <= right_dist
    active_contact = np.where(active_is_left, "left", "right")
    active_probe = [left_best_probe[i] if active_is_left[i] else right_best_probe[i] for i in range(len(ball_u))]
    min_contact_dist = np.minimum(left_dist, right_dist)

    left_motion = np.zeros(len(ball_u), dtype=np.float64)
    right_motion = np.zeros(len(ball_u), dtype=np.float64)
    if len(ball_u) >= 2:
        left_step = np.linalg.norm(left_best_uv[1:] - left_best_uv[:-1], axis=1)
        right_step = np.linalg.norm(right_best_uv[1:] - right_best_uv[:-1], axis=1)
        left_motion[1:] = np.where(left_valid_any[1:] & left_valid_any[:-1], left_step, 0.0)
        right_motion[1:] = np.where(right_valid_any[1:] & right_valid_any[:-1], right_step, 0.0)
    left_toward_ball = np.zeros(len(ball_u), dtype=np.float64)
    right_toward_ball = np.zeros(len(ball_u), dtype=np.float64)
    if len(ball_u) >= 2:
        left_toward_ball[1:] = np.clip(left_dist[:-1] - left_dist[1:], 0.0, None)
        right_toward_ball[1:] = np.clip(right_dist[:-1] - right_dist[1:], 0.0, None)

    min_contact_gap = np.maximum(min_contact_dist - ball_r - anchor_region_radius_px, 0.0)
    proximity_score = gaussian_score(min_contact_gap, score_sigma_px)

    inter_anchor_dist = np.linalg.norm(left_best_uv - right_best_uv, axis=1)
    inter_anchor_dist = np.where(left_valid_any & right_valid_any, inter_anchor_dist, np.inf)
    occlusion_penalty = np.zeros_like(min_contact_gap)
    object_contact_score = np.clip((ball_r + anchor_region_radius_px - min_contact_dist) / max(anchor_region_radius_px, 1e-6), 0.0, 1.0)

    prev_gap = np.empty_like(min_contact_gap)
    prev_gap[0] = min_contact_gap[0]
    prev_gap[1:] = min_contact_gap[:-1]
    approach_delta = np.clip(prev_gap - min_contact_gap, 0.0, None)
    approach_scale = max(dist_thresh_px * 0.15, 1.0)
    approaching_score = np.clip(approach_delta / approach_scale, 0.0, 1.0)

    centers = np.stack([ball_u, ball_v], axis=1)
    velocity = np.zeros_like(centers)
    velocity[1:] = centers[1:] - centers[:-1]
    accel = np.zeros(len(centers), dtype=np.float64)
    if len(centers) >= 3:
        accel[1:-1] = np.linalg.norm(velocity[2:] - velocity[1:-1], axis=1)
    response_scale = max(np.percentile(accel, 90), 1.0)
    object_response_score = np.clip(accel / response_scale, 0.0, 1.0)
    if audio_support is None:
        audio_support = np.zeros_like(min_contact_gap)
    else:
        audio_support = np.asarray(audio_support, dtype=np.float64)

    motion_scale = max(dist_thresh_px * 0.12, 1.0)
    left_motion_score = np.clip(left_motion / motion_scale, 0.0, 1.0)
    right_motion_score = np.clip(right_motion / motion_scale, 0.0, 1.0)
    toward_scale = max(dist_thresh_px * 0.10, 1.0)
    left_toward_score = np.clip(left_toward_ball / toward_scale, 0.0, 1.0)
    right_toward_score = np.clip(right_toward_ball / toward_scale, 0.0, 1.0)

    contact_score = 0.25 * proximity_score + 0.15 * approaching_score + 0.20 * object_response_score + 0.30 * object_contact_score + 0.10 * audio_support
    contact_local_min = local_min_mask(min_contact_gap, local_radius)
    response_local_peak = local_max_mask(object_response_score, local_radius)

    # Conservative near-touch path for visible early contacts that have not yet
    # produced overlap under our simplified foot-point geometry.
    near_touch_gap_thresh = max(anchor_region_radius_px * 0.75, 18.0)
    strong_near_touch = (
        (min_contact_gap <= near_touch_gap_thresh)
        & (contact_score >= max(state_score_thresh, 0.50))
        & (proximity_score >= 0.85)
        & (approaching_score >= 0.30)
    )

    # If a strong audio impact happens nearby, accept a slightly looser near-touch.
    audio_supported_near_touch = (
        (audio_support >= 0.45)
        & (min_contact_gap <= max(anchor_region_radius_px * 1.5, 32.0))
        & (contact_score >= max(state_score_thresh, 0.42))
        & (proximity_score >= 0.78)
        & (approaching_score >= 0.20)
    )

    near_touch_candidate = strong_near_touch | audio_supported_near_touch
    is_candidate = (
        (min_contact_gap <= dist_thresh_px)
        & (
            ((object_contact_score > 0.05) | near_touch_candidate)
            & ((contact_local_min | response_local_peak) | near_touch_candidate)
        )
    )
    state = (
        (min_contact_gap <= state_dist_thresh_px)
        & ((object_contact_score > 0.10) | near_touch_candidate)
        & (contact_score >= state_score_thresh)
    )
    state = bridge_short_gaps(state, gap_bridge)
    return {
        "left_dist": left_dist,
        "right_dist": right_dist,
        "active_contact": active_contact.tolist(),
        "left_best_probe": left_best_probe,
        "right_best_probe": right_best_probe,
        "active_probe": active_probe,
        "min_contact_dist": min_contact_dist,
        "min_contact_gap": min_contact_gap,
        "inter_anchor_dist": inter_anchor_dist,
        "occlusion_penalty": occlusion_penalty,
        "object_contact_score": object_contact_score,
        "proximity_score": proximity_score,
        "approaching_score": approaching_score,
        "object_response_score": object_response_score,
        "audio_support": audio_support,
        "score": contact_score,
        "left_motion_score": left_motion_score,
        "right_motion_score": right_motion_score,
        "left_toward_score": left_toward_score,
        "right_toward_score": right_toward_score,
        "candidate": is_candidate,
        "state": state,
    }


def detect_floor_contact(*, ball_v: np.ndarray, ball_r: np.ndarray, support_v: np.ndarray, gap_thresh_px: float, score_sigma_px: float, local_radius: int, state_gap_thresh_px: float, state_score_thresh: float, gap_bridge: int) -> dict[str, np.ndarray]:
    ball_bottom_v = ball_v + ball_r
    floor_gap = np.abs(ball_bottom_v - support_v)
    floor_gap = np.where(np.isnan(floor_gap), np.inf, floor_gap)
    floor_score = gaussian_score(floor_gap, score_sigma_px)
    floor_local_peak = local_max_mask(ball_bottom_v, local_radius)
    if len(ball_bottom_v) >= 3:
        vel = np.diff(ball_bottom_v)
        bounce_turn = np.zeros(len(ball_bottom_v), dtype=bool)
        bounce_turn[1:-1] = (vel[:-1] > 0.0) & (vel[1:] < 0.0)
    else:
        bounce_turn = np.zeros(len(ball_bottom_v), dtype=bool)

    # Geometric floor contact is primarily explained by the ball bottom approaching
    # a fixed support line. Kinematic turning points only add support.
    candidate_geom = floor_local_peak & (floor_gap <= gap_thresh_px)
    is_candidate = candidate_geom | (bounce_turn & (floor_gap <= gap_thresh_px * 1.5))

    state = (floor_gap <= state_gap_thresh_px) & (floor_score >= state_score_thresh)
    state = bridge_short_gaps(state, gap_bridge)
    return {
        "ball_bottom_v": ball_bottom_v,
        "gap": floor_gap,
        "score": floor_score,
        "local_peak": floor_local_peak,
        "bounce_turn": bounce_turn,
        "candidate": is_candidate,
        "state": state,
    }


def fuse_contact_types(*, anchor_mode: str, anchor_state: np.ndarray, floor_state: np.ndarray, anchor_score: np.ndarray, floor_score: np.ndarray, max_gap: int, anchor_floor_margin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    del anchor_mode, anchor_score, floor_score, anchor_floor_margin
    fused_anchor_state = anchor_state.copy()
    multi_contact_state = fused_anchor_state & floor_state
    transition_state = build_transition_state(fused_anchor_state, floor_state, max_gap)
    return fused_anchor_state, multi_contact_state, transition_state


def enforce_audio_contact_coverage(*, frames: np.ndarray, audio_rows: list[dict[str, float | int | str]],
    audio_support: np.ndarray, active_target: list[str], anchor_score: np.ndarray,
    floor_score: np.ndarray, min_contact_gap: np.ndarray, floor_gap: np.ndarray, contact_state: np.ndarray,
    floor_state: np.ndarray, is_contact_candidate: np.ndarray, is_floor_candidate: np.ndarray,
    left_motion_score: np.ndarray, right_motion_score: np.ndarray,
    left_toward_score: np.ndarray, right_toward_score: np.ndarray,
    min_audio_score: float = 0.45, radius: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_to_idx = {int(fr): i for i, fr in enumerate(frames.tolist())}
    contact_state = contact_state.copy()
    floor_state = floor_state.copy()
    is_contact_candidate = is_contact_candidate.copy()
    is_floor_candidate = is_floor_candidate.copy()
    for row in audio_rows:
        audio_score = float(row.get('audio_score', 0.0) or 0.0)
        if audio_score < min_audio_score:
            continue
        center = int(row['audio_frame'])
        idxs = [frame_to_idx[fr] for fr in range(center - radius, center + radius + 1) if fr in frame_to_idx]
        if not idxs:
            continue

        def best_center_dist(side_idxs: list[int]) -> float:
            if not side_idxs:
                return 1e9
            return min(abs(int(frames[i]) - center) for i in side_idxs)

        def point_score(side_name: str, i: int) -> float:
            if side_name == 'left_foot':
                motion_arr = left_motion_score
                toward_arr = left_toward_score
            else:
                motion_arr = right_motion_score
                toward_arr = right_toward_score
            gap_bonus = np.clip((40.0 - min_contact_gap[i]) / 40.0, 0.0, 1.0)
            return (
                anchor_score[i]
                + 0.18 * audio_support[i]
                + 0.30 * motion_arr[i]
                + 0.30 * toward_arr[i]
                + 0.22 * gap_bonus
                - 0.03 * abs(int(frames[i]) - center)
            )

        def build_side_bursts(side_name: str, side_idxs: list[int]) -> list[list[int]]:
            if not side_idxs:
                return []
            bursts: list[list[int]] = [[side_idxs[0]]]
            for idx in side_idxs[1:]:
                if idx == bursts[-1][-1] + 1:
                    bursts[-1].append(idx)
                else:
                    bursts.append([idx])
            return bursts

        def burst_score(side_name: str, burst: list[int]) -> float:
            vals = [point_score(side_name, i) for i in burst]
            vals.sort(reverse=True)
            total = vals[0]
            if len(vals) >= 2:
                total += 0.95 * vals[1]
            if len(burst) >= 2:
                total += 0.18
            return total

        foot_idxs = [i for i in idxs if str(active_target[i]).endswith('_foot')]
        side_buckets: dict[str, list[int]] = {'left_foot': [], 'right_foot': []}
        for i in foot_idxs:
            side_buckets.setdefault(str(active_target[i]), []).append(i)
        left_side = side_buckets.get('left_foot', [])
        right_side = side_buckets.get('right_foot', [])
        left_bursts = build_side_bursts('left_foot', left_side)
        right_bursts = build_side_bursts('right_foot', right_side)
        best_left_burst = max(left_bursts, key=lambda b: burst_score('left_foot', b), default=None)
        best_right_burst = max(right_bursts, key=lambda b: burst_score('right_foot', b), default=None)
        left_score = burst_score('left_foot', best_left_burst) if best_left_burst else -1e9
        right_score = burst_score('right_foot', best_right_burst) if best_right_burst else -1e9
        if right_score > left_score + 0.05:
            dominant_side = 'right_foot'
            dominant_burst = best_right_burst
        elif left_score > right_score + 0.05:
            dominant_side = 'left_foot'
            dominant_burst = best_left_burst
        else:
            choose_right = best_center_dist(right_side) < best_center_dist(left_side)
            dominant_side = 'right_foot' if choose_right else 'left_foot'
            dominant_burst = best_right_burst if choose_right else best_left_burst
        dominant_idxs = dominant_burst or []

        # Window-level side override: if this audio window already contains a foot
        # contact but the opposing side forms a stronger burst, reassign it.
        existing_anchor_idxs = [i for i in idxs if contact_state[i] and str(active_target[i]).endswith('_foot')]
        if existing_anchor_idxs and dominant_idxs:
            existing_best = max(existing_anchor_idxs, key=lambda i: anchor_score[i] + 0.10 * audio_support[i])
            existing_side = str(active_target[existing_best])
            existing_burst = best_left_burst if existing_side == 'left_foot' else best_right_burst
            existing_score = burst_score(existing_side, existing_burst) if existing_burst else -1e9
            dominant_score = burst_score(dominant_side, dominant_idxs)
            if existing_side != dominant_side and dominant_score >= existing_score + 0.02:
                replacement = max(dominant_idxs, key=lambda i: point_score(dominant_side, i))
                contact_state[existing_best] = False
                is_contact_candidate[existing_best] = False
                contact_state[replacement] = True
                is_contact_candidate[replacement] = True
                continue

        if np.any(contact_state[idxs] | floor_state[idxs]):
            continue

        # First try the normal anchor fallback inside this audio window.
        anchor_candidates = [i for i in idxs if anchor_score[i] >= 0.42 and min_contact_gap[i] <= 36.0]
        best_anchor = max(anchor_candidates, key=lambda i: anchor_score[i] + 0.10 * audio_support[i], default=None)
        best_floor = max(idxs, key=lambda i: floor_score[i] + 0.05 * audio_support[i])
        if best_anchor is not None:
            anchor_val = anchor_score[best_anchor] + 0.10 * audio_support[best_anchor]
            floor_val = floor_score[best_floor] + 0.05 * audio_support[best_floor]
            if anchor_val >= floor_val - 0.03:
                contact_state[best_anchor] = True
                is_contact_candidate[best_anchor] = True
                continue

        # If the audio window still has no resolved frame, do a relaxed foot-only
        # fallback locally, using the dominant side decided above.
        relaxed_anchor = None
        if dominant_idxs:
            relaxed_single = [
                i for i in dominant_idxs
                if min_contact_gap[i] <= 42.0 and anchor_score[i] >= 0.28 and audio_support[i] >= 0.20
            ]
            if relaxed_single:
                relaxed_anchor = max(relaxed_single, key=lambda i: point_score(dominant_side, i))
            if relaxed_anchor is None and len(dominant_idxs) >= 2:
                for a, b in zip(dominant_idxs[:-1], dominant_idxs[1:]):
                    if b != a + 1:
                        continue
                    pair_score = anchor_score[a] + anchor_score[b]
                    pair_audio = audio_support[a] + audio_support[b]
                    pair_gap = min(min_contact_gap[a], min_contact_gap[b])
                    if pair_audio >= 0.70 and pair_score >= 0.75 and pair_gap <= 36.0:
                        relaxed_anchor = a if point_score(dominant_side, a) >= point_score(dominant_side, b) else b
                        break
        if relaxed_anchor is not None:
            contact_state[relaxed_anchor] = True
            is_contact_candidate[relaxed_anchor] = True
            continue

        # Floor fallback must still be geometrically plausible; otherwise leave the
        # audio window unresolved rather than forcing a fake floor contact.
        if floor_gap[best_floor] <= 18.0 and floor_score[best_floor] >= 0.35:
            floor_state[best_floor] = True
            is_floor_candidate[best_floor] = True
    return contact_state, floor_state, is_contact_candidate, is_floor_candidate


def build_peak_index_from_selector(mask: np.ndarray, selector: np.ndarray, reducer: str) -> np.ndarray:
    peak_index = np.arange(len(mask), dtype=np.int32)
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        start = i
        while i + 1 < len(mask) and mask[i + 1]:
            i += 1
        end = i
        local = selector[start : end + 1]
        best_rel = int(np.argmin(local) if reducer == "min" else np.argmax(local))
        best_idx = start + best_rel
        peak_index[start : end + 1] = best_idx
        i += 1
    return peak_index


def maybe_override_anchor_center_from_audio_window(*, peak_frame: int, target: str, frames: np.ndarray,
    audio_rows: list[dict[str, float | int | str]], active_target: list[str], contact_score: np.ndarray,
    min_contact_gap: np.ndarray, audio_support: np.ndarray, left_motion_score: np.ndarray,
    right_motion_score: np.ndarray, left_toward_score: np.ndarray, right_toward_score: np.ndarray,
    min_audio_score: float = 0.45, radius: int = 2) -> tuple[int, str]:
    # Only reconsider anchor centers when a nearby strong audio peak exists.
    nearby_audio = [row for row in audio_rows if float(row.get('audio_score', 0.0) or 0.0) >= min_audio_score and abs(int(row['audio_frame']) - peak_frame) <= radius]
    if not nearby_audio:
        return peak_frame, target
    row = min(nearby_audio, key=lambda r: abs(int(r['audio_frame']) - peak_frame))
    center = int(row['audio_frame'])
    frame_to_idx = {int(fr): i for i, fr in enumerate(frames.tolist())}
    idxs = [frame_to_idx[fr] for fr in range(center - radius, center + radius + 1) if fr in frame_to_idx]
    if not idxs:
        return peak_frame, target

    def point_score(side_name: str, i: int) -> float:
        motion_arr = left_motion_score if side_name == 'left_foot' else right_motion_score
        toward_arr = left_toward_score if side_name == 'left_foot' else right_toward_score
        gap_bonus = np.clip((40.0 - min_contact_gap[i]) / 40.0, 0.0, 1.0)
        return (
            contact_score[i]
            + 0.18 * audio_support[i]
            + 0.30 * motion_arr[i]
            + 0.30 * toward_arr[i]
            + 0.22 * gap_bonus
            - 0.03 * abs(int(frames[i]) - center)
        )

    side_buckets: dict[str, list[int]] = {'left_foot': [], 'right_foot': []}
    for i in idxs:
        name = str(active_target[i])
        if name.endswith('_foot'):
            side_buckets.setdefault(name, []).append(i)
    def bursts(side_idxs: list[int]) -> list[list[int]]:
        if not side_idxs:
            return []
        out = [[side_idxs[0]]]
        for i in side_idxs[1:]:
            if i == out[-1][-1] + 1:
                out[-1].append(i)
            else:
                out.append([i])
        return out
    def burst_score(side_name: str, burst: list[int]) -> float:
        vals = sorted([point_score(side_name, i) for i in burst], reverse=True)
        total = vals[0]
        if len(vals) >= 2:
            total += 0.95 * vals[1]
        if len(burst) >= 2:
            total += 0.18
        return total

    left_bursts = bursts(side_buckets.get('left_foot', []))
    right_bursts = bursts(side_buckets.get('right_foot', []))
    best_left = max(left_bursts, key=lambda b: burst_score('left_foot', b), default=None)
    best_right = max(right_bursts, key=lambda b: burst_score('right_foot', b), default=None)
    left_score = burst_score('left_foot', best_left) if best_left else -1e9
    right_score = burst_score('right_foot', best_right) if best_right else -1e9
    current_side = str(target)
    current_burst = best_left if current_side == 'left_foot' else best_right if current_side == 'right_foot' else None
    current_score = burst_score(current_side, current_burst) if current_burst is not None else -1e9
    if right_score > left_score + 0.05:
        dominant_side, dominant_burst, dominant_score = 'right_foot', best_right, right_score
    elif left_score > right_score + 0.05:
        dominant_side, dominant_burst, dominant_score = 'left_foot', best_left, left_score
    else:
        return peak_frame, target
    if dominant_burst is None:
        return peak_frame, target
    if current_side != dominant_side and dominant_score >= current_score + 0.02:
        replacement_idx = max(dominant_burst, key=lambda i: point_score(dominant_side, i))
        return int(frames[replacement_idx]), dominant_side
    return peak_frame, target



def suppress_boundary_long_foot_segments(*,
    frames: np.ndarray,
    contact_state: np.ndarray,
    is_contact_candidate: np.ndarray,
    active_target: list[str],
    audio_support: np.ndarray,
    min_contact_gap: np.ndarray,
    min_frames: int = 6,
    boundary_context: int = 10,
    max_audio_support: float = 0.20,
    max_mean_gap: float = 18.0,
    protect_audio_support: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove long boundary foot segments from hard contact output.

    These segments may be persistent contact or pure occlusion, but without full
    temporal context at the boundary they are not reliable hard anchors.
    """
    contact_state = contact_state.copy()
    is_contact_candidate = is_contact_candidate.copy()
    n = len(contact_state)
    i = 0
    while i < n:
        if not contact_state[i]:
            i += 1
            continue
        start = i
        while i + 1 < n and contact_state[i + 1]:
            i += 1
        end = i
        idxs = list(range(start, end + 1))
        if len(idxs) < min_frames:
            i += 1
            continue
        targets = [str(active_target[j]) for j in idxs if str(active_target[j]).endswith('_foot')]
        if not targets:
            i += 1
            continue
        touches_boundary = start < boundary_context or end > (n - 1 - boundary_context)
        if not touches_boundary:
            i += 1
            continue
        max_audio = float(np.max(audio_support[idxs])) if idxs else 0.0
        mean_gap = float(np.mean(min_contact_gap[idxs])) if idxs else 1e9
        if max_audio >= protect_audio_support:
            i += 1
            continue
        if max_audio <= max_audio_support and mean_gap <= max_mean_gap:
            contact_state[idxs] = False
            is_contact_candidate[idxs] = False
        i += 1
    return contact_state, is_contact_candidate


def mask_to_intervals(frames: np.ndarray, times: np.ndarray, mask: np.ndarray, label: str, target: list[str], score: np.ndarray, peak_index: np.ndarray | None = None) -> list[dict[str, object]]:
    intervals: list[dict[str, object]] = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i + 1 < n and mask[i + 1]:
            i += 1
        end = i
        seg_scores = score[start : end + 1]
        if peak_index is None:
            peak_rel = int(np.argmax(seg_scores))
            peak_idx = start + peak_rel
        else:
            peak_idx = int(peak_index[start : end + 1][0])
        target_counts: dict[str, int] = {}
        for name in target[start : end + 1]:
            target_counts[name] = target_counts.get(name, 0) + 1
        major_target = max(target_counts.items(), key=lambda kv: kv[1])[0]
        intervals.append({
            "contact_type": label,
            "target": major_target,
            "start_frame": int(frames[start]),
            "end_frame": int(frames[end]),
            "start_time": float(times[start]),
            "end_time": float(times[end]),
            "peak_frame": int(frames[peak_idx]),
            "peak_time": float(times[peak_idx]),
            "mean_score": float(np.mean(seg_scores)),
            "max_score": float(np.max(seg_scores)),
            "num_frames": int(end - start + 1),
        })
        i += 1
    return intervals


def split_contact_label(target: str, default_part: str) -> tuple[str, str, str]:
    target = str(target or "").strip()
    if not target or target == "floor":
        return "floor", "", "floor"
    if "_" in target:
        side, part = target.split("_", 1)
        return part, side, target
    return default_part, "", target


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build generic 2D contact candidates before 3D refinement.")
    parser.add_argument("--sample-dir", type=Path, default=Path("samples/basketball_01"))
    parser.add_argument("--body-model-root", type=Path, default=Path("third-party/GVHMR/inputs/checkpoints/body_models"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--contact-anchor", type=str, choices=["hand", "foot"], default="hand")
    parser.add_argument("--contact-dist-thresh-px", type=float, default=95.0)
    parser.add_argument("--contact-score-sigma-px", type=float, default=45.0)
    parser.add_argument("--contact-local-radius", type=int, default=2)
    parser.add_argument("--contact-state-dist-thresh-px", type=float, default=70.0)
    parser.add_argument("--contact-state-score-thresh", type=float, default=0.40)
    parser.add_argument("--anchor-region-radius-px", type=float, default=8.0)
    parser.add_argument("--anchor-overlap-sigma-px", type=float, default=55.0)
    parser.add_argument("--anchor-occlusion-weight", type=float, default=0.35)
    parser.add_argument("--contact-gap-bridge", type=int, default=2)
    parser.add_argument("--contact-event-mode", type=str, choices=["min_palm_dist", "highest_point"], default="highest_point")
    parser.add_argument("--floor-gap-thresh-px", type=float, default=18.0)
    parser.add_argument("--floor-score-sigma-px", type=float, default=10.0)
    parser.add_argument("--floor-local-radius", type=int, default=2)
    parser.add_argument("--floor-state-gap-thresh-px", type=float, default=16.0)
    parser.add_argument("--floor-state-score-thresh", type=float, default=0.35)
    parser.add_argument("--floor-gap-bridge", type=int, default=1)
    parser.add_argument("--transition-gap-max", type=int, default=2)
    parser.add_argument("--anchor-floor-margin", type=float, default=0.20)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    sample_dir = args.sample_dir
    results_dir = sample_dir / "results"
    out_dir = args.out_dir or (results_dir / "contact_candidates")
    out_dir.mkdir(parents=True, exist_ok=True)

    ball_rows = read_ball_track(results_dir / "tracking" / "ball_trajectory.csv")
    human = read_human_result(results_dir / "gvhmr" / "result.pkl")
    joints = build_body_joints(args.body_model_root, human)

    K = np.asarray(human["K_fullimg"], dtype=np.float64)
    if len(joints) < len(ball_rows):
        raise RuntimeError(f"GVHMR has fewer frames than tracking: {len(joints)} < {len(ball_rows)}")
    if len(joints) != len(ball_rows):
        print(f"Warning: truncating GVHMR from {len(joints)} to first {len(ball_rows)} frames to match tracking")
        joints = joints[: len(ball_rows)]
        K = K[: len(ball_rows)]

    ball_v = np.asarray([float(r["ball_center_y"]) for r in ball_rows], dtype=np.float64)
    ball_r = np.asarray([float(r["radius"]) for r in ball_rows], dtype=np.float64)
    floor_v_scalar = estimate_floor_v_from_ball(ball_v, ball_r, args.floor_local_radius)
    support_v = np.full(len(ball_rows), floor_v_scalar, dtype=np.float64)

    if args.contact_anchor == "hand":
        left_contact_cam, right_contact_cam = build_palm_centers(joints)
        left_contact_uv, left_valid = project_points(left_contact_cam, K)
        right_contact_uv, right_valid = project_points(right_contact_cam, K)
        left_support_v = np.full(len(ball_rows), np.nan, dtype=np.float64)
        right_support_v = np.full(len(ball_rows), np.nan, dtype=np.float64)
        contact_side_name = "hand"
        active_name = "active_anchor_side"
        left_name = "left_palm"
        right_name = "right_palm"
        state_name = "anchor_contact_state"
        event_name = "anchor_contact_event"
        interval_name = "anchor_contact_state"
        transition_target = "anchor_floor_transition"
    else:
        left_contact_cam, right_contact_cam = build_foot_points(joints)
        left_contact_uv, left_valid = project_points(left_contact_cam, K)
        right_contact_uv, right_valid = project_points(right_contact_cam, K)
        left_probe_names = ["fixed_point"]
        right_probe_names = ["fixed_point"]
        _feet_support_v, left_support_v, right_support_v = compute_feet_support_v(joints, K)
        contact_side_name = "foot"
        active_name = "active_anchor_side"
        left_name = "left_foot"
        right_name = "right_foot"
        state_name = "anchor_contact_state"
        event_name = "anchor_contact_event"
        interval_name = "anchor_contact_state"
        transition_target = "anchor_floor_transition"

    frames = np.asarray([int(r["frame"]) for r in ball_rows], dtype=np.int32)
    times = np.asarray([float(r["time"]) for r in ball_rows], dtype=np.float64)
    ball_u = np.asarray([float(r["ball_center_x"]) for r in ball_rows], dtype=np.float64)
    audio_rows = read_audio_events(results_dir / "events" / "audio_events.csv")
    audio_support = build_audio_support(frames, audio_rows, radius=2)

    anchor_det = detect_anchor_contact(
        left_contact_uv=left_contact_uv,
        right_contact_uv=right_contact_uv,
        left_valid=left_valid,
        right_valid=right_valid,
        ball_u=ball_u,
        ball_v=ball_v,
        ball_r=ball_r,
        dist_thresh_px=args.contact_dist_thresh_px,
        score_sigma_px=args.contact_score_sigma_px,
        local_radius=args.contact_local_radius,
        state_dist_thresh_px=args.contact_state_dist_thresh_px,
        state_score_thresh=args.contact_state_score_thresh,
        gap_bridge=args.contact_gap_bridge,
        anchor_region_radius_px=args.anchor_region_radius_px,
        overlap_sigma_px=args.anchor_overlap_sigma_px,
        occlusion_weight=args.anchor_occlusion_weight,
        audio_support=audio_support,
        left_probe_names=(left_probe_names if args.contact_anchor == "foot" else None),
        right_probe_names=(right_probe_names if args.contact_anchor == "foot" else None),
    )
    floor_det = detect_floor_contact(
        ball_v=ball_v,
        ball_r=ball_r,
        support_v=support_v,
        gap_thresh_px=args.floor_gap_thresh_px,
        score_sigma_px=args.floor_score_sigma_px,
        local_radius=args.floor_local_radius,
        state_gap_thresh_px=args.floor_state_gap_thresh_px,
        state_score_thresh=args.floor_state_score_thresh,
        gap_bridge=args.floor_gap_bridge,
    )

    left_dist = np.asarray(anchor_det["left_dist"], dtype=np.float64)
    right_dist = np.asarray(anchor_det["right_dist"], dtype=np.float64)
    active_contact = np.asarray(anchor_det["active_contact"], dtype=object)
    active_probe = list(anchor_det["active_probe"])
    left_best_probe = list(anchor_det["left_best_probe"])
    right_best_probe = list(anchor_det["right_best_probe"])
    active_target = [f"{name}_{contact_side_name}" for name in active_contact.tolist()]

    if left_contact_uv.ndim == 3:
        left_probe_index = {name: idx for idx, name in enumerate(left_probe_names)}
        right_probe_index = {name: idx for idx, name in enumerate(right_probe_names)}
        left_best_idx = np.asarray([left_probe_index[name] for name in left_best_probe], dtype=np.int32)
        right_best_idx = np.asarray([right_probe_index[name] for name in right_best_probe], dtype=np.int32)
        left_display_uv = left_contact_uv[np.arange(len(frames)), left_best_idx, :]
        right_display_uv = right_contact_uv[np.arange(len(frames)), right_best_idx, :]
    else:
        left_display_uv = left_contact_uv
        right_display_uv = right_contact_uv
    min_contact_dist = np.asarray(anchor_det["min_contact_dist"], dtype=np.float64)
    min_contact_gap = np.asarray(anchor_det["min_contact_gap"], dtype=np.float64)
    contact_score = np.asarray(anchor_det["score"], dtype=np.float64)
    left_motion_score_arr = np.asarray(anchor_det["left_motion_score"], dtype=np.float64)
    right_motion_score_arr = np.asarray(anchor_det["right_motion_score"], dtype=np.float64)
    left_toward_score_arr = np.asarray(anchor_det["left_toward_score"], dtype=np.float64)
    right_toward_score_arr = np.asarray(anchor_det["right_toward_score"], dtype=np.float64)
    inter_anchor_dist = np.asarray(anchor_det["inter_anchor_dist"], dtype=np.float64)
    occlusion_penalty = np.asarray(anchor_det["occlusion_penalty"], dtype=np.float64)
    object_contact_score = np.asarray(anchor_det["object_contact_score"], dtype=np.float64)
    is_contact_candidate = np.asarray(anchor_det["candidate"], dtype=bool)
    raw_contact_state = np.asarray(anchor_det["state"], dtype=bool)

    ball_bottom_v = np.asarray(floor_det["ball_bottom_v"], dtype=np.float64)
    floor_gap = np.asarray(floor_det["gap"], dtype=np.float64)
    floor_score = np.asarray(floor_det["score"], dtype=np.float64)
    floor_local_peak = np.asarray(floor_det["local_peak"], dtype=bool)
    bounce_turn = np.asarray(floor_det["bounce_turn"], dtype=bool)
    is_floor_candidate = np.asarray(floor_det["candidate"], dtype=bool)
    floor_state = np.asarray(floor_det["state"], dtype=bool)

    contact_state, multi_contact_state, transition_state = fuse_contact_types(
        anchor_mode=args.contact_anchor,
        anchor_state=raw_contact_state,
        floor_state=floor_state,
        anchor_score=contact_score,
        floor_score=floor_score,
        max_gap=args.transition_gap_max,
        anchor_floor_margin=args.anchor_floor_margin,
    )
    contact_state, floor_state, is_contact_candidate, is_floor_candidate = enforce_audio_contact_coverage(
        frames=frames,
        audio_rows=audio_rows,
        audio_support=audio_support,
        active_target=active_target,
        anchor_score=contact_score,
        floor_score=floor_score,
        min_contact_gap=min_contact_gap,
        floor_gap=floor_gap,
        contact_state=contact_state,
        floor_state=floor_state,
        is_contact_candidate=is_contact_candidate,
        is_floor_candidate=is_floor_candidate,
        left_motion_score=np.asarray(anchor_det["left_motion_score"], dtype=np.float64),
        right_motion_score=np.asarray(anchor_det["right_motion_score"], dtype=np.float64),
        left_toward_score=np.asarray(anchor_det["left_toward_score"], dtype=np.float64),
        right_toward_score=np.asarray(anchor_det["right_toward_score"], dtype=np.float64),
        min_audio_score=0.45,
        radius=2,
    )
    contact_state, is_contact_candidate = suppress_boundary_long_foot_segments(
        frames=frames,
        contact_state=contact_state,
        is_contact_candidate=is_contact_candidate,
        active_target=active_target,
        audio_support=audio_support,
        min_contact_gap=min_contact_gap,
        min_frames=6,
        boundary_context=10,
        max_audio_support=0.20,
        max_mean_gap=18.0,
        protect_audio_support=0.45,
    )
    multi_contact_state = contact_state & floor_state
    transition_state = build_transition_state(contact_state, floor_state, args.transition_gap_max)

    contact_rows = []
    floor_rows = []
    frame_rows = []
    center_rows = []

    for i in range(len(frames)):
        active_label = f"{str(active_contact[i])}_{contact_side_name}"
        contact_rows.append({
            "frame": int(frames[i]),
            "time": float(times[i]),
            "anchor_type": contact_side_name,
            "contact_part": contact_side_name,
            "contact_side": str(active_contact[i]),
            "contact_label": active_label,
            "ball_center_x": float(ball_u[i]),
            "ball_center_y": float(ball_v[i]),
            "left_anchor_u": float(left_display_uv[i, 0]),
            "left_anchor_v": float(left_display_uv[i, 1]),
            "right_anchor_u": float(right_display_uv[i, 0]),
            "right_anchor_v": float(right_display_uv[i, 1]),
            "left_anchor_valid": int(bool(np.any(left_valid[i]))),
            "right_anchor_valid": int(bool(np.any(right_valid[i]))),
            "left_best_probe": left_best_probe[i],
            "right_best_probe": right_best_probe[i],
            "left_anchor_dist_px": float(left_dist[i]),
            "right_anchor_dist_px": float(right_dist[i]),
            "min_anchor_dist_px": float(min_contact_dist[i]),
            "min_anchor_boundary_gap_px": float(min_contact_gap[i]),
            "inter_anchor_dist_px": float(inter_anchor_dist[i]) if np.isfinite(inter_anchor_dist[i]) else "",
            "occlusion_penalty": float(occlusion_penalty[i]),
            "object_contact_score": float(object_contact_score[i]),
            active_name: str(active_contact[i]),
            "active_anchor_probe": active_probe[i],
            "proximity_score": float(anchor_det["proximity_score"][i]),
            "approaching_score": float(anchor_det["approaching_score"][i]),
            "object_response_score": float(anchor_det["object_response_score"][i]),
            "audio_support": float(anchor_det["audio_support"][i]),
            "anchor_score": float(contact_score[i]),
            "is_candidate": int(bool(is_contact_candidate[i])),
            state_name: int(bool(contact_state[i])),
        })
        floor_rows.append({
            "frame": int(frames[i]),
            "time": float(times[i]),
            "ball_center_y": float(ball_v[i]),
            "ball_radius_px": float(ball_r[i]),
            "ball_bottom_v": float(ball_bottom_v[i]),
            "floor_v": float(support_v[i]) if not np.isnan(support_v[i]) else "",
            "left_support_v": float(left_support_v[i]) if not np.isnan(left_support_v[i]) else "",
            "right_support_v": float(right_support_v[i]) if not np.isnan(right_support_v[i]) else "",
            "floor_gap_px": float(floor_gap[i]),
            "floor_score": float(floor_score[i]),
            "is_local_peak": int(bool(floor_local_peak[i])),
            "is_bounce_turn": int(bool(bounce_turn[i])),
            "is_candidate": int(bool(is_floor_candidate[i])),
            "floor_contact_state": int(bool(floor_state[i])),
        })
        frame_rows.append({
            "frame": int(frames[i]),
            "time": float(times[i]),
            "anchor_type": contact_side_name,
            "contact_part": contact_side_name,
            "contact_side": str(active_contact[i]),
            "contact_label": active_label,
            active_name: str(active_contact[i]),
            "active_anchor_probe": active_probe[i],
            "anchor_score": float(contact_score[i]),
            "occlusion_penalty": float(occlusion_penalty[i]),
            "object_contact_score": float(object_contact_score[i]),
            "floor_score": float(floor_score[i]),
            state_name: int(bool(contact_state[i])),
            "floor_contact_state": int(bool(floor_state[i])),
            "transition_contact_state": int(bool(transition_state[i])),
            "multi_contact_state": int(bool(multi_contact_state[i])),
        })

    if args.contact_event_mode == "highest_point":
        contact_event_index = build_peak_index_from_selector(contact_state, ball_v, "min")
    else:
        contact_event_index = build_peak_index_from_selector(contact_state, min_contact_dist, "min")
    floor_event_index = build_peak_index_from_selector(floor_state, floor_gap, "min")

    intervals = []
    intervals.extend(mask_to_intervals(frames, times, contact_state, interval_name, active_target, contact_score, peak_index=contact_event_index))
    intervals.extend(mask_to_intervals(frames, times, floor_state, "floor_contact_state", ["floor"] * len(frames), floor_score, peak_index=floor_event_index))
    transition_score = np.maximum(contact_score, floor_score)
    intervals.extend(mask_to_intervals(frames, times, transition_state, "transition_contact_state", [transition_target] * len(frames), transition_score))
    intervals.sort(key=lambda r: (int(r["start_frame"]), str(r["contact_type"])))

    for r in intervals:
        peak_frame = int(r["peak_frame"])
        peak_idx = peak_frame - int(frames[0])
        if r["contact_type"] == interval_name:
            final_frame, final_target = maybe_override_anchor_center_from_audio_window(
                peak_frame=peak_frame,
                target=str(r["target"]),
                frames=frames,
                audio_rows=audio_rows,
                active_target=active_target,
                contact_score=contact_score,
                min_contact_gap=min_contact_gap,
                audio_support=np.asarray(anchor_det["audio_support"], dtype=np.float64),
                left_motion_score=left_motion_score_arr,
                right_motion_score=right_motion_score_arr,
                left_toward_score=left_toward_score_arr,
                right_toward_score=right_toward_score_arr,
                min_audio_score=0.45,
                radius=2,
            )
            final_idx = final_frame - int(frames[0])
            center_part, center_side, center_label = split_contact_label(final_target, contact_side_name)
            center_rows.append({
                "frame": final_frame,
                "time": float(times[final_idx]),
                "contact_type": event_name,
                "anchor_type": contact_side_name,
                "contact_part": center_part,
                "contact_side": center_side,
                "contact_label": center_label,
                "target": final_target,
                "score": float(contact_score[final_idx]),
                "confidence": float(contact_score[final_idx]),
                "source": ("interval_highest_point" if args.contact_event_mode == "highest_point" else f"interval_min_{contact_side_name}_dist"),
            })
        elif r["contact_type"] == "floor_contact_state":
            center_rows.append({
                "frame": peak_frame,
                "time": float(r["peak_time"]),
                "contact_type": "floor_contact_event",
                "anchor_type": "floor",
                "contact_part": "floor",
                "contact_side": "",
                "contact_label": "floor",
                "target": "floor",
                "score": float(floor_score[peak_idx]),
                "confidence": float(floor_score[peak_idx]),
                "source": "interval_min_gap",
            })
    center_rows.sort(key=lambda r: (int(r["frame"]), str(r["contact_type"])))

    write_csv(
        out_dir / "anchor_contact_candidates.csv",
        contact_rows,
        [
            "frame", "time", "anchor_type", "contact_part", "contact_side", "contact_label", "ball_center_x", "ball_center_y",
            "left_anchor_u", "left_anchor_v", "right_anchor_u", "right_anchor_v",
            "left_anchor_valid", "right_anchor_valid",
            "left_best_probe", "right_best_probe",
            "left_anchor_dist_px", "right_anchor_dist_px", "min_anchor_dist_px", "min_anchor_boundary_gap_px", "inter_anchor_dist_px", "occlusion_penalty", "object_contact_score",
            active_name, "active_anchor_probe", "proximity_score", "approaching_score", "object_response_score", "audio_support", "anchor_score", "is_candidate", state_name,
        ],
    )
    write_csv(
        out_dir / "floor_contact_candidates.csv",
        floor_rows,
        [
            "frame", "time", "ball_center_y", "ball_radius_px", "ball_bottom_v",
            "floor_v", "left_support_v", "right_support_v", "floor_gap_px", "floor_score", "is_local_peak",
            "is_bounce_turn", "is_candidate", "floor_contact_state",
        ],
    )
    write_csv(
        out_dir / "contact_state_frames.csv",
        frame_rows,
        [
            "frame", "time", "anchor_type", "contact_part", "contact_side", "contact_label", "left_anchor_u", "left_anchor_v", "right_anchor_u", "right_anchor_v",
            "left_anchor_valid", "right_anchor_valid", "left_best_probe", "right_best_probe",
            "left_anchor_dist_px", "right_anchor_dist_px", "min_anchor_dist_px", "min_anchor_boundary_gap_px", "inter_anchor_dist_px", "occlusion_penalty", "object_contact_score",
            active_name, "active_anchor_probe", "anchor_score", "floor_score", state_name,
            "floor_contact_state", "transition_contact_state", "multi_contact_state",
        ],
    )
    write_csv(
        out_dir / "contact_intervals.csv",
        intervals,
        ["contact_type", "target", "start_frame", "end_frame", "start_time", "end_time", "peak_frame", "peak_time", "mean_score", "max_score", "num_frames"],
    )
    write_csv(
        out_dir / "contact_candidates_labeled.csv",
        center_rows,
        ["frame", "time", "contact_type", "anchor_type", "contact_part", "contact_side", "contact_label", "target", "score", "confidence", "source"],
    )

    print(f"Wrote {out_dir / 'anchor_contact_candidates.csv'}")
    print(f"Wrote {out_dir / 'floor_contact_candidates.csv'}")
    print(f"Wrote {out_dir / 'contact_state_frames.csv'}")
    print(f"Wrote {out_dir / 'contact_intervals.csv'}")
    print(f"Wrote {out_dir / 'contact_candidates_labeled.csv'}")
    print(f"estimated_floor_v: {floor_v_scalar:.3f}")
    print(f"Anchor center candidates: {int(np.sum(is_contact_candidate))}")
    print(f"Floor center candidates: {int(np.sum(is_floor_candidate))}")
    print(f"Anchor contact state frames: {int(np.sum(contact_state))}")
    print(f"Floor contact state frames: {int(np.sum(floor_state))}")
    print(f"Multi-contact frames: {int(np.sum(multi_contact_state))}")


if __name__ == "__main__":
    main()
