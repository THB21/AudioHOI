#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import smplx
import torch

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.shared.generic_contact_pipeline.components.contact.utils.contact_part_utils import build_contact_part_centers

LEFT_HAND_KEY = 'left_hand'
RIGHT_HAND_KEY = 'right_hand'
LEFT_FOOT_KEY = 'left_foot'
RIGHT_FOOT_KEY = 'right_foot'


@dataclass
class SupportHypothesis:
    source: str
    support_v: np.ndarray
    confidence: np.ndarray
    stability: np.ndarray
    kind: str


def parse_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, '')
    if value in {'', None}:
        return default
    return float(value)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f'No rows found in {path}')
    return rows


def read_object_proxy_track(path: Path) -> list[dict[str, float | int | str]]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('frame'):
                continue
            ref_u = parse_float(row, 'ref_u')
            ref_v = parse_float(row, 'ref_v')
            support_u = parse_float(row, 'support_u')
            support_v = parse_float(row, 'support_v')
            contact_u = parse_float(row, 'contact_u')
            contact_v = parse_float(row, 'contact_v')
            rows.append({
                'frame': int(row['frame']),
                'time': parse_float(row, 'time', 0.0),
                'ref_u': ref_u,
                'ref_v': ref_v,
                'ref_source': row.get('ref_source', ''),
                'support_u': support_u,
                'support_v': support_v,
                'support_v_raw': parse_float(row, 'support_v_raw', support_v),
                'support_source': row.get('support_source', ''),
                'support_dv': parse_float(row, 'support_dv', 0.0),
                'contact_u': contact_u,
                'contact_v': contact_v,
                'contact_source': row.get('contact_source', ''),
                'contact_proxy_name': row.get('contact_proxy_name', ''),
                'contact_depth_offset_m': parse_float(row, 'contact_depth_offset_m', 0.0),
                'observation_conf': parse_float(row, 'observation_conf', 1.0),
                'contact_conf': parse_float(row, 'contact_conf', 0.0),
                'support_conf': parse_float(row, 'support_conf', 0.0),
                'ref_conf': parse_float(row, 'ref_conf', 0.0),
                'depth_conf': parse_float(row, 'depth_conf', 0.0),
                'object_motion_score': parse_float(row, 'object_motion_score', 0.0),
                'audio_score': parse_float(row, 'audio_score', 0.0),
                'active_label': row.get('active_label', ''),
                'active_part_u': parse_float(row, 'active_part_u'),
                'active_part_v': parse_float(row, 'active_part_v'),
            })
    if not rows:
        raise RuntimeError(f'No object proxy rows found in {path}')
    return rows


def read_object_mesh_tracks(path: Path) -> dict[int, dict[str, np.ndarray]]:
    by_frame = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row['frame'])
            by_frame.setdefault(frame, []).append(row)
    out = {}
    for frame, rows in by_frame.items():
        xy = []
        visible = []
        is_boundary = []
        point_ids = []
        for r in rows:
            x = r.get('x', '')
            y = r.get('y', '')
            if x == '' or y == '':
                continue
            xy.append([float(x), float(y)])
            visible.append(float(r.get('visible', 1.0) or 1.0) > 0.5)
            is_boundary.append(str(r.get('point_type', '')) == 'boundary')
            point_ids.append(str(r.get('point_id', '')))
        if xy:
            out[frame] = {
                'xy': np.asarray(xy, dtype=np.float64),
                'visible': np.asarray(visible, dtype=bool),
                'is_boundary': np.asarray(is_boundary, dtype=bool),
                'point_ids': np.asarray(point_ids, dtype=object),
            }
    if not out:
        raise RuntimeError(f'No object mesh tracks found in {path}')
    return out


def read_audio_events(path: Path) -> list[dict[str, float | int | str]]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('audio_frame'):
                continue
            rows.append({
                'audio_frame': int(float(row['audio_frame'])),
                'audio_time': float(row.get('audio_time', 0.0) or 0.0),
                'audio_score': float(row.get('audio_score', 0.0) or 0.0),
            })
    return rows


def build_audio_support(frames: np.ndarray, audio_rows: list[dict[str, float | int | str]], radius: int = 2) -> np.ndarray:
    support = np.zeros(len(frames), dtype=np.float64)
    frame_to_idx = {int(fr): i for i, fr in enumerate(frames.tolist())}
    for row in audio_rows:
        center = int(row['audio_frame'])
        score = float(row['audio_score'])
        for fr in range(center - radius, center + radius + 1):
            idx = frame_to_idx.get(fr)
            if idx is None:
                continue
            dist = abs(fr - center)
            weight = max(0.0, 1.0 - 0.25 * dist)
            support[idx] = max(support[idx], score * weight)
    return support


def read_human_result(path: Path) -> dict[str, np.ndarray]:
    with path.open('rb') as f:
        data = pickle.load(f)
    params = data['smpl_params_incam']
    return {
        'body_pose': np.asarray(params['body_pose'], dtype=np.float32),
        'betas': np.asarray(params['betas'], dtype=np.float32),
        'global_orient': np.asarray(params['global_orient'], dtype=np.float32),
        'transl': np.asarray(params['transl'], dtype=np.float32),
        'K_fullimg': np.asarray(data['K_fullimg'], dtype=np.float32),
    }


def build_body_joints(body_models_root: Path, human_params: dict[str, np.ndarray]) -> np.ndarray:
    model = smplx.create(
        str(body_models_root),
        model_type='smplx',
        gender='neutral',
        ext='npz',
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        batch_size=human_params['transl'].shape[0],
    )
    with torch.inference_mode():
        output = model(
            body_pose=torch.from_numpy(human_params['body_pose']),
            betas=torch.from_numpy(human_params['betas']),
            global_orient=torch.from_numpy(human_params['global_orient']),
            transl=torch.from_numpy(human_params['transl']),
            return_verts=False,
        )
    return output.joints.detach().cpu().numpy().astype(np.float64)


def project_points(points_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.clip(points_cam[:, 2], 1e-6, None)
    u = K[:, 0, 0] * (points_cam[:, 0] / z) + K[:, 0, 2]
    v = K[:, 1, 1] * (points_cam[:, 1] / z) + K[:, 1, 2]
    valid = points_cam[:, 2] > 1e-6
    return np.stack([u, v], axis=1), valid


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


def rolling_stat(values: np.ndarray, radius: int, mode: str, q: float = 0.5) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        win = values[lo:hi]
        win = win[np.isfinite(win)]
        if len(win) == 0:
            continue
        if mode == 'median':
            out[i] = float(np.median(win))
        elif mode == 'std':
            out[i] = float(np.std(win))
        elif mode == 'quantile':
            out[i] = float(np.quantile(win, q))
        else:
            raise ValueError(mode)
    return out


def ema_smooth(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    prev = math.nan
    for i, v in enumerate(values):
        if not np.isfinite(v):
            out[i] = prev
            continue
        if not np.isfinite(prev):
            prev = float(v)
        else:
            prev = alpha * prev + (1.0 - alpha) * float(v)
        out[i] = prev
    return out


def gated_ema(values: np.ndarray, alpha: float, jump_thresh: float) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    prev = math.nan
    for i, v in enumerate(values):
        if not np.isfinite(v):
            out[i] = prev
            continue
        if not np.isfinite(prev):
            prev = float(v)
        else:
            a = 0.90 if abs(v - prev) > jump_thresh else alpha
            prev = a * prev + (1.0 - a) * float(v)
        out[i] = prev
    return out


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
        gap_len = end - start
        if 0 < gap_len <= max_gap and start > 0 and end < n and out[start - 1] and out[end]:
            out[start:end] = True
    return out


def hysteresis_from_scores(enter_mask: np.ndarray, exit_mask: np.ndarray) -> np.ndarray:
    state = np.zeros(len(enter_mask), dtype=bool)
    active = False
    for i in range(len(enter_mask)):
        if not active:
            if enter_mask[i]:
                active = True
        else:
            if exit_mask[i]:
                active = False
        state[i] = active
    return state


def nearest_object_points(anchor_uv, anchor_valid, frames, object_mesh_by_frame, prefer_boundary=True):
    n = len(frames)
    nearest_dist = np.full(n, np.inf, dtype=np.float64)
    nearest_uv = np.full((n, 2), np.nan, dtype=np.float64)
    nearest_point_id = [''] * n
    nearest_valid = np.zeros(n, dtype=bool)
    for i, frame in enumerate(frames.tolist()):
        if not anchor_valid[i]:
            continue
        mesh = object_mesh_by_frame.get(int(frame))
        if mesh is None:
            continue
        pts = mesh['xy']
        valid = mesh['visible'].copy()
        if prefer_boundary:
            boundary_valid = valid & mesh['is_boundary']
            if np.any(boundary_valid):
                valid = boundary_valid
        if not np.any(valid):
            continue
        candidates = pts[valid]
        candidate_ids = mesh['point_ids'][valid]
        dists = np.linalg.norm(candidates - anchor_uv[i][None, :], axis=1)
        j = int(np.argmin(dists))
        nearest_dist[i] = float(dists[j])
        nearest_uv[i] = candidates[j]
        nearest_point_id[i] = str(candidate_ids[j])
        nearest_valid[i] = True
    return {'dist': nearest_dist, 'uv': nearest_uv, 'point_id': nearest_point_id, 'valid': nearest_valid}


def split_active_label(label: str) -> tuple[str, str, str]:
    clean = str(label or "").strip()
    if not clean or clean == "floor":
        return "", "", clean
    if "_" not in clean:
        return "", "", clean
    side, part = clean.split("_", 1)
    return part, side, clean


def detect_object_anchor_contact_from_active(*, frames, active_contact_uv, active_valid, active_labels, object_mesh_by_frame, dist_thresh_px, score_sigma_px, local_radius, state_dist_thresh_px, state_score_thresh, gap_bridge, audio_support=None):
    active_nn = nearest_object_points(active_contact_uv, active_valid, frames, object_mesh_by_frame, prefer_boundary=True)
    min_contact_gap = np.asarray(active_nn['dist'], dtype=np.float64)
    if audio_support is None:
        audio_support = np.zeros_like(min_contact_gap)
    else:
        audio_support = np.asarray(audio_support, dtype=np.float64)
    proximity_score = gaussian_score(min_contact_gap, score_sigma_px)
    object_response_score = np.zeros_like(min_contact_gap)
    contact_score = 0.55 * proximity_score + 0.15 * object_response_score + 0.30 * audio_support
    contact_local_min = local_min_mask(min_contact_gap, local_radius)
    is_candidate = (min_contact_gap <= dist_thresh_px) & contact_local_min
    state = (min_contact_gap <= state_dist_thresh_px) & (contact_score >= state_score_thresh)
    state = bridge_short_gaps(state, gap_bridge)
    active_sides = []
    active_parts = []
    clean_labels = []
    for raw in active_labels:
        part, side, clean = split_active_label(raw)
        active_sides.append(side)
        active_parts.append(part)
        clean_labels.append(clean)
    active_uv = np.asarray(active_nn['uv'], dtype=np.float64)
    return {
        'active_contact': active_sides,
        'active_part': active_parts,
        'active_label': clean_labels,
        'min_contact_gap': min_contact_gap,
        'score': contact_score,
        'candidate': is_candidate,
        'state': state,
        'active_object_point_id': list(active_nn['point_id']),
        'active_object_u': active_uv[:, 0],
        'active_object_v': active_uv[:, 1],
    }


def detect_object_anchor_contact_from_part_pair(*, frames, left_uv, right_uv, left_valid, right_valid, object_mesh_by_frame, part_name, dist_thresh_px, score_sigma_px, local_radius, state_dist_thresh_px, state_score_thresh, gap_bridge, audio_support=None):
    left_nn = nearest_object_points(left_uv, left_valid, frames, object_mesh_by_frame, prefer_boundary=True)
    right_nn = nearest_object_points(right_uv, right_valid, frames, object_mesh_by_frame, prefer_boundary=True)
    left_gap = np.asarray(left_nn['dist'], dtype=np.float64)
    right_gap = np.asarray(right_nn['dist'], dtype=np.float64)
    active_is_left = left_gap <= right_gap
    min_contact_gap = np.minimum(left_gap, right_gap)
    if audio_support is None:
        audio_support = np.zeros_like(min_contact_gap)
    else:
        audio_support = np.asarray(audio_support, dtype=np.float64)
    proximity_score = gaussian_score(min_contact_gap, score_sigma_px)
    object_response_score = np.zeros_like(min_contact_gap)
    contact_score = 0.55 * proximity_score + 0.15 * object_response_score + 0.30 * audio_support
    contact_local_min = local_min_mask(min_contact_gap, local_radius)
    is_candidate = (min_contact_gap <= dist_thresh_px) & contact_local_min
    state = (min_contact_gap <= state_dist_thresh_px) & (contact_score >= state_score_thresh)
    state = bridge_short_gaps(state, gap_bridge)

    active_sides = ['left' if flag else 'right' for flag in active_is_left.tolist()]
    active_parts = [part_name] * len(active_sides)
    clean_labels = [f"left_{part_name}" if flag else f"right_{part_name}" for flag in active_is_left.tolist()]
    active_object_point_id = [left_nn['point_id'][i] if active_is_left[i] else right_nn['point_id'][i] for i in range(len(frames))]
    active_object_u = np.asarray([left_nn['uv'][i, 0] if active_is_left[i] else right_nn['uv'][i, 0] for i in range(len(frames))], dtype=np.float64)
    active_object_v = np.asarray([left_nn['uv'][i, 1] if active_is_left[i] else right_nn['uv'][i, 1] for i in range(len(frames))], dtype=np.float64)
    return {
        'active_contact': active_sides,
        'active_part': active_parts,
        'active_label': clean_labels,
        'min_contact_gap': min_contact_gap,
        'score': contact_score,
        'candidate': is_candidate,
        'state': state,
        'active_object_point_id': active_object_point_id,
        'active_object_u': active_object_u,
        'active_object_v': active_object_v,
    }


def detect_object_anchor_contact_from_feet(**kwargs):
    return detect_object_anchor_contact_from_part_pair(
        frames=kwargs['frames'],
        left_uv=kwargs['left_foot_uv'],
        right_uv=kwargs['right_foot_uv'],
        left_valid=kwargs['left_valid'],
        right_valid=kwargs['right_valid'],
        object_mesh_by_frame=kwargs['object_mesh_by_frame'],
        part_name='foot',
        dist_thresh_px=kwargs['dist_thresh_px'],
        score_sigma_px=kwargs['score_sigma_px'],
        local_radius=kwargs['local_radius'],
        state_dist_thresh_px=kwargs['state_dist_thresh_px'],
        state_score_thresh=kwargs['state_score_thresh'],
        gap_bridge=kwargs['gap_bridge'],
        audio_support=kwargs.get('audio_support'),
    )


def detect_object_anchor_contact_from_hands(**kwargs):
    return detect_object_anchor_contact_from_part_pair(
        frames=kwargs['frames'],
        left_uv=kwargs['left_hand_uv'],
        right_uv=kwargs['right_hand_uv'],
        left_valid=kwargs['left_valid'],
        right_valid=kwargs['right_valid'],
        object_mesh_by_frame=kwargs['object_mesh_by_frame'],
        part_name='hand',
        dist_thresh_px=kwargs['dist_thresh_px'],
        score_sigma_px=kwargs['score_sigma_px'],
        local_radius=kwargs['local_radius'],
        state_dist_thresh_px=kwargs['state_dist_thresh_px'],
        state_score_thresh=kwargs['state_score_thresh'],
        gap_bridge=kwargs['gap_bridge'],
        audio_support=kwargs.get('audio_support'),
    )


def build_support_hypotheses(object_support_raw: np.ndarray, left_foot_uv: np.ndarray, right_foot_uv: np.ndarray, left_valid: np.ndarray, right_valid: np.ndarray) -> list[SupportHypothesis]:
    feet_bottom = np.full(len(object_support_raw), np.nan, dtype=np.float64)
    for i in range(len(object_support_raw)):
        vals = []
        if left_valid[i]:
            vals.append(float(left_foot_uv[i, 1]))
        if right_valid[i]:
            vals.append(float(right_foot_uv[i, 1]))
        if vals:
            feet_bottom[i] = max(vals)
    global_floor_v = float(np.nanpercentile(feet_bottom, 90)) if np.any(np.isfinite(feet_bottom)) else math.nan
    global_floor_arr = np.full(len(object_support_raw), global_floor_v, dtype=np.float64)
    global_conf = np.full(len(object_support_raw), 0.75 if np.isfinite(global_floor_v) else 0.0, dtype=np.float64)
    global_stability = np.full(len(object_support_raw), 1.0 if np.isfinite(global_floor_v) else 0.0, dtype=np.float64)

    object_support_smooth = ema_smooth(object_support_raw, 0.70)
    local_plane_v = rolling_stat(object_support_smooth, radius=8, mode='quantile', q=0.90)
    local_std = rolling_stat(object_support_smooth, radius=8, mode='std')
    vel = np.abs(np.gradient(np.where(np.isfinite(object_support_smooth), object_support_smooth, np.nanmedian(object_support_smooth[np.isfinite(object_support_smooth)]))))
    local_conf = np.exp(-np.nan_to_num(local_std, nan=50.0) / 5.0) * np.exp(-np.nan_to_num(vel, nan=50.0) / 5.0)
    local_conf = np.clip(local_conf, 0.0, 1.0)
    local_stability = 1.0 / (1.0 + np.nan_to_num(local_std, nan=50.0) / 3.0)

    image_bottom = np.nanpercentile(object_support_raw[np.isfinite(object_support_raw)], 99) if np.any(np.isfinite(object_support_raw)) else math.nan
    image_bottom_arr = np.full(len(object_support_raw), image_bottom, dtype=np.float64)
    image_conf = np.full(len(object_support_raw), 0.15 if np.isfinite(image_bottom) else 0.0, dtype=np.float64)
    image_stability = np.full(len(object_support_raw), 1.0 if np.isfinite(image_bottom) else 0.0, dtype=np.float64)

    return [
        SupportHypothesis('gvhmr_floor', global_floor_arr, global_conf, global_stability, 'floor'),
        SupportHypothesis('local_plane', local_plane_v, local_conf, local_stability, 'unknown_plane'),
        SupportHypothesis('image_bottom', image_bottom_arr, image_conf, image_stability, 'fallback'),
    ]


def select_support_surface(object_support_raw: np.ndarray, anchor_score: np.ndarray, hypotheses: list[SupportHypothesis]) -> dict[str, np.ndarray | list[str]]:
    object_support_smooth = ema_smooth(object_support_raw, 0.70)
    n = len(object_support_raw)
    chosen_v_raw = np.full(n, np.nan, dtype=np.float64)
    chosen_conf = np.zeros(n, dtype=np.float64)
    chosen_source = ['missing'] * n
    chosen_type = ['unknown'] * n
    signed_gap_raw = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        best_score = -1e9
        best = None
        for hyp in hypotheses:
            sv = hyp.support_v[i]
            if not np.isfinite(sv) or not np.isfinite(object_support_smooth[i]):
                continue
            signed_gap = object_support_smooth[i] - sv
            gap_score = math.exp(-abs(signed_gap) / 12.0)
            if signed_gap > 3.0:
                gap_score *= math.exp(-(signed_gap - 3.0) / 4.0)
            competition_penalty = 0.35 * anchor_score[i]
            score = 0.45 * gap_score + 0.30 * hyp.confidence[i] + 0.25 * hyp.stability[i] - competition_penalty
            if score > best_score:
                best_score = score
                best = (sv, hyp.confidence[i], hyp.source, hyp.kind, signed_gap)
        if best is not None:
            chosen_v_raw[i], chosen_conf[i], chosen_source[i], chosen_type[i], signed_gap_raw[i] = best

    chosen_v_smooth = gated_ema(chosen_v_raw, alpha=0.70, jump_thresh=10.0)
    signed_gap = object_support_smooth - chosen_v_smooth
    abs_gap = np.abs(signed_gap)
    support_std = rolling_stat(chosen_v_smooth, radius=4, mode='std')
    gap_std = rolling_stat(signed_gap, radius=4, mode='std')
    gap_med = rolling_stat(abs_gap, radius=4, mode='median')

    return {
        'object_support_raw': object_support_raw,
        'object_support_smooth': object_support_smooth,
        'support_v_raw': chosen_v_raw,
        'support_v_smooth': chosen_v_smooth,
        'support_conf': chosen_conf,
        'support_source': chosen_source,
        'support_surface_type': chosen_type,
        'signed_gap': signed_gap,
        'abs_gap': abs_gap,
        'gap_std': gap_std,
        'gap_med': gap_med,
        'support_std': support_std,
    }


def detect_object_support_contact(*, support_track: dict[str, np.ndarray | list[str]], anchor_score: np.ndarray, gap_thresh_px, score_sigma_px, local_radius, state_gap_thresh_px, state_score_thresh, gap_bridge, enter_gap_thresh_px=10.0, exit_gap_thresh_px=16.0, gap_std_thresh_px=4.0, support_std_thresh_px=4.0):
    object_support_v = np.asarray(support_track['object_support_smooth'], dtype=np.float64)
    support_v = np.asarray(support_track['support_v_smooth'], dtype=np.float64)
    signed_gap = np.asarray(support_track['signed_gap'], dtype=np.float64)
    support_gap = np.asarray(support_track['abs_gap'], dtype=np.float64)
    gap_std = np.asarray(support_track['gap_std'], dtype=np.float64)
    gap_med = np.asarray(support_track['gap_med'], dtype=np.float64)
    support_std = np.asarray(support_track['support_std'], dtype=np.float64)
    support_conf = np.asarray(support_track['support_conf'], dtype=np.float64)

    support_gap = np.where(np.isnan(support_gap), np.inf, support_gap)
    support_score = gaussian_score(support_gap, score_sigma_px)
    direction_penalty = np.where(signed_gap <= 2.0, 1.0, np.exp(-(signed_gap - 2.0) / 3.0))
    stability_score = np.exp(-np.nan_to_num(gap_std, nan=50.0) / 4.0) * np.exp(-np.nan_to_num(support_std, nan=50.0) / 4.0)
    competition_penalty = 1.0 - np.clip(anchor_score, 0.0, 1.0)
    floor_score = support_score * direction_penalty * (0.4 + 0.6 * support_conf) * stability_score * (0.6 + 0.4 * competition_penalty)

    support_local_peak = local_max_mask(object_support_v, local_radius)
    is_candidate = support_local_peak & (support_gap <= gap_thresh_px) & (signed_gap <= 2.0)

    gap_ok = np.nan_to_num(gap_med, nan=np.inf) <= state_gap_thresh_px
    gap_stable = np.nan_to_num(gap_std, nan=np.inf) <= gap_std_thresh_px
    support_stable = np.nan_to_num(support_std, nan=np.inf) <= support_std_thresh_px
    enter_mask = gap_ok & gap_stable & support_stable & (support_gap <= enter_gap_thresh_px) & (floor_score >= state_score_thresh) & (anchor_score < 0.55) & (signed_gap <= 2.0)
    exit_mask = (support_gap >= exit_gap_thresh_px) | (signed_gap > 4.0) | (anchor_score > 0.65)
    state = hysteresis_from_scores(enter_mask, exit_mask)
    state = bridge_short_gaps(state, gap_bridge)
    return {
        'object_support_v_raw': np.asarray(support_track['object_support_raw'], dtype=np.float64),
        'object_support_v': object_support_v,
        'support_v_raw': np.asarray(support_track['support_v_raw'], dtype=np.float64),
        'support_v': support_v,
        'gap': support_gap,
        'signed_gap': signed_gap,
        'score': floor_score,
        'local_peak': support_local_peak,
        'candidate': is_candidate,
        'state': state,
        'support_conf': support_conf,
        'support_source': list(support_track['support_source']),
        'support_surface_type': list(support_track['support_surface_type']),
        'gap_std': gap_std,
        'support_std': support_std,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Experimental radius-free object contact candidate detection.')
    parser.add_argument('--sample-dir', type=Path, required=True)
    parser.add_argument('--object-proxy-observation-csv', type=Path, default=None)
    parser.add_argument('--object-mesh-csv', type=Path, default=None)
    parser.add_argument('--out-subdir', type=str, default='contact_candidates_object_proxy_test')
    parser.add_argument('--body-model-root', type=Path, default=Path('third-party/GVHMR/inputs/checkpoints/body_models'))
    parser.add_argument('--contact-dist-thresh-px', type=float, default=28.0)
    parser.add_argument('--contact-score-sigma-px', type=float, default=18.0)
    parser.add_argument('--contact-local-radius', type=int, default=2)
    parser.add_argument('--contact-state-dist-thresh-px', type=float, default=38.0)
    parser.add_argument('--contact-state-score-thresh', type=float, default=0.35)
    parser.add_argument('--contact-gap-bridge', type=int, default=2)
    parser.add_argument('--floor-gap-thresh-px', type=float, default=14.0)
    parser.add_argument('--floor-score-sigma-px', type=float, default=10.0)
    parser.add_argument('--floor-local-radius', type=int, default=2)
    parser.add_argument('--floor-state-gap-thresh-px', type=float, default=12.0)
    parser.add_argument('--floor-state-score-thresh', type=float, default=0.45)
    parser.add_argument('--floor-gap-bridge', type=int, default=2)
    args = parser.parse_args()

    sample_dir = args.sample_dir
    results_dir = sample_dir / 'results'
    out_dir = results_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    object_rows = read_object_proxy_track(args.object_proxy_observation_csv or (results_dir / 'object_proxy_observations_test' / 'object_proxy_observations.csv'))
    object_mesh_by_frame = read_object_mesh_tracks(args.object_mesh_csv or (results_dir / 'tracking' / 'object_mesh_tracks_test.csv'))
    audio_rows = read_audio_events(results_dir / 'events' / 'audio_events.csv')

    human = read_human_result(results_dir / 'gvhmr' / 'result.pkl')
    joints = build_body_joints(args.body_model_root, human)
    K = np.asarray(human['K_fullimg'], dtype=np.float64)
    if len(joints) < len(object_rows):
        raise RuntimeError(f'GVHMR has fewer frames than object proxy rows: {len(joints)} < {len(object_rows)}')
    joints = joints[:len(object_rows)]
    K = K[:len(object_rows)]

    frames = np.asarray([int(r['frame']) for r in object_rows], dtype=np.int32)
    times = np.asarray([float(r['time']) for r in object_rows], dtype=np.float64)
    object_support_v_raw = np.asarray([float(r['support_v']) for r in object_rows], dtype=np.float64)
    audio_support = build_audio_support(frames, audio_rows, radius=2)

    centers = build_contact_part_centers(joints)
    left_foot_uv, left_foot_valid = project_points(centers[LEFT_FOOT_KEY], K)
    right_foot_uv, right_foot_valid = project_points(centers[RIGHT_FOOT_KEY], K)
    active_contact_uv = np.asarray([[parse_float(r, 'active_part_u'), parse_float(r, 'active_part_v')] for r in object_rows], dtype=np.float64)
    active_valid = np.all(np.isfinite(active_contact_uv), axis=1)
    active_labels = [str(r.get('active_label', '') or '') for r in object_rows]
    foot_driven_mode = any(token in sample_dir.name.lower() for token in ('football', 'soccer', 'foot'))

    if foot_driven_mode:
        anchor_det = detect_object_anchor_contact_from_feet(
            frames=frames,
            left_foot_uv=left_foot_uv,
            right_foot_uv=right_foot_uv,
            left_valid=left_foot_valid,
            right_valid=right_foot_valid,
            object_mesh_by_frame=object_mesh_by_frame,
            dist_thresh_px=args.contact_dist_thresh_px,
            score_sigma_px=args.contact_score_sigma_px,
            local_radius=args.contact_local_radius,
            state_dist_thresh_px=args.contact_state_dist_thresh_px,
            state_score_thresh=args.contact_state_score_thresh,
            gap_bridge=args.contact_gap_bridge,
            audio_support=audio_support,
        )
    else:
        anchor_det = detect_object_anchor_contact_from_active(
            frames=frames,
            active_contact_uv=active_contact_uv,
            active_valid=active_valid,
            active_labels=active_labels,
            object_mesh_by_frame=object_mesh_by_frame,
            dist_thresh_px=args.contact_dist_thresh_px,
            score_sigma_px=args.contact_score_sigma_px,
            local_radius=args.contact_local_radius,
            state_dist_thresh_px=args.contact_state_dist_thresh_px,
            state_score_thresh=args.contact_state_score_thresh,
            gap_bridge=args.contact_gap_bridge,
            audio_support=audio_support,
        )

    support_hypotheses = build_support_hypotheses(object_support_v_raw, left_foot_uv, right_foot_uv, left_foot_valid, right_foot_valid)
    support_track = select_support_surface(object_support_v_raw, np.asarray(anchor_det['score'], dtype=np.float64), support_hypotheses)
    floor_det = detect_object_support_contact(
        support_track=support_track,
        anchor_score=np.asarray(anchor_det['score'], dtype=np.float64),
        gap_thresh_px=args.floor_gap_thresh_px,
        score_sigma_px=args.floor_score_sigma_px,
        local_radius=args.floor_local_radius,
        state_gap_thresh_px=args.floor_state_gap_thresh_px,
        state_score_thresh=args.floor_state_score_thresh,
        gap_bridge=args.floor_gap_bridge,
    )

    anchor_rows = []
    floor_rows = []
    state_rows = []
    labeled_rows = []
    intervals = []
    current_interval = None
    diagnostics_rows = []

    anchor_event_index = np.arange(len(frames), dtype=np.int32)
    i = 0
    anchor_state = np.asarray(anchor_det['state'], dtype=bool)
    while i < len(frames):
        if not anchor_state[i]:
            i += 1
            continue
        start = i
        while i + 1 < len(frames) and anchor_state[i + 1]:
            i += 1
        end = i
        local_gap = np.asarray(anchor_det['min_contact_gap'][start:end + 1], dtype=np.float64)
        best_rel = int(np.argmin(local_gap))
        best_idx = start + best_rel
        anchor_event_index[start:end + 1] = best_idx
        i += 1

    floor_event_index = np.arange(len(frames), dtype=np.int32)
    i = 0
    floor_state_mask = np.asarray(floor_det['state'], dtype=bool)
    while i < len(frames):
        if not floor_state_mask[i]:
            i += 1
            continue
        start = i
        while i + 1 < len(frames) and floor_state_mask[i + 1]:
            i += 1
        end = i
        local_gap = np.asarray(floor_det['gap'][start:end + 1], dtype=np.float64)
        best_rel = int(np.argmin(local_gap))
        best_idx = start + best_rel
        floor_event_index[start:end + 1] = best_idx
        i += 1
    for i, frame in enumerate(frames.tolist()):
        side = anchor_det['active_contact'][i] or ''
        active_label = anchor_det['active_label'][i]
        active_part = anchor_det['active_part'][i] or ''
        if bool(floor_det['candidate'][i]) and not bool(anchor_det['candidate'][i]):
            contact_part, contact_side, contact_label = 'floor', '', 'floor'
        else:
            contact_part, contact_side, contact_label = split_active_label(active_label)
        anchor_rows.append({'frame': frame, 'time': f'{times[i]:.6f}', 'active_side': side, 'active_object_point_id': anchor_det['active_object_point_id'][i], 'active_object_u': f"{anchor_det['active_object_u'][i]:.3f}" if np.isfinite(anchor_det['active_object_u'][i]) else '', 'active_object_v': f"{anchor_det['active_object_v'][i]:.3f}" if np.isfinite(anchor_det['active_object_v'][i]) else '', 'min_object_boundary_gap_px': f"{anchor_det['min_contact_gap'][i]:.6f}", 'anchor_score': f"{anchor_det['score'][i]:.6f}", 'is_candidate': int(bool(anchor_det['candidate'][i])), 'anchor_contact_state': int(bool(anchor_det['state'][i]))})
        floor_rows.append({'frame': frame, 'time': f'{times[i]:.6f}', 'object_support_v_raw': f"{floor_det['object_support_v_raw'][i]:.3f}", 'object_support_v': f"{floor_det['object_support_v'][i]:.3f}", 'support_v_raw': f"{floor_det['support_v_raw'][i]:.3f}" if np.isfinite(floor_det['support_v_raw'][i]) else '', 'support_v': f"{floor_det['support_v'][i]:.3f}" if np.isfinite(floor_det['support_v'][i]) else '', 'signed_gap': f"{floor_det['signed_gap'][i]:.6f}", 'gap': f"{floor_det['gap'][i]:.6f}", 'gap_std': f"{floor_det['gap_std'][i]:.6f}" if np.isfinite(floor_det['gap_std'][i]) else '', 'support_std': f"{floor_det['support_std'][i]:.6f}" if np.isfinite(floor_det['support_std'][i]) else '', 'floor_score': f"{floor_det['score'][i]:.6f}", 'support_conf': f"{floor_det['support_conf'][i]:.6f}", 'support_source': floor_det['support_source'][i], 'support_surface_type': floor_det['support_surface_type'][i], 'is_candidate': int(bool(floor_det['candidate'][i])), 'floor_contact_state': int(bool(floor_det['state'][i]))})
        state_rows.append({'frame': frame, 'time': f'{times[i]:.6f}', 'anchor_type': active_part, 'contact_part': contact_part, 'contact_side': contact_side, 'contact_label': contact_label, 'anchor_score': f"{anchor_det['score'][i]:.6f}", 'floor_score': f"{floor_det['score'][i]:.6f}", 'anchor_contact_state': int(bool(anchor_det['state'][i])), 'human_contact_state': int(bool(anchor_det['state'][i])), 'floor_contact_state': int(bool(floor_det['state'][i])), 'transition_contact_state': int(bool(anchor_det['state'][i]) != bool(floor_det['state'][i]) and (bool(anchor_det['state'][i]) or bool(floor_det['state'][i]))), 'multi_contact_state': int(bool(anchor_det['state'][i]) and bool(floor_det['state'][i])), 'support_surface_type': floor_det['support_surface_type'][i], 'support_source': floor_det['support_source'][i], 'signed_support_gap_px': f"{floor_det['signed_gap'][i]:.6f}", 'support_conf': f"{floor_det['support_conf'][i]:.6f}", 'active_object_point_id': anchor_det['active_object_point_id'][i], 'active_object_u': f"{anchor_det['active_object_u'][i]:.3f}" if np.isfinite(anchor_det['active_object_u'][i]) else '', 'active_object_v': f"{anchor_det['active_object_v'][i]:.3f}" if np.isfinite(anchor_det['active_object_v'][i]) else '', 'min_object_boundary_gap_px': f"{anchor_det['min_contact_gap'][i]:.6f}"})
        if bool(anchor_det['candidate'][i]) and int(anchor_event_index[i]) == i:
            labeled_rows.append({'frame': frame, 'time': f'{times[i]:.6f}', 'contact_type': 'anchor_contact_event', 'anchor_type': active_part, 'target': contact_label, 'score': f"{anchor_det['score'][i]:.6f}", 'confidence': f"{anchor_det['score'][i]:.6f}", 'source': 'object_mesh_boundary'})
        if bool(floor_det['candidate'][i]) and int(floor_event_index[i]) == i:
            labeled_rows.append({'frame': frame, 'time': f'{times[i]:.6f}', 'contact_type': 'plane_support_contact_event', 'anchor_type': floor_det['support_surface_type'][i], 'target': floor_det['support_surface_type'][i], 'score': f"{floor_det['score'][i]:.6f}", 'confidence': f"{floor_det['support_conf'][i]:.6f}", 'source': floor_det['support_source'][i]})
        active = bool(anchor_det['state'][i]) or bool(floor_det['state'][i])
        if active and current_interval is None:
            current_interval = {'start_frame': frame, 'end_frame': frame, 'support_surface_type': floor_det['support_surface_type'][i] if bool(floor_det['state'][i]) else ''}
        elif active and current_interval is not None:
            current_interval['end_frame'] = frame
        elif (not active) and current_interval is not None:
            intervals.append(current_interval)
            current_interval = None

        diagnostics_rows.append({
            'frame': frame,
            'time': f'{times[i]:.6f}',
            'object_support_v_raw': f"{support_track['object_support_raw'][i]:.6f}",
            'object_support_v_smoothed': f"{support_track['object_support_smooth'][i]:.6f}",
            'support_v_raw': f"{support_track['support_v_raw'][i]:.6f}" if np.isfinite(support_track['support_v_raw'][i]) else '',
            'support_v_smoothed': f"{support_track['support_v_smooth'][i]:.6f}" if np.isfinite(support_track['support_v_smooth'][i]) else '',
            'signed_gap': f"{support_track['signed_gap'][i]:.6f}" if np.isfinite(support_track['signed_gap'][i]) else '',
            'abs_gap': f"{support_track['abs_gap'][i]:.6f}" if np.isfinite(support_track['abs_gap'][i]) else '',
            'support_confidence': f"{support_track['support_conf'][i]:.6f}",
            'support_source': support_track['support_source'][i],
            'support_surface_type': support_track['support_surface_type'][i],
            'anchor_score': f"{anchor_det['score'][i]:.6f}",
            'floor_score': f"{floor_det['score'][i]:.6f}",
            'plane_support_state': int(bool(floor_det['state'][i])),
            'human_contact_state': int(bool(anchor_det['state'][i])),
        })
    if current_interval is not None:
        intervals.append(current_interval)

    write_csv(out_dir / 'anchor_contact_candidates.csv', anchor_rows, ['frame','time','active_side','active_object_point_id','active_object_u','active_object_v','min_object_boundary_gap_px','anchor_score','is_candidate','anchor_contact_state'])
    write_csv(out_dir / 'floor_contact_candidates.csv', floor_rows, ['frame','time','object_support_v_raw','object_support_v','support_v_raw','support_v','signed_gap','gap','gap_std','support_std','floor_score','support_conf','support_source','support_surface_type','is_candidate','floor_contact_state'])
    write_csv(out_dir / 'support_diagnostics.csv', diagnostics_rows, ['frame','time','object_support_v_raw','object_support_v_smoothed','support_v_raw','support_v_smoothed','signed_gap','abs_gap','support_confidence','support_source','support_surface_type','anchor_score','floor_score','plane_support_state','human_contact_state'])
    write_csv(out_dir / 'contact_state_frames.csv', state_rows, ['frame','time','anchor_type','contact_part','contact_side','contact_label','anchor_score','floor_score','anchor_contact_state','human_contact_state','floor_contact_state','transition_contact_state','multi_contact_state','support_surface_type','support_source','signed_support_gap_px','support_conf','active_object_point_id','active_object_u','active_object_v','min_object_boundary_gap_px'])
    write_csv(out_dir / 'contact_candidates_labeled.csv', labeled_rows, ['frame','time','contact_type','anchor_type','target','score','confidence','source'])
    write_csv(out_dir / 'contact_intervals.csv', intervals, ['start_frame','end_frame','support_surface_type'])
    print(f'contact_candidate_dir: {out_dir}')


if __name__ == '__main__':
    main()
