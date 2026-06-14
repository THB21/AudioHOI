#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares


BODY_R = 0.040
BODY_H = 0.096
BODY_D = 0.014
HANDLE_CENTER = np.array([-0.076, 0.000, 0.004], dtype=float)
HANDLE_RX = 0.030
HANDLE_RY = 0.031


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def ff(row: dict[str, str], key: str, default: float = math.nan) -> float:
    v = row.get(key, '')
    if v in ('', None):
        return default
    try:
        return float(v)
    except ValueError:
        return default


def load_K(sample: Path) -> np.ndarray:
    with (sample / 'results' / 'gvhmr' / 'result.pkl').open('rb') as f:
        data = pickle.load(f)
    return np.asarray(data['K_fullimg'], dtype=float)


def backproject(u: float, v: float, z: float, K: np.ndarray) -> np.ndarray:
    return np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], dtype=float)


def project(P: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = max(float(P[2]), 1e-6)
    return np.array([K[0, 0] * P[0] / z + K[0, 2], K[1, 1] * P[1] / z + K[1, 2]], dtype=float)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_R(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def pitch_R(pitch: float) -> np.ndarray:
    c, s = math.cos(pitch), math.sin(pitch)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def roll_R(roll: float) -> np.ndarray:
    c, s = math.cos(roll), math.sin(roll)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def object_R(yaw: float, pitch: float, roll: float) -> np.ndarray:
    return yaw_R(yaw) @ pitch_R(pitch) @ roll_R(roll)


def transform(params: np.ndarray, pts: np.ndarray) -> np.ndarray:
    x, y, z, yaw, pitch, roll, scale = params
    return pts @ (scale * object_R(yaw, pitch, roll)).T + np.array([x, y, z], dtype=float)


def project_pts(params: np.ndarray, pts: np.ndarray, K: np.ndarray) -> np.ndarray:
    P = transform(params, pts)
    return np.array([project(p, K) for p in P], dtype=float)


def body_points() -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    top = np.stack([BODY_R*np.cos(theta), np.full_like(theta, -BODY_H/2), BODY_D*np.sin(theta)], axis=1)
    bot = np.stack([BODY_R*np.cos(theta), np.full_like(theta, BODY_H/2), BODY_D*np.sin(theta)], axis=1)
    side = np.array([
        [-BODY_R, -BODY_H/2, 0], [BODY_R, -BODY_H/2, 0],
        [-BODY_R, BODY_H/2, 0], [BODY_R, BODY_H/2, 0],
        [0, -BODY_H/2, 0], [0, BODY_H/2, 0], [0, 0, 0],
    ], dtype=float)
    return np.vstack([top, bot, side])


def closest_point_residual(pred: np.ndarray, target: np.ndarray, sigma: float) -> list[float]:
    if target is None or len(target) == 0 or len(pred) == 0:
        return []
    out = []
    for t in target:
        d = pred - t[None, :]
        j = int(np.argmin(np.sum(d*d, axis=1)))
        out.extend(((pred[j] - t) / sigma).tolist())
    return out


def handle_points() -> np.ndarray:
    phi = np.deg2rad(np.linspace(70, 290, 72))
    handle = np.stack([
        HANDLE_CENTER[0] + HANDLE_RX*np.cos(phi),
        HANDLE_CENTER[1] + HANDLE_RY*np.sin(phi),
        np.full_like(phi, HANDLE_CENTER[2]),
    ], axis=1)
    upper = np.array([[-BODY_R, -0.035, 0.0], [-0.052, -0.035, 0.0]], dtype=float)
    lower = np.array([[-BODY_R, 0.027, 0.0], [-0.052, 0.027, 0.0]], dtype=float)
    return np.vstack([handle, upper, lower, HANDLE_CENTER[None, :]])


def all_points() -> np.ndarray:
    return np.vstack([body_points(), handle_points()])


def bbox_residual(uv: np.ndarray, target: list[float], sigma: float) -> list[float]:
    pred = [float(np.min(uv[:, 0])), float(np.min(uv[:, 1])), float(np.max(uv[:, 0])), float(np.max(uv[:, 1]))]
    return [(pred[i] - target[i]) / sigma for i in range(4)]


def outside_bbox_residual(uv: np.ndarray, bbox: list[float], margin: float, sigma: float) -> list[float]:
    if uv is None or len(uv) == 0:
        return []
    x1, y1, x2, y2 = [float(v) for v in bbox]
    vals: list[float] = []
    for u, v in uv:
        vals.append(max(0.0, x1 - margin - float(u)) / sigma)
        vals.append(max(0.0, float(u) - (x2 + margin)) / sigma)
        vals.append(max(0.0, y1 - margin - float(v)) / sigma)
        vals.append(max(0.0, float(v) - (y2 + margin)) / sigma)
    return vals


def mask_dt(mask_path: Path) -> np.ndarray | None:
    if not mask_path.exists():
        return None
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    b = (m > 0).astype(np.uint8)
    return cv2.distanceTransform(1 - b, cv2.DIST_L2, 5)


def dt_at(dt: np.ndarray, uv: np.ndarray) -> float:
    u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
    if u < 0 or v < 0 or v >= dt.shape[0] or u >= dt.shape[1]:
        return 200.0
    return float(dt[v, u])


def row_center(row: dict[str, str]) -> np.ndarray:
    cx = ff(row, 'body_center_x')
    cy = ff(row, 'body_center_y')
    if np.isfinite(cx) and np.isfinite(cy):
        return np.array([cx, cy], dtype=float)
    x1 = ff(row, 'body_bbox_x1', ff(row, 'bbox_x1'))
    y1 = ff(row, 'body_bbox_y1', ff(row, 'bbox_y1'))
    x2 = ff(row, 'body_bbox_x2', ff(row, 'bbox_x2'))
    y2 = ff(row, 'body_bbox_y2', ff(row, 'bbox_y2'))
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=float)


def sample_contact_frames(sample: Path, obs_rows: dict[int, dict[str, str]], step: int = 10) -> list[int]:
    intervals_path = sample / 'results' / 'contact_candidates_object_proxy' / 'contact_intervals.csv'
    if not intervals_path.exists():
        return []
    frames: set[int] = set()
    for row in read_csv(intervals_path):
        start = int(float(row['start_frame']))
        end = int(float(row['end_frame']))
        frames.add(start)
        frames.add(end)
        first = ((start + step - 1) // step) * step
        for fr in range(first, end + 1, step):
            if fr in obs_rows:
                frames.add(fr)
    return sorted(frames)


def shift_point(p: list[float], delta: np.ndarray) -> list[float]:
    q = np.asarray(p, dtype=float) + delta
    return [float(q[0]), float(q[1])]


def blend_point(a: list[float], b: list[float], t: float) -> list[float]:
    p = (1.0 - t) * np.asarray(a, dtype=float) + t * np.asarray(b, dtype=float)
    return [float(p[0]), float(p[1])]


def angle_lerp_deg(a: float, b: float, t: float) -> float:
    da = (float(b) - float(a) + 180.0) % 360.0 - 180.0
    return float(a) + t * da


def ellipse_relative_to_center(ellipse: dict, center: np.ndarray) -> dict:
    out = dict(ellipse)
    out['center'] = (np.asarray(ellipse['center'], dtype=float) - center).tolist()
    axis = ellipse.get('axis_points') or {}
    if axis:
        out['axis_points'] = {k: (np.asarray(v, dtype=float) - center).tolist() for k, v in axis.items()}
    pts = ellipse.get('visible_arc_centerline_points') or []
    if pts:
        out['visible_arc_centerline_points'] = [(np.asarray(v, dtype=float) - center).tolist() for v in pts]
    return out


def ellipse_from_relative(ellipse: dict, center: np.ndarray) -> dict:
    out = dict(ellipse)
    out['center'] = shift_point(ellipse['center'], center)
    axis = ellipse.get('axis_points') or {}
    if axis:
        out['axis_points'] = {k: shift_point(v, center) for k, v in axis.items()}
    pts = ellipse.get('visible_arc_centerline_points') or []
    if pts:
        out['visible_arc_centerline_points'] = [shift_point(v, center) for v in pts]
    return out


def blend_ellipse(a: dict, b: dict, t: float) -> dict:
    out = {
        'center': blend_point(a['center'], b['center'], t),
        'rx': float((1.0 - t) * float(a['rx']) + t * float(b['rx'])),
        'ry': float((1.0 - t) * float(a['ry']) + t * float(b['ry'])),
        'angle_deg': angle_lerp_deg(float(a.get('angle_deg', 0.0)), float(b.get('angle_deg', 0.0)), t),
    }
    axis_a = a.get('axis_points') or {}
    axis_b = b.get('axis_points') or {}
    keys = sorted(set(axis_a) & set(axis_b))
    if keys:
        out['axis_points'] = {k: blend_point(axis_a[k], axis_b[k], t) for k in keys}
    pts_a = a.get('visible_arc_centerline_points') or []
    pts_b = b.get('visible_arc_centerline_points') or []
    if pts_a and len(pts_a) == len(pts_b):
        out['visible_arc_centerline_points'] = [blend_point(pa, pb, t) for pa, pb in zip(pts_a, pts_b)]
    return out


def manual_anchor_frames(manual_ann: dict, obs_rows: dict[int, dict[str, str]]) -> list[tuple[int, dict]]:
    anchors: list[tuple[int, dict]] = []
    for key, item in (manual_ann.get('frames') or {}).items():
        fr = int(key)
        if fr not in obs_rows or 'rim_ellipse' not in item or 'bottom_ellipse' not in item:
            continue
        anchors.append((fr, item))
    return sorted(anchors, key=lambda x: x[0])


def propagated_annotation_from_manual(frame: int, row: dict[str, str], manual_ann: dict, obs_rows: dict[int, dict[str, str]]) -> dict | None:
    anchors = manual_anchor_frames(manual_ann, obs_rows)
    if not anchors:
        return None
    current_center = row_center(row)
    before = [(f, item) for f, item in anchors if f <= frame]
    after = [(f, item) for f, item in anchors if f >= frame]
    left = before[-1] if before else None
    right = after[0] if after else None
    if left and right and left[0] != right[0]:
        lf, li = left
        rf, ri = right
        t = (frame - lf) / float(rf - lf)
        lc = row_center(obs_rows[lf])
        rc = row_center(obs_rows[rf])
        cc = row_center(row)
        left_side = obs_rows[lf].get('hand_contact_side') or obs_rows[lf].get('contact_region_side') or obs_rows[lf].get('handle_side')
        right_side = obs_rows[rf].get('hand_contact_side') or obs_rows[rf].get('contact_region_side') or obs_rows[rf].get('handle_side')
        current_side = row.get('hand_contact_side') or row.get('contact_region_side') or row.get('handle_side')
        center_span = float(np.linalg.norm(rc - lc))
        center_from_left = float(np.linalg.norm(cc - lc))
        same_contact_side = bool(left_side and right_side and current_side and left_side == right_side == current_side)
        # Motion gate: if the object/contact hand barely moves and the contact side is unchanged,
        # do not let a later manual frame rotate the mug structure through the whole interval.
        # Keep the last reliable relative rim/bottom and only translate it with the object center.
        freeze_relative = same_contact_side and center_span < 35.0 and center_from_left < 35.0
        if freeze_relative:
            rim_rel = ellipse_relative_to_center(li['rim_ellipse'], lc)
            bot_rel = ellipse_relative_to_center(li['bottom_ellipse'], lc)
            source = f'motion_gated_hold_manual_{lf}_before_{rf}'
        else:
            rim_rel = blend_ellipse(ellipse_relative_to_center(li['rim_ellipse'], lc), ellipse_relative_to_center(ri['rim_ellipse'], rc), t)
            bot_rel = blend_ellipse(ellipse_relative_to_center(li['bottom_ellipse'], lc), ellipse_relative_to_center(ri['bottom_ellipse'], rc), t)
            source = f'propagated_between_manual_{lf}_{rf}'
        left_hidden = str(li.get('handle_visibility_policy', '')).lower() == 'hidden'
        right_hidden = str(ri.get('handle_visibility_policy', '')).lower() == 'hidden'
        hidden = left_hidden and right_hidden
    else:
        nf, ni = min(anchors, key=lambda x: abs(x[0] - frame))
        nc = row_center(obs_rows[nf])
        rim_rel = ellipse_relative_to_center(ni['rim_ellipse'], nc)
        bot_rel = ellipse_relative_to_center(ni['bottom_ellipse'], nc)
        source = f'propagated_from_nearest_manual_{nf}'
        hidden = str(ni.get('handle_visibility_policy', '')).lower() == 'hidden'
    rim = ellipse_from_relative(rim_rel, current_center)
    bottom = ellipse_from_relative(bot_rel, current_center)
    return {
        'label': source,
        'body_axis': [rim['center'], bottom['center']],
        'rim_ellipse': rim,
        'bottom_ellipse': bottom,
        'handle_visibility_policy': 'hidden' if hidden else 'visible',
        'source': source,
        'propagated_from_manual': True,
    }


def build_propagated_manual_annotations(sample: Path, manual_ann: dict, obs_rows: dict[int, dict[str, str]], frames: list[int]) -> dict:
    frames_out: dict[str, dict] = {}
    exact = manual_ann.get('frames') or {}
    for fr in sorted(set(frames)):
        if str(fr) in exact:
            continue
        item = propagated_annotation_from_manual(fr, obs_rows[fr], manual_ann, obs_rows)
        if item is not None:
            frames_out[str(fr)] = item
    out = {
        'source': 'propagated_from_manual_rim_bottom_keyframes',
        'coordinate_system': 'full_frame_pixel_xy',
        'notes': 'Generated review/optimization constraints. Exact manual annotations remain in mug_rim_bottom_manual.json.',
        'frames': frames_out,
    }
    out_path = sample / 'annotations' / 'mug_rim_bottom_propagated_from_manual.json'
    out_path.write_text(json.dumps(out, indent=2))
    return out


def edge_observation_from_manual(frame: int, ann: dict, shape: tuple[int, int]) -> dict | None:
    item = (ann.get('frames') or {}).get(str(frame))
    if not item:
        return None
    h, w = shape
    top_mask = np.zeros((h, w), dtype=np.uint8)
    bottom_mask = np.zeros((h, w), dtype=np.uint8)
    def draw_ellipse(mask: np.ndarray, e: dict) -> np.ndarray:
        center = tuple(int(round(v)) for v in e['center'])
        axes = (max(1, int(round(e['rx']))), max(1, int(round(e['ry']))))
        angle = float(e.get('angle_deg', 0.0))
        cv2.ellipse(mask, center, axes, angle, 0, 360, 1, 1, cv2.LINE_AA)
        return np.array(center, dtype=float)
    top_center = draw_ellipse(top_mask, item['rim_ellipse'])
    bottom_center = draw_ellipse(bottom_mask, item['bottom_ellipse'])
    axis = bottom_center - top_center
    roll_prior = math.atan2(float(axis[0]), float(axis[1])) if np.linalg.norm(axis) > 1e-6 else 0.0
    rim_pts = item.get('rim_ellipse', {}).get('visible_arc_centerline_points') or []
    rim_axis = item.get('rim_ellipse', {}).get('axis_points') or {}
    bot_axis = item.get('bottom_ellipse', {}).get('axis_points') or {}
    manual_points = {
        'rim_arc': np.asarray(rim_pts, dtype=float) if rim_pts else None,
        'rim_major': np.asarray([rim_axis['major_neg'], rim_axis['major_pos']], dtype=float) if 'major_neg' in rim_axis and 'major_pos' in rim_axis else None,
        'rim_minor': np.asarray([rim_axis['minor_neg'], rim_axis['minor_pos']], dtype=float) if 'minor_neg' in rim_axis and 'minor_pos' in rim_axis else None,
        'bottom_major': np.asarray([bot_axis['major_neg'], bot_axis['major_pos']], dtype=float) if 'major_neg' in bot_axis and 'major_pos' in bot_axis else None,
        'bottom_minor': np.asarray([bot_axis['minor_neg'], bot_axis['minor_pos']], dtype=float) if 'minor_neg' in bot_axis and 'minor_pos' in bot_axis else None,
    }
    return {
        'bbox': [
            min(item['rim_ellipse']['center'][0]-item['rim_ellipse']['rx'], item['bottom_ellipse']['center'][0]-item['bottom_ellipse']['rx']),
            min(item['rim_ellipse']['center'][1]-item['rim_ellipse']['ry'], item['bottom_ellipse']['center'][1]-item['bottom_ellipse']['ry']),
            max(item['rim_ellipse']['center'][0]+item['rim_ellipse']['rx'], item['bottom_ellipse']['center'][0]+item['bottom_ellipse']['rx']),
            max(item['rim_ellipse']['center'][1]+item['rim_ellipse']['ry'], item['bottom_ellipse']['center'][1]+item['bottom_ellipse']['ry']),
        ],
        'top_center': top_center,
        'bottom_center': bottom_center,
        'top_dt': cv2.distanceTransform(1 - top_mask, cv2.DIST_L2, 5),
        'bottom_dt': cv2.distanceTransform(1 - bottom_mask, cv2.DIST_L2, 5),
        'roll_prior': float(roll_prior),
        'manual': item,
        'manual_points': manual_points,
        'source': item.get('source', 'manual_rim_bottom_ellipse'),
    }


def edge_observation_from_mask(mask_path: Path) -> dict | None:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    b = (m > 0).astype(np.uint8)
    if int(b.sum()) < 20:
        return None
    x, y, w, h = cv2.boundingRect(b)
    top_pts = []
    bottom_pts = []
    for xx in range(x, x + w):
        ys = np.flatnonzero(b[:, xx] > 0)
        if len(ys) == 0:
            continue
        top_pts.append([xx, int(ys.min())])
        bottom_pts.append([xx, int(ys.max())])
    if len(top_pts) < 4 or len(bottom_pts) < 4:
        return None
    top_pts = np.asarray(top_pts, dtype=np.float32)
    bottom_pts = np.asarray(bottom_pts, dtype=np.float32)
    top_mask = np.zeros_like(b)
    bottom_mask = np.zeros_like(b)
    for u, v in np.round(top_pts).astype(int):
        cv2.circle(top_mask, (int(u), int(v)), 1, 1, -1)
    for u, v in np.round(bottom_pts).astype(int):
        cv2.circle(bottom_mask, (int(u), int(v)), 1, 1, -1)
    top_dt = cv2.distanceTransform(1 - top_mask, cv2.DIST_L2, 5)
    bottom_dt = cv2.distanceTransform(1 - bottom_mask, cv2.DIST_L2, 5)
    top_center = np.median(top_pts, axis=0)
    bottom_center = np.median(bottom_pts, axis=0)
    axis = bottom_center - top_center
    roll_prior = math.atan2(float(axis[0]), float(axis[1])) if np.linalg.norm(axis) > 1e-6 else 0.0
    return {
        'bbox': [float(x), float(y), float(x + w), float(y + h)],
        'top_center': top_center.astype(float),
        'bottom_center': bottom_center.astype(float),
        'top_dt': top_dt,
        'bottom_dt': bottom_dt,
        'roll_prior': float(roll_prior),
    }


def write_edge_debug(sample: Path, edge_obs: dict[int, dict]) -> None:
    out_dir = sample / 'annotations' / 'rim_bottom_observations'
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    for fr, obs in edge_obs.items():
        img = cv2.imread(str(sample / 'frames' / f'{fr:05d}.png'))
        if img is None:
            continue
        tc = tuple(np.round(obs['top_center']).astype(int))
        bc = tuple(np.round(obs['bottom_center']).astype(int))
        if 'manual' in obs:
            for key, color in [('rim_ellipse', (255, 0, 255)), ('bottom_ellipse', (0, 255, 255))]:
                e = obs['manual'][key]
                cv2.ellipse(img, tuple(int(round(v)) for v in e['center']), (int(round(e['rx'])), int(round(e['ry']))), float(e.get('angle_deg', 0.0)), 0, 360, color, 2, cv2.LINE_AA)
                for pt in e.get('visible_arc_centerline_points', []) or []:
                    cv2.circle(img, tuple(int(round(v)) for v in pt), 3, color, -1, cv2.LINE_AA)
                for pt in (e.get('axis_points') or {}).values():
                    cv2.circle(img, tuple(int(round(v)) for v in pt), 3, color, -1, cv2.LINE_AA)
        cv2.circle(img, tc, 5, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(img, bc, 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.line(img, tc, bc, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f'{fr:05d}_rim_bottom_debug.png'), img)
        meta[str(fr)] = {
            'bbox_xyxy': obs['bbox'],
            'top_center_xy': obs['top_center'].tolist(),
            'bottom_center_xy': obs['bottom_center'].tolist(),
            'roll_prior_rad': obs['roll_prior'],
            'source': obs.get('source', 'segmentation_mask_top_bottom_silhouette_edges'),
            'manual': obs.get('manual'),
            'manual_points_available': {k: (None if v is None else len(v)) for k, v in (obs.get('manual_points') or {}).items()},
        }
    (out_dir / 'rim_bottom_observations.json').write_text(json.dumps(meta, indent=2))


def pitch_prior_for_label(label: str) -> tuple[float, float, str]:
    label = (label or '').lower()
    if 'drinking' in label:
        return math.radians(-58.0), 0.75, 'drinking_bottom_visible_prior'
    if 'side_motion' in label:
        return math.radians(-38.0), 0.55, 'side_motion_tilt_prior'
    if 'annotated_contact' in label:
        return math.radians(-12.0), 0.30, 'contact_mask_weak_tilt_prior'
    if 'putdown' in label or 'table_stable' in label or 'final_static' in label:
        return math.radians(2.0), 0.55, 'upright_table_prior'
    return 0.0, 0.25, 'weak_upright_prior'


def initial_params(frame: int, obs: dict[str, str], proxy_obs: dict[str, str], old_pose: dict[str, str], K: np.ndarray, pitch_prior: float = 0.0, roll_prior: float = 0.0) -> np.ndarray:
    cx = ff(obs, 'body_center_x', ff(obs, 'center_x', ff(old_pose, 'cx')))
    cy = ff(obs, 'body_center_y', ff(obs, 'center_y', ff(old_pose, 'cy')))
    z = ff(proxy_obs, 'object_depth_smooth', ff(proxy_obs, 'da3_depth_smooth', 3.6))
    if not np.isfinite(z) or z < 0.2:
        z = 3.6
    xyz = backproject(cx, cy, z, K)
    bh = ff(obs, 'body_bbox_h_px', ff(obs, 'bbox_h_px', ff(old_pose, 'scale', 42.0)))
    scale = float(np.clip((bh * z) / max(K[1, 1] * BODY_H, 1.0), 0.45, 2.5))
    yaw = ff(old_pose, 'yaw_rad', 0.0)
    return np.array([xyz[0], xyz[1], z, yaw, pitch_prior, roll_prior, scale], dtype=float)


def fit_one(frame_item: dict, obs: dict[str, str], proxy_obs: dict[str, str], old_pose: dict[str, str], K: np.ndarray, dt: np.ndarray | None, edge_obs: dict | None) -> dict:
    frame = int(frame_item['frame'])
    pitch_prior, pitch_weight, pitch_source = pitch_prior_for_label(frame_item.get('label', ''))
    roll_prior = float(edge_obs.get('roll_prior', 0.0)) if edge_obs is not None else 0.0
    x0 = initial_params(frame, obs, proxy_obs, old_pose, K, pitch_prior=pitch_prior, roll_prior=roll_prior)
    body_bbox = [float(x) for x in frame_item['body_bbox_xyxy']]
    obj_bbox = [float(x) for x in frame_item['object_bbox_xyxy']]
    body_ctr = np.array([0.5 * (body_bbox[0] + body_bbox[2]), 0.5 * (body_bbox[1] + body_bbox[3])], dtype=float)
    handle_visible = str(frame_item.get('handle_visible', '0')) == '1'
    manual_policy = ''
    if edge_obs is not None and edge_obs.get('manual'):
        manual_policy = str(edge_obs['manual'].get('handle_visibility_policy', '')).lower()
    handle_must_be_hidden = (not handle_visible) or manual_policy == 'hidden'
    handle_conf = float(frame_item.get('handle_conf') or 0.0)
    hc = frame_item.get('handle_center_xy') or ['', '']
    handle_center = None
    try:
        if hc[0] != '' and hc[1] != '':
            handle_center = np.array([float(hc[0]), float(hc[1])], dtype=float)
    except Exception:
        handle_center = None
    yaw_prior = float(x0[3])
    z_prior = float(x0[2])
    pitch_prior = float(x0[4])
    roll_prior = float(x0[5])
    scale_prior = float(x0[6])
    bpts = body_points()
    hpts = handle_points()
    apts = all_points()

    def residual(p: np.ndarray) -> np.ndarray:
        res: list[float] = []
        buv = project_pts(p, bpts, K)
        auv = project_pts(p, apts, K)
        cuv = project_pts(p, np.array([[0.0, 0.0, 0.0]], dtype=float), K)[0]
        top_ring = project_pts(p, np.stack([BODY_R*np.cos(np.linspace(0, 2*np.pi, 36)), np.full(36, -BODY_H/2), BODY_D*np.sin(np.linspace(0, 2*np.pi, 36))], axis=1), K)
        bottom_ring = project_pts(p, np.stack([BODY_R*np.cos(np.linspace(0, 2*np.pi, 36)), np.full(36, BODY_H/2), BODY_D*np.sin(np.linspace(0, 2*np.pi, 36))], axis=1), K)
        edge_source = edge_obs.get('source', '') if edge_obs is not None else ''
        edge_is_manual_like = edge_source.startswith('manual') or edge_source.startswith('user_manual') or edge_source.startswith('propagated')
        center_sigma_xy = 8.0 if edge_is_manual_like else 4.0
        body_bbox_sigma = 9.0 if edge_is_manual_like else 4.5
        res.extend(((cuv - body_ctr) / np.array([center_sigma_xy, center_sigma_xy])).tolist())
        res.extend(bbox_residual(buv, body_bbox, sigma=body_bbox_sigma))
        # Object bbox includes handle. Use it strongly only when handle is visible; weakly otherwise.
        obj_sigma = 12.0 if (handle_visible and not edge_is_manual_like) else 24.0
        obj_weight = 0.7 if (handle_visible and not edge_is_manual_like) else 0.15
        res.extend([obj_weight * r for r in bbox_residual(auv, obj_bbox, sigma=obj_sigma)])
        huv = project_pts(p, hpts, K)
        hcenter_uv = project_pts(p, HANDLE_CENTER[None, :], K)[0]
        if handle_center is not None and handle_visible and not handle_must_be_hidden:
            w = float(np.clip(0.4 + 2.0 * handle_conf, 0.4, 1.3))
            res.extend((w * (hcenter_uv - handle_center) / 5.0).tolist())
        if handle_must_be_hidden:
            # If the handle is annotated/observed as hidden, its projected proxy should stay
            # behind the cup body silhouette instead of rotating into a visible side region.
            res.extend([1.45 * r for r in outside_bbox_residual(huv[::6], body_bbox, margin=3.0, sigma=5.0)])
        if dt is not None and frame == 131:
            res.append(dt_at(dt, hcenter_uv) / 6.0)
        if edge_obs is not None:
            # Manual rim/bottom observations constrain real visible cup geometry; handle is ignored when occluded.
            top_dt = edge_obs['top_dt']
            bottom_dt = edge_obs['bottom_dt']
            is_manual = edge_obs.get('source', '').startswith('manual') or edge_obs.get('source', '').startswith('user_manual') or edge_obs.get('source', '').startswith('propagated') or edge_obs.get('source', '').startswith('motion_gated')
            edge_sigma = 12.0 if is_manual else 10.0
            center_sigma = 1.8 if is_manual else 5.0
            res.extend([min(dt_at(top_dt, q), 80.0) / edge_sigma for q in top_ring[::6]])
            res.extend([min(dt_at(bottom_dt, q), 80.0) / edge_sigma for q in bottom_ring[::6]])
            top_center_uv = project_pts(p, np.array([[0.0, -BODY_H/2, 0.0]], dtype=float), K)[0]
            bottom_center_uv = project_pts(p, np.array([[0.0, BODY_H/2, 0.0]], dtype=float), K)[0]
            res.extend(((top_center_uv - edge_obs['top_center']) / center_sigma).tolist())
            res.extend(((bottom_center_uv - edge_obs['bottom_center']) / center_sigma).tolist())
            mp = edge_obs.get('manual_points') or {}
            point_sigma = 3.2 if is_manual else 7.0
            minor_sigma = 4.5 if is_manual else 9.0
            res.extend(closest_point_residual(top_ring, mp.get('rim_arc'), sigma=point_sigma))
            res.extend(closest_point_residual(top_ring, mp.get('rim_major'), sigma=point_sigma))
            res.extend(closest_point_residual(top_ring, mp.get('rim_minor'), sigma=minor_sigma))
            res.extend(closest_point_residual(bottom_ring, mp.get('bottom_major'), sigma=point_sigma))
            res.extend(closest_point_residual(bottom_ring, mp.get('bottom_minor'), sigma=minor_sigma))
            res.append(0.35 * wrap(float(p[5] - roll_prior)) / 0.45)
        res.append(0.20 * wrap(float(p[3] - yaw_prior)) / 0.45)
        res.append(pitch_weight * (float(p[4]) - pitch_prior) / 0.35)
        res.append(0.10 * (float(p[2]) - z_prior) / 0.25)
        res.append(0.20 * (float(p[6]) - scale_prior) / 0.35)
        return np.asarray(res, dtype=float)

    lower = np.array([x0[0] - 0.8, x0[1] - 0.8, max(0.8, z_prior - 1.2), -math.pi, math.radians(-80.0), math.radians(-80.0), 0.25], dtype=float)
    upper = np.array([x0[0] + 0.8, x0[1] + 0.8, z_prior + 1.2, math.pi, math.radians(70.0), math.radians(80.0), 3.0], dtype=float)
    result = least_squares(residual, x0, bounds=(lower, upper), loss='soft_l1', f_scale=1.0, max_nfev=250)
    p = result.x
    return {
        'frame': frame,
        'label': frame_item.get('label', ''),
        'time': (frame - 1) / 24.0,
        'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2]),
        'yaw': float(wrap(p[3])), 'yaw_deg': float(np.rad2deg(wrap(p[3]))),
        'pitch': float(p[4]), 'pitch_deg': float(np.rad2deg(p[4])),
        'roll': float(p[5]), 'roll_deg': float(np.rad2deg(p[5])),
        'scale': float(p[6]),
        'cost': float(result.cost), 'success': bool(result.success), 'nfev': int(result.nfev),
        'source': 'articraft_mug_keyframe_reprojection_fit',
        'used_body_bbox': body_bbox,
        'used_object_bbox': obj_bbox,
        'used_handle_center': handle_center.tolist() if handle_center is not None else None,
        'handle_visible': handle_visible,
        'handle_must_be_hidden': handle_must_be_hidden,
        'handle_visibility_policy': manual_policy,
        'pitch_prior_source': pitch_source,
    }


def interp_scalar(fr: int, items: list[dict], key: str) -> float:
    xs = np.array([int(i['frame']) for i in items], dtype=float)
    ys = np.array([float(i[key]) for i in items], dtype=float)
    return float(np.interp(fr, xs, ys))


def interp_angle(fr: int, items: list[dict]) -> float:
    xs = np.array([int(i['frame']) for i in items], dtype=float)
    ys = np.unwrap(np.array([float(i['yaw']) for i in items], dtype=float))
    return float(wrap(np.interp(fr, xs, ys)))


def make_params_from_frame(fr: int, fit_items: list[dict], obs: dict[str, str], proxy_obs: dict[str, str], K: np.ndarray) -> np.ndarray:
    exact = next((i for i in fit_items if int(i['frame']) == fr), None)
    if exact is not None:
        return np.array([exact['x'], exact['y'], exact['z'], exact['yaw'], exact.get('pitch', 0.0), exact.get('roll', 0.0), exact['scale']], dtype=float)
    # Between fitted keyframes, translation follows the observed mug body center so the overlay stays on the cup; yaw/scale comes from fitted keyframes.
    z = ff(proxy_obs, 'object_depth_smooth', ff(proxy_obs, 'da3_depth_smooth', interp_scalar(fr, fit_items, 'z')))
    if not np.isfinite(z) or z < 0.2:
        z = interp_scalar(fr, fit_items, 'z')
    u = ff(obs, 'body_center_x', ff(obs, 'center_x'))
    v = ff(obs, 'body_center_y', ff(obs, 'center_y'))
    if not np.isfinite(u) or not np.isfinite(v):
        nearest = min(fit_items, key=lambda x: abs(int(x['frame']) - fr))
        return np.array([nearest['x'], nearest['y'], nearest['z'], nearest['yaw'], nearest.get('pitch', 0.0), nearest.get('roll', 0.0), nearest['scale']], dtype=float)
    xyz = backproject(u, v, z, K)
    scale = interp_scalar(fr, fit_items, 'scale')
    yaw = interp_angle(fr, fit_items)
    pitch = interp_scalar(fr, fit_items, 'pitch')
    roll = interp_scalar(fr, fit_items, 'roll')
    return np.array([xyz[0], xyz[1], xyz[2], yaw, pitch, roll, scale], dtype=float)


def draw_poly(img: np.ndarray, uv: np.ndarray, color: tuple[int, int, int], thick: int, closed: bool = False) -> None:
    if np.all(np.isfinite(uv)):
        cv2.polylines(img, [np.round(uv).astype(np.int32)], closed, color, thick, cv2.LINE_AA)


def draw_mug(img: np.ndarray, params: np.ndarray, K: np.ndarray, active: bool = False, show_handle: bool = True) -> None:
    overlay = img.copy()
    theta = np.linspace(0, 2 * np.pi, 96, endpoint=True)
    top = np.stack([BODY_R*np.cos(theta), np.full_like(theta, -BODY_H/2), BODY_D*np.sin(theta)], axis=1)
    bot = np.stack([BODY_R*np.cos(theta), np.full_like(theta, BODY_H/2), BODY_D*np.sin(theta)], axis=1)
    phi = np.deg2rad(np.linspace(70, 290, 96))
    handle = np.stack([HANDLE_CENTER[0] + HANDLE_RX*np.cos(phi), HANDLE_CENTER[1] + HANDLE_RY*np.sin(phi), np.full_like(phi, HANDLE_CENTER[2])], axis=1)
    upper = np.array([[-BODY_R, -0.035, 0.0], [-0.052, -0.035, 0.0]], dtype=float)
    lower = np.array([[-BODY_R, 0.027, 0.0], [-0.052, 0.027, 0.0]], dtype=float)
    side_l = np.array([[-BODY_R, -BODY_H/2, 0.0], [-BODY_R, BODY_H/2, 0.0]], dtype=float)
    side_r = np.array([[BODY_R, -BODY_H/2, 0.0], [BODY_R, BODY_H/2, 0.0]], dtype=float)
    for pts in [top, bot, side_l, side_r]:
        draw_poly(overlay, project_pts(params, pts, K), (245,245,245), 2, closed=False)
    if show_handle:
        for pts in [handle, upper, lower]:
            draw_poly(overlay, project_pts(params, pts, K), (0,220,255), 3, closed=False)
        c = project_pts(params, HANDLE_CENTER[None, :], K)[0]
        if np.all(np.isfinite(c)):
            cc = tuple(np.round(c).astype(int))
            cv2.circle(overlay, cc, 5, (0,220,255), -1, cv2.LINE_AA)
            if active:
                cv2.circle(overlay, cc, 14, (40,40,230), 3, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.70, img, 0.30, 0, img)


def finalize(tmp: Path, out: Path) -> None:
    ffmpeg = shutil.which('ffmpeg') or '/usr/bin/ffmpeg'
    try:
        subprocess.run([ffmpeg, '-y', '-i', str(tmp), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out)], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        subprocess.run([ffmpeg, '-y', '-i', str(tmp), '-c:v', 'libopenh264', '-pix_fmt', 'yuv420p', str(out)], check=True, capture_output=True, text=True)
    tmp.unlink(missing_ok=True)


def show_handle_for_frame(fr: int, fit_items: list[dict]) -> bool:
    by_frame = {int(i['frame']): i for i in fit_items}
    if fr in by_frame:
        return not bool(by_frame[fr].get('handle_must_be_hidden'))
    frames = sorted(by_frame)
    prev = [f for f in frames if f <= fr]
    nxt = [f for f in frames if f >= fr]
    if not prev or not nxt:
        near = min(frames, key=lambda f: abs(f - fr))
        return not bool(by_frame[near].get('handle_must_be_hidden'))
    a, b = prev[-1], nxt[0]
    if a == b:
        return not bool(by_frame[a].get('handle_must_be_hidden'))
    # Hidden wins inside a propagated hidden span. This prevents transient raw handle detections
    # from making the handle pop out between two hidden anchor frames.
    return (not bool(by_frame[a].get('handle_must_be_hidden'))) and (not bool(by_frame[b].get('handle_must_be_hidden')))


def render(sample: Path, fit_items: list[dict], full_rows: list[dict], proxy_rows: dict[int, dict], K_all: np.ndarray, fps: float) -> Path:
    frames_dir = sample / 'frames'
    out_dir = sample / 'results' / 'renders' / 'M3_keyframe_optimized_pose'
    out_dir.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(frames_dir / '00001.png'))
    h, w = first.shape[:2]
    tmp = out_dir / 'overlay.tmp.mp4'
    out = out_dir / 'overlay.mp4'
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    keyset = {int(i['frame']) for i in fit_items}
    params_rows = []
    for row in full_rows:
        fr = int(row['frame'])
        img = cv2.imread(str(frames_dir / f'{fr:05d}.png'))
        if img is None:
            continue
        params = make_params_from_frame(fr, fit_items, row, proxy_rows.get(fr, {}), K_all[fr-1])
        show_handle = show_handle_for_frame(fr, fit_items)
        draw_mug(img, params, K_all[fr-1], active=fr in keyset, show_handle=show_handle)
        cv2.putText(img, f'M3_keyframe_optimized_pose frame {fr:03d}', (24, 36), cv2.FONT_HERSHEY_SIMPLEX, .7, (20,20,20), 3, cv2.LINE_AA)
        cv2.putText(img, f'M3_keyframe_optimized_pose frame {fr:03d}', (24, 36), cv2.FONT_HERSHEY_SIMPLEX, .7, (245,245,245), 1, cv2.LINE_AA)
        writer.write(img)
        params_rows.append({'frame': fr, 'time': ff(row, 'time', (fr-1)/fps), 'x': params[0], 'y': params[1], 'z': params[2], 'yaw': params[3], 'yaw_deg': np.rad2deg(params[3]), 'pitch': params[4], 'pitch_deg': np.rad2deg(params[4]), 'roll': params[5], 'roll_deg': np.rad2deg(params[5]), 'scale': params[6]})
    writer.release()
    finalize(tmp, out)
    csv_path = sample / 'proxy' / 'mug_optimized_pose_sequence.csv'
    with csv_path.open('w', newline='') as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(params_rows[0].keys()))
        writer_csv.writeheader(); writer_csv.writerows(params_rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-dir', type=Path, required=True)
    ap.add_argument('--fps', type=float, default=24.0)
    args = ap.parse_args()
    sample = args.sample_dir
    manifest = json.loads((sample / 'keyframes_pose_fit' / 'keyframes_pose_fit_manifest.json').read_text())
    manual_path = sample / 'annotations' / 'mug_rim_bottom_manual.json'
    manual_ann = json.loads(manual_path.read_text()) if manual_path.exists() else {'frames': {}}
    obs_rows = {int(r['frame']): r for r in read_csv(sample / 'results' / 'object_observations' / 'object_observations.csv')}
    proxy_rows = {int(r['frame']): r for r in read_csv(sample / 'results' / 'object_proxy_observations' / 'object_proxy_observations.csv')}
    old_rows = {int(r['frame']): r for r in read_csv(sample / 'results' / 'mug_proxy_3d_pose' / 'mug_proxy_3d_pose.csv')}
    K_all = load_K(sample)
    dt = mask_dt(sample / 'annotations' / '001_contact_region_mask.png')
    contact_frames = sample_contact_frames(sample, obs_rows, step=10)
    # Mug-specific dense fitting span: the drinking/contact interval is visually sensitive,
    # and sparse 10-frame keyframes leave interpolation drift between annotated poses.
    # Use every frame in 60-100 after motion-gated rim/bottom propagation.
    dense_motion_gated_frames = [fr for fr in range(60, 101) if fr in obs_rows]
    contact_frames = sorted(set(contact_frames) | set(dense_motion_gated_frames))
    propagated_ann = build_propagated_manual_annotations(sample, manual_ann, obs_rows, contact_frames)
    exact_manual_frames = set(int(k) for k in (manual_ann.get('frames') or {}))
    propagated_frames = set(int(k) for k in (propagated_ann.get('frames') or {}))

    # Add exact manual frames and propagated contact-interval frames to the fitting keyframes.
    existing = {int(item['frame']) for item in manifest}
    supplemental_frames = sorted((exact_manual_frames | propagated_frames) - existing)
    for fr in supplemental_frames:
        r = obs_rows[fr]
        ann_item = (manual_ann.get('frames') or {}).get(str(fr)) or (propagated_ann.get('frames') or {}).get(str(fr), {})
        manifest.append({
            'frame': fr,
            'label': ann_item.get('label', f'propagated_{fr}'),
            'note': 'manual/propagated rim-bottom supplemental keyframe',
            'body_bbox_xyxy': [ff(r, 'body_bbox_x1'), ff(r, 'body_bbox_y1'), ff(r, 'body_bbox_x2'), ff(r, 'body_bbox_y2')],
            'object_bbox_xyxy': [ff(r, 'bbox_x1'), ff(r, 'bbox_y1'), ff(r, 'bbox_x2'), ff(r, 'bbox_y2')],
            'handle_visible': '1' if (ann_item.get('handle_visibility_policy') == 'visible') else '0',
            'handle_side': r.get('handle_side') or r.get('contact_region_side') or 'left',
            'handle_conf': r.get('handle_conf') or '0',
            'handle_center_xy': [r.get('handle_center_x') or '', r.get('handle_center_y') or ''],
            'contact_region_side': r.get('contact_region_side') or r.get('handle_side') or 'left',
        })
    manifest = sorted(manifest, key=lambda x: int(x['frame']))

    frame0 = cv2.imread(str(sample / 'frames' / '00001.png'))
    shape = frame0.shape[:2]
    edge_obs = {}
    for item in manifest:
        fr = int(item['frame'])
        eo = edge_observation_from_manual(fr, manual_ann, shape)
        if eo is None:
            eo = edge_observation_from_manual(fr, propagated_ann, shape)
        if eo is None:
            eo = edge_observation_from_mask(sample / 'results' / 'segmentation' / 'masks' / f'{fr:05d}_mask.png')
        if eo is not None:
            edge_obs[fr] = eo
    write_edge_debug(sample, edge_obs)
    fit_items = []
    for item in manifest:
        fr = int(item['frame'])
        fit_items.append(fit_one(item, obs_rows[fr], proxy_rows.get(fr, {}), old_rows.get(fr, {}), K_all[fr-1], dt, edge_obs.get(fr)))
    proxy_dir = sample / 'proxy'
    proxy_dir.mkdir(exist_ok=True)
    out_json = proxy_dir / 'keyframe_pose_fit_optimized.json'
    out_json.write_text(json.dumps({'model': 'articraft_mug_keyframe_reprojection_fit_with_rim_bottom_edges', 'keyframes': fit_items}, indent=2))
    out_mp4 = render(sample, fit_items, [obs_rows[k] for k in sorted(obs_rows)], proxy_rows, K_all, args.fps)
    summary = {'keyframe_pose_fit_optimized': str(out_json), 'render': str(out_mp4), 'pose_sequence': str(proxy_dir / 'mug_optimized_pose_sequence.csv')}
    (proxy_dir / 'mug_articraft_pose_fit_outputs.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
