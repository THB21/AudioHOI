#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.shared.generic_contact_pipeline.components.observation.object_proxy_observation_utils import (
    build_body_joints,
    build_contact_part_centers,
    get_video_hw,
    load_depth,
    map_to_depth_uv,
    parse_float,
    project_points,
    read_da3_priors,
    read_human_result,
    read_index,
    read_object_mesh_tracks,
    read_rows,
    sample_depth_at_uv,
    select_active_body_proxy_from_mesh,
    select_contact_proxy_from_human_point,
    select_ref_proxy_from_mesh,
    visible_mesh_points,
    write_csv,
)


def rolling_stat(values: np.ndarray, radius: int, mode: str) -> np.ndarray:
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
        else:
            raise ValueError(mode)
    return out


def rolling_abs_deviation(values: np.ndarray, window: int = 7) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.full(len(values), np.nan, dtype=np.float64)
    half = window // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        win = values[lo:hi]
        win = win[np.isfinite(win)]
        if len(win) == 0 or not np.isfinite(values[i]):
            continue
        med = float(np.median(win))
        out[i] = abs(float(values[i]) - med)
    return out


def ema_filter(values: np.ndarray, alpha: float = 0.75) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    prev = math.nan
    for i, v in enumerate(values):
        if np.isfinite(v):
            prev = float(v) if not np.isfinite(prev) else alpha * prev + (1.0 - alpha) * float(v)
        out[i] = prev
    return out


def low_value_score(x: np.ndarray, p_good: float, p_bad: float) -> np.ndarray:
    s = (p_bad - x) / max(p_bad - p_good, 1e-6)
    return np.clip(s, 0.0, 1.0)


def high_value_score(x: np.ndarray, p_bad: float, p_good: float) -> np.ndarray:
    s = (x - p_bad) / max(p_good - p_bad, 1e-6)
    return np.clip(s, 0.0, 1.0)


def smooth_1d_signal(x: np.ndarray, median_radius: int = 2, ema_alpha: float = 0.75) -> np.ndarray:
    x_med = rolling_stat(np.asarray(x, dtype=np.float64), radius=median_radius, mode='median')
    return ema_filter(x_med, alpha=ema_alpha)


def robust_smooth_signal(x: np.ndarray, window: int = 5, polyorder: int = 2) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x.copy()
    finite = np.isfinite(x)
    if not np.any(finite):
        return x.copy()
    filled = x.copy()
    idx = np.arange(len(x))
    filled[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    med = rolling_stat(filled, radius=1, mode='median')
    if len(med) < 5:
        return med
    win = min(window, len(med) if len(med) % 2 == 1 else len(med) - 1)
    if win < 3:
        return med
    poly = min(polyorder, win - 1)
    return savgol_filter(med, window_length=win, polyorder=poly, mode='interp')


def parse_scalar(value: str | float | int | None, default: float = math.nan) -> float:
    if value in {'', None}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def select_stable_ref_proxy(row: dict[str, str]) -> tuple[float, float, float, str]:
    # Stable trajectory reference: use the object observation center, not a
    # boundary/contact/support mesh point whose physical identity can change.
    candidates = [
        ('center', 'center_x', 'center_y', 'observation_conf'),
        ('mask_center', 'mask_center_x', 'mask_center_y', 'mask_conf'),
        ('track_center', 'track_center_x', 'track_center_y', 'track_geometry_conf'),
    ]
    for name, ux, vy, ck in candidates:
        u = parse_scalar(row.get(ux, ''))
        v = parse_scalar(row.get(vy, ''))
        if np.isfinite(u) and np.isfinite(v):
            conf = parse_scalar(row.get(ck, ''), 1.0)
            if not np.isfinite(conf):
                conf = 1.0
            return float(u), float(v), float(np.clip(conf, 0.0, 1.0)), name
    return math.nan, math.nan, 0.0, 'missing_stable_ref'


def build_audio_support(frames: np.ndarray, audio_rows: list[dict[str, str]], radius: int = 2) -> np.ndarray:
    support = np.zeros(len(frames), dtype=np.float64)
    frame_to_idx = {int(fr): i for i, fr in enumerate(frames.tolist())}
    for row in audio_rows:
        if not row.get('audio_frame'):
            continue
        center = int(float(row['audio_frame']))
        score = float(row.get('audio_score', 0.0) or 0.0)
        for fr in range(center - radius, center + radius + 1):
            idx = frame_to_idx.get(fr)
            if idx is None:
                continue
            dist = abs(fr - center)
            weight = max(0.0, 1.0 - 0.25 * dist)
            support[idx] = max(support[idx], score * weight)
    return support


def select_support_proxy_bottom_percentile(mesh: dict[str, np.ndarray], q: float = 95.0) -> tuple[float, float, float, str]:
    pts, _ = visible_mesh_points(mesh, boundary_preferred=True)
    if len(pts) == 0:
        return math.nan, math.nan, 0.0, 'missing_mesh'
    support_v = float(np.percentile(pts[:, 1], q))
    support_u = float(np.median(pts[:, 0][pts[:, 1] >= support_v - 1e-6]))
    conf = min(1.0, len(pts) / 8.0)
    return support_u, support_v, conf, 'bottom_percentile_95'


def load_object_proxy(sample_dir: Path) -> dict[str, object]:
    proxy_path = sample_dir / 'proxy' / 'mug_proxy.json'
    if not proxy_path.exists():
        return {}
    try:
        return json.loads(proxy_path.read_text())
    except Exception:
        return {}


def _finite_bbox(row: dict[str, str], keys: tuple[str, str, str, str]) -> tuple[float, float, float, float] | None:
    vals = [parse_scalar(row.get(k, '')) for k in keys]
    if all(np.isfinite(v) for v in vals) and vals[2] > vals[0] and vals[3] > vals[1]:
        return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
    return None


def build_articraft_contact_region_track(
    sample_dir: Path,
    object_rows: list[dict[str, str]],
    object_proxy: dict[str, object],
    mesh_tracks: dict[int, dict[str, np.ndarray]],
) -> dict[int, tuple[float, float, float, str, str]]:
    contact_region = object_proxy.get('contact_region') if isinstance(object_proxy, dict) else None
    if not isinstance(contact_region, dict) or str(contact_region.get('type', '')).lower() not in {'handle', 'handle_or_side', 'handle_or_body_side', 'object_surface_contact_region', 'surface_contact_region'}:
        return {}

    ann_path = sample_dir / 'annotations' / '001_contact_region_mask.json'
    if not ann_path.exists():
        return {}
    try:
        ann = json.loads(ann_path.read_text())
        ann_frame = int(ann.get('frame'))
        bbox = ann.get('bbox_xyxy') or []
        if len(bbox) != 4:
            return {}
        x1, y1, x2, y2 = [float(v) for v in bbox]
        key_pt = np.asarray([0.5 * (x1 + x2), 0.5 * (y1 + y2), 1.0], dtype=np.float64)
    except Exception:
        return {}

    key_mesh = mesh_tracks.get(ann_frame)
    if key_mesh is None:
        return {}
    key_ids = np.asarray(key_mesh.get('point_ids', []), dtype=object)
    key_xy = np.asarray(key_mesh.get('xy', []), dtype=np.float64)
    key_visible = np.asarray(key_mesh.get('visible', np.ones(len(key_xy), dtype=bool)), dtype=bool)
    if len(key_xy) < 6:
        return {}
    key_by_id = {str(pid): (key_xy[i], bool(key_visible[i])) for i, pid in enumerate(key_ids)}

    out: dict[int, tuple[float, float, float, str, str]] = {}
    for row in object_rows:
        frame = int(row['frame'])
        mesh = mesh_tracks.get(frame)
        if mesh is None:
            continue
        ids = np.asarray(mesh.get('point_ids', []), dtype=object)
        xy = np.asarray(mesh.get('xy', []), dtype=np.float64)
        visible = np.asarray(mesh.get('visible', np.ones(len(xy), dtype=bool)), dtype=bool)
        src_pts = []
        dst_pts = []
        for i, pid in enumerate(ids):
            item = key_by_id.get(str(pid))
            if item is None or not visible[i] or not item[1]:
                continue
            src_pts.append([item[0][0], item[0][1], 1.0])
            dst_pts.append([xy[i, 0], xy[i, 1]])
        if len(src_pts) < 6:
            continue
        src = np.asarray(src_pts, dtype=np.float64)
        dst = np.asarray(dst_pts, dtype=np.float64)
        try:
            affine, *_ = np.linalg.lstsq(src, dst, rcond=None)  # 3x2, maps keyframe -> current frame.
            uv = key_pt @ affine
        except Exception:
            continue
        if not np.all(np.isfinite(uv)):
            continue
        # Side label is only descriptive; the point itself comes from rigid/affine
        # propagation of the painted keyframe contact point through object tracks.
        ref_u = parse_scalar(row.get('body_center_x', row.get('center_x', '')), math.nan)
        if not np.isfinite(ref_u):
            bbox_body = _finite_bbox(row, ('body_bbox_x1', 'body_bbox_y1', 'body_bbox_x2', 'body_bbox_y2'))
            if bbox_body is not None:
                ref_u = 0.5 * (bbox_body[0] + bbox_body[2])
        side = 'left' if np.isfinite(ref_u) and uv[0] < ref_u else 'right'
        src_name = 'articraft_surface_contact_region_keyframe_affine'
        if frame == ann_frame:
            src_name = 'articraft_surface_contact_region_keyframe_mask'
        out[frame] = (float(uv[0]), float(uv[1]), 1.0, src_name, f'contact_region:{side}:rigid_keyframe_surface_region')
    return out


def select_contact_proxy_from_articraft_region(
    row: dict[str, str],
    mesh: dict[str, np.ndarray],
    object_proxy: dict[str, object],
    active_uv: np.ndarray | None,
) -> tuple[float, float, float, str, str] | None:
    contact_region = object_proxy.get('contact_region') if isinstance(object_proxy, dict) else None
    if not isinstance(contact_region, dict):
        return None
    if str(contact_region.get('type', '')).lower() not in {'handle', 'handle_or_side', 'handle_or_body_side', 'object_surface_contact_region', 'surface_contact_region'}:
        return None
    proxy_side = str(contact_region.get('side', '') or '').strip().lower()
    row_side = str(row.get('contact_region_side', '') or row.get('hand_contact_side', '') or '').strip().lower()
    # Contact region is the hand-object surface region. It is not necessarily the
    # visual handle: during drinking the contact can be on the cup body/rim while
    # the handle is hidden. Use contact_region_side/hand_contact_side only; do not
    # let visual handle side drive the contact anchor.
    side = row_side if row_side in {'left', 'right'} else proxy_side
    if side not in {'left', 'right'}:
        return None

    handle_bbox = _finite_bbox(row, ('handle_bbox_x1', 'handle_bbox_y1', 'handle_bbox_x2', 'handle_bbox_y2'))
    body_bbox = _finite_bbox(row, ('body_bbox_x1', 'body_bbox_y1', 'body_bbox_x2', 'body_bbox_y2'))
    if body_bbox is None:
        body_bbox = _finite_bbox(row, ('bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'))

    if body_bbox is not None:
        x1, y1, x2, y2 = body_bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        sign = -1.0 if side == 'left' else 1.0
        # Mug/Articraft path: the object-side contact region is fixed by the
        # keyframe annotation + canonical mug proxy. Do not chase per-frame
        # visible handle pixels, because the handle is tiny/occluded and makes
        # the contact point jump. Use the stable cup body bbox as the 2D carrier
        # and place the region on the annotated side of the body/handle.
        target_u = (x1 if side == 'left' else x2) + sign * 0.30 * w
        target_v = (y1 + y2) * 0.5 - 0.04 * h
        target_source = 'articraft_surface_contact_region_from_body_bbox'
    elif active_uv is not None and np.all(np.isfinite(active_uv)):
        target_u, target_v = float(active_uv[0]), float(active_uv[1])
        target_source = 'articraft_surface_contact_region_active_part_fallback'
    else:
        return None

    return float(target_u), float(target_v), 1.0, target_source, f"contact_region:{side}:canonical_surface_region"


def main() -> None:
    parser = argparse.ArgumentParser(description='Build radius-free object proxy observations.')
    parser.add_argument('--sample-dir', type=Path, required=True)
    parser.add_argument('--out-subdir', type=str, default='object_proxy_observations')
    parser.add_argument('--object-observation-csv', type=Path, default=None)
    parser.add_argument('--da3-prior-csv', type=Path, default=None)
    parser.add_argument('--object-mesh-csv', type=Path, default=None)
    parser.add_argument('--body-model-root', type=Path, default=Path('third-party/GVHMR/inputs/checkpoints/body_models'))
    args = parser.parse_args()

    sample_dir = args.sample_dir
    results_dir = sample_dir / 'results'
    object_csv = args.object_observation_csv or (results_dir / 'object_observations' / 'object_observations.csv')
    da3_csv = args.da3_prior_csv or (results_dir / 'da3' / 'priors' / 'ball_depth_prior.csv')
    mesh_csv = args.object_mesh_csv or (results_dir / 'tracking' / 'object_mesh_tracks_test.csv')
    depth_index_csv = results_dir / 'da3' / 'scene_depth' / 'index.csv'
    audio_csv = results_dir / 'events' / 'audio_events.csv'
    out_dir = results_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    object_rows = read_rows(object_csv)
    da3 = read_da3_priors(da3_csv)
    mesh_tracks = read_object_mesh_tracks(mesh_csv) if mesh_csv.exists() else {}
    frame_to_depth = read_index(depth_index_csv) if depth_index_csv.exists() else {}
    video_hw = get_video_hw(sample_dir / 'video.mp4') if frame_to_depth else None
    audio_rows = read_rows(audio_csv) if audio_csv.exists() else []
    object_proxy = load_object_proxy(sample_dir)
    contact_region_track = build_articraft_contact_region_track(sample_dir, object_rows, object_proxy, mesh_tracks)

    human = read_human_result(results_dir / 'gvhmr' / 'result.pkl')
    joints = build_body_joints(args.body_model_root, human)
    K = np.asarray(human['K_fullimg'], dtype=np.float64)
    joints = joints[:len(object_rows)]
    K = K[:len(object_rows)]
    part_centers = build_contact_part_centers(joints)

    rows = []
    for idx, row in enumerate(object_rows):
        frame = int(row['frame'])
        mesh = mesh_tracks.get(frame)
        if mesh is None:
            continue
        label, active_uv, active_z, active_conf = select_active_body_proxy_from_mesh(mesh, part_centers, idx, K)
        ref_u, ref_v, ref_conf, ref_source = select_stable_ref_proxy(row)
        support_u, support_v, support_conf, support_source = select_support_proxy_bottom_percentile(mesh)
        region_contact = contact_region_track.get(frame)
        if region_contact is None:
            region_contact = select_contact_proxy_from_articraft_region(row, mesh, object_proxy, active_uv)
        if region_contact is not None:
            contact_u, contact_v, contact_proxy_conf, contact_source, contact_proxy_name = region_contact
        else:
            contact_u, contact_v, contact_proxy_conf, contact_source, contact_proxy_name = select_contact_proxy_from_human_point(mesh, active_uv)
        if not all(np.isfinite(v) for v in [ref_u, ref_v, support_u, support_v, contact_u, contact_v]):
            continue
        raw_depth, smooth_depth = da3.get(frame, (math.nan, math.nan))
        object_ref_depth_m = math.nan
        contact_proxy_depth_m = math.nan
        if frame in frame_to_depth and video_hw is not None:
            depth_map = load_depth(frame_to_depth[frame])
            ref_ud, ref_vd = map_to_depth_uv(ref_u, ref_v, video_hw, depth_map.shape[:2])
            object_ref_depth_m, _ = sample_depth_at_uv(depth_map, ref_ud, ref_vd)
            contact_ud, contact_vd = map_to_depth_uv(contact_u, contact_v, video_hw, depth_map.shape[:2])
            contact_proxy_depth_m, _ = sample_depth_at_uv(depth_map, contact_ud, contact_vd)
        if not np.isfinite(object_ref_depth_m) or not np.isfinite(contact_proxy_depth_m):
            continue
        rows.append({
            'frame': frame,
            'time': float(row['time']),
            'ref_u': ref_u,
            'ref_v': ref_v,
            'ref_source': ref_source,
            'support_u': support_u,
            'support_v_raw': support_v,
            'support_source': support_source,
            'contact_u': contact_u,
            'contact_v': contact_v,
            'contact_source': contact_source,
            'object_depth_raw': object_ref_depth_m,
            'contact_proxy_depth_m': contact_proxy_depth_m,
            'ref_conf': ref_conf,
            'support_conf_raw': support_conf,
            'contact_conf_raw': contact_proxy_conf,
            'active_label': label,
            'active_label_conf': active_conf,
            'active_part_u': active_uv[0] if active_uv is not None else math.nan,
            'active_part_v': active_uv[1] if active_uv is not None else math.nan,
            'active_part_z': active_z,
            'contact_proxy_name': contact_proxy_name,
            'da3_depth_raw': raw_depth,
            'da3_depth_smooth': smooth_depth,
        })

    if not rows:
        raise RuntimeError('No valid rows built')

    frames = np.asarray([r['frame'] for r in rows], dtype=np.int32)
    depth_raw = np.asarray([r['object_depth_raw'] for r in rows], dtype=np.float64)
    support_raw = np.asarray([r['support_v_raw'] for r in rows], dtype=np.float64)
    ref_u = np.asarray([r['ref_u'] for r in rows], dtype=np.float64)
    ref_v = np.asarray([r['ref_v'] for r in rows], dtype=np.float64)
    audio_score = build_audio_support(frames, audio_rows, radius=2)

    # Robust local depth prior: rely on a wide rolling median and clamp raw depth
    # toward that local trend instead of carrying stale old values across bad regions.
    depth_local_med = rolling_stat(depth_raw, radius=6, mode='median')
    depth_local_std = rolling_stat(depth_raw, radius=6, mode='std')
    depth_outlier = np.abs(depth_raw - depth_local_med)
    gate_base = np.nanpercentile(depth_outlier, 75)
    gate_dynamic = np.where(np.isfinite(depth_local_std), np.maximum(0.35, 2.0 * depth_local_std), 0.35)
    gate = np.minimum(np.maximum(gate_dynamic, 0.20), max(float(gate_base), 0.35))
    depth_clipped = np.where(
        np.isfinite(depth_local_med),
        np.clip(depth_raw, depth_local_med - gate, depth_local_med + gate),
        depth_raw,
    )
    depth_conf = low_value_score(depth_outlier, np.percentile(depth_outlier, 20), np.percentile(depth_outlier, 70))
    # For large outliers, fall back to the local robust median instead of keeping the
    # previous EMA value. This lets the trajectory re-enter a corrected depth layer.
    depth_input = np.where(depth_outlier <= gate, depth_clipped, depth_local_med)
    depth_smooth = ema_filter(depth_input, alpha=0.45)

    ref_u_smooth = robust_smooth_signal(ref_u, window=5, polyorder=2)
    ref_v_smooth = robust_smooth_signal(ref_v, window=5, polyorder=2)
    ref_u_fit = ref_u_smooth.copy()
    ref_v_fit = ref_v_smooth.copy()
    support_dv_raw = support_raw - ref_v
    support_dv_smooth = robust_smooth_signal(support_dv_raw, window=7, polyorder=2)
    support_smooth = ref_v_smooth + support_dv_smooth
    support_gap_px = np.abs(support_raw - support_smooth)
    support_gap_std = rolling_stat(support_gap_px, radius=3, mode='std')
    support_std = rolling_stat(support_smooth, radius=3, mode='std')
    support_conf = low_value_score(support_gap_px, np.percentile(support_gap_px, 10), np.percentile(support_gap_px, 60)) *         low_value_score(support_gap_std, np.percentile(support_gap_std, 20), np.percentile(support_gap_std, 80)) *         low_value_score(support_std, np.percentile(support_std, 20), np.percentile(support_std, 80))
    ref_jitter = np.sqrt(
        np.square(rolling_abs_deviation(ref_u, window=7))
        + np.square(rolling_abs_deviation(ref_v, window=7))
    )
    proxy_conf = low_value_score(
        ref_jitter,
        np.percentile(ref_jitter[np.isfinite(ref_jitter)], 30),
        np.percentile(ref_jitter[np.isfinite(ref_jitter)], 85),
    )
    support_jitter = rolling_abs_deviation(support_dv_raw, window=7)
    support_conf_jitter = low_value_score(
        support_jitter,
        np.percentile(support_jitter[np.isfinite(support_jitter)], 30),
        np.percentile(support_jitter[np.isfinite(support_jitter)], 85),
    )
    support_conf = support_conf * support_conf_jitter

    vx = np.gradient(ref_u)
    vy = np.gradient(ref_v)
    acc = np.sqrt(np.gradient(vx)**2 + np.gradient(vy)**2)
    motion_score = high_value_score(acc, np.percentile(acc, 50), np.percentile(acc, 90))

    out_rows = []
    for i, r in enumerate(rows):
        contact_depth_offset = float(r['contact_proxy_depth_m'] - depth_smooth[i])
        observation_conf = float(min(r['ref_conf'], max(depth_conf[i], 0.1)))
        out_rows.append({
            'frame': r['frame'],
            'time': f"{r['time']:.6f}",
            'ref_u': f"{r['ref_u']:.3f}",
            'ref_v': f"{r['ref_v']:.3f}",
            'ref_u_smooth': f"{ref_u_smooth[i]:.3f}",
            'ref_v_smooth': f"{ref_v_smooth[i]:.3f}",
            'ref_u_fit': f"{ref_u_fit[i]:.3f}",
            'ref_v_fit': f"{ref_v_fit[i]:.3f}",
            'ref_source': r['ref_source'],
            'ref_type': r['ref_source'],
            'support_u': f"{r['support_u']:.3f}",
            'support_v': f"{support_smooth[i]:.3f}",
            'support_v_raw': f"{support_raw[i]:.3f}",
            'support_dv': f"{(support_smooth[i] - r['ref_v']):.3f}",
            'support_dv_smooth': f"{support_dv_smooth[i]:.3f}",
            'support_source': r['support_source'],
            'contact_u': f"{r['contact_u']:.3f}",
            'contact_v': f"{r['contact_v']:.3f}",
            'contact_source': r['contact_source'],
            'object_ref_depth_m': f"{depth_smooth[i]:.6f}",
            'contact_proxy_depth_m': f"{r['contact_proxy_depth_m']:.6f}",
            'contact_depth_offset_m': f"{contact_depth_offset:.6f}",
            'ref_conf': f"{r['ref_conf']:.6f}",
            'support_conf': f"{support_conf[i]:.6f}",
            'contact_conf': f"{r['contact_conf_raw']:.6f}",
            'depth_conf': f"{depth_conf[i]:.6f}",
            'observation_conf': f"{observation_conf:.6f}",
            'proxy_conf': f"{proxy_conf[i]:.6f}",
            'proxy_jitter_px': f"{ref_jitter[i]:.6f}",
            'support_jitter_px': f"{support_jitter[i]:.6f}",
            'proxy_sigma_px': '5.000000',
            'support_sigma_px': '8.000000',
            'active_label': r['active_label'],
            'active_label_conf': f"{r['active_label_conf']:.6f}",
            'active_part_u': f"{r['active_part_u']:.3f}",
            'active_part_v': f"{r['active_part_v']:.3f}",
            'active_part_z': f"{r['active_part_z']:.6f}",
            'contact_proxy_name': r['contact_proxy_name'],
            'da3_depth_raw': f"{r['da3_depth_raw']:.6f}" if np.isfinite(r['da3_depth_raw']) else '',
            'da3_depth_smooth': f"{r['da3_depth_smooth']:.6f}" if np.isfinite(r['da3_depth_smooth']) else '',
            'object_depth_raw': f"{depth_raw[i]:.6f}",
            'object_depth_smooth': f"{depth_smooth[i]:.6f}",
            'object_depth_confidence': f"{depth_conf[i]:.6f}",
            'support_gap_px': f"{support_gap_px[i]:.6f}",
            'object_motion_score': f"{motion_score[i]:.6f}",
            'audio_score': f"{audio_score[i]:.6f}",
        })

    out_csv = out_dir / 'object_proxy_observations.csv'
    write_csv(out_csv, out_rows, list(out_rows[0].keys()))
    print(f'object_proxy_observations_csv: {out_csv}')


if __name__ == '__main__':
    main()
