#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_mug_articraft_keyframe_pose as base  # noqa: E402


BODY_R = base.BODY_R
BODY_H = base.BODY_H
BODY_D = base.BODY_R  # body-only debug must be a real cylinder, not the flattened handle proxy.


CANONICAL_POINT_LOCAL = {
    'rim_left': np.array([-BODY_R, -BODY_H / 2, 0.0], dtype=float),
    'rim_right': np.array([BODY_R, -BODY_H / 2, 0.0], dtype=float),
    'rim_front_or_visible_mid': np.array([0.0, -BODY_H / 2, -BODY_D], dtype=float),
    'bottom_left': np.array([-BODY_R, BODY_H / 2, 0.0], dtype=float),
    'bottom_right': np.array([BODY_R, BODY_H / 2, 0.0], dtype=float),
    'bottom_front_or_visible_mid': np.array([0.0, BODY_H / 2, -BODY_D], dtype=float),
}


def load_manual_point_observations(sample: Path) -> dict[int, dict[str, np.ndarray]]:
    path = sample / 'annotations' / 'mug_body_rim_bottom_manual_points.json'
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out: dict[int, dict[str, np.ndarray]] = {}
    for key, item in (data.get('frames') or {}).items():
        pts = {}
        for name, uv in (item.get('points') or {}).items():
            if uv is None or name not in CANONICAL_POINT_LOCAL:
                continue
            arr = np.asarray(uv, dtype=float)
            if arr.shape == (2,) and np.all(np.isfinite(arr)):
                pts[name] = arr
        if pts:
            out[int(key)] = pts
    return out


def manual_point_residual(params: np.ndarray, point_obs: dict[str, np.ndarray], K: np.ndarray, sigma_px: float = 2.5) -> list[float]:
    # The manual points are visible contour samples, not persistent semantic 3D vertices.
    # Match each rim/bottom observation to the closest projected point on the corresponding cylinder ring.
    theta = np.linspace(0, 2 * np.pi, 144, endpoint=False)
    rim_ring = np.stack([BODY_R * np.cos(theta), np.full_like(theta, -BODY_H / 2), BODY_D * np.sin(theta)], axis=1)
    bottom_ring = np.stack([BODY_R * np.cos(theta), np.full_like(theta, BODY_H / 2), BODY_D * np.sin(theta)], axis=1)
    rim_uv = project_pts(params, rim_ring, K)
    bottom_uv = project_pts(params, bottom_ring, K)
    res: list[float] = []
    for name, uv_obs in point_obs.items():
        ring_uv = rim_uv if name.startswith('rim_') else bottom_uv if name.startswith('bottom_') else None
        if ring_uv is None or len(ring_uv) == 0:
            continue
        d = ring_uv - uv_obs[None, :]
        j = int(np.argmin(np.sum(d * d, axis=1)))
        res.extend(((ring_uv[j] - uv_obs) / sigma_px).tolist())
    return res

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def ff(row: dict[str, str], key: str, default: float = math.nan) -> float:
    return base.ff(row, key, default)


def cylinder_points(n: int = 96) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    top = np.stack([BODY_R * np.cos(theta), np.full_like(theta, -BODY_H / 2), BODY_D * np.sin(theta)], axis=1)
    bot = np.stack([BODY_R * np.cos(theta), np.full_like(theta, BODY_H / 2), BODY_D * np.sin(theta)], axis=1)
    side = np.array([
        [-BODY_R, -BODY_H / 2, 0.0], [BODY_R, -BODY_H / 2, 0.0],
        [-BODY_R, BODY_H / 2, 0.0], [BODY_R, BODY_H / 2, 0.0],
        [0.0, -BODY_H / 2, 0.0], [0.0, BODY_H / 2, 0.0], [0.0, 0.0, 0.0],
    ], dtype=float)
    return top, bot, np.vstack([top, bot, side])


def project_pts(params: np.ndarray, pts: np.ndarray, K: np.ndarray) -> np.ndarray:
    return base.project_pts(params, pts, K)


def params_from_observation(row: dict[str, str], proxy: dict[str, str], K: np.ndarray) -> np.ndarray:
    z = ff(proxy, 'object_depth_smooth', ff(proxy, 'da3_depth_smooth', 3.8))
    if not np.isfinite(z) or z < 0.2:
        z = 3.8
    u = ff(row, 'body_center_x', ff(row, 'center_x'))
    v = ff(row, 'body_center_y', ff(row, 'center_y'))
    xyz = base.backproject(u, v, z, K)
    bh = ff(row, 'body_bbox_h_px', ff(row, 'bbox_h_px', 48.0))
    scale = float(np.clip((bh * z) / max(K[1, 1] * BODY_H, 1e-6), 0.30, 3.5))
    return np.array([xyz[0], xyz[1], xyz[2], 0.0, 0.0, 0.0, scale], dtype=float)


def build_edge_obs(sample: Path, frames: list[int], obs_rows: dict[int, dict[str, str]]) -> dict[int, dict]:
    manual_path = sample / 'annotations' / 'mug_rim_bottom_manual.json'
    manual_ann = json.loads(manual_path.read_text()) if manual_path.exists() else {'frames': {}}
    propagated_ann = base.build_propagated_manual_annotations(sample, manual_ann, obs_rows, frames)
    frame0 = cv2.imread(str(sample / 'frames' / '00001.png'))
    shape = frame0.shape[:2]
    out: dict[int, dict] = {}
    for fr in frames:
        eo = base.edge_observation_from_manual(fr, manual_ann, shape)
        if eo is None:
            eo = base.edge_observation_from_manual(fr, propagated_ann, shape)
        if eo is None:
            eo = base.edge_observation_from_mask(sample / 'results' / 'segmentation' / 'masks' / f'{fr:05d}_mask.png')
        if eo is not None:
            out[fr] = eo
    return out


def fit_body_only(sample: Path, start_frame: int, end_frame: int, max_nfev: int) -> tuple[list[int], np.ndarray, dict]:
    obs_rows = {int(r['frame']): r for r in read_csv(sample / 'results' / 'object_observations' / 'object_observations.csv')}
    proxy_rows = {int(r['frame']): r for r in read_csv(sample / 'results' / 'object_proxy_observations' / 'object_proxy_observations.csv')}
    frames = [fr for fr in range(start_frame, end_frame + 1) if fr in obs_rows]
    K_all = base.load_K(sample)
    edge_obs = build_edge_obs(sample, frames, obs_rows)
    manual_point_obs = load_manual_point_observations(sample)
    top_pts, bot_pts, body_pts = cylinder_points()
    center_pt = np.array([[0.0, 0.0, 0.0]], dtype=float)

    x0 = np.vstack([params_from_observation(obs_rows[fr], proxy_rows.get(fr, {}), K_all[fr - 1]) for fr in frames])
    n = len(frames)

    def residual(flat: np.ndarray) -> np.ndarray:
        P = flat.reshape(n, 7)
        res: list[float] = []
        for i, fr in enumerate(frames):
            p = P[i]
            obs = obs_rows[fr]
            proxy = proxy_rows.get(fr, {})
            K = K_all[fr - 1]
            point_obs = manual_point_obs.get(fr)
            body_bbox = [
                ff(obs, 'body_bbox_x1', ff(obs, 'bbox_x1')),
                ff(obs, 'body_bbox_y1', ff(obs, 'bbox_y1')),
                ff(obs, 'body_bbox_x2', ff(obs, 'bbox_x2')),
                ff(obs, 'body_bbox_y2', ff(obs, 'bbox_y2')),
            ]
            body_ctr = np.array([ff(obs, 'body_center_x', ff(obs, 'center_x')), ff(obs, 'body_center_y', ff(obs, 'center_y'))], dtype=float)
            cuv = project_pts(p, center_pt, K)[0]
            buv = project_pts(p, body_pts, K)
            res.extend(((cuv - body_ctr) / 7.0).tolist())
            res.extend(base.bbox_residual(buv, body_bbox, sigma=10.0))

            eo = edge_obs.get(fr)
            if point_obs:
                res.extend(manual_point_residual(p, point_obs, K, sigma_px=2.8))
            if eo is not None and not point_obs:
                src = str(eo.get('source', ''))
                manual_like = src.startswith('manual') or src.startswith('user_manual') or src.startswith('propagated') or src.startswith('motion_gated')
                top_uv = project_pts(p, top_pts, K)
                bot_uv = project_pts(p, bot_pts, K)
                top_center = project_pts(p, np.array([[0.0, -BODY_H / 2, 0.0]], dtype=float), K)[0]
                bot_center = project_pts(p, np.array([[0.0, BODY_H / 2, 0.0]], dtype=float), K)[0]
                center_sigma = 1.5 if manual_like else 5.0
                edge_sigma = 10.0 if manual_like else 12.0
                res.extend(((top_center - eo['top_center']) / center_sigma).tolist())
                res.extend(((bot_center - eo['bottom_center']) / center_sigma).tolist())
                res.extend([min(base.dt_at(eo['top_dt'], q), 80.0) / edge_sigma for q in top_uv[::6]])
                res.extend([min(base.dt_at(eo['bottom_dt'], q), 80.0) / edge_sigma for q in bot_uv[::6]])
                mp = eo.get('manual_points') or {}
                point_sigma = 2.8 if manual_like else 7.0
                minor_sigma = 4.0 if manual_like else 9.0
                res.extend(base.closest_point_residual(top_uv, mp.get('rim_arc'), sigma=point_sigma))
                res.extend(base.closest_point_residual(top_uv, mp.get('rim_major'), sigma=point_sigma))
                res.extend(base.closest_point_residual(top_uv, mp.get('rim_minor'), sigma=minor_sigma))
                res.extend(base.closest_point_residual(bot_uv, mp.get('bottom_major'), sigma=point_sigma))
                res.extend(base.closest_point_residual(bot_uv, mp.get('bottom_minor'), sigma=minor_sigma))
                res.append(0.35 * base.wrap(float(p[5] - eo['roll_prior'])) / 0.45)

            z_prior = ff(proxy, 'object_depth_smooth', ff(proxy, 'da3_depth_smooth', math.nan))
            if np.isfinite(z_prior):
                res.append((p[2] - z_prior) / 0.55)
            res.append((p[6] - x0[i, 6]) / 0.45)
            # Cylinder body has no yaw cue without handle/texture. Keep yaw weakly near the temporal seed.
            res.append(0.12 * base.wrap(float(p[3] - x0[i, 3])) / 0.7)

        for i in range(1, n):
            d = P[i] - P[i - 1]
            d[3] = base.wrap(d[3])
            d[4] = base.wrap(d[4])
            d[5] = base.wrap(d[5])
            res.extend((d[:3] / np.array([0.05, 0.05, 0.14])).tolist())
            res.extend((d[3:6] / np.array([0.25, 0.16, 0.16])).tolist())
            res.append(d[6] / 0.22)
        for i in range(1, n - 1):
            dd = P[i + 1] - 2 * P[i] + P[i - 1]
            dd[3] = base.wrap(P[i + 1, 3] - P[i, 3]) - base.wrap(P[i, 3] - P[i - 1, 3])
            dd[4] = base.wrap(P[i + 1, 4] - P[i, 4]) - base.wrap(P[i, 4] - P[i - 1, 4])
            dd[5] = base.wrap(P[i + 1, 5] - P[i, 5]) - base.wrap(P[i, 5] - P[i - 1, 5])
            res.extend((0.55 * dd[:3] / np.array([0.05, 0.05, 0.10])).tolist())
            res.extend((0.55 * dd[3:6] / np.array([0.18, 0.12, 0.12])).tolist())
        return np.asarray(res, dtype=float)

    lower = x0.copy()
    upper = x0.copy()
    lower[:, 0:2] -= 0.45
    upper[:, 0:2] += 0.45
    lower[:, 2] = np.maximum(0.8, x0[:, 2] - 0.9)
    upper[:, 2] = x0[:, 2] + 0.9
    lower[:, 3] = -math.pi
    upper[:, 3] = math.pi
    lower[:, 4] = math.radians(-85.0)
    upper[:, 4] = math.radians(80.0)
    lower[:, 5] = math.radians(-85.0)
    upper[:, 5] = math.radians(85.0)
    lower[:, 6] = 0.25
    upper[:, 6] = 3.5

    result = least_squares(
        residual,
        x0.reshape(-1),
        bounds=(lower.reshape(-1), upper.reshape(-1)),
        loss='soft_l1',
        f_scale=1.0,
        max_nfev=max_nfev,
        verbose=1,
    )
    P = result.x.reshape(n, 7)
    meta = {'success': bool(result.success), 'cost': float(result.cost), 'nfev': int(result.nfev), 'message': str(result.message)}
    return frames, P, meta


def draw_body(img: np.ndarray, params: np.ndarray, K: np.ndarray, active: bool = False) -> None:
    overlay = img.copy()
    top, bot, _body = cylinder_points()
    side_l = np.array([[-BODY_R, -BODY_H / 2, 0.0], [-BODY_R, BODY_H / 2, 0.0]], dtype=float)
    side_r = np.array([[BODY_R, -BODY_H / 2, 0.0], [BODY_R, BODY_H / 2, 0.0]], dtype=float)
    center_axis = np.array([[0.0, -BODY_H / 2, 0.0], [0.0, BODY_H / 2, 0.0]], dtype=float)
    for pts, color, thick, closed in [
        (top, (255, 0, 255), 2, True),
        (bot, (0, 255, 255), 2, True),
        (side_l, (245, 245, 245), 2, False),
        (side_r, (245, 245, 245), 2, False),
        (center_axis, (255, 255, 255), 1, False),
    ]:
        uv = project_pts(params, pts, K)
        if np.all(np.isfinite(uv)):
            cv2.polylines(overlay, [np.round(uv).astype(np.int32)], closed, color, thick, cv2.LINE_AA)
    c = project_pts(params, np.array([[0.0, 0.0, 0.0]], dtype=float), K)[0]
    if np.all(np.isfinite(c)):
        cv2.circle(overlay, tuple(np.round(c).astype(int)), 4, (255, 255, 255), -1, cv2.LINE_AA)
        if active:
            cv2.circle(overlay, tuple(np.round(c).astype(int)), 12, (40, 40, 230), 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)


def finalize(tmp: Path, out: Path) -> None:
    ffmpeg = shutil.which('ffmpeg') or '/usr/bin/ffmpeg'
    try:
        subprocess.run([ffmpeg, '-y', '-i', str(tmp), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out)], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        subprocess.run([ffmpeg, '-y', '-i', str(tmp), '-c:v', 'libopenh264', '-pix_fmt', 'yuv420p', str(out)], check=True, capture_output=True, text=True)
    tmp.unlink(missing_ok=True)


def render(sample: Path, frames: list[int], P: np.ndarray, fps: float) -> Path:
    out_dir = sample / 'results' / 'renders' / 'M5_body_only_cylinder_pose'
    out_dir.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(sample / 'frames' / '00001.png'))
    h, w = first.shape[:2]
    K_all = base.load_K(sample)
    tmp = out_dir / 'overlay.tmp.mp4'
    out = out_dir / 'overlay.mp4'
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for fr, p in zip(frames, P):
        img = cv2.imread(str(sample / 'frames' / f'{fr:05d}.png'))
        if img is None:
            continue
        draw_body(img, p, K_all[fr - 1], active=True)
        cv2.putText(img, f'M5_body_only_cylinder_pose frame {fr:03d}', (24, 36), cv2.FONT_HERSHEY_SIMPLEX, .7, (20,20,20), 3, cv2.LINE_AA)
        cv2.putText(img, f'M5_body_only_cylinder_pose frame {fr:03d}', (24, 36), cv2.FONT_HERSHEY_SIMPLEX, .7, (245,245,245), 1, cv2.LINE_AA)
        writer.write(img)
    writer.release()
    finalize(tmp, out)
    return out


def write_outputs(sample: Path, frames: list[int], P: np.ndarray, meta: dict, render_path: Path) -> None:
    proxy_dir = sample / 'proxy'
    proxy_dir.mkdir(exist_ok=True)
    rows = []
    for fr, p in zip(frames, P):
        rows.append({
            'frame': fr,
            'time': (fr - 1) / 24.0,
            'x': p[0],
            'y': p[1],
            'z': p[2],
            'yaw': base.wrap(p[3]),
            'yaw_deg': np.rad2deg(base.wrap(p[3])),
            'pitch': p[4],
            'pitch_deg': np.rad2deg(p[4]),
            'roll': p[5],
            'roll_deg': np.rad2deg(p[5]),
            'scale': p[6],
        })
    csv_path = proxy_dir / 'mug_body_only_cylinder_pose_sequence.csv'
    with csv_path.open('w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    fit_json = proxy_dir / 'mug_body_only_cylinder_pose_fit.json'
    fit_json.write_text(json.dumps({'model': 'body_only_true_cylinder_temporal_fit', 'meta': meta, 'frames': rows}, indent=2))
    outputs = {
        'pose_sequence': str(csv_path),
        'fit': str(fit_json),
        'render': str(render_path),
        'note': 'Body-only debug: circular cylinder cup body, no handle/contact region.',
    }
    (proxy_dir / 'mug_body_only_cylinder_pose_outputs.json').write_text(json.dumps(outputs, indent=2))
    print(json.dumps(outputs, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-dir', type=Path, required=True)
    ap.add_argument('--start-frame', type=int, default=1)
    ap.add_argument('--end-frame', type=int, default=240)
    ap.add_argument('--fps', type=float, default=24.0)
    ap.add_argument('--max-nfev', type=int, default=80)
    args = ap.parse_args()
    frames, P, meta = fit_body_only(args.sample_dir, args.start_frame, args.end_frame, args.max_nfev)
    out = render(args.sample_dir, frames, P, args.fps)
    write_outputs(args.sample_dir, frames, P, meta, out)


if __name__ == '__main__':
    main()
