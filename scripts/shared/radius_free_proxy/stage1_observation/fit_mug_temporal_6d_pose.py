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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def ff(row: dict[str, str], key: str, default: float = math.nan) -> float:
    return base.ff(row, key, default)


def params_from_m3_sequence(sample: Path, frames: list[int], obs_rows: dict[int, dict], proxy_rows: dict[int, dict], K_all: np.ndarray) -> np.ndarray:
    seq_path = sample / 'proxy' / 'mug_optimized_pose_sequence.csv'
    seq = {int(r['frame']): r for r in read_csv(seq_path)} if seq_path.exists() else {}
    out = []
    for fr in frames:
        row = seq.get(fr)
        if row:
            out.append([ff(row, 'x'), ff(row, 'y'), ff(row, 'z'), ff(row, 'yaw'), ff(row, 'pitch'), ff(row, 'roll'), ff(row, 'scale')])
        else:
            obs = obs_rows[fr]
            proxy = proxy_rows.get(fr, {})
            z = ff(proxy, 'object_depth_smooth', 3.8)
            xyz = base.backproject(ff(obs, 'body_center_x', ff(obs, 'center_x')), ff(obs, 'body_center_y', ff(obs, 'center_y')), z, K_all[fr - 1])
            bh = ff(obs, 'body_bbox_h_px', ff(obs, 'bbox_h_px', 42.0))
            scale = float(np.clip((bh * z) / max(K_all[fr - 1][1, 1] * base.BODY_H, 1.0), 0.35, 3.2))
            out.append([xyz[0], xyz[1], xyz[2], 0.0, 0.0, 0.0, scale])
    return np.asarray(out, dtype=float)


def build_edge_obs(sample: Path, frames: list[int], obs_rows: dict[int, dict]) -> dict[int, dict]:
    manual_path = sample / 'annotations' / 'mug_rim_bottom_manual.json'
    manual_ann = json.loads(manual_path.read_text()) if manual_path.exists() else {'frames': {}}
    propagated_ann = base.build_propagated_manual_annotations(sample, manual_ann, obs_rows, frames)
    frame0 = cv2.imread(str(sample / 'frames' / '00001.png'))
    shape = frame0.shape[:2]
    edge_obs = {}
    for fr in frames:
        eo = base.edge_observation_from_manual(fr, manual_ann, shape)
        if eo is None:
            eo = base.edge_observation_from_manual(fr, propagated_ann, shape)
        if eo is None:
            eo = base.edge_observation_from_mask(sample / 'results' / 'segmentation' / 'masks' / f'{fr:05d}_mask.png')
        if eo is not None:
            edge_obs[fr] = eo
    return edge_obs


def write_edge_debug_temporal(sample: Path, edge_obs: dict[int, dict]) -> None:
    out_dir = sample / 'annotations' / 'rim_bottom_observations_temporal'
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    for fr, obs in edge_obs.items():
        meta[str(fr)] = {
            'bbox_xyxy': obs['bbox'],
            'top_center_xy': obs['top_center'].tolist(),
            'bottom_center_xy': obs['bottom_center'].tolist(),
            'roll_prior_rad': obs['roll_prior'],
            'source': obs.get('source', 'segmentation_mask_top_bottom_silhouette_edges'),
            'manual': obs.get('manual'),
            'manual_points_available': {k: (None if v is None else len(v)) for k, v in (obs.get('manual_points') or {}).items()},
        }
    (out_dir / 'rim_bottom_observations_temporal.json').write_text(json.dumps(meta, indent=2))


def outside_bbox_residual(uv: np.ndarray, bbox: list[float], margin: float, sigma: float) -> list[float]:
    return base.outside_bbox_residual(uv, bbox, margin, sigma)


def temporal_fit(sample: Path, start_frame: int, end_frame: int, fps: float, max_nfev: int) -> tuple[list[int], np.ndarray, dict]:
    obs_rows = {int(r['frame']): r for r in read_csv(sample / 'results' / 'object_observations' / 'object_observations.csv')}
    proxy_rows = {int(r['frame']): r for r in read_csv(sample / 'results' / 'object_proxy_observations' / 'object_proxy_observations.csv')}
    frames = [fr for fr in range(start_frame, end_frame + 1) if fr in obs_rows]
    K_all = base.load_K(sample)
    edge_obs = build_edge_obs(sample, frames, obs_rows)
    write_edge_debug_temporal(sample, edge_obs)

    x0 = params_from_m3_sequence(sample, frames, obs_rows, proxy_rows, K_all)
    n = len(frames)
    bpts = base.body_points()
    hpts = base.handle_points()
    apts = base.all_points()
    top_pts = np.stack([base.BODY_R*np.cos(np.linspace(0, 2*np.pi, 24)), np.full(24, -base.BODY_H/2), base.BODY_D*np.sin(np.linspace(0, 2*np.pi, 24))], axis=1)
    bot_pts = np.stack([base.BODY_R*np.cos(np.linspace(0, 2*np.pi, 24)), np.full(24, base.BODY_H/2), base.BODY_D*np.sin(np.linspace(0, 2*np.pi, 24))], axis=1)
    center_pt = np.array([[0.0, 0.0, 0.0]], dtype=float)
    contact_pt = base.HANDLE_CENTER[None, :]

    def residual(flat: np.ndarray) -> np.ndarray:
        P = flat.reshape(n, 7)
        res: list[float] = []
        for i, fr in enumerate(frames):
            p = P[i]
            obs = obs_rows[fr]
            proxy = proxy_rows.get(fr, {})
            K = K_all[fr - 1]
            body_bbox = [ff(obs, 'body_bbox_x1', ff(obs, 'bbox_x1')), ff(obs, 'body_bbox_y1', ff(obs, 'bbox_y1')), ff(obs, 'body_bbox_x2', ff(obs, 'bbox_x2')), ff(obs, 'body_bbox_y2', ff(obs, 'bbox_y2'))]
            body_ctr = np.array([ff(obs, 'body_center_x', ff(obs, 'center_x')), ff(obs, 'body_center_y', ff(obs, 'center_y'))], dtype=float)
            cuv = base.project_pts(p, center_pt, K)[0]
            buv = base.project_pts(p, bpts, K)
            auv = base.project_pts(p, apts, K)
            huv = base.project_pts(p, hpts, K)
            res.extend(((cuv - body_ctr) / 10.0).tolist())
            res.extend(base.bbox_residual(buv, body_bbox, sigma=12.0))
            eo = edge_obs.get(fr)
            if eo is not None:
                src = eo.get('source', '')
                manual_like = src.startswith('manual') or src.startswith('user_manual') or src.startswith('propagated') or src.startswith('motion_gated')
                top_ring = base.project_pts(p, top_pts, K)
                bot_ring = base.project_pts(p, bot_pts, K)
                top_center = base.project_pts(p, np.array([[0.0, -base.BODY_H/2, 0.0]], dtype=float), K)[0]
                bot_center = base.project_pts(p, np.array([[0.0, base.BODY_H/2, 0.0]], dtype=float), K)[0]
                center_sigma = 1.6 if manual_like else 5.0
                res.extend(((top_center - eo['top_center']) / center_sigma).tolist())
                res.extend(((bot_center - eo['bottom_center']) / center_sigma).tolist())
                edge_sigma = 12.0 if manual_like else 10.0
                res.extend([min(base.dt_at(eo['top_dt'], q), 80.0) / edge_sigma for q in top_ring[::6]])
                res.extend([min(base.dt_at(eo['bottom_dt'], q), 80.0) / edge_sigma for q in bot_ring[::6]])
                mp = eo.get('manual_points') or {}
                point_sigma = 3.0 if manual_like else 7.0
                minor_sigma = 4.5 if manual_like else 9.0
                res.extend(base.closest_point_residual(top_ring, mp.get('rim_arc'), sigma=point_sigma))
                res.extend(base.closest_point_residual(top_ring, mp.get('rim_major'), sigma=point_sigma))
                res.extend(base.closest_point_residual(top_ring, mp.get('rim_minor'), sigma=minor_sigma))
                res.extend(base.closest_point_residual(bot_ring, mp.get('bottom_major'), sigma=point_sigma))
                res.extend(base.closest_point_residual(bot_ring, mp.get('bottom_minor'), sigma=minor_sigma))
                res.append(0.3 * base.wrap(float(p[5] - eo['roll_prior'])) / 0.5)
                manual_policy = str((eo.get('manual') or {}).get('handle_visibility_policy', '')).lower()
                if manual_policy == 'hidden':
                    res.extend([1.0 * r for r in outside_bbox_residual(huv[::6], body_bbox, margin=3.0, sigma=5.0)])
            z_prior = ff(proxy, 'object_depth_smooth', ff(proxy, 'da3_depth_smooth', p[2]))
            if np.isfinite(z_prior):
                res.append((p[2] - z_prior) / 0.45)
            # Weak hand-object attachment from radius-free active human part. This is not a full palm rigid term yet,
            # but it keeps the canonical handle/contact region from drifting far away from the active hand proxy.
            au = ff(proxy, 'active_part_u')
            av = ff(proxy, 'active_part_v')
            conf = ff(proxy, 'active_label_conf', 0.0)
            if np.isfinite(au) and np.isfinite(av) and conf > 0.15:
                contact_uv = base.project_pts(p, contact_pt, K)[0]
                res.extend((0.20 * min(1.0, conf) * (contact_uv - np.array([au, av])) / 22.0).tolist())
            res.append((p[6] - x0[i, 6]) / 0.45)
        # temporal smoothness: first and second order, with angles wrapped.
        for i in range(1, n):
            d = P[i] - P[i - 1]
            d[3] = base.wrap(d[3])
            d[4] = base.wrap(d[4])
            d[5] = base.wrap(d[5])
            res.extend((d[:3] / np.array([0.06, 0.06, 0.14])).tolist())
            res.extend((d[3:6] / np.array([0.16, 0.13, 0.15])).tolist())
            res.append(d[6] / 0.20)
        for i in range(1, n - 1):
            dd = P[i + 1] - 2 * P[i] + P[i - 1]
            dd[3] = base.wrap(P[i + 1, 3] - P[i, 3]) - base.wrap(P[i, 3] - P[i - 1, 3])
            dd[4] = base.wrap(P[i + 1, 4] - P[i, 4]) - base.wrap(P[i, 4] - P[i - 1, 4])
            dd[5] = base.wrap(P[i + 1, 5] - P[i, 5]) - base.wrap(P[i, 5] - P[i - 1, 5])
            res.extend((0.7 * dd[:3] / np.array([0.05, 0.05, 0.10])).tolist())
            res.extend((0.7 * dd[3:6] / np.array([0.12, 0.10, 0.12])).tolist())
        return np.asarray(res, dtype=float)

    lower = x0.copy(); upper = x0.copy()
    lower[:, 0:2] -= 0.45; upper[:, 0:2] += 0.45
    lower[:, 2] = np.maximum(0.8, x0[:, 2] - 0.9); upper[:, 2] = x0[:, 2] + 0.9
    lower[:, 3:6] = -math.pi; upper[:, 3:6] = math.pi
    lower[:, 4] = math.radians(-85.0); upper[:, 4] = math.radians(75.0)
    lower[:, 5] = math.radians(-85.0); upper[:, 5] = math.radians(85.0)
    lower[:, 6] = 0.25; upper[:, 6] = 3.3
    result = least_squares(residual, x0.reshape(-1), bounds=(lower.reshape(-1), upper.reshape(-1)), loss='soft_l1', f_scale=1.0, max_nfev=max_nfev, verbose=1)
    P = result.x.reshape(n, 7)
    meta = {'success': bool(result.success), 'cost': float(result.cost), 'nfev': int(result.nfev), 'message': str(result.message)}
    return frames, P, meta


def finalize(tmp: Path, out: Path) -> None:
    ffmpeg = shutil.which('ffmpeg') or '/usr/bin/ffmpeg'
    try:
        subprocess.run([ffmpeg, '-y', '-i', str(tmp), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out)], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        subprocess.run([ffmpeg, '-y', '-i', str(tmp), '-c:v', 'libopenh264', '-pix_fmt', 'yuv420p', str(out)], check=True, capture_output=True, text=True)
    tmp.unlink(missing_ok=True)


def render(sample: Path, frames: list[int], P: np.ndarray, fps: float) -> Path:
    out_dir = sample / 'results' / 'renders' / 'M4_temporal_6d_pose'
    out_dir.mkdir(parents=True, exist_ok=True)
    K_all = base.load_K(sample)
    first = cv2.imread(str(sample / 'frames' / '00001.png'))
    h, w = first.shape[:2]
    tmp = out_dir / 'overlay.tmp.mp4'
    out = out_dir / 'overlay.mp4'
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    params_by_frame = {fr: P[i] for i, fr in enumerate(frames)}
    for fr in frames:
        img = cv2.imread(str(sample / 'frames' / f'{fr:05d}.png'))
        if img is None:
            continue
        base.draw_mug(img, params_by_frame[fr], K_all[fr - 1], active=True, show_handle=False)
        cv2.putText(img, f'M4_temporal_6d_pose frame {fr:03d}', (24, 36), cv2.FONT_HERSHEY_SIMPLEX, .7, (20,20,20), 3, cv2.LINE_AA)
        cv2.putText(img, f'M4_temporal_6d_pose frame {fr:03d}', (24, 36), cv2.FONT_HERSHEY_SIMPLEX, .7, (245,245,245), 1, cv2.LINE_AA)
        writer.write(img)
    writer.release()
    finalize(tmp, out)
    return out


def write_outputs(sample: Path, frames: list[int], P: np.ndarray, meta: dict, render_path: Path) -> None:
    proxy = sample / 'proxy'
    proxy.mkdir(exist_ok=True)
    rows = []
    for fr, p in zip(frames, P):
        rows.append({
            'frame': fr, 'time': (fr - 1) / 24.0,
            'x': p[0], 'y': p[1], 'z': p[2],
            'yaw': base.wrap(p[3]), 'yaw_deg': np.rad2deg(base.wrap(p[3])),
            'pitch': p[4], 'pitch_deg': np.rad2deg(p[4]),
            'roll': p[5], 'roll_deg': np.rad2deg(p[5]),
            'scale': p[6],
        })
    csv_path = proxy / 'mug_temporal_6d_pose_sequence.csv'
    with csv_path.open('w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    json_path = proxy / 'mug_temporal_6d_pose_fit.json'
    json_path.write_text(json.dumps({'model': 'mug_temporal_6d_pose_fit', 'meta': meta, 'frames': rows, 'render': str(render_path)}, indent=2))
    summary = {'pose_sequence': str(csv_path), 'pose_fit': str(json_path), 'render': str(render_path), **meta}
    (proxy / 'mug_temporal_6d_pose_outputs.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-dir', type=Path, default=Path('samples_known_object/02_mug'))
    ap.add_argument('--start-frame', type=int, default=60)
    ap.add_argument('--end-frame', type=int, default=100)
    ap.add_argument('--fps', type=float, default=24.0)
    ap.add_argument('--max-nfev', type=int, default=20)
    args = ap.parse_args()
    frames, P, meta = temporal_fit(args.sample_dir, args.start_frame, args.end_frame, args.fps, args.max_nfev)
    out = render(args.sample_dir, frames, P, args.fps)
    write_outputs(args.sample_dir, frames, P, meta, out)


if __name__ == '__main__':
    main()
