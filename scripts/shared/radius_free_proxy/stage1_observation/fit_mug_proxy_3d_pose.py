#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
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


def smooth_nan(values: np.ndarray, radius: int = 4) -> np.ndarray:
    out = values.astype(float).copy()
    n = len(out)
    for i in range(n):
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        win = out[lo:hi]
        win = win[np.isfinite(win)]
        if len(win):
            out[i] = float(np.median(win))
    return out


def interp_nan(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    idx = np.arange(len(values))
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    out = values.copy()
    out[~finite] = np.interp(idx[~finite], idx[finite], values[finite])
    return out


def estimate_yaw(row: dict[str, str]) -> tuple[float, float, str]:
    # Yaw is a weak-perspective proxy: yaw=0 means handle broad/C-shaped in image;
    # yaw=90deg means the handle plane is side-on and collapses toward a vertical line.
    bw = ff(row, 'body_bbox_w_px', ff(row, 'bbox_w_px'))
    side = (row.get('handle_side') or row.get('visual_handle_side') or row.get('contact_region_side') or 'left').strip().lower()
    if side not in {'left', 'right'}:
        side = 'left'
    sign = -1.0 if side == 'left' else 1.0
    visible = str(row.get('handle_visible', '') or '') == '1'
    conf = ff(row, 'handle_conf', 0.0)
    extra = max(ff(row, 'handle_extra_left_px', 0.0), ff(row, 'handle_extra_right_px', 0.0))
    if np.isfinite(bw) and bw > 1 and visible and conf > 0.06:
        # Expected fully visible handle protrusion is roughly 0.55 body widths in this
        # simple proxy. If observed protrusion is small, the handle is rotating side-on.
        ratio = np.clip(extra / max(0.55 * bw, 1.0), 0.05, 1.0)
        yaw_abs = math.acos(float(ratio))
        # Very visible handle center on the side should stay closer to open C-shape.
        yaw_abs *= float(np.clip(1.0 - 0.75 * conf, 0.35, 1.0))
        return sign * yaw_abs, float(np.clip(conf + 0.35, 0.0, 1.0)), 'handle_visible'
    contact_side = (row.get('contact_region_side') or row.get('hand_contact_side') or '').strip().lower()
    if contact_side in {'left', 'right'}:
        sign = -1.0 if contact_side == 'left' else 1.0
        # Hand-contact frames with handle hidden are usually side-on/occluded.
        return sign * math.radians(68.0), 0.45, 'contact_side_occluded'
    return sign * math.radians(75.0), 0.15, 'latent_side_default'


def build_pose(rows: list[dict[str, str]]) -> list[dict[str, float | int | str]]:
    frames = np.array([int(r['frame']) for r in rows], dtype=int)
    cx = np.array([ff(r, 'body_center_x', ff(r, 'center_x')) for r in rows], dtype=float)
    cy = np.array([ff(r, 'body_center_y', ff(r, 'center_y')) for r in rows], dtype=float)
    bh = np.array([ff(r, 'body_bbox_h_px', ff(r, 'bbox_h_px')) for r in rows], dtype=float)
    bw = np.array([ff(r, 'body_bbox_w_px', ff(r, 'bbox_w_px')) for r in rows], dtype=float)

    yaw_raw, yaw_conf, yaw_src = [], [], []
    for r in rows:
        y, c, s = estimate_yaw(r)
        yaw_raw.append(y)
        yaw_conf.append(c)
        yaw_src.append(s)
    yaw_raw = np.array(yaw_raw, dtype=float)
    yaw_conf = np.array(yaw_conf, dtype=float)

    # Smooth body pose separately from object mesh jitter. This is the rigid object pose,
    # not a per-frame handle drawing path.
    cx_s = smooth_nan(interp_nan(cx), radius=3)
    cy_s = smooth_nan(interp_nan(cy), radius=3)
    scale_s = smooth_nan(interp_nan(bh / 1.05), radius=4)

    key = yaw_conf >= 0.35
    yaw_key = np.full_like(yaw_raw, np.nan)
    yaw_key[key] = yaw_raw[key]
    yaw_interp = interp_nan(yaw_key)
    yaw_s = smooth_nan(yaw_interp, radius=5)

    out = []
    for i, r in enumerate(rows):
        out.append({
            'frame': int(frames[i]),
            'time': ff(r, 'time', 0.0),
            'cx': float(cx_s[i]),
            'cy': float(cy_s[i]),
            'scale': float(scale_s[i]),
            'yaw_rad': float(yaw_s[i]),
            'yaw_deg': float(np.rad2deg(yaw_s[i])),
            'yaw_raw_deg': float(np.rad2deg(yaw_raw[i])),
            'yaw_conf': float(yaw_conf[i]),
            'yaw_source': yaw_src[i],
            'body_w_px': float(bw[i]),
            'body_h_px': float(bh[i]),
        })
    return out


def project_proxy(cx: float, cy: float, scale: float, yaw: float) -> dict[str, np.ndarray]:
    r = 0.38
    h = 1.05
    zc = np.linspace(-h / 2, h / 2, 2)
    theta = np.linspace(0, 2 * np.pi, 64)
    def proj(points: np.ndarray) -> np.ndarray:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        xr = np.cos(yaw) * x - np.sin(yaw) * y
        # weak perspective: vertical stays z; tiny depth cue keeps bottom/top readable
        yr = z + 0.10 * (np.sin(yaw) * x + np.cos(yaw) * y)
        return np.stack([cx + scale * xr, cy - scale * yr], axis=1)
    top = np.stack([r * np.cos(theta), r * np.sin(theta), np.full_like(theta, h / 2)], axis=1)
    bot = np.stack([r * np.cos(theta), r * np.sin(theta), np.full_like(theta, -h / 2)], axis=1)
    # Handle is a C in the x-z plane attached to left side in canonical contact-side.
    phi = np.deg2rad(np.linspace(72, 288, 64))
    hx = -0.72 + 0.30 * np.cos(phi)
    hz = 0.02 + 0.36 * np.sin(phi)
    hy = np.zeros_like(phi)
    handle = np.stack([hx, hy, hz], axis=1)
    upper = np.array([[-r, 0.0, 0.30], [-0.53, 0.0, 0.30]])
    lower = np.array([[-r, 0.0, -0.27], [-0.53, 0.0, -0.27]])
    side1 = np.array([[-r, 0, -h/2], [-r, 0, h/2]])
    side2 = np.array([[r, 0, -h/2], [r, 0, h/2]])
    return {
        'top': proj(top), 'bottom': proj(bot), 'handle': proj(handle),
        'upper': proj(upper), 'lower': proj(lower), 'side1': proj(side1), 'side2': proj(side2),
    }


def draw_poly(img, pts, color, thick):
    cv2.polylines(img, [np.round(pts).astype(np.int32)], False, color, thick, cv2.LINE_AA)


def render(sample_dir: Path, poses: list[dict[str, float | int | str]], fps: float) -> None:
    frames_dir = sample_dir / 'frames'
    out_dir = sample_dir / 'results' / 'renders' / 'mug_proxy_3d_pose_debug'
    out_dir.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(frames_dir / '00001.png'))
    if first is None:
        raise RuntimeError('missing frames')
    h, w = first.shape[:2]
    tmp = out_dir / 'overlay.tmp.mp4'
    mp4 = out_dir / 'overlay.mp4'
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for p in poses:
        fr = int(p['frame'])
        img = cv2.imread(str(frames_dir / f'{fr:05d}.png'))
        if img is None:
            continue
        geom = project_proxy(float(p['cx']), float(p['cy']), float(p['scale']), float(p['yaw_rad']))
        overlay = img.copy()
        for k in ['top', 'bottom', 'side1', 'side2']:
            draw_poly(overlay, geom[k], (235, 245, 255), 2)
        for k in ['handle', 'upper', 'lower']:
            draw_poly(overlay, geom[k], (40, 220, 255), 3)
        cv2.addWeighted(overlay, 0.68, img, 0.32, 0, img)
        cv2.putText(img, f"frame {fr:03d} yaw={float(p['yaw_deg']):.1f} src={p['yaw_source']}", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,20,20), 3, cv2.LINE_AA)
        cv2.putText(img, f"frame {fr:03d} yaw={float(p['yaw_deg']):.1f} src={p['yaw_source']}", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245,245,245), 1, cv2.LINE_AA)
        writer.write(img)
    writer.release()
    import subprocess, shutil
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        subprocess.run([ffmpeg, '-y', '-i', str(tmp), '-c:v', 'libopenh264', '-pix_fmt', 'yuv420p', str(mp4)], check=True, capture_output=True, text=True)
        tmp.unlink(missing_ok=True)
    else:
        tmp.replace(mp4)
    cv2.imwrite(str(out_dir / 'overlay_preview.png'), cv2.imread(str(frames_dir / '00155.png')))
    print(f'mug_3d_pose_render: {mp4}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-dir', type=Path, required=True)
    ap.add_argument('--fps', type=float, default=24.0)
    ap.add_argument('--render', action='store_true')
    args = ap.parse_args()
    obs_path = args.sample_dir / 'results' / 'object_observations' / 'object_observations.csv'
    rows = read_rows(obs_path)
    poses = build_pose(rows)
    out_dir = args.sample_dir / 'results' / 'mug_proxy_3d_pose'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'mug_proxy_3d_pose.csv'
    with out_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(poses[0].keys()))
        writer.writeheader(); writer.writerows(poses)
    print(f'mug_3d_pose_csv: {out_csv}')
    if args.render:
        render(args.sample_dir, poses, args.fps)

if __name__ == '__main__':
    main()
