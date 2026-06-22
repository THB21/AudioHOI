#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_mug_articraft_keyframe_pose as base  # noqa: E402
import fit_mug_body_only_cylinder_pose as bodyfit  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def ff(row: dict[str, str], key: str, default: float = math.nan) -> float:
    return base.ff(row, key, default)


def load_pose_sequence(path: Path) -> dict[int, np.ndarray]:
    rows = read_csv(path)
    out = {}
    for r in rows:
        fr = int(float(r['frame']))
        out[fr] = np.array([
            ff(r, 'x'), ff(r, 'y'), ff(r, 'z'), ff(r, 'yaw'), ff(r, 'pitch'), ff(r, 'roll'), ff(r, 'scale')
        ], dtype=float)
    return out


def ellipse_fields(item: dict | None, name: str) -> tuple[float, float, float, float, float]:
    if not item or name not in item:
        return (math.nan, math.nan, math.nan, math.nan, math.nan)
    e = item[name]
    c = e.get('center') or [math.nan, math.nan]
    return (float(c[0]), float(c[1]), float(e.get('rx', math.nan)), float(e.get('ry', math.nan)), float(e.get('angle_deg', math.nan)))


def mean_dt_error(dt: np.ndarray, pts: np.ndarray) -> float:
    vals = [base.dt_at(dt, q) for q in pts if np.all(np.isfinite(q))]
    if not vals:
        return math.nan
    return float(np.mean(vals))


def draw_observation(img: np.ndarray, eo: dict) -> None:
    item = eo.get('manual') or {}
    for key, color in [('rim_ellipse', (255, 0, 255)), ('bottom_ellipse', (0, 255, 255))]:
        e = item.get(key)
        if not e:
            continue
        center = tuple(int(round(v)) for v in e['center'])
        axes = (max(1, int(round(e['rx']))), max(1, int(round(e['ry']))))
        angle = float(e.get('angle_deg', 0.0))
        cv2.ellipse(img, center, axes, angle, 0, 360, color, 2, cv2.LINE_AA)
        cv2.circle(img, center, 3, color, -1, cv2.LINE_AA)
    if not item:
        cv2.circle(img, tuple(np.round(eo['top_center']).astype(int)), 5, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(np.round(eo['bottom_center']).astype(int)), 5, (0, 255, 255), -1, cv2.LINE_AA)


def draw_poly(img: np.ndarray, uv: np.ndarray, color: tuple[int, int, int], thick: int, closed: bool) -> None:
    if np.all(np.isfinite(uv)):
        cv2.polylines(img, [np.round(uv).astype(np.int32)], closed, color, thick, cv2.LINE_AA)


def audit(sample: Path, start_frame: int, end_frame: int) -> Path:
    obs_rows = {int(r['frame']): r for r in read_csv(sample / 'results' / 'object_observations' / 'object_observations.csv')}
    frames = [fr for fr in range(start_frame, end_frame + 1) if fr in obs_rows]
    pose_path = sample / 'proxy' / 'mug_body_only_cylinder_pose_sequence.csv'
    poses = load_pose_sequence(pose_path)
    K_all = base.load_K(sample)
    edge_obs = bodyfit.build_edge_obs(sample, frames, obs_rows)
    manual_point_obs = bodyfit.load_manual_point_observations(sample)
    top_pts, bot_pts, body_pts = bodyfit.cylinder_points()

    out_dir = sample / 'results' / 'renders' / 'M5_body_only_cylinder_pose' / 'audit'
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fr in frames:
        row = obs_rows[fr]
        p = poses.get(fr)
        eo = edge_obs.get(fr)
        if p is None or eo is None:
            continue
        K = K_all[fr - 1]
        img = cv2.imread(str(sample / 'frames' / f'{fr:05d}.png'))
        if img is None:
            continue
        body_bbox = [ff(row, 'body_bbox_x1', ff(row, 'bbox_x1')), ff(row, 'body_bbox_y1', ff(row, 'bbox_y1')), ff(row, 'body_bbox_x2', ff(row, 'bbox_x2')), ff(row, 'body_bbox_y2', ff(row, 'bbox_y2'))]
        body_ctr = np.array([ff(row, 'body_center_x', ff(row, 'center_x')), ff(row, 'body_center_y', ff(row, 'center_y'))], dtype=float)
        top_uv = bodyfit.project_pts(p, top_pts, K)
        bot_uv = bodyfit.project_pts(p, bot_pts, K)
        body_uv = bodyfit.project_pts(p, body_pts, K)
        top_center_pred = bodyfit.project_pts(p, np.array([[0.0, -bodyfit.BODY_H / 2, 0.0]], dtype=float), K)[0]
        bot_center_pred = bodyfit.project_pts(p, np.array([[0.0, bodyfit.BODY_H / 2, 0.0]], dtype=float), K)[0]
        body_center_pred = bodyfit.project_pts(p, np.array([[0.0, 0.0, 0.0]], dtype=float), K)[0]
        rim_center_error = float(np.linalg.norm(top_center_pred - eo['top_center']))
        bottom_center_error = float(np.linalg.norm(bot_center_pred - eo['bottom_center']))
        rim_dt_error = mean_dt_error(eo['top_dt'], top_uv[::3])
        bottom_dt_error = mean_dt_error(eo['bottom_dt'], bot_uv[::3])
        body_center_error = float(np.linalg.norm(body_center_pred - body_ctr))
        point_obs = manual_point_obs.get(fr, {})
        point_errors = []
        theta = np.linspace(0, 2 * np.pi, 144, endpoint=False)
        rim_ring = np.stack([bodyfit.BODY_R * np.cos(theta), np.full_like(theta, -bodyfit.BODY_H / 2), bodyfit.BODY_D * np.sin(theta)], axis=1)
        bottom_ring = np.stack([bodyfit.BODY_R * np.cos(theta), np.full_like(theta, bodyfit.BODY_H / 2), bodyfit.BODY_D * np.sin(theta)], axis=1)
        rim_uv_ring = bodyfit.project_pts(p, rim_ring, K)
        bottom_uv_ring = bodyfit.project_pts(p, bottom_ring, K)
        for name, uv_obs in point_obs.items():
            ring = rim_uv_ring if name.startswith('rim_') else bottom_uv_ring if name.startswith('bottom_') else None
            if ring is None:
                continue
            d = ring - uv_obs[None, :]
            point_errors.append(float(np.sqrt(np.min(np.sum(d * d, axis=1)))))
        manual_point_mean_error = float(np.mean(point_errors)) if point_errors else math.nan
        manual_point_max_error = float(np.max(point_errors)) if point_errors else math.nan
        pred_bbox = [float(np.min(body_uv[:, 0])), float(np.min(body_uv[:, 1])), float(np.max(body_uv[:, 0])), float(np.max(body_uv[:, 1]))]
        bbox_error = float(np.mean(np.abs(np.asarray(pred_bbox) - np.asarray(body_bbox))))
        item = eo.get('manual') or {}
        rim_u, rim_v, rim_rx, rim_ry, rim_angle = ellipse_fields(item, 'rim_ellipse')
        bot_u, bot_v, bot_rx, bot_ry, bot_angle = ellipse_fields(item, 'bottom_ellipse')
        rows.append({
            'frame': fr,
            'time': (fr - 1) / 24.0,
            'source_label': eo.get('source', ''),
            'rim_obs_center_u': rim_u if np.isfinite(rim_u) else float(eo['top_center'][0]),
            'rim_obs_center_v': rim_v if np.isfinite(rim_v) else float(eo['top_center'][1]),
            'bottom_obs_center_u': bot_u if np.isfinite(bot_u) else float(eo['bottom_center'][0]),
            'bottom_obs_center_v': bot_v if np.isfinite(bot_v) else float(eo['bottom_center'][1]),
            'rim_rx': rim_rx,
            'rim_ry': rim_ry,
            'rim_angle_deg': rim_angle,
            'bottom_rx': bot_rx,
            'bottom_ry': bot_ry,
            'bottom_angle_deg': bot_angle,
            'fit_reproj_error_px': 0.5 * (rim_center_error + bottom_center_error),
            'fit_rim_center_error_px': rim_center_error,
            'fit_bottom_center_error_px': bottom_center_error,
            'fit_rim_dt_error_px': rim_dt_error,
            'fit_bottom_dt_error_px': bottom_dt_error,
            'body_center_error_px': body_center_error,
            'body_bbox_mean_abs_error_px': bbox_error,
            'manual_point_count': len(point_errors),
            'manual_point_mean_error_px': manual_point_mean_error,
            'manual_point_max_error_px': manual_point_max_error,
            'fit_yaw_deg': float(np.rad2deg(base.wrap(p[3]))),
            'fit_pitch_deg': float(np.rad2deg(p[4])),
            'fit_roll_deg': float(np.rad2deg(p[5])),
            'fit_scale': float(p[6]),
        })

        vis = img.copy()
        draw_observation(vis, eo)
        for name, uv in point_obs.items():
            cv2.circle(vis, tuple(np.round(uv).astype(int)), 5, (255, 128, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, name.replace('_or_visible_mid',''), tuple(np.round(uv + np.array([4, -4])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,128,0), 1, cv2.LINE_AA)
        cv2.rectangle(vis, (int(body_bbox[0]), int(body_bbox[1])), (int(body_bbox[2]), int(body_bbox[3])), (255, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(vis, tuple(np.round(body_ctr).astype(int)), 4, (255, 255, 0), -1, cv2.LINE_AA)
        draw_poly(vis, top_uv, (0, 0, 255), 2, True)
        draw_poly(vis, bot_uv, (0, 180, 0), 2, True)
        draw_poly(vis, np.array([top_center_pred, bot_center_pred]), (255, 255, 255), 1, False)
        cv2.circle(vis, tuple(np.round(top_center_pred).astype(int)), 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, tuple(np.round(bot_center_pred).astype(int)), 4, (0, 180, 0), -1, cv2.LINE_AA)
        text = f"frame {fr:03d} src={eo.get('source','')} rim={rim_center_error:.1f}px bottom={bottom_center_error:.1f}px dt={rim_dt_error:.1f}/{bottom_dt_error:.1f}"
        cv2.putText(vis, text, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(vis, text, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f'frame_{fr:05d}_audit.png'), vis)

    csv_path = out_dir / 'm5_body_observation_audit.csv'
    with csv_path.open('w', newline='') as f:
        if rows:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader(); wr.writerows(rows)
    summary = {
        'frames': len(rows),
        'csv': str(csv_path),
        'debug_dir': str(out_dir),
        'note': 'Observation audit only. Magenta/yellow are observed rim/bottom; red/green are M5 projected rim/bottom.',
    }
    (out_dir / 'm5_body_observation_audit_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return csv_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-dir', type=Path, required=True)
    ap.add_argument('--start-frame', type=int, default=60)
    ap.add_argument('--end-frame', type=int, default=100)
    args = ap.parse_args()
    audit(args.sample_dir, args.start_frame, args.end_frame)


if __name__ == '__main__':
    main()
