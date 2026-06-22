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
from object_proxy_observation_utils import read_object_mesh_tracks  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def row_center(row: dict[str, str]) -> np.ndarray:
    return np.array([base.ff(row, 'body_center_x', base.ff(row, 'center_x')), base.ff(row, 'body_center_y', base.ff(row, 'center_y'))], dtype=float)


def collect_manual_points(item: dict) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    rim = item.get('rim_ellipse') or {}
    bot = item.get('bottom_ellipse') or {}
    if 'center' in rim:
        out['rim_center'] = np.asarray(rim['center'], dtype=float)
    if 'center' in bot:
        out['bottom_center'] = np.asarray(bot['center'], dtype=float)
    for prefix, e in [('rim', rim), ('bottom', bot)]:
        axis = e.get('axis_points') or {}
        for name, pt in axis.items():
            out[f'{prefix}_{name}'] = np.asarray(pt, dtype=float)
    for i, pt in enumerate(rim.get('visible_arc_centerline_points') or []):
        out[f'rim_arc_{i:02d}'] = np.asarray(pt, dtype=float)
    return out


def bind_points_to_mesh(manual_ann: dict, mesh_tracks: dict[int, dict[str, np.ndarray]], max_dist_px: float, source_min_frame: int | None = None, source_max_frame: int | None = None) -> dict[str, dict]:
    bindings: dict[str, dict] = {}
    for key, item in sorted((manual_ann.get('frames') or {}).items(), key=lambda kv: int(kv[0])):
        fr = int(key)
        if source_min_frame is not None and fr < source_min_frame:
            continue
        if source_max_frame is not None and fr > source_max_frame:
            continue
        mesh = mesh_tracks.get(fr)
        if mesh is None:
            continue
        xy = np.asarray(mesh['xy'], dtype=float)
        ids = np.asarray(mesh['point_ids'], dtype=object)
        visible = np.asarray(mesh['visible'], dtype=bool)
        if len(xy) == 0:
            continue
        pts = collect_manual_points(item)
        for name, uv in pts.items():
            valid = visible & np.all(np.isfinite(xy), axis=1)
            if not np.any(valid):
                continue
            cand_xy = xy[valid]
            cand_ids = ids[valid]
            d = np.linalg.norm(cand_xy - uv[None, :], axis=1)
            j = int(np.argmin(d))
            if float(d[j]) > max_dist_px:
                continue
            # Prefer the closest binding for the same semantic point if multiple manual frames bind it.
            old = bindings.get(name)
            if old is None or float(d[j]) < float(old['bind_distance_px']):
                bindings[name] = {
                    'point_id': str(cand_ids[j]),
                    'source_frame': fr,
                    'source_uv': uv.tolist(),
                    'bind_uv': cand_xy[j].tolist(),
                    'bind_distance_px': float(d[j]),
                    'semantic_point': name,
                }
    return bindings


def point_from_id(mesh: dict[str, np.ndarray], point_id: str) -> tuple[np.ndarray | None, float]:
    ids = np.asarray(mesh['point_ids'], dtype=object)
    hits = np.flatnonzero(ids.astype(str) == point_id)
    if len(hits) == 0:
        return None, 0.0
    i = int(hits[0])
    visible = float(mesh['visible'][i]) if 'visible' in mesh else 1.0
    return np.asarray(mesh['xy'][i], dtype=float), visible


def fit_ellipse_or_center(points: list[np.ndarray]) -> dict:
    arr = np.asarray(points, dtype=np.float32)
    center = np.mean(arr, axis=0)
    if len(arr) >= 5:
        try:
            (cx, cy), (w, h), angle = cv2.fitEllipse(arr.reshape(-1, 1, 2))
            return {'center': [float(cx), float(cy)], 'rx': float(w / 2), 'ry': float(h / 2), 'angle_deg': float(angle)}
        except cv2.error:
            pass
    if len(arr) >= 2:
        cov = np.cov(arr.T)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        angle = math.degrees(math.atan2(float(vecs[1, 0]), float(vecs[0, 0])))
        return {'center': center.astype(float).tolist(), 'rx': float(max(2.0, 2.0 * math.sqrt(max(vals[0], 1e-6)))), 'ry': float(max(2.0, 2.0 * math.sqrt(max(vals[-1], 1e-6)))), 'angle_deg': float(angle)}
    return {'center': center.astype(float).tolist(), 'rx': 3.0, 'ry': 2.0, 'angle_deg': 0.0}


def propagate(sample: Path, start_frame: int, end_frame: int, max_dist_px: float) -> Path:
    manual_path = sample / 'annotations' / 'mug_rim_bottom_manual.json'
    manual_ann = json.loads(manual_path.read_text())
    mesh_tracks = read_object_mesh_tracks(sample / 'results' / 'tracking' / 'object_mesh_tracks_test.csv')
    bindings = bind_points_to_mesh(manual_ann, mesh_tracks, max_dist_px=max_dist_px, source_min_frame=start_frame, source_max_frame=end_frame)
    frames_out: dict[str, dict] = {}
    out_dir = sample / 'annotations' / 'mug_rim_bottom_mesh_track_propagation'
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_rows = {int(r['frame']): r for r in read_csv(sample / 'results' / 'object_observations' / 'object_observations.csv')}

    for fr in range(start_frame, end_frame + 1):
        mesh = mesh_tracks.get(fr)
        if mesh is None:
            continue
        rim_pts: list[np.ndarray] = []
        bot_pts: list[np.ndarray] = []
        point_records = {}
        for name, bind in bindings.items():
            uv, vis = point_from_id(mesh, bind['point_id'])
            if uv is None or vis <= 0.1:
                continue
            if not np.all(np.isfinite(uv)):
                continue
            # CoTracker points can drift onto the hand/background. Keep only points close to the current mug body box.
            row = obs_rows.get(fr, {})
            bx1 = base.ff(row, 'body_bbox_x1', base.ff(row, 'bbox_x1', math.nan))
            by1 = base.ff(row, 'body_bbox_y1', base.ff(row, 'bbox_y1', math.nan))
            bx2 = base.ff(row, 'body_bbox_x2', base.ff(row, 'bbox_x2', math.nan))
            by2 = base.ff(row, 'body_bbox_y2', base.ff(row, 'bbox_y2', math.nan))
            margin = 18.0
            if all(math.isfinite(v) for v in [bx1, by1, bx2, by2]):
                if uv[0] < bx1 - margin or uv[0] > bx2 + margin or uv[1] < by1 - margin or uv[1] > by2 + margin:
                    continue
            point_records[name] = {'uv': uv.tolist(), 'confidence': float(vis), 'point_id': bind['point_id'], 'source_frame': bind['source_frame']}
            if name.startswith('rim_'):
                rim_pts.append(uv)
            elif name.startswith('bottom_'):
                bot_pts.append(uv)
        if len(rim_pts) < 2 or len(bot_pts) < 2:
            continue
        rim = fit_ellipse_or_center(rim_pts)
        bot = fit_ellipse_or_center(bot_pts)
        frames_out[str(fr)] = {
            'label': f'mesh_track_rim_bottom_{fr}',
            'source': 'cotracker_mesh_point_id_propagation_from_manual_rim_bottom',
            'rim_ellipse': rim,
            'bottom_ellipse': bot,
            'tracked_points': point_records,
            'handle_visibility_policy': 'hidden',
            'body_axis': [rim['center'], bot['center']],
        }

        img = cv2.imread(str(sample / 'frames' / f'{fr:05d}.png'))
        if img is not None:
            for pt in rim_pts:
                cv2.circle(img, tuple(np.round(pt).astype(int)), 3, (255, 0, 255), -1, cv2.LINE_AA)
            for pt in bot_pts:
                cv2.circle(img, tuple(np.round(pt).astype(int)), 3, (0, 255, 255), -1, cv2.LINE_AA)
            for e, color in [(rim, (255, 0, 255)), (bot, (0, 255, 255))]:
                c = tuple(int(round(v)) for v in e['center'])
                axes = (max(1, int(round(e['rx']))), max(1, int(round(e['ry']))))
                cv2.ellipse(img, c, axes, float(e.get('angle_deg', 0.0)), 0, 360, color, 1, cv2.LINE_AA)
            ctr = row_center(obs_rows.get(fr, {})) if fr in obs_rows else np.array([math.nan, math.nan])
            if np.all(np.isfinite(ctr)):
                cv2.circle(img, tuple(np.round(ctr).astype(int)), 4, (255, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(img, f'mesh-rim-bottom frame {fr:03d} rim={len(rim_pts)} bottom={len(bot_pts)}', (18, 30), cv2.FONT_HERSHEY_SIMPLEX, .55, (20,20,20), 3, cv2.LINE_AA)
            cv2.putText(img, f'mesh-rim-bottom frame {fr:03d} rim={len(rim_pts)} bottom={len(bot_pts)}', (18, 30), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1, cv2.LINE_AA)
            cv2.imwrite(str(out_dir / f'frame_{fr:05d}_mesh_rim_bottom.png'), img)

    out = {
        'source': 'cotracker_mesh_point_id_propagation_from_manual_rim_bottom',
        'coordinate_system': 'full_frame_pixel_xy',
        'bindings': bindings,
        'frames': frames_out,
        'notes': 'Rim/bottom points are propagated by stable object_mesh_tracks_test.csv point_id. Review before using as optimizer input.',
    }
    out_json = out_dir / 'mug_rim_bottom_mesh_propagated.json'
    out_json.write_text(json.dumps(out, indent=2))
    summary = {'json': str(out_json), 'debug_dir': str(out_dir), 'bindings': len(bindings), 'frames': len(frames_out)}
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return out_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-dir', type=Path, required=True)
    ap.add_argument('--start-frame', type=int, default=60)
    ap.add_argument('--end-frame', type=int, default=100)
    ap.add_argument('--max-bind-dist-px', type=float, default=18.0)
    args = ap.parse_args()
    propagate(args.sample_dir, args.start_frame, args.end_frame, args.max_bind_dist_px)


if __name__ == '__main__':
    main()
