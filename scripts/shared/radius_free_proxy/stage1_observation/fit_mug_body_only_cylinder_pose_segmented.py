#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_mug_body_only_cylinder_pose as bodyfit  # noqa: E402
import fit_mug_articraft_keyframe_pose as base  # noqa: E402


def finalize(tmp: Path, out: Path) -> None:
    ffmpeg = shutil.which('ffmpeg') or '/usr/bin/ffmpeg'
    try:
        subprocess.run([ffmpeg, '-y', '-i', str(tmp), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out)], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        subprocess.run([ffmpeg, '-y', '-i', str(tmp), '-c:v', 'libopenh264', '-pix_fmt', 'yuv420p', str(out)], check=True, capture_output=True, text=True)
    tmp.unlink(missing_ok=True)


def render_segmented(sample: Path, frames: list[int], P: np.ndarray, fps: float) -> Path:
    out_dir = sample / 'results' / 'renders' / 'M5_body_only_cylinder_pose_segmented'
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
        bodyfit.draw_body(img, p, K_all[fr - 1], active=True)
        cv2.putText(img, f'M5_segmented_body_only frame {fr:03d}', (24, 36), cv2.FONT_HERSHEY_SIMPLEX, .7, (20,20,20), 3, cv2.LINE_AA)
        cv2.putText(img, f'M5_segmented_body_only frame {fr:03d}', (24, 36), cv2.FONT_HERSHEY_SIMPLEX, .7, (245,245,245), 1, cv2.LINE_AA)
        writer.write(img)
    writer.release()
    finalize(tmp, out)
    return out


def write_outputs(sample: Path, frames: list[int], P: np.ndarray, metas: list[dict], out: Path) -> None:
    proxy = sample / 'proxy'
    proxy.mkdir(exist_ok=True)
    rows = []
    for fr, p in zip(frames, P):
        rows.append({
            'frame': fr,
            'time': (fr - 1) / 24.0,
            'x': p[0], 'y': p[1], 'z': p[2],
            'yaw': base.wrap(p[3]), 'yaw_deg': np.rad2deg(base.wrap(p[3])),
            'pitch': p[4], 'pitch_deg': np.rad2deg(p[4]),
            'roll': p[5], 'roll_deg': np.rad2deg(p[5]),
            'scale': p[6],
        })
    csv_path = proxy / 'mug_body_only_cylinder_pose_segmented_sequence.csv'
    with csv_path.open('w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    fit_path = proxy / 'mug_body_only_cylinder_pose_segmented_fit.json'
    fit_path.write_text(json.dumps({'model': 'body_only_true_cylinder_segmented_fit', 'segments': metas, 'frames': rows}, indent=2))
    outputs = {'pose_sequence': str(csv_path), 'fit': str(fit_path), 'render': str(out), 'segments': metas}
    (proxy / 'mug_body_only_cylinder_pose_segmented_outputs.json').write_text(json.dumps(outputs, indent=2))
    print(json.dumps(outputs, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-dir', type=Path, required=True)
    ap.add_argument('--segments', type=str, default='1-60,60-100,100-160,160-240')
    ap.add_argument('--max-nfev', type=int, default=80)
    ap.add_argument('--fps', type=float, default=24.0)
    args = ap.parse_args()
    all_frames: list[int] = []
    all_params = []
    metas = []
    seen = set()
    for spec in args.segments.split(','):
        a, b = [int(x) for x in spec.split('-')]
        print(f'=== segment {a}-{b} ===', flush=True)
        frames, P, meta = bodyfit.fit_body_only(args.sample_dir, a, b, args.max_nfev)
        meta = dict(meta)
        meta['segment'] = spec
        meta['num_frames'] = len(frames)
        metas.append(meta)
        for fr, p in zip(frames, P):
            if fr in seen:
                continue
            seen.add(fr)
            all_frames.append(fr)
            all_params.append(p)
    order = np.argsort(np.asarray(all_frames))
    frames_sorted = [all_frames[int(i)] for i in order]
    P_sorted = np.vstack([all_params[int(i)] for i in order])
    out = render_segmented(args.sample_dir, frames_sorted, P_sorted, args.fps)
    write_outputs(args.sample_dir, frames_sorted, P_sorted, metas, out)


if __name__ == '__main__':
    main()
