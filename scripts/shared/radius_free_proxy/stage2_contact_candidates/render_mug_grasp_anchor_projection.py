#!/usr/bin/env python3
"""Render/evaluate the stable mug grasp anchor projected through the M14 pose.

This is the first step after VLM keyframe contact labeling. It does not refit the
mug pose. It answers a narrower question: given the current mug pose + phase and
the stable object-local grasp anchor, where does that anchor land in the image,
and how far is it from the active hand proxy?
"""

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

STAGE1 = Path(__file__).resolve().parents[1] / 'stage1_observation'
sys.path.insert(0, str(STAGE1))
import fit_mug_articraft_keyframe_pose as base  # noqa: E402
import render_mug_articraft_joint_contact_phase as m14  # noqa: E402
import render_mug_articraft_rigid_mesh_vlm as rigid  # noqa: E402

FIELDS = [
    'frame', 'time', 'frame_mode',
    'stable_grasp_source_frame', 'stable_grasp_object_part', 'stable_grasp_object_region',
    'stable_grasp_hand_side', 'stable_grasp_primary_finger', 'stable_grasp_type',
    'anchor_local_x', 'anchor_local_y', 'anchor_local_z',
    'anchor_u', 'anchor_v', 'anchor_z',
    'active_label', 'active_part_u', 'active_part_v', 'active_label_conf',
    'hand_anchor_dist_px', 'residual_weight', 'use_as_attachment_residual', 'reason',
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def ff(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, '')
        if value == '':
            return default
        return float(value)
    except Exception:
        return default


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y'}


def read_phase_sequence(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in read_csv(path):
        fr = int(float(row['frame']))
        out[fr] = ff(row, 'mug_axial_phase_rad', ff(row, 'handle_phase_rad', 0.0))
    return out


def residual_policy(state: dict[str, str], dist_px: float, conf: float) -> tuple[float, int, str]:
    mode = state.get('frame_mode', '')
    has_anchor = state.get('stable_grasp_local_x', '') != ''
    if not has_anchor:
        return 0.0, 0, 'no stable grasp anchor exists yet'
    if state.get('stable_grasp_object_part') != 'handle':
        return 0.0, 0, 'stable grasp is not handle semantic region'
    if not np.isfinite(dist_px):
        return 0.0, 0, 'active hand uv missing'
    if mode == 'direct_grasp_anchor':
        return 1.0, 1, 'VISIBLE_CONFIRMED_CONTACT: update stable upper_handle grasp anchor'
    if mode == 'rim_contact_keep_previous_grasp_anchor':
        return 0.45, 1, 'DRINKING_OCCLUDED_HAND_CONTACT: rim/mouth visible, keep previous hand-handle grasp; do not update anchor'
    if mode == 'keep_previous_grasp_anchor':
        # Hidden/misaligned frames are still continuous grasp during the mug
        # contact interval. The visible keyframes define the object-local
        # handle contact region; occluded frames should carry that same anchor
        # forward instead of detecting a new body/rim point.
        weight = 0.85 if dist_px < 90.0 and conf >= 0.15 else 0.45
        return weight, 1, 'OCCLUDED_CONTINUOUS_CONTACT: keep previous upper_handle grasp; still physical attachment'
    return 0.0, 0, 'mode does not use attachment residual'


def contact_state_label(mode: str) -> str:
    if mode == 'direct_grasp_anchor':
        return 'CONTACT=VISIBLE_CONFIRMED_UPDATE_ANCHOR'
    if mode == 'keep_previous_grasp_anchor':
        return 'CONTACT=OCCLUDED_CONTINUOUS_KEEP_ANCHOR'
    if mode == 'rim_contact_keep_previous_grasp_anchor':
        return 'CONTACT=RIM_VISIBLE_HAND_GRASP_OCCLUDED_KEEP_ANCHOR'
    return 'CONTACT=INACTIVE'


def draw_circle(img: np.ndarray, uv: tuple[float, float], color: tuple[int, int, int], radius: int, thick: int = 2) -> None:
    if not (np.isfinite(uv[0]) and np.isfinite(uv[1])):
        return
    cv2.circle(img, (int(round(uv[0])), int(round(uv[1]))), radius, color, thick, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-dir', type=Path, default=Path('samples_known_object/02_mug'))
    ap.add_argument('--pose-csv', type=Path, default=None)
    ap.add_argument('--phase-csv', type=Path, default=None)
    ap.add_argument('--state-csv', type=Path, default=None)
    ap.add_argument('--contact-csv', type=Path, default=None)
    ap.add_argument('--fps', type=float, default=24.0)
    ap.add_argument('--mesh-root', type=Path, default=Path('samples_known_object/02_mug/articraft/materialized_mug_mesh'))
    ap.add_argument('--out-dir', type=Path, default=None)
    args = ap.parse_args()

    sample = args.sample_dir
    default_static_pose = sample / 'proxy' / 'mug_body_only_cylinder_pose_table_static_sequence.csv'
    pose_csv = args.pose_csv or (default_static_pose if default_static_pose.exists() else (sample / 'proxy' / 'mug_body_only_cylinder_pose_segmented_sequence.csv'))
    phase_csv = args.phase_csv or (sample / 'results' / 'renders' / 'M14_joint_contact_handle_phase' / 'handle_phase_joint_contact.csv')
    state_csv = args.state_csv or (sample / 'results' / 'mug_grasp_anchor_state' / 'mug_grasp_anchor_state.csv')
    contact_csv = args.contact_csv or (sample / 'results' / 'mug_articraft_contact_points' / 'mug_articraft_contact_points.csv')
    out_dir = args.out_dir or (sample / 'results' / 'renders' / 'M15_grasp_anchor_projection')
    out_dir.mkdir(parents=True, exist_ok=True)

    poses = m14.read_pose_sequence(pose_csv)
    phases = read_phase_sequence(phase_csv)
    states = {int(float(r['frame'])): r for r in read_csv(state_csv)}
    contacts = {int(float(r['frame'])): r for r in read_csv(contact_csv)}
    K_all = base.load_K(sample)
    meshes = rigid.load_articraft_meshes(args.mesh_root)

    rows: list[dict[str, str]] = []
    frames = sorted(set(poses) & set(states))

    first = cv2.imread(str(sample / 'frames' / '00001.png'))
    if first is None:
        raise FileNotFoundError(sample / 'frames' / '00001.png')
    h, w = first.shape[:2]
    tmp = out_dir / 'overlay_tmp.mp4'
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))

    for fr in frames:
        state = states[fr]
        contact = contacts.get(fr, {})
        img = cv2.imread(str(sample / 'frames' / f'{fr:05d}.png'))
        if img is None:
            continue
        overlay = img.copy()
        phase = phases.get(fr, 0.0)
        # Draw the complete Articraft mug body first: body, rim, bottom, and handle.
        # M15 then overlays the stable grasp anchor on top of the full mug.
        for name in ['body_shell', 'rim_ring', 'bottom_disk', 'handle_loop']:
            if name not in meshes:
                continue
            color, thick = m14.PART_STYLE[name]
            verts, edges = meshes[name]
            m14.draw_mesh_edges(overlay, poses[fr], K_all[fr - 1], verts, edges, color, thick, phase)
        cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
        overlay = img.copy()
        local = np.array([
            ff(state, 'stable_grasp_local_x'),
            ff(state, 'stable_grasp_local_y'),
            ff(state, 'stable_grasp_local_z'),
        ], dtype=float)
        anchor_cam = np.full(3, math.nan)
        anchor_uv = np.full(2, math.nan)
        if np.all(np.isfinite(local)):
            cam, uv = m14.project_mesh(poses[fr], K_all[fr - 1], local[None, :], phase)
            anchor_cam = cam[0]
            anchor_uv = uv[0]

        hu = ff(contact, 'active_part_u')
        hv = ff(contact, 'active_part_v')
        conf = ff(contact, 'active_label_conf', 0.0)
        dist = float(np.linalg.norm(anchor_uv - np.array([hu, hv]))) if np.all(np.isfinite(anchor_uv)) and np.isfinite(hu) and np.isfinite(hv) else math.nan
        weight, use_res, reason = residual_policy(state, dist, conf)

        # Cyan: projected stable object-local anchor. Yellow: active hand proxy.
        draw_circle(overlay, (anchor_uv[0], anchor_uv[1]), (255, 220, 0), 12, 3)
        draw_circle(overlay, (hu, hv), (0, 220, 255), 9, 2)
        if np.all(np.isfinite(anchor_uv)) and np.isfinite(hu) and np.isfinite(hv):
            cv2.line(overlay, tuple(np.round(anchor_uv).astype(int)), (int(round(hu)), int(round(hv))), (255, 255, 255), 1, cv2.LINE_AA)
        color = (60, 220, 60) if use_res else (80, 80, 220)
        cv2.putText(overlay, f'M15 stable grasp frame {fr:03d} mode={state.get("frame_mode", "")}', (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(overlay, f'M15 stable grasp frame {fr:03d} mode={state.get("frame_mode", "")}', (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
        label2 = f'{contact_state_label(state.get("frame_mode", ""))}'
        cv2.putText(overlay, label2, (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(overlay, label2, (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.66, color, 2, cv2.LINE_AA)
        label3 = f'{state.get("stable_grasp_hand_side", "")}/{state.get("stable_grasp_primary_finger", "")} -> {state.get("stable_grasp_object_region", "")}  d={dist:.1f}px w={weight:.2f}'
        cv2.putText(overlay, label3, (18, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(overlay, label3, (18, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        writer.write(overlay)

        rows.append({
            'frame': str(fr),
            'time': state.get('time', ''),
            'frame_mode': state.get('frame_mode', ''),
            'stable_grasp_source_frame': state.get('stable_grasp_source_frame', ''),
            'stable_grasp_object_part': state.get('stable_grasp_object_part', ''),
            'stable_grasp_object_region': state.get('stable_grasp_object_region', ''),
            'stable_grasp_hand_side': state.get('stable_grasp_hand_side', ''),
            'stable_grasp_primary_finger': state.get('stable_grasp_primary_finger', ''),
            'stable_grasp_type': state.get('stable_grasp_type', ''),
            'anchor_local_x': state.get('stable_grasp_local_x', ''),
            'anchor_local_y': state.get('stable_grasp_local_y', ''),
            'anchor_local_z': state.get('stable_grasp_local_z', ''),
            'anchor_u': f'{anchor_uv[0]:.3f}' if np.isfinite(anchor_uv[0]) else '',
            'anchor_v': f'{anchor_uv[1]:.3f}' if np.isfinite(anchor_uv[1]) else '',
            'anchor_z': f'{anchor_cam[2]:.6f}' if np.isfinite(anchor_cam[2]) else '',
            'active_label': contact.get('active_label', state.get('active_label', '')),
            'active_part_u': f'{hu:.3f}' if np.isfinite(hu) else '',
            'active_part_v': f'{hv:.3f}' if np.isfinite(hv) else '',
            'active_label_conf': f'{conf:.6f}' if np.isfinite(conf) else '',
            'hand_anchor_dist_px': f'{dist:.3f}' if np.isfinite(dist) else '',
            'residual_weight': f'{weight:.3f}',
            'use_as_attachment_residual': str(use_res),
            'reason': reason,
        })

    writer.release()
    out_mp4 = out_dir / 'overlay.mp4'
    h264_tmp = out_dir / 'overlay_h264_tmp.mp4'
    cmd = ['ffmpeg', '-y', '-i', str(tmp), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(h264_tmp)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.move(str(h264_tmp), str(out_mp4))
        tmp.unlink(missing_ok=True)
    except Exception:
        shutil.move(str(tmp), str(out_mp4))

    out_csv = out_dir / 'grasp_anchor_projection.csv'
    with out_csv.open('w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader(); wr.writerows(rows)

    dists = [float(r['hand_anchor_dist_px']) for r in rows if r['hand_anchor_dist_px']]
    used = [r for r in rows if r['use_as_attachment_residual'] == '1']
    summary = {
        'pose_csv': str(pose_csv),
        'phase_csv': str(phase_csv),
        'state_csv': str(state_csv),
        'contact_csv': str(contact_csv),
        'mesh_root': str(args.mesh_root),
        'csv': str(out_csv),
        'render': str(out_mp4),
        'num_frames': len(rows),
        'num_residual_frames': len(used),
        'mean_hand_anchor_dist_px': float(np.mean(dists)) if dists else None,
        'median_hand_anchor_dist_px': float(np.median(dists)) if dists else None,
        'note': 'M15 diagnostic only: renders full Articraft mug body/handle, projects stable object-local upper_handle grasp anchor through M14 pose/phase, and compares it to active hand proxy. Hidden frames are treated as occluded continuous grasp that carries the previous handle anchor; M15 does not refit pose yet.',
    }
    (out_dir / 'outputs.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
