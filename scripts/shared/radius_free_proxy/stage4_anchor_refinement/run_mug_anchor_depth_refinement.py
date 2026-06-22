#!/usr/bin/env python3
"""Mug-specific GVHMR hand-depth anchor interpolation.

This is the mug/Articraft counterpart of the ball anchorinterp stage.  It keeps
M14's fitted 2D/canonical mug pose, but refines the camera-depth trajectory with
continuous hand-mug contact anchors from candidate frames and GVHMR hand depth.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
STAGE1_DIR = THIS_DIR.parent / 'stage1_observation'
for path in (REPO_ROOT, STAGE1_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fit_mug_articraft_keyframe_pose as base  # noqa: E402
import render_mug_articraft_rigid_mesh_vlm as rigid  # noqa: E402
from scripts.shared.radius_free_proxy.stage1_observation.object_proxy_observation_utils import (  # noqa: E402
    read_human_result,
    build_body_joints,
)
from scripts.shared.human_ball.contact.contact_part_utils import build_contact_part_centers  # noqa: E402


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f'No rows in {path}')
    return rows


def ff(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, '')
    if value in ('', None):
        return default
    try:
        return float(value)
    except Exception:
        return default


def ii(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, '')
    if value in ('', None):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def read_pose(path: Path) -> list[dict[str, float | int]]:
    rows = []
    for row in read_rows(path):
        rows.append({
            'frame': int(float(row['frame'])),
            'time': float(row.get('time', 0.0) or 0.0),
            'x': ff(row, 'x'),
            'y': ff(row, 'y'),
            'z': ff(row, 'z'),
            'yaw': ff(row, 'yaw'),
            'yaw_deg': ff(row, 'yaw_deg', math.degrees(ff(row, 'yaw', 0.0))),
            'pitch': ff(row, 'pitch'),
            'pitch_deg': ff(row, 'pitch_deg', math.degrees(ff(row, 'pitch', 0.0))),
            'roll': ff(row, 'roll'),
            'roll_deg': ff(row, 'roll_deg', math.degrees(ff(row, 'roll', 0.0))),
            'scale': ff(row, 'scale', 1.0),
        })
    return rows


def read_phase(path: Path, col: str | None = None) -> dict[int, float]:
    out = {}
    for row in read_rows(path):
        fr = int(float(row['frame']))
        if col and col in row:
            out[fr] = ff(row, col, 0.0)
        elif 'phase_rad' in row:
            out[fr] = ff(row, 'phase_rad', 0.0)
        elif 'phase' in row:
            val = ff(row, 'phase', 0.0)
            out[fr] = math.radians(val) if abs(val) > 2 * math.pi else val
        else:
            out[fr] = math.radians(ff(row, 'phase_deg', 0.0))
    return out


def pose_params(row: dict[str, float | int]) -> np.ndarray:
    return np.array([row['x'], row['y'], row['z'], row['yaw'], row['pitch'], row['roll'], row['scale']], dtype=float)


def center_uv(params: np.ndarray, K: np.ndarray) -> tuple[float, float]:
    z = max(float(params[2]), 1e-6)
    return float(K[0, 0] * params[0] / z + K[0, 2]), float(K[1, 1] * params[1] / z + K[1, 2])


def xyz_from_uvz(u: np.ndarray, v: np.ndarray, z: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return x, y


def finite3(vals: list[float]) -> bool:
    return all(np.isfinite(vals))


def robust_offset(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    # Physical hand-mug contact gap should be <8cm; larger values indicate proxy tracking error.
    return float(np.clip(value, -0.08, 0.08))


def choose_hand_label(grasp_row: dict[str, str], state_row: dict[str, str] | None) -> str | None:
    for key in ('stable_grasp_hand_side', 'active_label'):
        label = str(grasp_row.get(key, '') or '').strip()
        if label in {'left_hand', 'right_hand'}:
            return label
    if state_row:
        label = str(state_row.get('contact_label', '') or '').strip()
        if label in {'left_hand', 'right_hand'}:
            return label
    return None


def find_table_static_start(state_by_frame: dict[int, dict[str, str]], min_run: int = 10) -> int | None:
    frames = sorted(state_by_frame)
    run: list[int] = []
    for fr in frames:
        row = state_by_frame[fr]
        floor_on = ii(row, 'floor_contact_state', 0) == 1
        support_conf = ff(row, 'support_conf', 0.0)
        if floor_on and support_conf >= 0.65:
            run.append(fr)
            if len(run) >= min_run:
                return run[0]
        else:
            run = []
    return None


def build_anchor_targets(
    pose_rows: list[dict[str, float | int]],
    phases: dict[int, float],
    grasp_by_frame: dict[int, dict[str, str]],
    state_by_frame: dict[int, dict[str, str]],
    part_centers: dict[str, np.ndarray],
) -> dict[int, dict[str, object]]:
    anchors: dict[int, dict[str, object]] = {}
    for row in pose_rows:
        fr = int(row['frame'])
        g = grasp_by_frame.get(fr)
        if not g:
            continue
        state = state_by_frame.get(fr)
        mode = str(g.get('frame_mode', '') or '')
        stable = [ff(g, 'stable_grasp_local_x'), ff(g, 'stable_grasp_local_y'), ff(g, 'stable_grasp_local_z')]
        if not finite3(stable):
            continue
        human_state = ii(state, 'human_contact_state', 0) if state else 0
        anchor_state = ii(state, 'anchor_contact_state', 0) if state else 0
        # This is deliberately continuous: visible keyframes create/update the object-local
        # anchor; hidden/occluded frames still constrain depth by carrying the same local point.
        continuous_grasp = human_state == 1 or anchor_state == 1 or ii(g, 'use_previous_grasp_for_hand_attachment', 0) == 1 or ii(g, 'use_this_point_for_hand_attachment', 0) == 1
        if not continuous_grasp:
            continue
        label = choose_hand_label(g, state)
        if not label or label not in part_centers:
            continue
        idx = fr - 1
        if idx < 0 or idx >= len(part_centers[label]):
            continue
        hand_xyz = part_centers[label][idx]
        hand_x, hand_y, hand_z = float(hand_xyz[0]), float(hand_xyz[1]), float(hand_xyz[2])
        if not np.isfinite(hand_z) or hand_z <= 0.2:
            continue
        p = pose_params(row)
        phase = phases.get(fr, 0.0)
        local = np.asarray(stable, dtype=float)[None, :] @ rigid.rot_y(phase).T
        anchor_cam = base.transform(p, local)[0]
        local_delta_z = float(anchor_cam[2] - p[2])
        local_delta_x = float(anchor_cam[0] - p[0])
        local_delta_y = float(anchor_cam[1] - p[1])
        offset = robust_offset(ff(state, 'contact_depth_offset_m', 0.0) if state else ff(g, 'hand_object_z_gap_m', 0.0))
        target_z = hand_z - offset - local_delta_z
        # XY target: mug center = hand position - local offset from mug center to grasp point
        target_x = hand_x - local_delta_x
        target_y = hand_y - local_delta_y
        conf = max(ff(g, 'stable_grasp_conf', 0.25), ff(state, 'anchor_score', 0.0) if state else 0.0, 0.20)
        if ii(g, 'use_this_point_for_hand_attachment', 0) == 1:
            kind = 'confirmed_update'
            weight = 3.0 * conf
        elif ii(g, 'use_previous_grasp_for_hand_attachment', 0) == 1 or 'keep' in mode:
            kind = 'continuous_keep'
            weight = 1.4 * conf
        else:
            kind = 'candidate_continuous'
            weight = 1.0 * conf
        anchors[fr] = {
            'target_z': float(target_z),
            'target_x': float(target_x),
            'target_y': float(target_y),
            'hand_z': hand_z,
            'hand_x': hand_x,
            'hand_y': hand_y,
            'local_delta_z': local_delta_z,
            'local_delta_x': local_delta_x,
            'local_delta_y': local_delta_y,
            'offset': offset,
            'label': label,
            'mode': mode,
            'weight': float(np.clip(weight, 0.25, 4.0)),
            'kind': kind,
            'human_state': human_state,
            'floor_state': ii(state, 'floor_contact_state', 0) if state else 0,
            'support_conf': ff(state, 'support_conf', 0.0) if state else 0.0,
        }
    return anchors


def interp_reference(z_init: np.ndarray, anchors: dict[int, dict[str, object]], frames: list[int]) -> np.ndarray:
    z_ref = z_init.copy()
    idx_by_frame = {fr: i for i, fr in enumerate(frames)}
    anchor_idx = [idx_by_frame[fr] for fr in sorted(anchors) if fr in idx_by_frame and np.isfinite(float(anchors[fr]['target_z']))]
    if not anchor_idx:
        return z_ref
    first = anchor_idx[0]
    last = anchor_idx[-1]
    z_ref[:first + 1] = float(anchors[frames[first]]['target_z'])
    for a, b in zip(anchor_idx[:-1], anchor_idx[1:]):
        za = float(anchors[frames[a]]['target_z'])
        zb = float(anchors[frames[b]]['target_z'])
        alpha = np.linspace(0.0, 1.0, b - a + 1)
        z_ref[a:b + 1] = (1 - alpha) * za + alpha * zb
    z_ref[last:] = float(anchors[frames[last]]['target_z'])
    return np.maximum(z_ref, 0.30)


def optimize_z(z_init: np.ndarray, z_ref: np.ndarray, frames: list[int], anchors: dict[int, dict[str, object]], table_start: int | None) -> np.ndarray:
    idx_by_frame = {fr: i for i, fr in enumerate(frames)}
    anchor_indices = [(idx_by_frame[fr], anchors[fr]) for fr in sorted(anchors) if fr in idx_by_frame]
    static_idx = idx_by_frame.get(table_start) if table_start in idx_by_frame else None

    def residual(z: np.ndarray) -> np.ndarray:
        out: list[float] = []
        out.extend(((z - z_ref) * 0.45).tolist())
        for idx, a in anchor_indices:
            out.append((z[idx] - float(a['target_z'])) * float(a['weight']))
        if len(z) > 1:
            out.extend((np.diff(z) * 4.0).tolist())
        if len(z) > 2:
            out.extend((np.diff(z, n=2) * 12.0).tolist())
        if static_idx is not None:
            z_static = z[static_idx]
            for j in range(static_idx, len(z)):
                out.append((z[j] - z_static) * 20.0)
        return np.asarray(out, dtype=float)

    res = least_squares(residual, z_ref, loss='soft_l1', f_scale=0.03, max_nfev=300)
    return np.maximum(res.x, 0.30)


def optimize_xy(
    xy_init: np.ndarray,
    xy_ref: np.ndarray,
    frames: list[int],
    anchors: dict[int, dict[str, object]],
    table_start: int | None,
    anchor_key: str,
) -> np.ndarray:
    """Smooth + anchor optimization for a single camera-space XY axis."""
    idx_by_frame = {fr: i for i, fr in enumerate(frames)}
    anchor_indices = [
        (idx_by_frame[fr], anchors[fr])
        for fr in sorted(anchors)
        if fr in idx_by_frame and anchor_key in anchors[fr] and np.isfinite(float(anchors[fr][anchor_key]))
    ]
    static_idx = idx_by_frame.get(table_start) if table_start and table_start in idx_by_frame else None

    def residual(xy: np.ndarray) -> np.ndarray:
        out: list[float] = []
        out.extend(((xy - xy_ref) * 0.30).tolist())
        for idx, a in anchor_indices:
            out.append((xy[idx] - float(a[anchor_key])) * float(a['weight']) * 0.75)
        if len(xy) > 1:
            out.extend((np.diff(xy) * 3.0).tolist())
        if len(xy) > 2:
            out.extend((np.diff(xy, n=2) * 8.0).tolist())
        if static_idx is not None:
            xy_static = xy[static_idx]
            for j in range(static_idx, len(xy)):
                out.append((xy[j] - xy_static) * 15.0)
        return np.asarray(out, dtype=float)

    res = least_squares(residual, xy_ref.copy(), loss='soft_l1', f_scale=0.04, max_nfev=300)
    return res.x


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-dir', type=Path, default=Path('samples_known_object/02_mug'))
    ap.add_argument('--pose-csv', type=Path, default=None)
    ap.add_argument('--phase-csv', type=Path, default=None)
    ap.add_argument('--grasp-state-csv', type=Path, default=None)
    ap.add_argument('--contact-state-csv', type=Path, default=None)
    ap.add_argument('--body-model-root', type=Path, default=Path('third-party/GVHMR/inputs/checkpoints/body_models'))
    ap.add_argument('--phase-col', type=str, default=None, help='Column name for phase in phase CSV (default: auto-detect phase_rad/phase/phase_deg).')
    ap.add_argument('--out-csv', type=Path, default=None)
    ap.add_argument('--out-dir', type=Path, default=None)
    args = ap.parse_args()

    sample = args.sample_dir
    pose_csv = args.pose_csv or (sample / 'proxy' / 'mug_body_only_cylinder_pose_table_static_sequence.csv')
    if not pose_csv.exists():
        pose_csv = sample / 'proxy' / 'mug_body_only_cylinder_pose_segmented_sequence.csv'
    phase_csv = args.phase_csv or (sample / 'results' / 'renders' / 'M17_phase_corrected' / 'corrected_handle_phase.csv')
    if not phase_csv.exists():
        phase_csv = sample / 'results' / 'renders' / 'M14_joint_contact_handle_phase' / 'handle_phase_joint_contact.csv'
    grasp_csv = args.grasp_state_csv or (sample / 'results' / 'mug_grasp_anchor_state' / 'mug_grasp_anchor_state.csv')
    state_csv = args.contact_state_csv or (sample / 'results' / 'contact_candidates_object_proxy' / 'contact_state_frames.csv')
    out_csv = args.out_csv or (sample / 'proxy' / 'mug_body_only_cylinder_pose_anchor_depth_sequence.csv')
    out_dir = args.out_dir or (sample / 'results' / 'pose6d_mug_articraft_anchor_depth')

    pose_rows = read_pose(pose_csv)
    # Auto-detect phase column: prefer explicit --phase-col, then check CSV header for
    # m17_phase_rad (M17 corrected output), then fall back to generic auto-detect.
    if args.phase_col:
        phase_col = args.phase_col
    else:
        with phase_csv.open() as _f:
            _header = next(csv.reader(_f))
        phase_col = 'm17_phase_rad' if 'm17_phase_rad' in _header else None
    phases = read_phase(phase_csv, col=phase_col)
    grasp_by_frame = {int(float(r['frame'])): r for r in read_rows(grasp_csv)}
    state_by_frame = {int(float(r['frame'])): r for r in read_rows(state_csv)}

    human = read_human_result(sample / 'results' / 'gvhmr' / 'result.pkl')
    K_all = np.asarray(human['K_fullimg'], dtype=float)
    K = K_all[0] if K_all.ndim == 3 else K_all
    joints = build_body_joints(args.body_model_root, human)
    part_centers = build_contact_part_centers(joints)

    frames = [int(r['frame']) for r in pose_rows]
    z_init = np.asarray([float(r['z']) for r in pose_rows], dtype=float)
    x_init = np.asarray([float(r['x']) for r in pose_rows], dtype=float)
    y_init = np.asarray([float(r['y']) for r in pose_rows], dtype=float)

    anchors = build_anchor_targets(pose_rows, phases, grasp_by_frame, state_by_frame, part_centers)
    table_start = find_table_static_start(state_by_frame)

    # Step 1: anchor depth (Z)
    z_ref = interp_reference(z_init, anchors, frames)
    z_final = optimize_z(z_init, z_ref, frames, anchors, table_start)

    # Step 2: anchor XY — reference is the original proxy XY re-projected with the new depth
    u0 = np.asarray([center_uv(pose_params(r), K)[0] for r in pose_rows], dtype=float)
    v0 = np.asarray([center_uv(pose_params(r), K)[1] for r in pose_rows], dtype=float)
    x_ref, y_ref = xyz_from_uvz(u0, v0, z_final, K)
    x_final = optimize_xy(x_init, x_ref, frames, anchors, table_start, 'target_x')
    y_final = optimize_xy(y_init, y_ref, frames, anchors, table_start, 'target_y')

    out_rows = []
    diag_rows = []
    for i, r in enumerate(pose_rows):
        fr = int(r['frame'])
        out = dict(r)
        out['x'] = f'{x_final[i]:.9f}'
        out['y'] = f'{y_final[i]:.9f}'
        out['z'] = f'{z_final[i]:.9f}'
        out_rows.append(out)
        a = anchors.get(fr, {})
        diag_rows.append({
            'frame': fr,
            'time': f"{float(r['time']):.6f}",
            'z_init': f'{z_init[i]:.9f}',
            'z_ref': f'{z_ref[i]:.9f}',
            'z_final': f'{z_final[i]:.9f}',
            'anchor_used': int(fr in anchors),
            'anchor_kind': a.get('kind', ''),
            'anchor_weight': f"{float(a.get('weight', 0.0)):.6f}" if a else '',
            'target_center_z': f"{float(a.get('target_z', math.nan)):.9f}" if a else '',
            'hand_z': f"{float(a.get('hand_z', math.nan)):.9f}" if a else '',
            'local_delta_z': f"{float(a.get('local_delta_z', math.nan)):.9f}" if a else '',
            'contact_depth_offset_m': f"{float(a.get('offset', math.nan)):.9f}" if a else '',
            'contact_label': a.get('label', ''),
            'frame_mode': a.get('mode', ''),
            'human_contact_state': a.get('human_state', ''),
            'floor_contact_state': a.get('floor_state', ''),
            'support_conf': a.get('support_conf', ''),
            'table_static_start_frame': table_start or '',
        })

    fields = ['frame', 'time', 'x', 'y', 'z', 'yaw', 'yaw_deg', 'pitch', 'pitch_deg', 'roll', 'roll_deg', 'scale']
    write_csv(out_csv, out_rows, fields)
    diag_csv = out_dir / 'anchor_depth_diagnostics.csv'
    diag_fields = list(diag_rows[0].keys())
    write_csv(diag_csv, diag_rows, diag_fields)
    summary = {
        'pose_input_csv': str(pose_csv),
        'phase_csv': str(phase_csv),
        'phase_col': phase_col or 'auto',
        'grasp_state_csv': str(grasp_csv),
        'contact_state_csv': str(state_csv),
        'output_pose_csv': str(out_csv),
        'diagnostics_csv': str(diag_csv),
        'num_frames': len(frames),
        'num_anchor_frames': len(anchors),
        'table_static_start_frame': table_start,
        'z_init_range': [float(np.nanmin(z_init)), float(np.nanmax(z_init))],
        'z_final_range': [float(np.nanmin(z_final)), float(np.nanmax(z_final))],
        'policy': 'continuous hand-mug candidate frames use GVHMR hand depth; stable object-local grasp point is transformed by mug body pose plus Articraft handle phase; post-table-support frames are depth-frozen.',
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'anchor_depth_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out_dir / 'anchor_depth_summary.txt').write_text('\n'.join(f'{k}: {v}' for k, v in summary.items()) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
