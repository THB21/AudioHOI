#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import smplx
import torch

THIS_DIR = Path(__file__).resolve().parent
CONTACT_DIR = THIS_DIR.parent / "human_ball" / "contact"
if str(CONTACT_DIR) not in sys.path:
    sys.path.insert(0, str(CONTACT_DIR))

from contact_part_utils import (
    build_contact_identity,
    build_contact_part_centers,
    infer_default_part,
    normalize_contact_label,
)


def parse_float(row: dict[str, str], key: str, default: float = np.nan) -> float:
    value = row.get(key, '')
    if value is None or value == '':
        return default
    return float(value)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f'No rows found in {path}')
    return rows


def read_object_proxy_track(path: Path) -> list[dict[str, float | int | str]]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref_u = row.get('ref_u') or row.get('center_x')
            ref_v = row.get('ref_v') or row.get('center_y')
            support_u = row.get('support_u') or row.get('support_proxy_u') or ref_u
            support_v = row.get('support_v') or row.get('support_proxy_v') or row.get('lowest_visible_y') or row.get('bbox_y2')
            if not ref_u or not ref_v:
                continue
            rows.append({
                'frame': int(row['frame']),
                'time': float(row.get('time', 0.0) or 0.0),
                'ref_u': float(ref_u),
                'ref_v': float(ref_v),
                'support_u': float(support_u) if support_u else float(ref_u),
                'support_v': float(support_v) if support_v else float(ref_v),
                'observation_conf': float(row.get('observation_conf', 1.0) or 1.0),
            })
    if not rows:
        raise RuntimeError(f'No object proxy rows found in {path}')
    return rows


def read_object_mesh_tracks(path: Path) -> dict[int, dict[str, np.ndarray]]:
    by_frame = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row['frame'])
            by_frame.setdefault(frame, []).append(row)
    out = {}
    for frame, rows in by_frame.items():
        xy = []
        visible = []
        is_boundary = []
        point_ids = []
        for r in rows:
            x = r.get('x', '')
            y = r.get('y', '')
            if x == '' or y == '':
                continue
            xy.append([float(x), float(y)])
            visible.append(float(r.get('visible', 1.0) or 1.0) > 0.5)
            is_boundary.append(str(r.get('point_type', '')) == 'boundary')
            point_ids.append(str(r.get('point_id', '')))
        if xy:
            out[frame] = {
                'xy': np.asarray(xy, dtype=np.float64),
                'visible': np.asarray(visible, dtype=bool),
                'is_boundary': np.asarray(is_boundary, dtype=bool),
                'point_ids': np.asarray(point_ids, dtype=object),
            }
    if not out:
        raise RuntimeError(f'No object mesh tracks found in {path}')
    return out


def read_audio_events(path: Path) -> list[dict[str, float | int | str]]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('audio_frame'):
                continue
            rows.append({
                'audio_frame': int(float(row['audio_frame'])),
                'audio_time': float(row.get('audio_time', 0.0) or 0.0),
                'audio_score': float(row.get('audio_score', 0.0) or 0.0),
            })
    return rows


def build_audio_support(frames: np.ndarray, audio_rows: list[dict[str, float | int | str]], radius: int = 2) -> np.ndarray:
    support = np.zeros(len(frames), dtype=np.float64)
    frame_to_idx = {int(fr): i for i, fr in enumerate(frames.tolist())}
    for row in audio_rows:
        center = int(row['audio_frame'])
        score = float(row['audio_score'])
        for fr in range(center - radius, center + radius + 1):
            idx = frame_to_idx.get(fr)
            if idx is None:
                continue
            dist = abs(fr - center)
            weight = max(0.0, 1.0 - 0.25 * dist)
            support[idx] = max(support[idx], score * weight)
    return support


def read_human_result(path: Path) -> dict[str, np.ndarray]:
    with path.open('rb') as f:
        data = pickle.load(f)
    params = data['smpl_params_incam']
    return {
        'body_pose': np.asarray(params['body_pose'], dtype=np.float32),
        'betas': np.asarray(params['betas'], dtype=np.float32),
        'global_orient': np.asarray(params['global_orient'], dtype=np.float32),
        'transl': np.asarray(params['transl'], dtype=np.float32),
        'K_fullimg': np.asarray(data['K_fullimg'], dtype=np.float32),
    }


def build_body_joints(body_models_root: Path, human_params: dict[str, np.ndarray]) -> np.ndarray:
    model = smplx.create(
        str(body_models_root),
        model_type='smplx',
        gender='neutral',
        ext='npz',
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        batch_size=human_params['transl'].shape[0],
    )
    with torch.inference_mode():
        output = model(
            body_pose=torch.from_numpy(human_params['body_pose']),
            betas=torch.from_numpy(human_params['betas']),
            global_orient=torch.from_numpy(human_params['global_orient']),
            transl=torch.from_numpy(human_params['transl']),
            return_verts=False,
        )
    return output.joints.detach().cpu().numpy().astype(np.float64)


def project_points(points_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.clip(points_cam[:, 2], 1e-6, None)
    u = K[:, 0, 0] * (points_cam[:, 0] / z) + K[:, 0, 2]
    v = K[:, 1, 1] * (points_cam[:, 1] / z) + K[:, 1, 2]
    valid = points_cam[:, 2] > 1e-6
    return np.stack([u, v], axis=1), valid


def gaussian_score(x: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)


def local_min_mask(values: np.ndarray, radius: int) -> np.ndarray:
    mask = np.zeros(len(values), dtype=bool)
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        if values[i] <= np.min(values[lo:hi]):
            mask[i] = True
    return mask


def local_max_mask(values: np.ndarray, radius: int) -> np.ndarray:
    mask = np.zeros(len(values), dtype=bool)
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        if values[i] >= np.max(values[lo:hi]):
            mask[i] = True
    return mask


def bridge_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    if max_gap <= 0:
        return mask.copy()
    out = mask.copy()
    n = len(mask)
    i = 0
    while i < n:
        if out[i]:
            i += 1
            continue
        start = i
        while i < n and not out[i]:
            i += 1
        end = i
        gap_len = end - start
        if 0 < gap_len <= max_gap and start > 0 and end < n and out[start - 1] and out[end]:
            out[start:end] = True
    return out


def nearest_object_points(anchor_uv, anchor_valid, frames, object_mesh_by_frame, prefer_boundary=True):
    n = len(frames)
    nearest_dist = np.full(n, np.inf, dtype=np.float64)
    nearest_uv = np.full((n, 2), np.nan, dtype=np.float64)
    nearest_point_id = [''] * n
    nearest_valid = np.zeros(n, dtype=bool)
    for i, frame in enumerate(frames.tolist()):
        if not anchor_valid[i]:
            continue
        mesh = object_mesh_by_frame.get(int(frame))
        if mesh is None:
            continue
        pts = mesh['xy']
        valid = mesh['visible'].copy()
        if prefer_boundary:
            boundary_valid = valid & mesh['is_boundary']
            if np.any(boundary_valid):
                valid = boundary_valid
        if not np.any(valid):
            continue
        candidates = pts[valid]
        candidate_ids = mesh['point_ids'][valid]
        dists = np.linalg.norm(candidates - anchor_uv[i][None, :], axis=1)
        j = int(np.argmin(dists))
        nearest_dist[i] = float(dists[j])
        nearest_uv[i] = candidates[j]
        nearest_point_id[i] = str(candidate_ids[j])
        nearest_valid[i] = True
    return {'dist': nearest_dist, 'uv': nearest_uv, 'point_id': nearest_point_id, 'valid': nearest_valid}


def detect_object_anchor_contact(*, frames, left_contact_uv, right_contact_uv, left_valid, right_valid, object_mesh_by_frame, dist_thresh_px, score_sigma_px, local_radius, state_dist_thresh_px, state_score_thresh, gap_bridge, audio_support=None):
    left_nn = nearest_object_points(left_contact_uv, left_valid, frames, object_mesh_by_frame, prefer_boundary=True)
    right_nn = nearest_object_points(right_contact_uv, right_valid, frames, object_mesh_by_frame, prefer_boundary=True)
    left_dist = np.asarray(left_nn['dist'], dtype=np.float64)
    right_dist = np.asarray(right_nn['dist'], dtype=np.float64)
    active_is_left = left_dist <= right_dist
    min_contact_gap = np.minimum(left_dist, right_dist)
    if audio_support is None:
        audio_support = np.zeros_like(min_contact_gap)
    else:
        audio_support = np.asarray(audio_support, dtype=np.float64)
    proximity_score = gaussian_score(min_contact_gap, score_sigma_px)
    object_response_score = np.zeros_like(min_contact_gap)
    contact_score = 0.55 * proximity_score + 0.15 * object_response_score + 0.30 * audio_support
    contact_local_min = local_min_mask(min_contact_gap, local_radius)
    is_candidate = (min_contact_gap <= dist_thresh_px) & contact_local_min
    state = (min_contact_gap <= state_dist_thresh_px) & (contact_score >= state_score_thresh)
    state = bridge_short_gaps(state, gap_bridge)
    left_ids = list(left_nn['point_id'])
    right_ids = list(right_nn['point_id'])
    left_uv = np.asarray(left_nn['uv'], dtype=np.float64)
    right_uv = np.asarray(right_nn['uv'], dtype=np.float64)
    active_point_id = []
    active_object_u = np.full(len(frames), np.nan, dtype=np.float64)
    active_object_v = np.full(len(frames), np.nan, dtype=np.float64)
    active_contact = np.where(active_is_left, 'left', 'right')
    for i in range(len(frames)):
        if active_is_left[i]:
            active_point_id.append(left_ids[i])
            active_object_u[i] = left_uv[i, 0]
            active_object_v[i] = left_uv[i, 1]
        else:
            active_point_id.append(right_ids[i])
            active_object_u[i] = right_uv[i, 0]
            active_object_v[i] = right_uv[i, 1]
    return {
        'left_dist': left_dist,
        'right_dist': right_dist,
        'active_contact': active_contact.tolist(),
        'min_contact_gap': min_contact_gap,
        'proximity_score': proximity_score,
        'object_response_score': object_response_score,
        'audio_support': audio_support,
        'score': contact_score,
        'candidate': is_candidate,
        'state': state,
        'active_object_point_id': active_point_id,
        'active_object_u': active_object_u,
        'active_object_v': active_object_v,
    }


def detect_object_support_contact(*, object_support_v, support_v, gap_thresh_px, score_sigma_px, local_radius, state_gap_thresh_px, state_score_thresh, gap_bridge):
    support_gap = np.abs(object_support_v - support_v)
    support_gap = np.where(np.isnan(support_gap), np.inf, support_gap)
    support_score = gaussian_score(support_gap, score_sigma_px)
    support_local_peak = local_max_mask(object_support_v, local_radius)
    is_candidate = support_local_peak & (support_gap <= gap_thresh_px)
    state = (support_gap <= state_gap_thresh_px) & (support_score >= state_score_thresh)
    state = bridge_short_gaps(state, gap_bridge)
    return {
        'object_support_v': object_support_v,
        'gap': support_gap,
        'score': support_score,
        'local_peak': support_local_peak,
        'candidate': is_candidate,
        'state': state,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Experimental radius-free object contact candidate detection.')
    parser.add_argument('--sample-dir', type=Path, required=True)
    parser.add_argument('--object-proxy-observation-csv', type=Path, default=None)
    parser.add_argument('--object-mesh-csv', type=Path, default=None)
    parser.add_argument('--out-subdir', type=str, default='contact_candidates_object_proxy_test')
    parser.add_argument('--body-model-root', type=Path, default=Path('third-party/GVHMR/inputs/checkpoints/body_models'))
    parser.add_argument('--contact-dist-thresh-px', type=float, default=28.0)
    parser.add_argument('--contact-score-sigma-px', type=float, default=18.0)
    parser.add_argument('--contact-local-radius', type=int, default=2)
    parser.add_argument('--contact-state-dist-thresh-px', type=float, default=38.0)
    parser.add_argument('--contact-state-score-thresh', type=float, default=0.35)
    parser.add_argument('--contact-gap-bridge', type=int, default=2)
    parser.add_argument('--floor-gap-thresh-px', type=float, default=18.0)
    parser.add_argument('--floor-score-sigma-px', type=float, default=10.0)
    parser.add_argument('--floor-local-radius', type=int, default=2)
    parser.add_argument('--floor-state-gap-thresh-px', type=float, default=24.0)
    parser.add_argument('--floor-state-score-thresh', type=float, default=0.40)
    parser.add_argument('--floor-gap-bridge', type=int, default=2)
    args = parser.parse_args()

    sample_dir = args.sample_dir
    results_dir = sample_dir / 'results'
    out_dir = results_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    object_rows = read_object_proxy_track(args.object_proxy_observation_csv or (results_dir / 'object_proxy_observations_test' / 'object_proxy_observations.csv'))
    object_mesh_by_frame = read_object_mesh_tracks(args.object_mesh_csv or (results_dir / 'tracking' / 'object_mesh_tracks_test.csv'))
    audio_rows = read_audio_events(results_dir / 'events' / 'audio_visual_alignment.csv')
    state_rows_existing = read_rows(results_dir / 'contact_candidates' / 'contact_state_frames.csv')
    event_rows_existing = read_rows(results_dir / 'contact_candidates' / 'contact_candidates_labeled.csv')

    human = read_human_result(results_dir / 'gvhmr' / 'result.pkl')
    joints = build_body_joints(args.body_model_root, human)
    K = np.asarray(human['K_fullimg'], dtype=np.float64)
    if len(joints) < len(object_rows):
        raise RuntimeError(f'GVHMR has fewer frames than object proxy rows: {len(joints)} < {len(object_rows)}')
    joints = joints[:len(object_rows)]
    K = K[:len(object_rows)]

    frames = np.asarray([int(r['frame']) for r in object_rows], dtype=np.int32)
    times = np.asarray([float(r['time']) for r in object_rows], dtype=np.float64)
    object_support_v = np.asarray([float(r['support_v']) for r in object_rows], dtype=np.float64)
    audio_support = build_audio_support(frames, audio_rows, radius=2)

    default_part = infer_default_part([*state_rows_existing, *event_rows_existing], fallback='hand')
    state_by_frame = {int(r['frame']): r for r in state_rows_existing}
    labels = [normalize_contact_label(state_by_frame.get(int(fr), {}), default_part=default_part, fallback_side='right') for fr in frames]
    centers = build_contact_part_centers(joints)
    left_contact_cam = centers['left_hand'] if default_part == 'hand' else centers['left_foot']
    right_contact_cam = centers['right_hand'] if default_part == 'hand' else centers['right_foot']
    left_contact_uv, left_valid = project_points(left_contact_cam, K)
    right_contact_uv, right_valid = project_points(right_contact_cam, K)

    anchor_det = detect_object_anchor_contact(
        frames=frames,
        left_contact_uv=left_contact_uv,
        right_contact_uv=right_contact_uv,
        left_valid=left_valid,
        right_valid=right_valid,
        object_mesh_by_frame=object_mesh_by_frame,
        dist_thresh_px=args.contact_dist_thresh_px,
        score_sigma_px=args.contact_score_sigma_px,
        local_radius=args.contact_local_radius,
        state_dist_thresh_px=args.contact_state_dist_thresh_px,
        state_score_thresh=args.contact_state_score_thresh,
        gap_bridge=args.contact_gap_bridge,
        audio_support=audio_support,
    )
    support_v = float(np.nanmedian(object_support_v[np.isfinite(object_support_v)]))
    support_v_arr = np.full(len(frames), support_v, dtype=np.float64)
    floor_det = detect_object_support_contact(
        object_support_v=object_support_v,
        support_v=support_v_arr,
        gap_thresh_px=args.floor_gap_thresh_px,
        score_sigma_px=args.floor_score_sigma_px,
        local_radius=args.floor_local_radius,
        state_gap_thresh_px=args.floor_state_gap_thresh_px,
        state_score_thresh=args.floor_state_score_thresh,
        gap_bridge=args.floor_gap_bridge,
    )

    anchor_rows = []
    floor_rows = []
    state_rows = []
    labeled_rows = []
    intervals = []
    current_interval = None
    for i, frame in enumerate(frames.tolist()):
        side = 'left' if anchor_det['active_contact'][i] == 'left' else 'right'
        active_label = f'{side}_{default_part}'
        contact_part, contact_side, contact_label = build_contact_identity(active_label=active_label, event_on=bool(anchor_det['candidate'][i]), floor_event_on=bool(floor_det['candidate'][i]), default_part=default_part)
        anchor_rows.append({'frame': frame, 'time': f'{times[i]:.6f}', 'active_side': side, 'active_object_point_id': anchor_det['active_object_point_id'][i], 'active_object_u': f"{anchor_det['active_object_u'][i]:.3f}" if np.isfinite(anchor_det['active_object_u'][i]) else '', 'active_object_v': f"{anchor_det['active_object_v'][i]:.3f}" if np.isfinite(anchor_det['active_object_v'][i]) else '', 'min_object_boundary_gap_px': f"{anchor_det['min_contact_gap'][i]:.6f}", 'anchor_score': f"{anchor_det['score'][i]:.6f}", 'is_candidate': int(bool(anchor_det['candidate'][i])), 'anchor_contact_state': int(bool(anchor_det['state'][i]))})
        floor_rows.append({'frame': frame, 'time': f'{times[i]:.6f}', 'object_support_v': f"{floor_det['object_support_v'][i]:.3f}", 'support_v': f'{support_v_arr[i]:.3f}', 'gap': f"{floor_det['gap'][i]:.6f}", 'floor_score': f"{floor_det['score'][i]:.6f}", 'is_candidate': int(bool(floor_det['candidate'][i])), 'floor_contact_state': int(bool(floor_det['state'][i]))})
        state_rows.append({'frame': frame, 'time': f'{times[i]:.6f}', 'anchor_type': default_part, 'contact_part': contact_part, 'contact_side': contact_side, 'contact_label': contact_label, 'anchor_score': f"{anchor_det['score'][i]:.6f}", 'floor_score': f"{floor_det['score'][i]:.6f}", 'anchor_contact_state': int(bool(anchor_det['state'][i])), 'human_contact_state': int(bool(anchor_det['state'][i])), 'floor_contact_state': int(bool(floor_det['state'][i])), 'transition_contact_state': 0, 'multi_contact_state': int(bool(anchor_det['state'][i]) and bool(floor_det['state'][i])), 'active_object_point_id': anchor_det['active_object_point_id'][i], 'active_object_u': f"{anchor_det['active_object_u'][i]:.3f}" if np.isfinite(anchor_det['active_object_u'][i]) else '', 'active_object_v': f"{anchor_det['active_object_v'][i]:.3f}" if np.isfinite(anchor_det['active_object_v'][i]) else '', 'min_object_boundary_gap_px': f"{anchor_det['min_contact_gap'][i]:.6f}"})
        if bool(anchor_det['candidate'][i]):
            labeled_rows.append({'frame': frame, 'time': f'{times[i]:.6f}', 'contact_type': 'anchor_contact_event', 'anchor_type': default_part, 'target': contact_label, 'score': f"{anchor_det['score'][i]:.6f}", 'confidence': f"{anchor_det['score'][i]:.6f}", 'source': 'object_mesh_boundary'})
        if bool(floor_det['candidate'][i]):
            labeled_rows.append({'frame': frame, 'time': f'{times[i]:.6f}', 'contact_type': 'floor_contact_event', 'anchor_type': 'floor', 'target': 'floor', 'score': f"{floor_det['score'][i]:.6f}", 'confidence': f"{floor_det['score'][i]:.6f}", 'source': 'support_proxy'})
        active = bool(anchor_det['state'][i]) or bool(floor_det['state'][i])
        if active and current_interval is None:
            current_interval = {'start_frame': frame, 'end_frame': frame}
        elif active and current_interval is not None:
            current_interval['end_frame'] = frame
        elif (not active) and current_interval is not None:
            intervals.append(current_interval)
            current_interval = None
    if current_interval is not None:
        intervals.append(current_interval)

    write_csv(out_dir / 'anchor_contact_candidates.csv', anchor_rows, ['frame','time','active_side','active_object_point_id','active_object_u','active_object_v','min_object_boundary_gap_px','anchor_score','is_candidate','anchor_contact_state'])
    write_csv(out_dir / 'floor_contact_candidates.csv', floor_rows, ['frame','time','object_support_v','support_v','gap','floor_score','is_candidate','floor_contact_state'])
    write_csv(out_dir / 'contact_state_frames.csv', state_rows, ['frame','time','anchor_type','contact_part','contact_side','contact_label','anchor_score','floor_score','anchor_contact_state','human_contact_state','floor_contact_state','transition_contact_state','multi_contact_state','active_object_point_id','active_object_u','active_object_v','min_object_boundary_gap_px'])
    write_csv(out_dir / 'contact_candidates_labeled.csv', labeled_rows, ['frame','time','contact_type','anchor_type','target','score','confidence','source'])
    write_csv(out_dir / 'contact_intervals.csv', intervals, ['start_frame','end_frame'])
    print(f'contact_candidate_dir: {out_dir}')


if __name__ == '__main__':
    main()
