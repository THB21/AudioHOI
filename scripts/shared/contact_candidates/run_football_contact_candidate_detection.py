#!/usr/bin/env python3
"""Build first-pass 2D foot/floor contact candidates for football scenes."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import torch
import smplx

LEFT_FOOT_IDS = [7, 10, 57, 58, 59]
RIGHT_FOOT_IDS = [8, 11, 60, 61, 62]
SUPPORT_IDS = [7, 8, 10, 11]


def read_ball_track(path: Path) -> list[dict[str, float | int | str]]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "frame": int(row["frame"]),
                    "time": float(row["time"]),
                    "ball_center_x": float(row["ball_center_x"]),
                    "ball_center_y": float(row["ball_center_y"]),
                    "radius": float(row["radius"]),
                    "mask_area": float(row.get("mask_area", 0.0) or 0.0),
                    "source": row.get("source", ""),
                }
            )
    if not rows:
        raise RuntimeError(f"No ball rows found in {path}")
    return rows


def read_human_result(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as f:
        data = pickle.load(f)
    params = data["smpl_params_incam"]
    return {
        "body_pose": np.asarray(params["body_pose"], dtype=np.float32),
        "betas": np.asarray(params["betas"], dtype=np.float32),
        "global_orient": np.asarray(params["global_orient"], dtype=np.float32),
        "transl": np.asarray(params["transl"], dtype=np.float32),
        "K_fullimg": np.asarray(data["K_fullimg"], dtype=np.float32),
    }


def build_body_joints(body_models_root: Path, human_params: dict[str, np.ndarray]) -> np.ndarray:
    model = smplx.create(
        str(body_models_root),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        batch_size=human_params["transl"].shape[0],
    )
    with torch.inference_mode():
        output = model(
            body_pose=torch.from_numpy(human_params["body_pose"]),
            betas=torch.from_numpy(human_params["betas"]),
            global_orient=torch.from_numpy(human_params["global_orient"]),
            transl=torch.from_numpy(human_params["transl"]),
            return_verts=False,
        )
    return output.joints.detach().cpu().numpy().astype(np.float64)


def project_points(points_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.clip(points_cam[:, 2], 1e-6, None)
    u = K[:, 0, 0] * (points_cam[:, 0] / z) + K[:, 0, 2]
    v = K[:, 1, 1] * (points_cam[:, 1] / z) + K[:, 1, 2]
    valid = points_cam[:, 2] > 1e-6
    return np.stack([u, v], axis=1), valid


def build_foot_centers(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = joints[:, LEFT_FOOT_IDS, :].mean(axis=1)
    right = joints[:, RIGHT_FOOT_IDS, :].mean(axis=1)
    return left, right


def compute_feet_support_v(joints: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = joints[:, SUPPORT_IDS, :]
    B = pts.shape[0]
    uv, valid = project_points(pts.reshape(-1, 3), np.repeat(K, len(SUPPORT_IDS), axis=0))
    uv = uv.reshape(B, len(SUPPORT_IDS), 2)
    valid = valid.reshape(B, len(SUPPORT_IDS))
    left_v = np.full(B, np.nan, dtype=np.float64)
    right_v = np.full(B, np.nan, dtype=np.float64)
    support_v = np.full(B, np.nan, dtype=np.float64)
    for i in range(B):
        lv = []
        rv = []
        if valid[i, 0]: lv.append(float(uv[i, 0, 1]))
        if valid[i, 2]: lv.append(float(uv[i, 2, 1]))
        if valid[i, 1]: rv.append(float(uv[i, 1, 1]))
        if valid[i, 3]: rv.append(float(uv[i, 3, 1]))
        if lv:
            left_v[i] = max(lv)
        if rv:
            right_v[i] = max(rv)
        vals = [v for v in [left_v[i], right_v[i]] if not np.isnan(v)]
        if vals:
            support_v[i] = max(vals)
    return support_v, left_v, right_v


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
        left_on = start - 1 >= 0 and out[start - 1]
        right_on = end < n and out[end]
        if left_on and right_on and (end - start) <= max_gap:
            out[start:end] = True
    return out


def build_transition_state(foot_state: np.ndarray, floor_state: np.ndarray, max_gap: int) -> np.ndarray:
    if max_gap <= 0:
        return np.zeros(len(foot_state), dtype=bool)
    transition = np.zeros(len(foot_state), dtype=bool)
    occupied = foot_state | floor_state
    n = len(occupied)
    i = 0
    while i < n:
        if occupied[i]:
            i += 1
            continue
        start = i
        while i < n and not occupied[i]:
            i += 1
        end = i
        gap = end - start
        if gap <= 0 or gap > max_gap:
            continue
        left_idx = start - 1
        right_idx = end if end < n else -1
        if left_idx < 0 or right_idx < 0:
            continue
        left_foot = bool(foot_state[left_idx])
        left_floor = bool(floor_state[left_idx])
        right_foot = bool(foot_state[right_idx])
        right_floor = bool(floor_state[right_idx])
        if (left_foot and right_floor) or (left_floor and right_foot):
            transition[start:end] = True
    return transition


def mask_to_intervals(frames, times, mask, label, target, score, peak_index=None):
    intervals = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i + 1 < n and mask[i + 1]:
            i += 1
        end = i
        seg_scores = score[start:end + 1]
        if peak_index is None:
            peak_rel = int(np.argmax(seg_scores))
            peak_idx = start + peak_rel
        else:
            peak_idx = int(peak_index[start:end + 1][0])
        target_counts = {}
        for name in target[start:end + 1]:
            target_counts[name] = target_counts.get(name, 0) + 1
        major_target = max(target_counts.items(), key=lambda kv: kv[1])[0]
        intervals.append({
            'contact_type': label,
            'target': major_target,
            'start_frame': int(frames[start]),
            'end_frame': int(frames[end]),
            'start_time': float(times[start]),
            'end_time': float(times[end]),
            'peak_frame': int(frames[peak_idx]),
            'peak_time': float(times[peak_idx]),
            'mean_score': float(np.mean(seg_scores)),
            'max_score': float(np.max(seg_scores)),
            'num_frames': int(end - start + 1),
        })
        i += 1
    return intervals


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build 2D football contact candidates from feet/floor cues.')
    parser.add_argument('--sample-dir', type=Path, default=Path('samples/football_10'))
    parser.add_argument('--body-model-root', type=Path, default=Path('third-party/GVHMR/inputs/checkpoints/body_models'))
    parser.add_argument('--out-dir', type=Path, default=None)
    parser.add_argument('--foot-dist-thresh-px', type=float, default=95.0)
    parser.add_argument('--foot-score-sigma-px', type=float, default=45.0)
    parser.add_argument('--foot-local-radius', type=int, default=2)
    parser.add_argument('--foot-state-dist-thresh-px', type=float, default=75.0)
    parser.add_argument('--foot-state-score-thresh', type=float, default=0.38)
    parser.add_argument('--foot-gap-bridge', type=int, default=2)
    parser.add_argument('--floor-gap-thresh-px', type=float, default=18.0)
    parser.add_argument('--floor-score-sigma-px', type=float, default=10.0)
    parser.add_argument('--floor-local-radius', type=int, default=2)
    parser.add_argument('--floor-state-gap-thresh-px', type=float, default=16.0)
    parser.add_argument('--floor-state-score-thresh', type=float, default=0.35)
    parser.add_argument('--floor-gap-bridge', type=int, default=1)
    parser.add_argument('--transition-gap-max', type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sample_dir = args.sample_dir
    results_dir = sample_dir / 'results'
    out_dir = args.out_dir or (results_dir / 'contact_candidates')
    out_dir.mkdir(parents=True, exist_ok=True)

    ball_rows = read_ball_track(results_dir / 'tracking' / 'ball_trajectory.csv')
    human = read_human_result(results_dir / 'gvhmr' / 'result.pkl')
    joints = build_body_joints(args.body_model_root, human)
    K = np.asarray(human['K_fullimg'], dtype=np.float64)
    if len(joints) < len(ball_rows):
        raise RuntimeError(f'GVHMR has fewer frames than ball tracking: {len(joints)} < {len(ball_rows)}')
    if len(joints) != len(ball_rows):
        print(f'Warning: truncating GVHMR from {len(joints)} to first {len(ball_rows)} frames to match ball tracking')
        joints = joints[:len(ball_rows)]
        K = K[:len(ball_rows)]
    left_foot_cam, right_foot_cam = build_foot_centers(joints)
    left_foot_uv, left_valid = project_points(left_foot_cam, K)
    right_foot_uv, right_valid = project_points(right_foot_cam, K)
    feet_support_v, left_support_v, right_support_v = compute_feet_support_v(joints, K)

    frames = np.asarray([int(r['frame']) for r in ball_rows], dtype=np.int32)
    times = np.asarray([float(r['time']) for r in ball_rows], dtype=np.float64)
    ball_u = np.asarray([float(r['ball_center_x']) for r in ball_rows], dtype=np.float64)
    ball_v = np.asarray([float(r['ball_center_y']) for r in ball_rows], dtype=np.float64)
    ball_r = np.asarray([float(r['radius']) for r in ball_rows], dtype=np.float64)
    ball_bottom_v = ball_v + ball_r

    left_dist = np.linalg.norm(np.stack([ball_u, ball_v], axis=1) - left_foot_uv, axis=1)
    right_dist = np.linalg.norm(np.stack([ball_u, ball_v], axis=1) - right_foot_uv, axis=1)
    left_dist = np.where(left_valid, left_dist, np.inf)
    right_dist = np.where(right_valid, right_dist, np.inf)
    active_is_left = left_dist <= right_dist
    active_foot = np.where(active_is_left, 'left', 'right')
    active_target = [f'{name}_foot' for name in active_foot]
    min_foot_dist = np.minimum(left_dist, right_dist)
    foot_score = gaussian_score(min_foot_dist, args.foot_score_sigma_px)
    foot_local_min = local_min_mask(min_foot_dist, args.foot_local_radius)
    is_foot_candidate = foot_local_min & (min_foot_dist <= args.foot_dist_thresh_px)

    floor_gap = np.abs(ball_bottom_v - feet_support_v)
    floor_gap = np.where(np.isnan(floor_gap), np.inf, floor_gap)
    floor_score = gaussian_score(floor_gap, args.floor_score_sigma_px)
    floor_local_peak = local_max_mask(ball_v, args.floor_local_radius)
    if len(ball_v) >= 3:
        vel = np.diff(ball_v)
        bounce_turn = np.zeros(len(ball_v), dtype=bool)
        bounce_turn[1:-1] = (vel[:-1] > 0.0) & (vel[1:] < 0.0)
    else:
        bounce_turn = np.zeros(len(ball_v), dtype=bool)
    is_floor_candidate = floor_local_peak & bounce_turn & (floor_gap <= args.floor_gap_thresh_px)

    foot_state = (min_foot_dist <= args.foot_state_dist_thresh_px) & (foot_score >= args.foot_state_score_thresh)
    foot_state = bridge_short_gaps(foot_state, args.foot_gap_bridge)
    floor_state = (floor_gap <= args.floor_state_gap_thresh_px) & (floor_score >= args.floor_state_score_thresh)
    floor_state = bridge_short_gaps(floor_state, args.floor_gap_bridge)
    transition_state = build_transition_state(foot_state, floor_state, args.transition_gap_max)

    foot_rows = []
    floor_rows = []
    frame_rows = []
    center_rows = []
    for i in range(len(frames)):
        foot_rows.append({
            'frame': int(frames[i]), 'time': float(times[i]),
            'ball_center_x': float(ball_u[i]), 'ball_center_y': float(ball_v[i]),
            'left_foot_u': float(left_foot_uv[i, 0]), 'left_foot_v': float(left_foot_uv[i, 1]),
            'right_foot_u': float(right_foot_uv[i, 0]), 'right_foot_v': float(right_foot_uv[i, 1]),
            'left_foot_valid': int(bool(left_valid[i])), 'right_foot_valid': int(bool(right_valid[i])),
            'left_foot_dist_px': float(left_dist[i]), 'right_foot_dist_px': float(right_dist[i]),
            'min_foot_dist_px': float(min_foot_dist[i]), 'active_foot': str(active_foot[i]),
            'foot_score': float(foot_score[i]), 'is_candidate': int(bool(is_foot_candidate[i])),
            'foot_contact_state': int(bool(foot_state[i])),
        })
        floor_rows.append({
            'frame': int(frames[i]), 'time': float(times[i]),
            'ball_center_y': float(ball_v[i]), 'ball_radius_px': float(ball_r[i]),
            'ball_bottom_v': float(ball_bottom_v[i]), 'feet_support_v': '' if np.isnan(feet_support_v[i]) else float(feet_support_v[i]),
            'left_support_v': '' if np.isnan(left_support_v[i]) else float(left_support_v[i]),
            'right_support_v': '' if np.isnan(right_support_v[i]) else float(right_support_v[i]),
            'floor_gap_px': float(floor_gap[i]), 'floor_score': float(floor_score[i]),
            'is_local_peak': int(bool(floor_local_peak[i])), 'is_bounce_turn': int(bool(bounce_turn[i])),
            'is_candidate': int(bool(is_floor_candidate[i])), 'floor_contact_state': int(bool(floor_state[i])),
        })
        frame_rows.append({
            'frame': int(frames[i]), 'time': float(times[i]), 'active_foot': str(active_foot[i]),
            'foot_score': float(foot_score[i]), 'floor_score': float(floor_score[i]),
            'foot_contact_state': int(bool(foot_state[i])), 'floor_contact_state': int(bool(floor_state[i])),
            'transition_contact_state': int(bool(transition_state[i])), 'multi_contact_state': int(bool(foot_state[i] and floor_state[i])),
        })
        if is_foot_candidate[i]:
            center_rows.append({'frame': int(frames[i]), 'time': float(times[i]), 'contact_type': 'foot_contact_event', 'target': active_target[i], 'score': float(foot_score[i]), 'confidence': float(foot_score[i]), 'source': 'rule2d_peak'})
        if is_floor_candidate[i]:
            center_rows.append({'frame': int(frames[i]), 'time': float(times[i]), 'contact_type': 'floor_contact_event', 'target': 'floor', 'score': float(floor_score[i]), 'confidence': float(floor_score[i]), 'source': 'rule2d_peak'})

    foot_event_index = np.arange(len(frames), dtype=np.int32)
    floor_event_index = np.arange(len(frames), dtype=np.int32)
    i = 0
    while i < len(frames):
        if not foot_state[i]:
            i += 1
            continue
        start = i
        while i + 1 < len(frames) and foot_state[i + 1]:
            i += 1
        end = i
        local_dist = min_foot_dist[start:end + 1]
        best_rel = int(np.argmin(local_dist))
        best_idx = start + best_rel
        foot_event_index[start:end + 1] = best_idx
        i += 1
    i = 0
    while i < len(frames):
        if not floor_state[i]:
            i += 1
            continue
        start = i
        while i + 1 < len(frames) and floor_state[i + 1]:
            i += 1
        end = i
        local_gap = floor_gap[start:end + 1]
        best_rel = int(np.argmin(local_gap))
        best_idx = start + best_rel
        floor_event_index[start:end + 1] = best_idx
        i += 1

    intervals = []
    intervals.extend(mask_to_intervals(frames, times, foot_state, 'foot_contact_state', active_target, foot_score, peak_index=foot_event_index))
    intervals.extend(mask_to_intervals(frames, times, floor_state, 'floor_contact_state', ['floor'] * len(frames), floor_score, peak_index=floor_event_index))
    transition_score = np.maximum(foot_score, floor_score)
    intervals.extend(mask_to_intervals(frames, times, transition_state, 'transition_contact_state', ['foot_floor_transition'] * len(frames), transition_score))
    intervals.sort(key=lambda r: (int(r['start_frame']), str(r['contact_type'])))

    foot_event_frames = {int(r['peak_frame']) for r in intervals if r['contact_type'] == 'foot_contact_state'}
    floor_event_frames = {int(r['peak_frame']) for r in intervals if r['contact_type'] == 'floor_contact_state'}
    center_rows = [r for r in center_rows if (r['contact_type'] != 'foot_contact_event' or int(r['frame']) in foot_event_frames) and (r['contact_type'] != 'floor_contact_event' or int(r['frame']) in floor_event_frames)]
    existing_foot_frames = {int(r['frame']) for r in center_rows if r['contact_type'] == 'foot_contact_event'}
    existing_floor_frames = {int(r['frame']) for r in center_rows if r['contact_type'] == 'floor_contact_event'}
    for r in intervals:
        peak_frame = int(r['peak_frame'])
        peak_idx = peak_frame - int(frames[0])
        if r['contact_type'] == 'foot_contact_state' and peak_frame not in existing_foot_frames:
            center_rows.append({'frame': peak_frame, 'time': float(r['peak_time']), 'contact_type': 'foot_contact_event', 'target': r['target'], 'score': float(foot_score[peak_idx]), 'confidence': float(foot_score[peak_idx]), 'source': 'interval_min_foot_dist'})
        elif r['contact_type'] == 'floor_contact_state' and peak_frame not in existing_floor_frames:
            center_rows.append({'frame': peak_frame, 'time': float(r['peak_time']), 'contact_type': 'floor_contact_event', 'target': 'floor', 'score': float(floor_score[peak_idx]), 'confidence': float(floor_score[peak_idx]), 'source': 'interval_min_gap'})
    center_rows.sort(key=lambda r: (int(r['frame']), str(r['contact_type'])))

    write_csv(out_dir / 'foot_contact_candidates.csv', foot_rows, ['frame','time','ball_center_x','ball_center_y','left_foot_u','left_foot_v','right_foot_u','right_foot_v','left_foot_valid','right_foot_valid','left_foot_dist_px','right_foot_dist_px','min_foot_dist_px','active_foot','foot_score','is_candidate','foot_contact_state'])
    write_csv(out_dir / 'floor_contact_candidates.csv', floor_rows, ['frame','time','ball_center_y','ball_radius_px','ball_bottom_v','feet_support_v','left_support_v','right_support_v','floor_gap_px','floor_score','is_local_peak','is_bounce_turn','is_candidate','floor_contact_state'])
    write_csv(out_dir / 'contact_state_frames.csv', frame_rows, ['frame','time','active_foot','foot_score','floor_score','foot_contact_state','floor_contact_state','transition_contact_state','multi_contact_state'])
    write_csv(out_dir / 'contact_intervals.csv', intervals, ['contact_type','target','start_frame','end_frame','start_time','end_time','peak_frame','peak_time','mean_score','max_score','num_frames'])
    write_csv(out_dir / 'contact_candidates_labeled.csv', center_rows, ['frame','time','contact_type','target','score','confidence','source'])

    print(f'Wrote {out_dir / "foot_contact_candidates.csv"}')
    print(f'Wrote {out_dir / "floor_contact_candidates.csv"}')
    print(f'Wrote {out_dir / "contact_state_frames.csv"}')
    print(f'Wrote {out_dir / "contact_intervals.csv"}')
    print(f'Wrote {out_dir / "contact_candidates_labeled.csv"}')
    print(f'Foot center candidates: {int(np.sum(is_foot_candidate))}')
    print(f'Floor center candidates: {int(np.sum(is_floor_candidate))}')
    print(f'Foot contact state frames: {int(np.sum(foot_state))}')
    print(f'Floor contact state frames: {int(np.sum(floor_state))}')
    print(f'Multi-contact frames: {int(np.sum(foot_state & floor_state))}')


if __name__ == '__main__':
    main()
