#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.shared.generic_contact_pipeline.components.contact.ball_contact_candidate_utils import (
    LEFT_HAND_KEY,
    RIGHT_HAND_KEY,
    LEFT_FOOT_KEY,
    RIGHT_FOOT_KEY,
    build_audio_support,
    build_body_joints,
    build_contact_part_centers,
    build_support_hypotheses,
    detect_object_anchor_contact_from_feet,
    detect_object_anchor_contact_from_hands,
    detect_object_anchor_contact_from_active,
    detect_object_support_contact,
    parse_float,
    project_points,
    read_audio_events,
    read_human_result,
    read_object_mesh_tracks,
    read_object_proxy_track,
    select_support_surface,
    split_active_label,
    bridge_short_gaps,
    local_min_mask,
    local_max_mask,
    write_csv,
)


def dilate_frames(frames: set[int], valid_frames: list[int], radius: int) -> set[int]:
    valid = set(valid_frames)
    out = set()
    for t in frames:
        for dt in range(-radius, radius + 1):
            tt = t + dt
            if tt in valid:
                out.add(tt)
    return out


def low_value_score(x: np.ndarray, p_good: float, p_bad: float) -> np.ndarray:
    s = (p_bad - x) / max(p_bad - p_good, 1e-6)
    return np.clip(s, 0.0, 1.0)


def topk_indices(score: np.ndarray, k: int) -> set[int]:
    if len(score) == 0:
        return set()
    k = min(k, len(score))
    return set(np.argsort(score)[-k:].tolist())


def confidence_level(v: float) -> str:
    if v > 0.70:
        return 'strong'
    if v > 0.45:
        return 'soft'
    if v > 0.25:
        return 'proposal'
    return 'none'


def contiguous_windows(mask: np.ndarray, frames: np.ndarray) -> dict[int, tuple[int,int]]:
    out = {}
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i + 1 < n and mask[i+1]:
            i += 1
        end = i
        for j in range(start, end+1):
            out[int(frames[j])] = (int(frames[start]), int(frames[end]))
        i += 1
    return out


def build_peak_index_from_selector(mask: np.ndarray, selector: np.ndarray, mode: str) -> np.ndarray:
    peak_index = np.arange(len(mask), dtype=np.int32)
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        start = i
        while i + 1 < len(mask) and mask[i + 1]:
            i += 1
        end = i
        local = np.asarray(selector[start:end + 1], dtype=np.float64)
        finite = np.isfinite(local)
        if np.any(finite):
            safe = local.copy()
            safe[~finite] = np.inf if mode == 'min' else -np.inf
            best_rel = int(np.argmin(safe) if mode == 'min' else np.argmax(safe))
        else:
            best_rel = 0
        peak_index[start:end + 1] = start + best_rel
        i += 1
    return peak_index


def mask_to_intervals(
    frames: np.ndarray,
    times: np.ndarray,
    mask: np.ndarray,
    label: str,
    target: list[str],
    score: np.ndarray,
    peak_index: np.ndarray | None = None,
) -> list[dict[str, object]]:
    intervals: list[dict[str, object]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i + 1 < n and mask[i + 1]:
            i += 1
        end = i
        peak_idx = int(peak_index[start]) if peak_index is not None else int(start + np.argmax(score[start:end + 1]))
        peak_idx = min(max(peak_idx, start), end)
        window_score = np.asarray(score[start:end + 1], dtype=np.float64)
        intervals.append({
            'contact_type': label,
            'target': target[peak_idx],
            'start_frame': int(frames[start]),
            'end_frame': int(frames[end]),
            'start_time': float(times[start]),
            'end_time': float(times[end]),
            'peak_frame': int(frames[peak_idx]),
            'peak_time': float(times[peak_idx]),
            'peak_index': peak_idx,
            'mean_score': float(np.nanmean(window_score)) if len(window_score) else 0.0,
            'max_score': float(np.nanmax(window_score)) if len(window_score) else 0.0,
            'num_frames': int(end - start + 1),
        })
        i += 1
    return intervals


def infer_anchor_event_mode(sample_dir: Path) -> str:
    metadata_path = sample_dir / 'metadata.json'
    if not metadata_path.exists():
        return 'interval_peak'
    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception:
        return 'interval_peak'
    text = ' '.join(
        str(metadata.get(k, '') or '').lower()
        for k in ('name', 'prompt', 'detection_text')
    )
    continuous_words = (
        'hold', 'holding', 'grasp', 'grabbing', 'grab',
        'lift', 'lifts', 'carry', 'carries', 'drink', 'sip',
        'hand', 'handle', 'place', 'places', 'lower', 'lowers',
    )
    continuous_objects = ('mug', 'cup', 'hammer', 'broom', 'drawer', 'chair')
    impulse_objects = ('basketball', 'football', 'soccer ball', 'ball')
    if any(w in text for w in continuous_objects):
        return 'continuous_state'
    if any(w in text for w in impulse_objects):
        return 'interval_peak'
    if any(w in text for w in continuous_words):
        return 'continuous_state'
    return 'interval_peak'


def infer_contact_part_policy(sample_dir: Path) -> str:
    metadata_path = sample_dir / 'metadata.json'
    text = sample_dir.name.lower()
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            text += ' ' + ' '.join(str(metadata.get(k, '') or '').lower() for k in ('name', 'prompt', 'detection_text'))
        except Exception:
            pass
    foot_words = ('football', 'soccer', 'kick', 'kicks', 'juggling', 'foot', 'feet')
    hand_words = ('basketball', 'dribble', 'bounce', 'shoot', 'throw', 'catch', 'hand')
    if any(w in text for w in foot_words):
        return 'feet'
    if any(w in text for w in hand_words):
        return 'hands'
    return 'active'


def main() -> None:
    parser = argparse.ArgumentParser(description='Run radius-free object contact candidate detection.')
    parser.add_argument('--sample-dir', type=Path, required=True)
    parser.add_argument('--object-proxy-observation-csv', type=Path, default=None)
    parser.add_argument('--object-mesh-csv', type=Path, default=None)
    parser.add_argument('--out-dir', type=Path, default=None)
    parser.add_argument('--body-model-root', type=Path, default=Path('third-party/GVHMR/inputs/checkpoints/body_models'))
    parser.add_argument('--anchor-event-mode', choices=['auto', 'interval_peak', 'continuous_state'], default='auto')
    args = parser.parse_args()

    sample_dir = args.sample_dir
    results_dir = sample_dir / 'results'
    out_dir = args.out_dir or (results_dir / 'generic_pipeline_v1' / 'contact_candidates_internal')
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_csv = results_dir / 'events' / 'audio_events.csv'
    if audio_csv.exists():
        shutil.copy2(audio_csv, out_dir / 'audio_events.csv')

    object_rows = read_object_proxy_track(args.object_proxy_observation_csv or (results_dir / 'generic_pipeline_v1' / 'ball_proxy_depth.csv'))
    object_mesh_by_frame = read_object_mesh_tracks(args.object_mesh_csv or (results_dir / 'tracking' / 'object_mesh_tracks_test.csv'))
    audio_rows = read_audio_events(audio_csv)

    human = read_human_result(results_dir / 'gvhmr' / 'result.pkl')
    joints = build_body_joints(args.body_model_root, human)
    K = np.asarray(human['K_fullimg'], dtype=np.float64)[:len(object_rows)]
    joints = joints[:len(object_rows)]
    centers = build_contact_part_centers(joints)

    frames = np.asarray([int(r['frame']) for r in object_rows], dtype=np.int32)
    times = np.asarray([float(r['time']) for r in object_rows], dtype=np.float64)
    valid_frames = frames.tolist()
    audio_support = build_audio_support(frames, audio_rows, radius=2)
    object_support_v_raw = np.asarray([parse_float(r, 'support_v_raw', parse_float(r, 'support_v')) for r in object_rows], dtype=np.float64)
    motion_score = np.asarray([parse_float(r, 'object_motion_score', 0.0) for r in object_rows], dtype=np.float64)
    depth_score = np.asarray([parse_float(r, 'object_depth_confidence', parse_float(r, 'depth_conf', 0.0)) for r in object_rows], dtype=np.float64)
    contact_depth_offset = np.abs(np.asarray([parse_float(r, 'contact_depth_offset_m', 0.0) for r in object_rows], dtype=np.float64))

    left_hand_uv, left_hand_valid = project_points(centers[LEFT_HAND_KEY], K)
    right_hand_uv, right_hand_valid = project_points(centers[RIGHT_HAND_KEY], K)
    left_foot_uv, left_foot_valid = project_points(centers[LEFT_FOOT_KEY], K)
    right_foot_uv, right_foot_valid = project_points(centers[RIGHT_FOOT_KEY], K)
    active_contact_uv = np.asarray([[parse_float(r, 'active_part_u'), parse_float(r, 'active_part_v')] for r in object_rows], dtype=np.float64)
    active_valid = np.all(np.isfinite(active_contact_uv), axis=1)
    active_labels = [str(r.get('active_label', '') or '') for r in object_rows]
    contact_part_policy = infer_contact_part_policy(sample_dir)
    anchor_event_mode = args.anchor_event_mode
    if anchor_event_mode == 'auto':
        anchor_event_mode = infer_anchor_event_mode(sample_dir)
    use_proxy_contact_region = anchor_event_mode == 'continuous_state' and any(
        str(r.get('contact_proxy_name', '')).startswith('handle:') for r in object_rows
    )
    if use_proxy_contact_region:
        contact_uv = np.asarray([[parse_float(r, 'contact_u'), parse_float(r, 'contact_v')] for r in object_rows], dtype=np.float64)
        contact_valid = np.all(np.isfinite(contact_uv), axis=1)
        left_dist = np.linalg.norm(left_hand_uv - contact_uv, axis=1)
        right_dist = np.linalg.norm(right_hand_uv - contact_uv, axis=1)
        left_dist[~(left_hand_valid & contact_valid)] = np.inf
        right_dist[~(right_hand_valid & contact_valid)] = np.inf
        use_left = left_dist <= right_dist
        min_gap_proxy = np.where(use_left, left_dist, right_dist)
        score_proxy = np.exp(-0.5 * (min_gap_proxy / 30.0) ** 2)
        score_proxy[~np.isfinite(min_gap_proxy)] = 0.0
        state_proxy = (min_gap_proxy <= 48.0) & (score_proxy >= 0.25)
        state_proxy = bridge_short_gaps(state_proxy, 3)
        candidate_proxy = state_proxy | ((min_gap_proxy <= 62.0) & (score_proxy >= 0.12))
        active_contact = np.where(use_left, 'left', 'right').astype(object)
        active_label_proxy = np.where(use_left, 'left_hand', 'right_hand').astype(object)
        active_part_proxy = np.where(use_left, 'hand', 'hand').astype(object)
        anchor_det = {
            'score': score_proxy,
            'candidate': candidate_proxy,
            'state': state_proxy,
            'min_contact_gap': min_gap_proxy,
            'active_label': active_label_proxy.tolist(),
            'active_part': active_part_proxy.tolist(),
            'active_contact': active_contact.tolist(),
            'active_object_point_id': [str(r.get('contact_proxy_name', 'canonical_contact_region')) for r in object_rows],
            'active_object_u': contact_uv[:, 0],
            'active_object_v': contact_uv[:, 1],
        }
    elif contact_part_policy == 'feet':
        anchor_det = detect_object_anchor_contact_from_feet(
            frames=frames, left_foot_uv=left_foot_uv, right_foot_uv=right_foot_uv,
            left_valid=left_foot_valid, right_valid=right_foot_valid,
            object_mesh_by_frame=object_mesh_by_frame, dist_thresh_px=28.0, score_sigma_px=18.0,
            local_radius=2, state_dist_thresh_px=38.0, state_score_thresh=0.35, gap_bridge=2,
            audio_support=audio_support,
        )
    elif contact_part_policy == 'hands':
        anchor_det = detect_object_anchor_contact_from_hands(
            frames=frames, left_hand_uv=left_hand_uv, right_hand_uv=right_hand_uv,
            left_valid=left_hand_valid, right_valid=right_hand_valid,
            object_mesh_by_frame=object_mesh_by_frame, dist_thresh_px=28.0, score_sigma_px=18.0,
            local_radius=2, state_dist_thresh_px=38.0, state_score_thresh=0.35, gap_bridge=2,
            audio_support=audio_support,
        )
    else:
        anchor_det = detect_object_anchor_contact_from_active(
            frames=frames, active_contact_uv=active_contact_uv, active_valid=active_valid,
            active_labels=active_labels, object_mesh_by_frame=object_mesh_by_frame,
            dist_thresh_px=28.0, score_sigma_px=18.0, local_radius=2,
            state_dist_thresh_px=38.0, state_score_thresh=0.35, gap_bridge=2,
            audio_support=audio_support,
        )

    support_hypotheses = build_support_hypotheses(object_support_v_raw, left_foot_uv, right_foot_uv, left_foot_valid, right_foot_valid)
    support_track = select_support_surface(object_support_v_raw, np.asarray(anchor_det['score'], dtype=np.float64), support_hypotheses)
    floor_det = detect_object_support_contact(
        support_track=support_track,
        anchor_score=np.asarray(anchor_det['score'], dtype=np.float64),
        gap_thresh_px=14.0, score_sigma_px=10.0, local_radius=2,
        state_gap_thresh_px=12.0, state_score_thresh=0.45, gap_bridge=2,
    )

    # Mainline-style candidate core for the radius-free proxy path:
    # proxy geometry provides absolute measurements; state/event selection uses
    # absolute thresholds and interval peaks. Do not let depth/support percentile
    # scores promote far human gaps into anchor contacts.
    min_gap = np.asarray(anchor_det['min_contact_gap'], dtype=np.float64)
    anchor_score = np.asarray(anchor_det['score'], dtype=np.float64)
    floor_score = np.asarray(floor_det['score'], dtype=np.float64)
    support_score = floor_score.copy()
    motion_score = np.asarray(motion_score, dtype=np.float64)
    depth_consistency_score = np.zeros_like(anchor_score)
    proposal_score = np.maximum.reduce([anchor_score, floor_score, motion_score, audio_support])

    anchor_candidate = np.asarray(anchor_det['candidate'], dtype=bool)
    base_anchor_state = np.asarray(anchor_det['state'], dtype=bool)
    if anchor_event_mode == 'interval_peak':
        # Generic impulse-contact state: proxy proximity is the base, while
        # object/audio motion can open a wider interval. This is the mainline
        # interval idea with radius-free measurements.
        motion_gate = (motion_score >= 0.80) | (audio_support >= 0.20)
        impulse_state = (min_gap <= 52.0) & motion_gate
        # For impulse contacts, proximity alone is not a state. A stationary
        # object resting near a body part after the action must not become a
        # human anchor; this matches the mainline interval logic, where late
        # near-foot frames can have nonzero anchor_score but anchor_state=0.
        anchor_state = bridge_short_gaps(impulse_state, 2)
        anchor_candidate = anchor_candidate | anchor_state
    else:
        anchor_state = base_anchor_state

    floor_gap = np.asarray(floor_det['gap'], dtype=np.float64)
    floor_candidate = np.asarray(floor_det['candidate'], dtype=bool)
    floor_event_selector = floor_gap
    if anchor_event_mode == 'interval_peak':
        # Radius-free ball/object impulse floor contact: use the object's own
        # bottom/support proxy peak. The absolute support-plane gap can be wrong
        # when GVHMR floor is biased, so it is diagnostic only in this mode.
        object_support_peak_v = np.asarray(floor_det['object_support_v_raw'], dtype=np.float64)
        finite_support = object_support_peak_v[np.isfinite(object_support_peak_v)]
        if finite_support.size:
            support_enter = float(np.nanpercentile(finite_support, 85))
            support_soft = float(np.nanpercentile(finite_support, 65))
        else:
            support_enter = np.inf
            support_soft = np.inf
        support_span = max(support_enter - support_soft, 1e-6)
        proxy_floor_score = np.clip((object_support_peak_v - support_soft) / support_span, 0.0, 1.0)
        support_local_peak = local_max_mask(object_support_peak_v, 2)
        floor_state = support_local_peak & (object_support_peak_v >= support_enter)
        floor_candidate = floor_candidate | floor_state
        floor_score = np.maximum(floor_score, proxy_floor_score)
        support_score = floor_score.copy()
        floor_event_selector = object_support_peak_v
    else:
        floor_state = np.asarray(floor_det['state'], dtype=bool)
    transition_state = np.asarray((anchor_state != floor_state) & (anchor_state | floor_state), dtype=bool)
    multi_state = np.asarray(anchor_state & floor_state, dtype=bool)

    object_ref_v = np.asarray([parse_float(r, 'ref_v', parse_float(r, 'support_v', np.nan)) for r in object_rows], dtype=np.float64)
    if anchor_event_mode == 'interval_peak':
        if contact_part_policy == 'feet':
            # Foot contacts are best localized by foot-object proximity; using
            # image height can shift the event to a nearby flight frame.
            anchor_event_index = build_peak_index_from_selector(anchor_state, anchor_score, 'max')
        else:
            # Mainline impulse default: choose the highest object point in each
            # contact interval. Smaller image-v means higher in the image.
            anchor_event_index = build_peak_index_from_selector(anchor_state, object_ref_v, 'min')
    else:
        anchor_event_index = build_peak_index_from_selector(anchor_state, anchor_score, 'max')
    floor_event_index = build_peak_index_from_selector(floor_state, floor_event_selector, 'max' if anchor_event_mode == 'interval_peak' else 'min')
    for idx in np.unique(anchor_event_index[anchor_state]):
        anchor_candidate[int(idx)] = True
    for idx in np.unique(floor_event_index[floor_state]):
        floor_candidate[int(idx)] = True

    anchor_windows = contiguous_windows(anchor_state, frames)
    floor_windows = contiguous_windows(floor_state, frames)

    anchor_rows = []
    floor_rows = []
    state_rows = []
    labeled_rows = []
    intervals = []
    current = None
    for i, frame in enumerate(frames.tolist()):
        active_label = anchor_det['active_label'][i]
        active_part = anchor_det['active_part'][i] or ''
        side = anchor_det['active_contact'][i] or ''
        contact_part, contact_side, contact_label = split_active_label(active_label)
        if floor_state[i] and not anchor_state[i]:
            contact_part, contact_side, contact_label = 'floor', '', 'floor'
        state_part_u = math.nan
        state_part_v = math.nan
        if contact_label == 'left_foot' and left_foot_valid[i]:
            state_part_u, state_part_v = left_foot_uv[i]
        elif contact_label == 'right_foot' and right_foot_valid[i]:
            state_part_u, state_part_v = right_foot_uv[i]
        elif contact_label == active_label:
            state_part_u = parse_float(object_rows[i], 'active_part_u')
            state_part_v = parse_float(object_rows[i], 'active_part_v')
        anchor_rows.append({
            'frame': frame,
            'time': f'{times[i]:.6f}',
            'active_side': side,
            'active_object_point_id': anchor_det['active_object_point_id'][i],
            'active_object_u': f"{anchor_det['active_object_u'][i]:.3f}" if np.isfinite(anchor_det['active_object_u'][i]) else '',
            'active_object_v': f"{anchor_det['active_object_v'][i]:.3f}" if np.isfinite(anchor_det['active_object_v'][i]) else '',
            'min_object_boundary_gap_px': f"{min_gap[i]:.6f}",
            'active_part_distance_px': f"{min_gap[i]:.6f}",
            'proposal_score': f"{proposal_score[i]:.6f}",
            'anchor_score': f"{anchor_score[i]:.6f}",
            'confidence_level': confidence_level(anchor_score[i]),
            'is_candidate': int(bool(anchor_candidate[i])),
            'anchor_contact_state': int(bool(anchor_state[i])),
        })
        floor_rows.append({
            'frame': frame,
            'time': f'{times[i]:.6f}',
            'object_support_v_raw': f"{floor_det['object_support_v_raw'][i]:.3f}",
            'object_support_v': f"{floor_det['object_support_v'][i]:.3f}",
            'support_v_raw': f"{floor_det['support_v_raw'][i]:.3f}" if np.isfinite(floor_det['support_v_raw'][i]) else '',
            'support_v': f"{floor_det['support_v'][i]:.3f}" if np.isfinite(floor_det['support_v'][i]) else '',
            'signed_gap': f"{floor_det['signed_gap'][i]:.6f}",
            'gap': f"{floor_det['gap'][i]:.6f}",
            'floor_score': f"{floor_score[i]:.6f}",
            'proposal_score': f"{proposal_score[i]:.6f}",
            'confidence_level': confidence_level(floor_score[i]),
            'is_candidate': int(bool(floor_candidate[i])),
            'floor_contact_state': int(bool(floor_state[i])),
        })
        state_rows.append({
            'frame': frame,
            'time': f'{times[i]:.6f}',
            'proposal_score': f"{proposal_score[i]:.6f}",
            'anchor_score': f"{anchor_score[i]:.6f}",
            'floor_score': f"{floor_score[i]:.6f}",
            'anchor_contact_state': int(bool(anchor_state[i])),
            'human_contact_state': int(bool(anchor_state[i])),
            'floor_contact_state': int(bool(floor_state[i])),
            'transition_contact_state': int(bool(transition_state[i])),
            'multi_contact_state': int(bool(multi_state[i])),
            'confidence_level': confidence_level(max(anchor_score[i], floor_score[i])),
            'contact_part': contact_part,
            'contact_label': contact_label,
            'contact_side': contact_side,
            'anchor_type': active_part,
            'min_object_boundary_gap_px': f"{min_gap[i]:.6f}",
            'active_part_distance_px': f"{min_gap[i]:.6f}",
            'contact_depth_offset_m': f"{parse_float(object_rows[i], 'contact_depth_offset_m', 0.0):.6f}",
            'object_acceleration': f"{motion_score[i]:.6f}",
            'support_surface_type': floor_det['support_surface_type'][i],
            'support_source': floor_det['support_source'][i],
            'signed_support_gap_px': f"{floor_det['signed_gap'][i]:.6f}",
            'support_conf': f"{floor_det['support_conf'][i]:.6f}",
            'active_object_point_id': anchor_det['active_object_point_id'][i],
            'active_object_u': f"{anchor_det['active_object_u'][i]:.3f}" if np.isfinite(anchor_det['active_object_u'][i]) else '',
            'active_object_v': f"{anchor_det['active_object_v'][i]:.3f}" if np.isfinite(anchor_det['active_object_v'][i]) else '',
            'state_active_part_u': f"{state_part_u:.3f}" if np.isfinite(state_part_u) else '',
            'state_active_part_v': f"{state_part_v:.3f}" if np.isfinite(state_part_v) else '',
        })
        emit_anchor = bool(anchor_state[i]) if anchor_event_mode == 'continuous_state' else bool(anchor_state[i] and int(anchor_event_index[i]) == i and anchor_score[i] >= 0.15)
        if emit_anchor:
            ws, we = anchor_windows.get(frame, (frame, frame))
            labeled_rows.append({
                'frame': frame,
                'window_start': ws,
                'window_end': we,
                'time': f'{times[i]:.6f}',
                'contact_type': 'anchor_contact_event',
                'target': contact_label,
                'score': f"{anchor_score[i]:.6f}",
                'confidence_level': confidence_level(anchor_score[i]),
                'source_support': f"{support_score[i]:.6f}",
                'source_proximity': f"{anchor_score[i]:.6f}",
                'source_motion': f"{motion_score[i]:.6f}",
                'source_audio': f"{audio_support[i]:.6f}",
                'source_depth': f"{depth_consistency_score[i]:.6f}",
                'occlusion_score': '0.000000',
            })
        if floor_state[i] and int(floor_event_index[i]) == i:
            ws, we = floor_windows.get(frame, (frame, frame))
            labeled_rows.append({
                'frame': frame,
                'window_start': ws,
                'window_end': we,
                'time': f'{times[i]:.6f}',
                'contact_type': 'plane_support_contact_event',
                'target': floor_det['support_surface_type'][i],
                'score': f"{floor_score[i]:.6f}",
                'confidence_level': confidence_level(floor_score[i]),
                'source_support': f"{support_score[i]:.6f}",
                'source_proximity': f"{anchor_score[i]:.6f}",
                'source_motion': f"{motion_score[i]:.6f}",
                'source_audio': f"{audio_support[i]:.6f}",
                'source_depth': f"{depth_consistency_score[i]:.6f}",
                'occlusion_score': '0.000000',
            })
        active = bool(anchor_state[i]) or bool(floor_state[i])
        if active and current is None:
            current = {'start_frame': frame, 'end_frame': frame, 'support_surface_type': floor_det['support_surface_type'][i] if floor_state[i] else ''}
        elif active and current is not None:
            current['end_frame'] = frame
        elif (not active) and current is not None:
            intervals.append(current)
            current = None
    if current is not None:
        intervals.append(current)

    write_csv(out_dir / 'anchor_contact_candidates.csv', anchor_rows, list(anchor_rows[0].keys()))
    write_csv(out_dir / 'floor_contact_candidates.csv', floor_rows, list(floor_rows[0].keys()))
    write_csv(out_dir / 'contact_state_frames.csv', state_rows, list(state_rows[0].keys()))
    write_csv(out_dir / 'contact_candidates_labeled.csv', labeled_rows, list(labeled_rows[0].keys()) if labeled_rows else ['frame','window_start','window_end','time','contact_type','target','score','confidence_level','source_support','source_proximity','source_motion','source_audio','source_depth','occlusion_score'])
    write_csv(out_dir / 'contact_intervals.csv', intervals, ['start_frame','end_frame','support_surface_type'])
    print(f'contact_candidate_dir: {out_dir}')


if __name__ == '__main__':
    main()
