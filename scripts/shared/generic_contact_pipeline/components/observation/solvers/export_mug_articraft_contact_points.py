#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.shared.generic_contact_pipeline.components.pose.solvers import fit_mug_articraft_keyframe_pose as base  # noqa: E402
from scripts.shared.generic_contact_pipeline.components.pose.solvers import fit_mug_body_only_cylinder_pose as bodyfit  # noqa: E402
from scripts.shared.generic_contact_pipeline.components.render.scenes import render_mug_articraft_rigid_mesh_vlm as rigid  # noqa: E402
from scripts.shared.generic_contact_pipeline.components.render.scenes import render_mug_segmented_body_with_visual_handle_contact_region as vismod  # noqa: E402

MAX_CONFIRMED_HANDLE_UV_DIST_PX = 25.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def ff(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, '') or default)
    except Exception:
        return default


def read_pose_sequence(path: Path) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for row in read_csv(path):
        fr = int(float(row['frame']))
        out[fr] = np.array([ff(row, k) for k in ['x', 'y', 'z', 'yaw', 'pitch', 'roll', 'scale']], dtype=float)
    return out


def read_phase_sequence(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in read_csv(path):
        fr = int(float(row['frame']))
        out[fr] = ff(row, 'mug_axial_phase_rad', ff(row, 'handle_phase_rad', 0.0))
    return out


def backproject(u: float, v: float, z: float, K: np.ndarray) -> np.ndarray:
    return np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], dtype=float)


def camera_to_mug_canonical(P_cam: np.ndarray, params: np.ndarray, phase: float) -> np.ndarray:
    x, y, z, yaw, pitch, roll, scale = [float(v) for v in params]
    R = base.object_R(yaw, pitch, roll)
    local_phase = (P_cam - np.array([x, y, z], dtype=float)) @ R / max(scale, 1e-8)
    # render path uses canonical @ rot_y(phase).T before body transform.
    # Therefore inverse is local_phase @ rot_y(phase).
    return local_phase @ rigid.rot_y(phase)


def nearest_part(local: np.ndarray, meshes: dict[str, tuple[np.ndarray, list[tuple[int, int]]]]) -> tuple[str, int, float, np.ndarray]:
    best = ('unknown', -1, math.inf, np.full(3, math.nan))
    for name, (verts, _edges) in meshes.items():
        if len(verts) == 0:
            continue
        d2 = np.sum((verts - local[None, :]) ** 2, axis=1)
        idx = int(np.argmin(d2))
        dist = float(math.sqrt(d2[idx]))
        if dist < best[2]:
            best = (name, idx, dist, verts[idx].copy())
    return best



def handle_contact_from_active_hand(
    obj_row: dict[str, str],
    params: np.ndarray,
    K: np.ndarray,
    phase: float,
    meshes: dict[str, tuple[np.ndarray, list[tuple[int, int]]]],
) -> dict[str, Any] | None:
    """Pick the Articraft handle surface point closest to the active hand in image space."""
    if 'handle_loop' not in meshes:
        return None
    hu = ff(obj_row, 'active_part_u')
    hv = ff(obj_row, 'active_part_v')
    if not (np.isfinite(hu) and np.isfinite(hv)):
        return None
    verts, _edges = meshes['handle_loop']
    if len(verts) == 0:
        return None
    local_phase = verts @ rigid.rot_y(phase).T
    cam = base.transform(params, local_phase)
    uv = bodyfit.project_pts(params, local_phase, K)
    valid = np.all(np.isfinite(uv), axis=1) & np.all(np.isfinite(cam), axis=1) & (cam[:, 2] > 0.2)
    if not np.any(valid):
        return None
    d2 = np.sum((uv - np.array([[hu, hv]], dtype=float)) ** 2, axis=1)
    d2[~valid] = np.inf
    idx = int(np.argmin(d2))
    if not np.isfinite(d2[idx]):
        return None
    return {
        'u': float(uv[idx, 0]),
        'v': float(uv[idx, 1]),
        'P_cam': cam[idx].copy(),
        'local': verts[idx].copy(),
        'vertex_index': idx,
        'nearest': verts[idx].copy(),
        'dist': 0.0,
        'source': 'articraft_handle_mesh_nearest_active_hand',
        'conf': ff(obj_row, 'active_label_conf', 0.0),
        'hand_handle_uv_dist_px': float(math.sqrt(d2[idx])),
    }

def candidate_uv(row: dict[str, str], obj_row: dict[str, str]) -> tuple[float, float, str, float]:
    cu = ff(obj_row, 'contact_u')
    cv = ff(obj_row, 'contact_v')
    conf = ff(obj_row, 'contact_conf', 0.0)
    src = obj_row.get('contact_source') or 'object_proxy_contact_u_v'
    if np.isfinite(cu) and np.isfinite(cv) and conf > 0.0:
        return cu, cv, src, conf
    uv, src2 = vismod.contact_region_uv_from_obs(row)
    if uv is not None and np.all(np.isfinite(uv)):
        return float(uv[0]), float(uv[1]), src2, ff(row, 'hand_contact_conf', 0.0)
    return math.nan, math.nan, 'missing_contact_proxy', 0.0


def semantic_from_vlm(vlm_row: dict[str, str] | None) -> dict[str, Any]:
    if not vlm_row:
        return {}
    return {
        'vlm_visibility': vlm_row.get('visibility', ''),
        'vlm_visible_side': vlm_row.get('visible_side', ''),
        'vlm_occlusion_reason': vlm_row.get('occlusion_reason', ''),
        'vlm_hand_contact_part': vlm_row.get('hand_contact_part', ''),
        'vlm_handle_contact': vlm_row.get('handle_contact', ''),
        'vlm_body_contact': vlm_row.get('body_contact', ''),
        'vlm_rim_contact': vlm_row.get('rim_contact', ''),
        'vlm_constraint': vlm_row.get('recommended_visibility_constraint', ''),
        'vlm_confidence': vlm_row.get('confidence', ''),
    }


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y'}


def filter_vlm_short_visible_islands(rows: dict[int, dict[str, str]], max_len: int = 3) -> dict[int, dict[str, str]]:
    # Hidden/body evidence is conservative for mug handle contact: a far-side
    # handle should not be pulled back into view by short visible blips. Suppress
    # only short visible-handle islands near hidden runs; never convert hidden
    # far-side/body frames into visible handle frames.
    if not rows:
        return rows
    frames = sorted(rows)
    state: dict[int, bool] = {}
    for fr in frames:
        row = rows[fr]
        part = str(row.get('hand_contact_part', '')).strip().lower()
        visible = str(row.get('visibility', '')).strip().lower() == 'visible'
        handle = truthy(row.get('handle_contact', '')) or part == 'handle'
        state[fr] = bool(visible and handle)

    runs: list[tuple[int, int, bool]] = []
    start = frames[0]
    last = frames[0]
    cur = state[start]
    for fr in frames[1:]:
        if state[fr] != cur or fr != last + 1:
            runs.append((start, last, cur))
            start = fr
            cur = state[fr]
        last = fr
    runs.append((start, last, cur))

    suppress_visible: set[int] = set()
    for i, (a, b, val) in enumerate(runs):
        if not val or (b - a + 1) > max_len:
            continue
        prev_hidden = i > 0 and not runs[i - 1][2] and runs[i - 1][1] + 1 == a
        next_hidden = i + 1 < len(runs) and not runs[i + 1][2] and b + 1 == runs[i + 1][0]
        if prev_hidden or next_hidden:
            suppress_visible.update(fr for fr in range(a, b + 1) if fr in state)

    out = {fr: dict(row) for fr, row in rows.items()}
    for fr in sorted(suppress_visible):
        if fr not in out:
            continue
        out[fr]['visibility'] = 'hidden'
        out[fr]['handle_visible'] = 'False'
        out[fr]['visible_side'] = 'unknown'
        out[fr]['occlusion_reason'] = 'temporal_short_visible_island'
        out[fr]['handle_shape_visible'] = 'not_reliable_short_visible_island'
        out[fr]['hand_contact_part'] = 'body'
        out[fr]['handle_contact'] = 'False'
        out[fr]['body_contact'] = 'True'
        out[fr]['yaw_anchor_quality'] = 'low'
        out[fr]['recommended_visibility_constraint'] = 'force_hidden_temporal_filter'
        out[fr]['short_reason'] = 'Suppressed short visible-handle island by hidden/body conservative filter.'
    return out


def hand_distance_px(obj_row: dict[str, str], u: float, v: float) -> float:
    hu = ff(obj_row, 'active_part_u')
    hv = ff(obj_row, 'active_part_v')
    if not (np.isfinite(hu) and np.isfinite(hv) and np.isfinite(u) and np.isfinite(v)):
        return math.nan
    return float(math.hypot(hu - u, hv - v))


def active_hand_camera_point(obj_row: dict[str, str], K: np.ndarray) -> np.ndarray:
    hu = ff(obj_row, 'active_part_u')
    hv = ff(obj_row, 'active_part_v')
    hz = ff(obj_row, 'active_part_z')
    if not (np.isfinite(hu) and np.isfinite(hv) and np.isfinite(hz) and hz > 0.2):
        return np.full(3, math.nan)
    return backproject(hu, hv, hz, K)


def classify_contact_usage(
    obj_row: dict[str, str],
    semantic_part: str,
    nearest_part_name: str,
    nearest_dist_m: float,
    u: float,
    v: float,
    contact_cam: np.ndarray,
    hand_cam: np.ndarray,
    vlm_info: dict[str, Any],
) -> dict[str, Any]:
    """Separate object-side contact type from hand-attachment usage.

    A single projected object contact point can land on rim/body/handle. During
    drinking the rim contact is real, but it must not replace the hand grasp
    anchor. We therefore export explicit usage flags for later optimization.
    """
    active_label = str(obj_row.get('active_label', '') or '')
    active_conf = ff(obj_row, 'active_label_conf', 0.0)
    is_hand = active_label.endswith('_hand') or active_label in {'left_hand', 'right_hand', 'hand'}
    d_hand = hand_distance_px(obj_row, u, v)
    hand_gap_m = float(np.linalg.norm(hand_cam - contact_cam)) if np.all(np.isfinite(hand_cam)) and np.all(np.isfinite(contact_cam)) else math.nan
    hand_z_gap_m = float(hand_cam[2] - contact_cam[2]) if np.all(np.isfinite(hand_cam)) and np.all(np.isfinite(contact_cam)) else math.nan
    near_hand = is_hand and active_conf >= 0.20 and (not np.isfinite(d_hand) or d_hand <= 65.0)
    geom_ok = np.isfinite(nearest_dist_m) and nearest_dist_m <= 0.14
    depth_misaligned = np.isfinite(hand_gap_m) and hand_gap_m > 0.12

    vlm_part = str(vlm_info.get('vlm_hand_contact_part', '') or '').strip().lower()
    vlm_rim = truthy(vlm_info.get('vlm_rim_contact', '')) or vlm_part in {'rim', 'lip', 'mouth', 'cup_rim'}
    vlm_handle = truthy(vlm_info.get('vlm_handle_contact', '')) or vlm_part == 'handle'
    vlm_body = truthy(vlm_info.get('vlm_body_contact', '')) or vlm_part in {'body', 'cup_body', 'wall'}

    part = semantic_part or nearest_part_name or ''
    part_l = part.lower()
    nearest_l = (nearest_part_name or '').lower()
    src_l = str(obj_row.get('contact_source', '') or '').lower()
    proxy_l = str(obj_row.get('contact_proxy_name', '') or '').lower()
    explicit_handle_observation = vlm_handle or proxy_l.startswith('handle') or src_l.startswith('handle')
    is_rim_point = 'rim' in part_l or vlm_rim
    visibility = str(vlm_info.get('vlm_visibility', '') or '').strip().lower()
    occlusion = str(vlm_info.get('vlm_occlusion_reason', '') or '').strip().lower()
    hidden_by_vlm = visibility.startswith('hidden') or 'hidden' in visibility or occlusion in {'far_side', 'hand', 'body', 'self_occluded'}
    # Semantic/VLM body evidence wins over a geometric nearest-handle result.
    # Nearest mesh is only a fallback and can be wrong when the contact point is
    # near the handle/body interface or the handle is occluded.
    is_body_point = ('body' in part_l or 'shell' in part_l or vlm_body) and not is_rim_point
    is_handle_point = ((part_l.startswith('handle') or vlm_handle) or (nearest_l.startswith('handle') and not is_body_point)) and not is_rim_point

    usage = {
        'object_contact_event': 'object_surface_contact',
        'hand_mug_contact_state': 'not_hand_contact',
        'use_this_point_for_hand_attachment': 0,
        'use_previous_grasp_for_hand_attachment': 0,
        'rim_drinking_contact': 0,
        'hand_contact_u': '',
        'hand_contact_v': '',
        'hand_to_object_contact_px': f'{d_hand:.3f}' if np.isfinite(d_hand) else '',
        'hand_contact_3d_gap_m': f'{hand_gap_m:.6f}' if np.isfinite(hand_gap_m) else '',
        'hand_object_z_gap_m': f'{hand_z_gap_m:.6f}' if np.isfinite(hand_z_gap_m) else '',
        'hand_depth_alignment_needed': int(depth_misaligned),
        'hand_contact_reason': '',
    }

    if is_rim_point:
        usage['object_contact_event'] = 'rim_drinking_contact'
        usage['rim_drinking_contact'] = 1
        if near_hand:
            usage['hand_mug_contact_state'] = 'held_during_rim_drinking'
            usage['use_previous_grasp_for_hand_attachment'] = 1
            usage['hand_contact_reason'] = 'rim is mouth/drinking contact; keep prior hand grasp anchor'
        else:
            usage['hand_contact_reason'] = 'rim contact without reliable active hand gate'
        return usage

    if hidden_by_vlm and near_hand:
        usage['object_contact_event'] = 'occluded_hand_mug_grasp'
        usage['hand_mug_contact_state'] = 'held_during_occlusion'
        usage['use_previous_grasp_for_hand_attachment'] = 1
        usage['hand_contact_reason'] = 'VLM marks contact/handle region hidden; current visible point is not a direct hand-handle observation, keep prior grasp anchor'
        return usage

    if not geom_ok:
        usage['object_contact_event'] = 'invalid_geometry_gap'
        usage['hand_contact_reason'] = 'object contact point is too far from Articraft mesh'
        return usage

    if not near_hand:
        usage['object_contact_event'] = 'object_surface_contact'
        usage['hand_contact_reason'] = 'no reliable active hand gate near this object contact point'
        return usage

    if is_handle_point and not explicit_handle_observation:
        usage['object_contact_event'] = 'unconfirmed_handle_candidate'
        usage['hand_mug_contact_state'] = 'held_but_handle_not_observed'
        usage['use_previous_grasp_for_hand_attachment'] = 1
        usage['hand_contact_reason'] = 'nearest mesh part is handle, but there is no VLM/explicit handle observation; do not use current visible point as direct hand-handle anchor'
        return usage

    if is_handle_point:
        if np.isfinite(d_hand) and d_hand > MAX_CONFIRMED_HANDLE_UV_DIST_PX:
            usage['object_contact_event'] = 'handle_contact_point_misaligned'
            usage['hand_mug_contact_state'] = 'confirmed_handle_but_projected_point_not_on_hand'
            usage['use_this_point_for_hand_attachment'] = 0
            usage['use_previous_grasp_for_hand_attachment'] = 1
            usage['hand_contact_reason'] = 'VLM confirms handle contact, but projected Articraft handle point is too far from active hand; do not use this wrong point, fit handle/mug pose first'
            return usage
        usage['object_contact_event'] = 'hand_handle_grasp_depth_misaligned' if depth_misaligned else 'hand_handle_grasp'
        usage['hand_mug_contact_state'] = 'visual_contact_needs_depth_alignment' if depth_misaligned else 'direct_hand_grasp_point'
        usage['use_this_point_for_hand_attachment'] = 1
        usage['hand_contact_u'] = f'{u:.3f}'
        usage['hand_contact_v'] = f'{v:.3f}'
        usage['hand_contact_reason'] = 'visual/VLM handle contact is valid; GVHMR/object depth gap should be solved as an alignment residual' if depth_misaligned else 'active hand is near Articraft handle contact point'
    elif is_body_point:
        usage['object_contact_event'] = 'non_handle_body_contact'
        usage['hand_mug_contact_state'] = 'not_handle_contact_point'
        usage['use_this_point_for_hand_attachment'] = 0
        if near_hand:
            usage['use_previous_grasp_for_hand_attachment'] = 1
            usage['hand_mug_contact_state'] = 'held_but_visible_point_is_not_handle'
        usage['hand_contact_reason'] = 'visible point is mug body, not handle; do not use it as hand-handle attachment anchor'
    else:
        usage['object_contact_event'] = 'non_handle_unknown_surface_contact'
        usage['hand_mug_contact_state'] = 'not_handle_contact_point'
        usage['use_this_point_for_hand_attachment'] = 0
        if near_hand:
            usage['use_previous_grasp_for_hand_attachment'] = 1
            usage['hand_mug_contact_state'] = 'held_but_visible_point_is_not_handle'
        usage['hand_contact_reason'] = 'active hand is near object, but this is not a confirmed handle contact point'
    return usage


def main() -> None:
    ap = argparse.ArgumentParser(description='Export concrete mug object-side contact points on Articraft canonical mesh.')
    ap.add_argument('--sample-dir', type=Path, default=Path('samples_known_object/02_mug'))
    ap.add_argument('--mesh-root', type=Path, default=Path('samples_known_object/02_mug/articraft/materialized_mug_mesh'))
    ap.add_argument('--pose-csv', type=Path, default=None)
    ap.add_argument('--phase-csv', type=Path, default=None)
    ap.add_argument('--vlm-csv', type=Path, default=None)
    ap.add_argument('--object-proxy-csv', type=Path, default=None)
    ap.add_argument('--min-contact-conf', type=float, default=0.05)
    ap.add_argument('--out-dir', type=Path, default=None)
    args = ap.parse_args()

    sample = args.sample_dir
    pose_csv = args.pose_csv or (sample / 'proxy' / 'mug_body_only_cylinder_pose_segmented_sequence.csv')
    phase_csv = args.phase_csv or (sample / 'results' / 'renders' / 'M12_articraft_rigid_mesh_vlm' / 'handle_phase_all.csv')

    if args.vlm_csv:
        vlm_csv = args.vlm_csv
    else:
        vlm_full = sample / 'annotations' / 'vlm_handle_visibility_full' / 'qwen_handle_visibility.csv'
        vlm_dense = sample / 'annotations' / 'vlm_handle_visibility_dense' / 'qwen_handle_visibility.csv'
        vlm_csv = vlm_full if vlm_full.exists() else vlm_dense
    out_dir = args.out_dir or (sample / 'results' / 'mug_articraft_contact_points')
    out_dir.mkdir(parents=True, exist_ok=True)

    poses = read_pose_sequence(pose_csv)
    phases = read_phase_sequence(phase_csv)
    obs_rows = {int(r['frame']): r for r in read_csv(sample / 'results' / 'object_observations' / 'object_observations.csv')}
    object_proxy_csv = args.object_proxy_csv or (sample / 'results' / 'object_proxy_observations' / 'object_proxy_observations.csv')
    proxy_rows = {int(r['frame']): r for r in read_csv(object_proxy_csv)}
    vlm_rows = filter_vlm_short_visible_islands({int(r['frame']): r for r in read_csv(vlm_csv)}) if vlm_csv.exists() else {}
    K_all = base.load_K(sample)
    meshes = rigid.load_articraft_meshes(args.mesh_root)

    rows: list[dict[str, Any]] = []
    for fr in sorted(set(poses) & set(proxy_rows) & set(obs_rows)):
        obj = proxy_rows[fr]
        obs = obs_rows[fr]
        contact_conf = ff(obj, 'contact_conf', 0.0)
        if contact_conf < args.min_contact_conf and ff(obs, 'hand_contact_conf', 0.0) < args.min_contact_conf:
            continue
        u, v, src, conf = candidate_uv(obs, obj)
        z = ff(obj, 'contact_proxy_depth_m', ff(obj, 'object_ref_depth_m', ff(obj, 'object_depth_smooth')))
        if not (np.isfinite(u) and np.isfinite(v) and np.isfinite(z) and z > 0.2):
            continue
        K = K_all[fr - 1]
        phase = phases.get(fr, 0.0)
        vlm_info = semantic_from_vlm(vlm_rows.get(fr))
        vlm_part = str(vlm_info.get('vlm_hand_contact_part', '') or '').strip().lower()
        explicit_handle = truthy(vlm_info.get('vlm_handle_contact', '')) or vlm_part == 'handle'
        handle_pick = handle_contact_from_active_hand(obj, poses[fr], K, phase, meshes) if explicit_handle else None
        if handle_pick is not None:
            u = handle_pick['u']
            v = handle_pick['v']
            P_cam = handle_pick['P_cam']
            local = handle_pick['local']
            z = float(P_cam[2])
            src = handle_pick['source']
            conf = handle_pick['conf']
            part = 'handle_loop'
            vidx = int(handle_pick['vertex_index'])
            dist = float(handle_pick['dist'])
            nearest = handle_pick['nearest']
            contact_depth_offset_out = f"{(z - ff(obj, 'object_ref_depth_m')):.6f}" if np.isfinite(ff(obj, 'object_ref_depth_m')) else obj.get('contact_depth_offset_m', '')
        else:
            P_cam = backproject(u, v, z, K)
            local = camera_to_mug_canonical(P_cam, poses[fr], phase)
            part, vidx, dist, nearest = nearest_part(local, meshes)
            contact_depth_offset_out = obj.get('contact_depth_offset_m', '')
        # The nearest mesh part is geometric; VLM/contact labels are semantic evidence.
        semantic_part = part
        if str(vlm_info.get('vlm_handle_contact', '')).lower() == 'true':
            semantic_part = 'handle_loop'
        elif str(vlm_info.get('vlm_rim_contact', '')).lower() == 'true':
            semantic_part = 'rim_ring'
        elif str(vlm_info.get('vlm_body_contact', '')).lower() == 'true' and part == 'handle_loop' and handle_pick is None:
            semantic_part = 'body_shell_or_occluded_handle_region'
        hand_cam = active_hand_camera_point(obj, K)
        usage = classify_contact_usage(obj, semantic_part, part, dist, u, v, P_cam, hand_cam, vlm_info)
        rows.append({
            'frame': fr,
            'time': obj.get('time', ''),
            'contact_u': f'{u:.3f}',
            'contact_v': f'{v:.3f}',
            'contact_source': src,
            'contact_conf': f'{conf:.6f}',
            'contact_proxy_depth_m': f'{z:.6f}',
            'object_ref_depth_m': obj.get('object_ref_depth_m', ''),
            'contact_depth_offset_m': contact_depth_offset_out,
            'camera_x': f'{P_cam[0]:.6f}',
            'camera_y': f'{P_cam[1]:.6f}',
            'camera_z': f'{P_cam[2]:.6f}',
            'mug_local_x': f'{local[0]:.6f}',
            'mug_local_y': f'{local[1]:.6f}',
            'mug_local_z': f'{local[2]:.6f}',
            'nearest_articraft_part': part,
            'semantic_contact_part': semantic_part,
            'nearest_vertex_index': vidx,
            'nearest_vertex_dist_m_canonical': f'{dist:.6f}',
            'nearest_vertex_local_x': f'{nearest[0]:.6f}',
            'nearest_vertex_local_y': f'{nearest[1]:.6f}',
            'nearest_vertex_local_z': f'{nearest[2]:.6f}',
            'active_label': obj.get('active_label', ''),
            'active_label_conf': obj.get('active_label_conf', ''),
            'active_part_u': obj.get('active_part_u', ''),
            'active_part_v': obj.get('active_part_v', ''),
            'handle_contact_selection': src if src.startswith('articraft_handle_mesh') else '',
            **usage,
            **vlm_info,
        })

    csv_path = out_dir / 'mug_articraft_contact_points.csv'
    if rows:
        with csv_path.open('w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader(); wr.writerows(rows)
    summary = {
        'num_points': len(rows),
        'pose_csv': str(pose_csv),
        'phase_csv': str(phase_csv),
        'object_proxy_csv': str(object_proxy_csv),
        'vlm_csv': str(vlm_csv),
        'mesh_root': str(args.mesh_root),
        'csv': str(csv_path),
        'policy': 'contact_u/v + contact_proxy_depth_m are backprojected to camera 3D, transformed into mug canonical coordinates, then assigned to nearest Articraft semantic mesh part. The export separates object_contact_event from hand_mug_contact_state: rim/drinking contacts are real but do not replace the hand grasp anchor; handle/body points near the active GVHMR hand proxy can be used for hand attachment.',
    }
    (out_dir / 'mug_articraft_contact_points_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
