#!/usr/bin/env python3
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
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_mug_articraft_keyframe_pose as base  # noqa: E402
import fit_mug_body_only_cylinder_pose as bodyfit  # noqa: E402
import render_mug_articraft_rigid_mesh_vlm as rigid  # noqa: E402
import render_mug_segmented_body_with_visual_handle_contact_region as m9  # noqa: E402


PART_STYLE = {
    'body_shell': ((225, 225, 225), 1),
    'rim_ring': ((255, 0, 255), 2),
    'bottom_disk': ((0, 255, 255), 2),
    'handle_loop': ((0, 210, 255), 2),
}


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


def read_vlm(sample: Path, vlm_csv: Path | None) -> dict[int, dict[str, str]]:
    path = vlm_csv or (sample / 'annotations' / 'vlm_handle_visibility_full' / 'qwen_handle_visibility.csv')
    if not path.exists():
        path = sample / 'annotations' / 'vlm_handle_visibility_dense' / 'qwen_handle_visibility.csv'
    # M14 is the first-pass upstream solve. It must read the VLM labels as-is;
    # temporal visibility cleanup and far-side repair belong to later passes.
    return {int(float(r['frame'])): r for r in read_csv(path)} if path.exists() else {}


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y'}


def filter_vlm_short_visible_islands(rows: dict[int, dict[str, str]], max_len: int = 3) -> dict[int, dict[str, str]]:
    # Iteratively merge short visibility runs surrounded by the opposite state.
    # This handles both isolated blips and H/V/H/V transition jitter.
    if not rows:
        return rows
    frames = sorted(rows)
    state = {}
    for fr in frames:
        row = rows[fr]
        part = str(row.get('hand_contact_part', '')).strip().lower()
        visible = str(row.get('visibility', '')).strip().lower() == 'visible'
        handle = truthy(row.get('handle_contact', '')) or part == 'handle'
        state[fr] = bool(visible and handle)

    changed = True
    while changed:
        changed = False
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
        for i, (a, b, val) in enumerate(runs):
            length = b - a + 1
            if length > max_len or i == 0 or i + 1 >= len(runs):
                continue
            prev_val = runs[i - 1][2]
            next_val = runs[i + 1][2]
            if prev_val == next_val and prev_val != val:
                for fr in range(a, b + 1):
                    if fr in state:
                        state[fr] = prev_val
                changed = True
                break

    out = {fr: dict(row) for fr, row in rows.items()}
    for fr in frames:
        raw_part = str(rows[fr].get('hand_contact_part', '')).strip().lower()
        raw_visible = str(rows[fr].get('visibility', '')).strip().lower() == 'visible'
        raw_handle = truthy(rows[fr].get('handle_contact', '')) or raw_part == 'handle'
        raw_state = bool(raw_visible and raw_handle)
        if state[fr] == raw_state:
            continue
        if state[fr]:
            # Short hidden/body island inside a visible-handle run.
            out[fr]['visibility'] = 'visible'
            out[fr]['handle_visible'] = 'True'
            out[fr]['visible_side'] = out[fr].get('visible_side') if out[fr].get('visible_side') in {'left', 'right'} else 'right'
            out[fr]['occlusion_reason'] = 'temporal_short_hidden_island'
            out[fr]['handle_shape_visible'] = out[fr].get('handle_shape_visible') or 'temporal_visible_interpolated'
            out[fr]['hand_contact_part'] = 'handle'
            out[fr]['handle_contact'] = 'True'
            out[fr]['body_contact'] = 'False'
            out[fr]['yaw_anchor_quality'] = 'medium'
            out[fr]['recommended_visibility_constraint'] = 'force_visible_temporal_filter'
            out[fr]['short_reason'] = 'Suppressed short hidden-body island by temporal consistency filter.'
        else:
            # Short visible/handle island inside a hidden/body run.
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
            out[fr]['short_reason'] = 'Suppressed short visible-handle island by temporal consistency filter.'
    return out


def is_visible_handle_vlm(row: dict[str, str]) -> bool:
    visibility = str(row.get('visibility', '')).strip().lower()
    constraint = str(row.get('recommended_visibility_constraint', '')).strip().lower()
    part = str(row.get('hand_contact_part', '')).strip().lower()
    handle = truthy(row.get('handle_contact', '')) or part == 'handle'
    return (visibility == 'visible' or constraint.startswith('force_visible')) and handle


def compute_visibility_alpha(
    frames: list[int],
    vlm_rows: dict[int, dict[str, str]],
    transition: int = 5,
) -> dict[int, float]:
    """Continuous handle visibility weight for rigid emergence/disappearance.

    VLM labels are semantic states, but the rendered Articraft handle is a rigid
    part of the mug. Near visible/hidden boundaries, the handle should gradually
    emerge or disappear instead of switching as a binary detector.
    """
    if not frames:
        return {}
    visible = {fr: is_visible_handle_vlm(vlm_rows.get(fr, {})) for fr in frames}
    alpha = {fr: (1.0 if visible[fr] else 0.0) for fr in frames}

    runs: list[tuple[int, int, bool]] = []
    start = frames[0]
    last = frames[0]
    state = visible[start]
    for fr in frames[1:]:
        if fr != last + 1 or visible[fr] != state:
            runs.append((start, last, state))
            start = fr
            state = visible[fr]
        last = fr
    runs.append((start, last, state))

    for run_idx, (a, b, state) in enumerate(runs):
        if not state:
            continue
        has_hidden_before = run_idx > 0 and not runs[run_idx - 1][2]
        has_hidden_after = run_idx + 1 < len(runs) and not runs[run_idx + 1][2]
        length = b - a + 1
        for fr in range(a, b + 1):
            val = 1.0
            if has_hidden_before:
                val = min(val, (fr - a + 1) / float(transition + 1))
            if has_hidden_after:
                val = min(val, (b - fr + 1) / float(transition + 1))
            if length <= 2 * transition:
                val = min(val, 0.85)
            alpha[fr] = float(np.clip(val, 0.0, 1.0))

    for run_idx, (a, b, state) in enumerate(runs):
        if state:
            continue
        if run_idx > 0 and runs[run_idx - 1][2]:
            for d, fr in enumerate(range(a, min(b, a + transition - 1) + 1), start=1):
                alpha[fr] = max(alpha[fr], max(0.0, (transition - d + 1) / float(transition + 1)) * 0.35)
        if run_idx + 1 < len(runs) and runs[run_idx + 1][2]:
            for d, fr in enumerate(range(b, max(a, b - transition + 1) - 1, -1), start=1):
                alpha[fr] = max(alpha[fr], max(0.0, (transition - d + 1) / float(transition + 1)) * 0.35)
    return alpha


def compute_visibility_side(
    frames: list[int],
    vlm_rows: dict[int, dict[str, str]],
    transition: int = 5,
) -> dict[int, str]:
    """Propagate visible-side labels through transition frames.

    A right-visible handle should emerge from and retreat toward the right side;
    the same applies to the left side. This is separate from alpha magnitude.
    """
    side: dict[int, str] = {}
    if not frames:
        return side
    visible = {fr: is_visible_handle_vlm(vlm_rows.get(fr, {})) for fr in frames}
    runs: list[tuple[int, int, bool]] = []
    start = frames[0]
    last = frames[0]
    state = visible[start]
    for fr in frames[1:]:
        if fr != last + 1 or visible[fr] != state:
            runs.append((start, last, state))
            start = fr
            state = visible[fr]
        last = fr
    runs.append((start, last, state))

    visible_run_sides: list[tuple[int, int, str]] = []
    for run_idx, (a, b, state) in enumerate(runs):
        if not state:
            continue
        counts = {'left': 0, 'right': 0}
        for fr in range(a, b + 1):
            s = str(vlm_rows.get(fr, {}).get('visible_side', '')).strip().lower()
            if s in counts:
                counts[s] += 1
        run_side = 'right' if counts['right'] >= counts['left'] else 'left'
        visible_run_sides.append((a, b, run_side))
        for fr in range(a, b + 1):
            side[fr] = run_side
        # Extend the same side into the nearby hidden transition shoulders.
        if run_idx > 0 and not runs[run_idx - 1][2]:
            prev_a, prev_b, _ = runs[run_idx - 1]
            for fr in range(max(prev_a, prev_b - transition + 1), prev_b + 1):
                side[fr] = run_side
        if run_idx + 1 < len(runs) and not runs[run_idx + 1][2]:
            next_a, next_b, _ = runs[run_idx + 1]
            for fr in range(next_a, min(next_b, next_a + transition - 1) + 1):
                side[fr] = run_side

    return side


def project_mesh(params: np.ndarray, K: np.ndarray, verts: np.ndarray, phase: float) -> tuple[np.ndarray, np.ndarray]:
    pts = verts @ rigid.rot_y(phase).T
    cam = base.transform(params, pts)
    uv = bodyfit.project_pts(params, pts, K)
    return cam, uv


def unwrap_near(value: float, reference: float) -> float:
    return reference + base.wrap(float(value - reference))


def draw_mesh_edges(
    img: np.ndarray,
    params: np.ndarray,
    K: np.ndarray,
    verts: np.ndarray,
    edges: list[tuple[int, int]],
    color: tuple[int, int, int],
    thick: int,
    phase: float,
) -> None:
    _cam, uv = project_mesh(params, K, verts, phase)
    h, w = img.shape[:2]
    for a, b in edges:
        p = uv[a]
        q = uv[b]
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(q))):
            continue
        if p[0] < -w or p[0] > 2 * w or q[0] < -w or q[0] > 2 * w:
            continue
        if p[1] < -h or p[1] > 2 * h or q[1] < -h or q[1] > 2 * h:
            continue
        cv2.line(img, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), color, thick, cv2.LINE_AA)


def draw_phase_direction_arrow(
    img: np.ndarray,
    params: np.ndarray,
    K: np.ndarray,
    phase: float,
    next_phase: float | None,
) -> None:
    if next_phase is None:
        return
    dphase = base.wrap(float(next_phase - phase))
    ddeg = math.degrees(dphase)
    body_local = np.zeros((1, 3), dtype=float)
    _cam, uv = project_mesh(params, K, body_local, phase)
    if not np.all(np.isfinite(uv)):
        return
    cx, cy = uv[0]
    # Draw near the projected mug center but offset upward/right to avoid hiding the body.
    center = np.array([cx + 46.0, cy - 42.0], dtype=float)
    radius = 22.0 + min(12.0, abs(ddeg) * 1.5)
    color = (70, 220, 60) if ddeg >= 0 else (40, 120, 255)
    start_ang = -35.0 if ddeg >= 0 else 215.0
    end_ang = 250.0 if ddeg >= 0 else -70.0
    cv2.ellipse(img, tuple(np.round(center).astype(int)), (int(radius), int(radius)), 0, start_ang, end_ang, color, 3, cv2.LINE_AA)
    # Arrow head tangent to the arc. Positive/negative use opposite directions.
    head_ang = math.radians(end_ang)
    head = center + radius * np.array([math.cos(head_ang), math.sin(head_ang)])
    tangent = np.array([-math.sin(head_ang), math.cos(head_ang)], dtype=float)
    if ddeg < 0:
        tangent = -tangent
    left = head - 10.0 * tangent + 6.0 * np.array([math.cos(head_ang), math.sin(head_ang)])
    right = head - 10.0 * tangent - 6.0 * np.array([math.cos(head_ang), math.sin(head_ang)])
    pts = np.round(np.vstack([head, left, right])).astype(np.int32)
    cv2.fillConvexPoly(img, pts, color, cv2.LINE_AA)
    txt = f'dphase={ddeg:+.1f} deg/fr'
    org = tuple(np.round(center + np.array([-42.0, radius + 24.0])).astype(int))
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, .50, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, .50, color, 1, cv2.LINE_AA)


def body_bbox(row: dict[str, str]) -> np.ndarray | None:
    vals = [ff(row, k) for k in ['body_bbox_x1', 'body_bbox_y1', 'body_bbox_x2', 'body_bbox_y2']]
    if not all(np.isfinite(v) for v in vals) or vals[2] <= vals[0] or vals[3] <= vals[1]:
        vals = [ff(row, k) for k in ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']]
    if not all(np.isfinite(v) for v in vals) or vals[2] <= vals[0] or vals[3] <= vals[1]:
        return None
    return np.asarray(vals, dtype=float)


def is_confirmed_handle_anchor(row: dict[str, str]) -> bool:
    event = str(row.get('object_contact_event', '')).strip()
    state = str(row.get('hand_mug_contact_state', '')).strip()
    use_flag = str(row.get('use_this_point_for_hand_attachment', '')).strip() == '1'
    if not use_flag:
        return False
    # Depth-misaligned visual hints may indicate the handle is starting to emerge,
    # but they are not allowed to overwrite the object-local hand grasp anchor.
    return event == 'hand_handle_grasp' and state == 'direct_hand_grasp_point'


def should_keep_previous_grasp(row: dict[str, str]) -> bool:
    if str(row.get('use_previous_grasp_for_hand_attachment', '')).strip() == '1':
        return True
    event = str(row.get('object_contact_event', '')).strip()
    state = str(row.get('hand_mug_contact_state', '')).strip()
    return event == 'hand_handle_grasp_depth_misaligned' or state == 'visual_contact_needs_depth_alignment'


def optimize_phase(
    sample: Path,
    poses: dict[int, np.ndarray],
    phase0: dict[int, float],
    meshes: dict[str, tuple[np.ndarray, list[tuple[int, int]]]],
    vlm_rows: dict[int, dict[str, str]],
    contact_rows: dict[int, dict[str, str]],
    max_nfev: int,
    boundary_phase: float | None = None,
    boundary_velocity: float | None = None,
) -> tuple[dict[int, float], dict[str, object]]:
    frames = sorted(poses)
    n = len(frames)
    idx = {fr: i for i, fr in enumerate(frames)}
    theta0 = np.unwrap(np.asarray([phase0.get(fr, 0.0) for fr in frames], dtype=float))
    boundary_target0 = None
    # Do not shift the segment initialization to a predicted boundary velocity
    # in M14. The first pass must start from upstream Articraft/VLM phase and
    # let contact constraints select the branch; boundary-velocity steering was
    # an experimental later change and moved frames 64-100 onto the wrong branch.
    visibility_alpha = compute_visibility_alpha(frames, vlm_rows)
    K_all = base.load_K(sample)
    obs_rows = {int(r['frame']): r for r in read_csv(sample / 'results' / 'object_observations' / 'object_observations.csv')}
    proxy_obs_path = sample / 'results' / 'object_proxy_observations' / 'object_proxy_observations.csv'
    proxy_obs_rows = {int(r['frame']): r for r in read_csv(proxy_obs_path)} if proxy_obs_path.exists() else {}
    audio_path = sample / 'results' / 'events' / 'audio_events.csv'
    if not audio_path.exists():
        audio_path = sample / 'results' / 'contact_candidates_object_proxy' / 'audio_events.csv'
    audio_events = []
    if audio_path.exists():
        for ar in read_csv(audio_path):
            af = ff(ar, 'audio_frame')
            asc = ff(ar, 'audio_score', 0.0)
            if np.isfinite(af):
                audio_events.append((int(round(af)), float(asc)))
    hverts = meshes['handle_loop'][0]
    hsample = hverts[:: max(1, len(hverts) // 80)]
    hc_local = np.mean(hverts, axis=0, keepdims=True)
    body_center_local = np.zeros((1, 3), dtype=float)

    stable_grasp_items = []
    stable_hand_by_index: dict[int, tuple[np.ndarray, float]] = {}
    visible_handle_items = []
    hidden_items = []
    stable_grasp_local: np.ndarray | None = None
    stable_grasp_source = 'none'
    num_initial_grasp_carry = 0
    # Segmented optimization must still respect global grasp history. Before
    # the current segment starts, scan earlier confirmed handle frames and carry
    # the latest object-local grasp anchor into this segment. Rim/hidden/body
    # frames before the segment never overwrite the anchor.
    first_frame = frames[0] if frames else 0
    for prev_fr in sorted(k for k in contact_rows if k < first_frame):
        prev = contact_rows[prev_fr]
        if is_confirmed_handle_anchor(prev):
            local = np.array([ff(prev, 'mug_local_x'), ff(prev, 'mug_local_y'), ff(prev, 'mug_local_z')], dtype=float)
            if np.all(np.isfinite(local)):
                stable_grasp_local = local
                stable_grasp_source = f'carried_from_confirmed_handle_frame_{prev_fr}'
                num_initial_grasp_carry += 1
    num_grasp_updates = 0
    num_grasp_keeps = 0
    num_rim_keeps = 0
    for fr in frames:
        p = poses[fr]
        K = K_all[fr - 1]
        vlm = vlm_rows.get(fr, {})
        contact = contact_rows.get(fr, {})
        visibility = str(vlm.get('visibility', '')).strip().lower()
        constraint = str(vlm.get('recommended_visibility_constraint', '')).strip().lower()
        hidden = visibility == 'hidden' or constraint.startswith('force_hidden')
        if hidden:
            bbox = body_bbox(obs_rows.get(fr, {}))
            if bbox is not None:
                hidden_items.append((idx[fr], p, K, bbox, 1.0 - visibility_alpha.get(fr, 0.0)))
        if visibility == 'visible' or constraint.startswith('force_visible'):
            visible_handle_items.append((idx[fr], p, K, str(vlm.get('visible_side', '')).strip().lower(), visibility_alpha.get(fr, 1.0)))
        # Stable hand grasp state machine. Only confirmed handle-contact frames
        # may create/update the object-local grasp anchor. Rim drinking and
        # hidden/occluded frames are hold states and must never overwrite it.
        use_now = is_confirmed_handle_anchor(contact)
        use_prev = should_keep_previous_grasp(contact)
        rim = str(contact.get('rim_drinking_contact', '')).strip() == '1'
        if use_now:
            local = np.array([ff(contact, 'mug_local_x'), ff(contact, 'mug_local_y'), ff(contact, 'mug_local_z')], dtype=float)
            if np.all(np.isfinite(local)):
                stable_grasp_local = local
                stable_grasp_source = f'updated_from_confirmed_handle_frame_{fr}'
                num_grasp_updates += 1
        elif rim:
            num_rim_keeps += 1
        elif use_prev:
            num_grasp_keeps += 1

        if stable_grasp_local is not None and (use_now or use_prev or rim):
            hu = ff(contact, 'active_part_u')
            hv = ff(contact, 'active_part_v')
            hconf = ff(contact, 'active_label_conf', 0.0)
            if np.isfinite(hu) and np.isfinite(hv) and hconf >= 0.15:
                hand_uv = np.array([hu, hv], dtype=float)
                stable_hand_by_index[idx[fr]] = (hand_uv, min(1.0, hconf))
                # M15-compatible stable grasp policy: every frame that is
                # inside the continuous mug grasp interval uses the same
                # object-local grasp anchor. Confirmed visible frames update the
                # anchor; hidden/rim/depth-misaligned frames keep the previous
                # anchor but still constrain the rigid phase against the active
                # hand proxy. This intentionally restores the behavior that
                # produced the better M15 visible alignment. Far-side/hidden
                # visibility remains only a weak auxiliary cue below.
                stable_grasp_items.append((
                    idx[fr], p, K, hand_uv, min(1.0, hconf), stable_grasp_local.copy(),
                    stable_grasp_source, use_now, use_prev, rim, hidden, visibility_alpha.get(fr, 0.0)
                ))
    attachment_indices = {item[0] for item in stable_grasp_items}

    hand_motion_gate_items: list[tuple[int, float, float]] = []
    for i in range(1, n):
        if i not in stable_hand_by_index or (i - 1) not in stable_hand_by_index:
            continue
        uv0, c0 = stable_hand_by_index[i - 1]
        uv1, c1 = stable_hand_by_index[i]
        hand_step_px = float(np.linalg.norm(uv1 - uv0))
        conf = min(c0, c1)
        # If the grasping hand barely moves, the rigid handle phase should not
        # jump. The gate fades out for large hand motion where real rotation is
        # possible.
        gate = max(0.0, min(1.0, 1.0 - hand_step_px / 22.0)) * conf
        if gate > 0.05:
            hand_motion_gate_items.append((i, hand_step_px, gate))

    table_static_items: list[tuple[int, float]] = []
    audio_table_transition_items: list[tuple[int, float]] = []
    table_static_start_idx: int | None = None
    table_audio_event_frame: int | None = None
    if proxy_obs_rows:
        ok = []
        for fr in frames:
            row = proxy_obs_rows.get(fr, {})
            support_conf = ff(row, 'support_conf', 0.0)
            motion_score = ff(row, 'object_motion_score', 1.0)
            ok.append(bool(support_conf >= 0.65 and motion_score <= 0.15))
        # Require a sustained run so a single noisy support frame cannot lock pose.
        for start_i in range(n):
            end_i = start_i
            while end_i < n and ok[end_i]:
                end_i += 1
            if end_i - start_i >= 12:
                table_static_start_idx = start_i
                break
        if table_static_start_idx is not None:
            support_start_frame = frames[table_static_start_idx]
            prior_audio = [(af, sc) for af, sc in audio_events if af <= support_start_frame and support_start_frame - af <= 72 and sc >= 0.25]
            if prior_audio:
                # Use the last strong audio event before sustained visual support
                # as the putdown/contact boundary, but only ramp the static lock
                # until visual support confirms the table state.
                table_audio_event_frame = sorted(prior_audio, key=lambda x: (x[0], x[1]))[-1][0]
                audio_idx = min(range(n), key=lambda j: abs(frames[j] - table_audio_event_frame))
                span = max(1, table_static_start_idx - audio_idx)
                for j in range(audio_idx, table_static_start_idx):
                    progress = (j - audio_idx + 1) / float(span)
                    audio_table_transition_items.append((j, 0.35 * progress))
            for i in range(table_static_start_idx, n):
                fr = frames[i]
                row = proxy_obs_rows.get(fr, {})
                support_conf = ff(row, 'support_conf', 0.0)
                motion_score = ff(row, 'object_motion_score', 1.0)
                if support_conf >= 0.45 and motion_score <= 0.25:
                    weight = float(np.clip(support_conf * (1.0 - motion_score), 0.0, 1.0))
                    table_static_items.append((i, weight))

    def residual(theta: np.ndarray) -> np.ndarray:
        res: list[float] = []
        # Upstream branch prior. This is deliberately weak: M14 must not read
        # the recovered M15 answer as a prior, and it must be able to move away
        # from noisy M12/VLM phase when hand-handle attachment provides a
        # stronger cue.
        for i in range(n):
            res.append(0.35 * base.wrap(float(theta[i] - theta0[i])) / math.radians(25.0))

        # No segment-boundary residual in the M15-compatible first pass. Branch
        # continuity comes from the upstream initialization, multi-start choice,
        # hand attachment terms, and temporal smoothness.

        # Positive hand attachment residual. Direct visible handle contacts
        # update the local anchor. Hidden/use_prev frames keep that object-local
        # anchor and continue to use the active hand proxy as the 2D attachment
        # cue; later far-side/occlusion correction is intentionally not part of
        # this first-pass optimizer.
        for i, p, K, hand_uv, conf, grasp_local, _src, use_now, use_prev, rim, hidden, alpha in stable_grasp_items:
            _cam, uv = project_mesh(p, K, grasp_local[None, :], float(theta[i]))
            if not np.all(np.isfinite(uv)):
                continue
            target_uv = hand_uv.copy()
            d = float(np.linalg.norm(uv[0] - target_uv))
            if use_now:
                policy_weight = 1.00
                scale_px = 5.5
            elif rim:
                policy_weight = 0.30
                scale_px = 10.0
            else:
                policy_weight = 0.60 if d < 90.0 and conf >= 0.15 else 0.35
                scale_px = 8.5
            res.append((24.0 * policy_weight * max(0.35, conf)) * d / scale_px)

        # No visible-side, hidden far-side, or flip repair residuals in M14.
        # Those are second-pass corrections and must not change the accepted
        # first-pass M15 branch.

        for i, hand_step_px, gate in hand_motion_gate_items:
            # Motion-aware rigidity: when the attached hand is nearly static in
            # image space, the handle phase cannot rotate quickly.
            allowed = math.radians(3.0 + 0.20 * hand_step_px)
            # This gate should suppress implausible jitter, not overrule the
            # stronger attachment residual that actually aligns the handle to
            # the hand. Keep it lightweight.
            res.append((0.65 * gate) * base.wrap(float(theta[i] - theta[i - 1])) / max(math.radians(1.5), allowed))

        # Table/static locking: once visual support says the mug is on the
        # table, the handle phase should stop drifting. The preceding audio
        # event only ramps this lock toward the visually confirmed support
        # frame; it does not create a new grasp anchor.
        if table_static_start_idx is not None:
            ref = table_static_start_idx
            for i, weight in audio_table_transition_items:
                res.append((0.8 * weight) * base.wrap(float(theta[i] - theta[ref])) / math.radians(18.0))
            for i, weight in table_static_items:
                res.append((3.5 * weight) * base.wrap(float(theta[i] - theta[ref])) / math.radians(4.0))

        for i in range(1, n):
            vel = float(theta[i] - theta[i - 1])
            res.append(0.8 * vel / math.radians(10.0))
            # A rigid mug handle may rotate, but it cannot teleport out of
            # occlusion in one frame. Penalize only the excess so normal motion
            # around 5-7 deg/frame is still possible.
            excess = max(0.0, abs(vel) - math.radians(8.0))
            res.append(2.0 * excess / math.radians(3.0))
        for i in range(1, n - 1):
            v0 = float(theta[i] - theta[i - 1])
            v1 = float(theta[i + 1] - theta[i])
            acc = v1 - v0
            res.append(1.2 * acc / math.radians(5.0))
            excess_acc = max(0.0, abs(acc) - math.radians(3.0))
            res.append(2.0 * excess_acc / math.radians(3.0))
        return np.asarray(res, dtype=float)

    # Restore M14 to the original first-pass behavior: one solve initialized
    # from the upstream Articraft/VLM phase. Branch flipping, far-side repair,
    # and recovered-M15 priors belong to later diagnostics/second-pass stages.
    init_offsets = [0.0]
    best_offset = 0.0
    opt = least_squares(residual, theta0.copy(), loss='soft_l1', f_scale=1.0, max_nfev=max_nfev)
    theta = np.unwrap(opt.x)

    out = {fr: base.wrap(float(theta[i])) for i, fr in enumerate(frames)}

    meta = {
        'success': bool(opt.success),
        'cost': float(opt.cost),
        'nfev': int(opt.nfev),
        'message': str(opt.message),
        'num_initializations': len(init_offsets),
        'best_initial_offset_deg': float(math.degrees(best_offset)),
        'num_stable_grasp_items': len(stable_grasp_items),
        'num_initial_grasp_carry': num_initial_grasp_carry,
        'num_grasp_updates': num_grasp_updates,
        'num_grasp_keeps': num_grasp_keeps,
        'num_rim_keeps': num_rim_keeps,
        'num_visible_handle_items': len(visible_handle_items),
        'num_side_alpha_items': 0,
        'num_hidden_transition_side_items': 0,
        'num_hidden_items': len(hidden_items),
        'num_hand_motion_gate_items': len(hand_motion_gate_items),
        'num_table_static_items': len(table_static_items),
        'num_audio_table_transition_items': len(audio_table_transition_items),
        'table_audio_event_frame': table_audio_event_frame,
        'table_static_start_frame': frames[table_static_start_idx] if table_static_start_idx is not None else None,
        'visibility_transition': 'computed_for_diagnostics_only_not_used_as_phase_repair',
        'anchor_update_policy': 'M15-compatible first-pass: only confirmed visible hand-handle grasp updates the stable object-local anchor; hidden/keep_previous keeps the previous local anchor and uses the active hand proxy as the attachment cue; rim does not update the grasp anchor',
        'hand_motion_gate_policy': 'small active-hand motion penalizes fast handle phase changes',
        'table_static_policy': 'audio-ramped table support softly locks phase to the sustained visual support frame; render also freezes pose/phase after support',
        'visible_side_policy': 'disabled_in_M14_first_pass_second_pass_only',
        'boundary_phase': boundary_phase,
        'boundary_velocity': boundary_velocity,
        'boundary_continuity_policy': 'segment-boundary velocity/far-side repair disabled in M14 first pass',
        'postprocess_policy': 'disabled; M14 first-pass returns the direct optimizer result',
    }
    return out, meta


def finalize(tmp: Path, out: Path) -> None:
    ffmpeg = shutil.which('ffmpeg') or '/usr/bin/ffmpeg'
    try:
        subprocess.run([ffmpeg, '-y', '-i', str(tmp), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out)], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        subprocess.run([ffmpeg, '-y', '-i', str(tmp), '-c:v', 'libopenh264', '-pix_fmt', 'yuv420p', str(out)], check=True, capture_output=True, text=True)
    tmp.unlink(missing_ok=True)


def render(
    sample: Path,
    meshes: dict[str, tuple[np.ndarray, list[tuple[int, int]]]],
    poses: dict[int, np.ndarray],
    phases: dict[int, float],
    vlm_rows: dict[int, dict[str, str]],
    meta: dict[str, object],
    fps: float,
    out_dir: Path | None = None,
) -> dict[str, str]:
    out_dir = out_dir or (sample / 'results' / 'renders' / 'M14_joint_contact_handle_phase')
    out_dir.mkdir(parents=True, exist_ok=True)
    K_all = base.load_K(sample)
    first = cv2.imread(str(sample / 'frames' / '00001.png'))
    h, w = first.shape[:2]
    tmp = out_dir / 'overlay.tmp.mp4'
    out = out_dir / 'overlay.mp4'
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    visibility_alpha = compute_visibility_alpha(sorted(poses), vlm_rows)
    rows = []
    ordered_frames = sorted(poses)
    next_phase_by_frame = {fr: phases.get(ordered_frames[i + 1]) for i, fr in enumerate(ordered_frames[:-1])}
    next_phase_by_frame[ordered_frames[-1]] = None
    for fr in ordered_frames:
        img = cv2.imread(str(sample / 'frames' / f'{fr:05d}.png'))
        if img is None:
            continue
        overlay = img.copy()
        phase = phases[fr]
        vlm = vlm_rows.get(fr, {})
        alpha = visibility_alpha.get(fr, 0.0)
        visible_state = is_visible_handle_vlm(vlm)
        show_handle = visible_state and alpha > 0.12
        for name in ['body_shell', 'rim_ring', 'bottom_disk']:
            color, thick = PART_STYLE[name]
            verts, edges = meshes[name]
            draw_mesh_edges(overlay, poses[fr], K_all[fr - 1], verts, edges, color, thick, phase)
        cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
        if show_handle:
            # A low-alpha transition frame means the handle is only emerging from
            # occlusion. Draw it after body blending so it is not overwritten.
            handle_layer = img.copy()
            color, thick = PART_STYLE['handle_loop']
            verts, edges = meshes['handle_loop']
            draw_mesh_edges(handle_layer, poses[fr], K_all[fr - 1], verts, edges, color, max(1, thick if alpha > 0.55 else 1), phase)
            handle_blend = 0.22 + 0.58 * min(1.0, alpha)
            cv2.addWeighted(handle_layer, handle_blend, img, 1.0 - handle_blend, 0, img)
        draw_phase_direction_arrow(img, poses[fr], K_all[fr - 1], phase, next_phase_by_frame.get(fr))
        label = f'M14_joint_contact_handle_phase frame {fr:03d} phase={math.degrees(phase):.1f} alpha={alpha:.2f} vlm={vlm.get("visibility","")}/{vlm.get("hand_contact_part","")}'
        cv2.putText(img, label, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, .65, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(img, label, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, .65, (245, 245, 245), 1, cv2.LINE_AA)
        writer.write(img)
        rows.append({
            'frame': fr,
            'time': (fr - 1) / fps,
            'mug_axial_phase_rad': phase,
            'mug_axial_phase_deg': math.degrees(phase),
            'vlm_visibility': vlm.get('visibility', ''),
            'vlm_hand_contact_part': vlm.get('hand_contact_part', ''),
            'vlm_handle_contact': vlm.get('handle_contact', ''),
            'visibility_alpha': alpha,
        })
    writer.release()
    finalize(tmp, out)
    phase_csv = out_dir / 'handle_phase_joint_contact.csv'
    with phase_csv.open('w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    summary = {
        'render': str(out),
        'phase_csv': str(phase_csv),
        'note': 'M14 first-pass optimizer starts from upstream M12/Articraft/VLM axial phase by default, then uses stable object-local hand grasp attachment plus lightweight smooth/hand-motion/table-static terms to solve the mug handle phase. Recovered M15 phase is not used unless explicitly passed with --phase-csv. Visible-side, hidden far-side, and flip repair are intentionally disabled here; they belong to later second-pass processing.',
        **meta,
    }
    (out_dir / 'outputs.json').write_text(json.dumps(summary, indent=2))
    return summary


def freeze_table_static_pose(
    pose_csv: Path,
    poses: dict[int, np.ndarray],
    phases: dict[int, float],
    static_start: int | None,
    sample: Path,
) -> tuple[dict[int, np.ndarray], dict[int, float], Path | None]:
    if static_start is None or static_start not in poses or static_start not in phases:
        return poses, phases, None
    frozen_poses = {fr: p.copy() for fr, p in poses.items()}
    frozen_phases = dict(phases)
    ref_pose = poses[static_start].copy()
    ref_phase = phases[static_start]
    for fr in frozen_poses:
        if fr >= static_start:
            frozen_poses[fr] = ref_pose.copy()
            frozen_phases[fr] = ref_phase

    out_csv = sample / 'proxy' / 'mug_body_only_cylinder_pose_table_static_sequence.csv'
    rows = read_csv(pose_csv)
    out_rows = []
    for row in rows:
        fr = int(float(row['frame']))
        new = dict(row)
        if fr >= static_start:
            p = ref_pose
            vals = {
                'x': p[0], 'y': p[1], 'z': p[2],
                'yaw': p[3], 'pitch': p[4], 'roll': p[5], 'scale': p[6],
                'yaw_deg': math.degrees(p[3]),
                'pitch_deg': math.degrees(p[4]),
                'roll_deg': math.degrees(p[5]),
            }
            for k, v in vals.items():
                if k in new:
                    new[k] = f'{float(v):.12f}'
        out_rows.append(new)
    if out_rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open('w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            wr.writeheader()
            wr.writerows(out_rows)
    return frozen_poses, frozen_phases, out_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-dir', type=Path, default=Path('samples_known_object/02_mug'))
    ap.add_argument('--mesh-root', type=Path, default=Path('samples_known_object/02_mug/articraft/materialized_mug_mesh'))
    ap.add_argument('--pose-csv', type=Path, default=None)
    ap.add_argument('--phase-csv', type=Path, default=None)
    ap.add_argument('--contact-csv', type=Path, default=None)
    ap.add_argument('--vlm-csv', type=Path, default=None)
    ap.add_argument('--fps', type=float, default=24.0)
    ap.add_argument('--max-nfev', type=int, default=96)
    ap.add_argument('--segments', type=str, default='1-60,60-100,100-160,160-240')
    ap.add_argument('--out-dir', type=Path, default=None)
    args = ap.parse_args()

    sample = args.sample_dir
    default_static_pose = sample / 'proxy' / 'mug_body_only_cylinder_pose_table_static_sequence.csv'
    default_segmented_pose = sample / 'proxy' / 'mug_body_only_cylinder_pose_segmented_sequence.csv'
    pose_csv = args.pose_csv or (default_static_pose if default_static_pose.exists() else default_segmented_pose)
    # Clean first-pass input: default to the upstream Articraft/VLM phase. The
    # recovered M15 phase is a debug/regression artifact only and must be passed
    # explicitly with --phase-csv if someone wants to compare against it.
    default_m12_phase = sample / 'results' / 'renders' / 'M12_articraft_rigid_mesh_vlm' / 'handle_phase_all.csv'
    phase_csv = args.phase_csv or default_m12_phase
    contact_csv = args.contact_csv or (sample / 'results' / 'mug_articraft_contact_points' / 'mug_articraft_contact_points.csv')
    poses = read_pose_sequence(pose_csv)
    phase0 = read_phase_sequence(phase_csv)
    vlm_rows = read_vlm(sample, args.vlm_csv)
    contact_rows = {int(float(r['frame'])): r for r in read_csv(contact_csv)}
    meshes = rigid.load_articraft_meshes(args.mesh_root)
    phases: dict[int, float] = {}
    segment_meta: list[dict[str, object]] = []
    if args.segments.strip():
        seen: set[int] = set()
        prev_boundary_phase: float | None = None
        prev_boundary_velocity: float | None = None
        for spec in args.segments.split(','):
            start_s, end_s = [int(x) for x in spec.split('-')]
            sub_poses = {fr: p for fr, p in poses.items() if start_s <= fr <= end_s and fr not in seen}
            if not sub_poses:
                continue
            sub_phases, sub_meta = optimize_phase(
                sample,
                sub_poses,
                phase0,
                meshes,
                vlm_rows,
                contact_rows,
                args.max_nfev,
                boundary_phase=prev_boundary_phase,
                boundary_velocity=prev_boundary_velocity,
            )
            ordered_sub = sorted(sub_phases)
            for fr in ordered_sub:
                ph = sub_phases[fr]
                if prev_boundary_phase is not None and fr == ordered_sub[0]:
                    ph = unwrap_near(ph, prev_boundary_phase)
                phases[fr] = ph
                seen.add(fr)
            if len(ordered_sub) >= 2:
                last_fr = ordered_sub[-1]
                prev_fr = ordered_sub[-2]
                last_phase = phases[last_fr]
                prev_phase = unwrap_near(phases[prev_fr], last_phase)
                prev_boundary_velocity = base.wrap(last_phase - prev_phase)
                prev_boundary_phase = last_phase + prev_boundary_velocity
            elif ordered_sub:
                prev_boundary_phase = phases[ordered_sub[-1]]
                prev_boundary_velocity = 0.0
            sub_meta = dict(sub_meta)
            sub_meta['segment'] = spec
            sub_meta['num_frames'] = len(sub_poses)
            segment_meta.append(sub_meta)
        missing = [fr for fr in poses if fr not in phases]
        if missing:
            sub_poses = {fr: poses[fr] for fr in missing}
            sub_phases, sub_meta = optimize_phase(sample, sub_poses, phase0, meshes, vlm_rows, contact_rows, max(8, args.max_nfev // 2))
            phases.update(sub_phases)
            sub_meta = dict(sub_meta)
            sub_meta['segment'] = 'missing'
            sub_meta['num_frames'] = len(sub_poses)
            segment_meta.append(sub_meta)
        meta = {
            'success': all(bool(m.get('success')) for m in segment_meta),
            'segments': segment_meta,
            'mode': 'segmented_joint_phase',
        }
    else:
        phases, meta = optimize_phase(sample, poses, phase0, meshes, vlm_rows, contact_rows, args.max_nfev)
        meta['mode'] = 'full_sequence_joint_phase'
    static_candidates = [int(m['table_static_start_frame']) for m in segment_meta if m.get('table_static_start_frame')] if 'segment_meta' in locals() else []
    global_static_start = min(static_candidates) if static_candidates else meta.get('table_static_start_frame')
    if global_static_start is not None:
        global_static_start = int(global_static_start)
    # Restore original M15-style table behavior: after sustained table support,
    # pose and phase are held fixed for rendering.
    render_poses, render_phases, static_pose_csv = freeze_table_static_pose(pose_csv, poses, phases, global_static_start, sample)

    meta.update({
        'pose_csv': str(pose_csv),
        'phase_prior_csv': str(phase_csv),
        'contact_csv': str(contact_csv),
        'vlm_csv': str(args.vlm_csv or (sample / 'annotations' / 'vlm_handle_visibility_full' / 'qwen_handle_visibility.csv')),
        'mesh_root': str(args.mesh_root),
        'table_static_freeze_start_frame': global_static_start if global_static_start is not None else '',
        'table_static_pose_csv': str(static_pose_csv) if static_pose_csv is not None else '',
        'table_static_freeze_policy': 'enabled: hold pose and phase fixed after sustained table support',
    })
    print(json.dumps(render(sample, meshes, render_poses, render_phases, vlm_rows, meta, args.fps, args.out_dir), indent=2))


if __name__ == '__main__':
    main()
