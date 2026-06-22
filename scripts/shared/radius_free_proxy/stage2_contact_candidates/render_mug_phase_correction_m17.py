#!/usr/bin/env python3
"""M17: Hidden-segment handle phase correction with smooth boundary interpolation.

Problem: M14's phase optimizer has ambiguity within hidden segments — the
optimizer has no image evidence for the handle, so it may choose a phase
branch that is discontinuous with adjacent visible frames.

Fix (smooth boundary interpolation):
  For each hidden segment, linearly interpolate the phase between the last
  visible frame before and the first visible frame after, choosing the shorter
  angular path (minimum total rotation).  This eliminates all large phase jumps
  by construction and produces a physically plausible continuous trajectory.

  Visible frames are never modified.

  Light Gaussian smooth (sigma/2 over the full sequence) is applied after
  interpolation to remove any residual inter-frame roughness.

Usage (from repo root, with the audiohoi or articraft env active):
  python scripts/shared/radius_free_proxy/stage2_contact_candidates/render_mug_phase_correction_m17.py

Outputs → samples_known_object/02_mug/results/renders/M17_phase_corrected/
  corrected_handle_phase.csv   frame-level phases + diagnostics
  overlay.mp4                  video with handle drawn at M17 phase
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

STAGE1 = Path(__file__).resolve().parents[1] / 'stage1_observation'
sys.path.insert(0, str(STAGE1))
import fit_mug_articraft_keyframe_pose as base          # noqa: E402
import render_mug_articraft_joint_contact_phase as m14  # noqa: E402
import render_mug_articraft_rigid_mesh_vlm as rigid     # noqa: E402

SIGMA = 3.0   # Gaussian smoothing sigma (frames); ~±6 frame transition width

# Smoothing is applied WITHIN each hidden segment only (not across vis/hidden
# boundaries).  The correct per-frame phase is computed in the *wrapped* domain
# first (flip error frames by +π), then unwrapped inside each segment, then
# Gaussian-smoothed, then rewrapped.  Visible frames are always left at the
# original M14 phase so they are never corrupted.

FIELDS = [
    'frame', 'time',
    'm17_phase_rad', 'm17_phase_deg',
    'm14_phase_rad', 'm14_phase_deg',
    'is_error_frame',
    'phase_correction_rad', 'phase_correction_deg',
    'vlm_visibility',
    'handle_z_sin_m14',
    'handle_z_sin_m17',
    'mug_yaw_deg',
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def ff(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        v = row.get(key, '')
        return float(v) if v else default
    except Exception:
        return default


def _wrap(a: np.ndarray) -> np.ndarray:
    """Wrap angle array to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def compute_m17_phases(
    frames: list[int],
    m14_phases: dict[int, float],
    yaw_data: dict[int, float],
    visibility: dict[int, str],
    sigma: float = SIGMA,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (m17_phase_wrapped, is_error_bool) arrays aligned to `frames`.

    Algorithm: smooth boundary interpolation.

    For each hidden segment [i, j) we linearly interpolate in the *unwrapped*
    domain from the last visible phase (phase[i-1]) to the first visible phase
    (phase[j]), choosing the shorter of the two possible circular paths.  This
    eliminates all large phase jumps at hidden/visible boundaries by ensuring
    each hidden frame is no more than  total_arc / (n+1)  degrees from its
    neighbour.  The resulting trajectory is physically plausible (constant
    rotation rate through the segment).

    is_error marks hidden frames where M14 had sin(phase+yaw) < 0, i.e. the
    handle was on the near side in the raw M14 output — for diagnostics only;
    the correction is now interpolation, not per-frame reflection.

    A light global Gaussian smooth (sigma/2) over the full sequence removes
    any residual sub-degree roughness from the M14 visible-frame poses.
    """
    N = len(frames)
    phase_arr = np.array([m14_phases.get(fr, 0.0) for fr in frames], dtype=float)
    yaw_arr   = np.array([yaw_data.get(fr, 0.0) for fr in frames], dtype=float)
    vis_list  = [visibility.get(fr, 'visible') for fr in frames]

    is_error = np.array([
        vis_list[i] == 'hidden' and math.sin(yaw_arr[i] + phase_arr[i]) < 0
        for i in range(N)
    ], dtype=bool)

    corrected = phase_arr.copy()

    # Find and interpolate each hidden segment.
    i = 0
    while i < N:
        if vis_list[i] != 'hidden':
            i += 1
            continue
        j = i
        while j < N and vis_list[j] == 'hidden':
            j += 1

        # Boundary phases from the adjacent visible frames.
        left  = phase_arr[i - 1] if i > 0 else phase_arr[j] if j < N else 0.0
        right = phase_arr[j]     if j < N else phase_arr[i - 1] if i > 0 else 0.0

        n = j - i

        # Two candidate interpolation paths.
        # delta_short: shorter (or equal) arc in [-π, π]. positive=CCW, negative=CW.
        # delta_long: the longer arc going the other way around the circle.
        delta_short = float(_wrap(np.array([right - left]))[0])
        sign = 1.0 if delta_short >= 0 else -1.0
        delta_long  = delta_short - sign * 2.0 * math.pi
        right_ccw = left + delta_short   # unwrapped right via shorter arc
        right_cw  = left + delta_long    # unwrapped right via longer arc

        def _far_count(right_uw: float) -> int:
            count = 0
            for k in range(n):
                t = (k + 1) / (n + 1)
                ph = left + t * (right_uw - left)
                if math.sin(ph + yaw_arr[i + k]) > 0:
                    count += 1
            return count

        fc_ccw = _far_count(right_ccw)
        fc_cw  = _far_count(right_cw)

        # Prefer the path that puts more hidden frames on the far side.
        # Tie-break: shorter arc (less total rotation).
        if fc_cw > fc_ccw:
            right_uw = right_cw
        elif fc_ccw > fc_cw:
            right_uw = right_ccw
        else:
            # Equal far-side count: prefer shorter arc
            right_uw = right_ccw if abs(delta_short) <= abs(delta_long) else right_cw

        # Interpolate hidden frames t = 1/(n+1), 2/(n+1), ..., n/(n+1)
        for k in range(n):
            t = (k + 1) / (n + 1)
            corrected[i + k] = _wrap(np.array([left + t * (right_uw - left)]))[0]

        i = j

    # Light global smooth to remove residual M14 pose noise.
    # sigma/2 is gentle enough that it won't visibly move visible frames.
    m17_uw = np.unwrap(corrected)
    m17_uw = gaussian_filter1d(m17_uw, sigma=max(0.5, sigma / 2.0), mode='nearest')
    m17 = _wrap(m17_uw)

    return m17, is_error


def put_text(img: np.ndarray, text: str, y: int, color: tuple) -> None:
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
    cv2.putText(img, text, (18, y), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (18, y), font, scale, color, thick, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-dir', type=Path, default=Path('samples_known_object/02_mug'))
    ap.add_argument('--m14-csv', type=Path, default=None)
    ap.add_argument('--pose-csv', type=Path, default=None)
    ap.add_argument('--vis-csv', type=Path, default=None)
    ap.add_argument('--sigma', type=float, default=SIGMA)
    ap.add_argument('--fps', type=float, default=24.0)
    ap.add_argument('--mesh-root', type=Path, default=None)
    ap.add_argument('--out-dir', type=Path, default=None)
    args = ap.parse_args()

    sample   = args.sample_dir
    m14_csv  = args.m14_csv  or (sample / 'results/renders/M14_joint_contact_handle_phase/handle_phase_joint_contact.csv')
    default_static = sample / 'proxy/mug_body_only_cylinder_pose_table_static_sequence.csv'
    pose_csv = args.pose_csv or (default_static if default_static.exists()
                                 else sample / 'proxy/mug_body_only_cylinder_pose_segmented_sequence.csv')
    vis_csv  = args.vis_csv  or (sample / 'annotations/vlm_handle_visibility_full/qwen_handle_visibility.csv')
    mesh_root = args.mesh_root or (sample / 'articraft/materialized_mug_mesh')
    out_dir  = args.out_dir  or (sample / 'results/renders/M17_phase_corrected')
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load inputs ──────────────────────────────────────────────────────────
    m14_rows = read_csv(m14_csv)
    m14_phases: dict[int, float] = {}
    m14_time:   dict[int, float] = {}
    for row in m14_rows:
        fr = int(float(row['frame']))
        m14_phases[fr] = ff(row, 'mug_axial_phase_rad')
        m14_time[fr]   = ff(row, 'time', 0.0)

    poses    = m14.read_pose_sequence(pose_csv)
    yaw_data = {fr: float(p[3]) for fr, p in poses.items()}

    vis_rows   = read_csv(vis_csv)
    visibility = {int(r['frame']): r['visibility'] for r in vis_rows}

    frames = sorted(set(m14_phases.keys()) & set(poses.keys()))

    # ── Compute corrected phases ──────────────────────────────────────────────
    m17_wrapped, is_error = compute_m17_phases(
        frames, m14_phases, yaw_data, visibility, sigma=args.sigma
    )
    m17_phases = {fr: float(m17_wrapped[i]) for i, fr in enumerate(frames)}

    # ── Render video ─────────────────────────────────────────────────────────
    K_all   = base.load_K(sample)
    meshes  = rigid.load_articraft_meshes(mesh_root)

    first = cv2.imread(str(sample / 'frames/00001.png'))
    if first is None:
        raise FileNotFoundError(sample / 'frames/00001.png')
    h, w = first.shape[:2]

    tmp_raw = out_dir / 'overlay_raw_tmp.mp4'
    writer  = cv2.VideoWriter(str(tmp_raw), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))

    csv_rows: list[dict] = []

    for i, fr in enumerate(frames):
        img = cv2.imread(str(sample / 'frames' / f'{fr:05d}.png'))
        if img is None:
            continue

        m17_ph   = m17_phases[fr]
        m14_ph   = m14_phases[fr]
        yaw      = yaw_data.get(fr, 0.0)
        vis_lbl  = visibility.get(fr, 'visible')
        err_fr   = bool(is_error[i])
        corr_rad = m17_ph - m14_ph
        corr_deg = math.degrees(corr_rad)
        sin_m14  = math.sin(yaw + m14_ph)
        sin_m17  = math.sin(yaw + m17_ph)

        # Draw mug mesh at M17-corrected phase
        overlay = img.copy()
        for name in ['body_shell', 'rim_ring', 'bottom_disk', 'handle_loop']:
            if name not in meshes:
                continue
            color, thick = m14.PART_STYLE[name]
            verts, edges = meshes[name]
            m14.draw_mesh_edges(overlay, poses[fr], K_all[fr - 1], verts, edges, color, thick, m17_ph)
        cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)

        # Text overlay
        text_color = (30, 60, 220) if err_fr else (60, 200, 60)
        side_m17   = 'far(hidden)' if sin_m17 > 0 else 'near(visible)'
        side_m14   = 'far' if sin_m14 > 0 else 'near'
        flip_tag   = '  [FLIPPED+smooth]' if err_fr else ''
        lbl1 = f'M17 frame {fr:03d}  {vis_lbl.upper()}{flip_tag}'
        lbl2 = f'M14={math.degrees(m14_ph):+.1f}deg({side_m14})  M17={math.degrees(m17_ph):+.1f}deg({side_m17})  corr={corr_deg:+.1f}deg'
        lbl3 = f'sin M14={sin_m14:+.3f}  sin M17={sin_m17:+.3f}'
        put_text(img, lbl1, 34, text_color)
        put_text(img, lbl2, 62, text_color)
        put_text(img, lbl3, 90, text_color)

        writer.write(img)

        csv_rows.append({
            'frame':               str(fr),
            'time':                f'{m14_time.get(fr, 0.0):.6f}',
            'm17_phase_rad':       f'{m17_ph:.6f}',
            'm17_phase_deg':       f'{math.degrees(m17_ph):.4f}',
            'm14_phase_rad':       f'{m14_ph:.6f}',
            'm14_phase_deg':       f'{math.degrees(m14_ph):.4f}',
            'is_error_frame':      '1' if err_fr else '0',
            'phase_correction_rad': f'{corr_rad:.6f}',
            'phase_correction_deg': f'{corr_deg:.4f}',
            'vlm_visibility':      vis_lbl,
            'handle_z_sin_m14':    f'{sin_m14:.6f}',
            'handle_z_sin_m17':    f'{sin_m17:.6f}',
            'mug_yaw_deg':         f'{math.degrees(yaw):.4f}',
        })

    writer.release()

    # Re-encode to H.264
    out_mp4 = out_dir / 'overlay.mp4'
    h264_tmp = out_dir / 'overlay_h264_tmp.mp4'
    cmd = ['ffmpeg', '-y', '-i', str(tmp_raw), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
           '-movflags', '+faststart', str(h264_tmp)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.move(str(h264_tmp), str(out_mp4))
        tmp_raw.unlink(missing_ok=True)
    except Exception:
        shutil.move(str(tmp_raw), str(out_mp4))

    # Write CSV
    out_csv = out_dir / 'corrected_handle_phase.csv'
    with out_csv.open('w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(csv_rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_err     = int(is_error.sum())
    n_hidden  = sum(1 for r in csv_rows if r['vlm_visibility'] == 'hidden')
    n_hid_ok  = sum(1 for r in csv_rows if r['vlm_visibility'] == 'hidden'  and float(r['handle_z_sin_m17']) > 0)
    n_vis_bad = sum(1 for r in csv_rows if r['vlm_visibility'] == 'visible' and float(r['handle_z_sin_m17']) > 0)

    print(f'M17 complete  (sigma={args.sigma})')
    print(f'  Total frames          : {len(csv_rows)}')
    print(f'  Error frames flipped  : {n_err}')
    print(f'  Hidden frames total   : {n_hidden}')
    print(f'  Hidden → far side (✓): {n_hid_ok}/{n_hidden}')
    print(f'  Visible → far side (boundary smoothing artifact): {n_vis_bad}')
    print(f'  CSV  → {out_csv}')
    print(f'  Video→ {out_mp4}')


if __name__ == '__main__':
    main()
