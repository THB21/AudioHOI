#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
STAGE1 = HERE.parent / 'stage1_observation'
sys.path.insert(0, str(STAGE1))

import render_mug_articraft_joint_contact_phase as m14  # noqa: E402
import fit_mug_articraft_keyframe_pose as base  # noqa: E402


FIELDS = [
    'frame',
    'time',
    'mug_axial_phase_rad',
    'mug_axial_phase_deg',
    'vlm_visibility',
    'vlm_hand_contact_part',
    'vlm_handle_contact',
    'visibility_alpha',
    'source',
    'reprojection_error_px',
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def ff(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, '') or default)
    except Exception:
        return default


def wrap(a: float) -> float:
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def unwrap_near(value: float, reference: float) -> float:
    value = float(value)
    reference = float(reference)
    return value + round((reference - value) / (2.0 * math.pi)) * 2.0 * math.pi


def read_phase_context(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    return {int(float(r['frame'])): r for r in read_csv(path)}


def solve_phase_for_anchor(
    pose: np.ndarray,
    K: np.ndarray,
    local: np.ndarray,
    target_uv: np.ndarray,
    previous_phase: float | None,
) -> tuple[float, float]:
    def cost(theta: float) -> float:
        _cam, uv = m14.project_mesh(pose, K, local[None, :], wrap(theta))
        if not np.all(np.isfinite(uv)):
            return 1e12
        return float(np.sum((uv[0] - target_uv) ** 2))

    # Multi-start grid avoids choosing the wrong periodic branch before the
    # continuity unwrap below.
    grid = np.linspace(-math.pi, math.pi, 721)
    costs = np.asarray([cost(t) for t in grid])
    center = float(grid[int(np.argmin(costs))])
    step = 2.0 * math.pi / 720.0
    lo = center - 4.0 * step
    hi = center + 4.0 * step
    res = minimize_scalar(cost, bounds=(lo, hi), method='bounded', options={'xatol': 1e-7})
    phase = wrap(float(res.x if res.success else center))
    if previous_phase is not None:
        phase = unwrap_near(phase, previous_phase)
    return phase, math.sqrt(max(0.0, float(res.fun if res.success else np.min(costs))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=Path, default=Path('samples_known_object/02_mug'))
    ap.add_argument('--m15-csv', type=Path, default=None)
    ap.add_argument('--pose-csv', type=Path, default=None)
    ap.add_argument('--context-phase-csv', type=Path, default=None)
    ap.add_argument('--out-dir', type=Path, default=None)
    args = ap.parse_args()

    sample = args.sample
    m15_csv = args.m15_csv or (sample / 'results' / 'renders' / 'M15_grasp_anchor_projection' / 'grasp_anchor_projection.csv')
    pose_csv = args.pose_csv or (sample / 'proxy' / 'mug_body_only_cylinder_pose_table_static_sequence.csv')
    context_phase_csv = args.context_phase_csv or (sample / 'results' / 'renders' / 'M14_joint_contact_handle_phase' / 'handle_phase_joint_contact.csv')
    out_dir = args.out_dir or (sample / 'results' / 'renders' / 'M15_original_phase_recovered')
    out_dir.mkdir(parents=True, exist_ok=True)

    m15_rows = read_csv(m15_csv)
    poses = m14.read_pose_sequence(pose_csv)
    K_all = base.load_K(sample)
    context = read_phase_context(context_phase_csv)

    rows: list[dict[str, object]] = []
    prev_phase: float | None = None
    for row in m15_rows:
        fr = int(float(row['frame']))
        if fr not in poses:
            continue
        local = np.array([ff(row, 'anchor_local_x'), ff(row, 'anchor_local_y'), ff(row, 'anchor_local_z')], dtype=float)
        target_uv = np.array([ff(row, 'anchor_u'), ff(row, 'anchor_v')], dtype=float)
        if not np.all(np.isfinite(local)) or not np.all(np.isfinite(target_uv)):
            continue
        phase, err = solve_phase_for_anchor(poses[fr], K_all[fr - 1], local, target_uv, prev_phase)
        prev_phase = phase
        ctx = context.get(fr, {})
        rows.append({
            'frame': fr,
            'time': ff(row, 'time', (fr - 1) / 24.0),
            'mug_axial_phase_rad': wrap(phase),
            'mug_axial_phase_deg': math.degrees(wrap(phase)),
            'vlm_visibility': ctx.get('vlm_visibility', ''),
            'vlm_hand_contact_part': ctx.get('vlm_hand_contact_part', ''),
            'vlm_handle_contact': ctx.get('vlm_handle_contact', ''),
            'visibility_alpha': ctx.get('visibility_alpha', ''),
            'source': 'recovered_from_M15_anchor_projection',
            'reprojection_error_px': err,
        })

    out_csv = out_dir / 'handle_phase_joint_contact.csv'
    with out_csv.open('w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)

    errs = [float(r['reprojection_error_px']) for r in rows]
    meta = {
        'csv': str(out_csv),
        'm15_csv': str(m15_csv),
        'pose_csv': str(pose_csv),
        'context_phase_csv': str(context_phase_csv),
        'num_frames': len(rows),
        'mean_reprojection_error_px': float(np.mean(errs)) if errs else None,
        'median_reprojection_error_px': float(np.median(errs)) if errs else None,
        'note': (
            'Recovered the original phase branch used by the preserved M15 render by inverting '
            'the saved stable grasp anchor projection. This is the frozen baseline for second-pass '
            'far-side/occlusion correction; it does not rerun the M14 full optimizer and does not '
            'overwrite M15.'
        ),
    }
    out_json = out_dir / 'outputs.json'
    out_json.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
