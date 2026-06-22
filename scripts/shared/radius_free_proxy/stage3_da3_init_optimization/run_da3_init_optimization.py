#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from scripts.shared.radius_free_proxy.stage1_observation.object_proxy_observation_utils import (  # noqa: E402
    build_body_joints,
    build_contact_part_centers,
    read_human_result,
)
from scripts.shared.human_ball.contact.contact_part_utils import normalize_contact_label  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_outputs(out_dir: Path, times: np.ndarray, init_t: np.ndarray, opt_t: np.ndarray, residual_px: np.ndarray) -> None:
    fig = plt.figure(figsize=(8.5, 6.5), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(opt_t[:, 0], opt_t[:, 2], opt_t[:, 1], linewidth=2.2, color="#4c72b0")
    ax.scatter(opt_t[:, 0], opt_t[:, 2], opt_t[:, 1], c=times, cmap="plasma", s=14)
    ax.set_xlabel("X_cam (m)")
    ax.set_ylabel("Z_cam (m)")
    ax.set_zlabel("Y_cam (m)")
    ax.set_title("Radius-free DA3 contact-aware init")
    ax.view_init(elev=22, azim=-58)
    fig.tight_layout()
    fig.savefig(out_dir / "object_pose6d_sharedcam_plot.png")
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(9, 8.5), dpi=140, sharex=True)
    labels = [("X_cam", 0), ("Y_cam", 1), ("Z_cam", 2)]
    for ax, (name, col) in zip(axes[:3], labels):
        ax.plot(times, init_t[:, col], label=f"{name} init", color="#c7d2e8")
        ax.plot(times, opt_t[:, col], label=f"{name} opt", color="#4c72b0")
        ax.set_ylabel(f"{name} (m)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.25)
    axes[3].plot(times, residual_px, color="#dd8452", label="reprojection err")
    axes[3].set_ylabel("err (px)")
    axes[3].set_xlabel("time (s)")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "object_pose6d_sharedcam_components.png")
    plt.close(fig)


def parse_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value in {"", None}:
        return default
    return float(value)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def project_xyz(xyz: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.maximum(xyz[:, 2], 1e-6)
    u = K[:, 0, 2] + K[:, 0, 0] * xyz[:, 0] / z
    v = K[:, 1, 2] + K[:, 1, 1] * xyz[:, 1] / z
    return u, v


def choose_part_z(contact_label: str, part_centers: dict[str, np.ndarray], idx: int) -> float:
    key = contact_label if contact_label in part_centers else ''
    if not key:
        return math.nan
    pts = part_centers[key]
    if idx >= len(pts):
        return math.nan
    return float(pts[idx, 2])


def merge_rows(obs_rows: list[dict[str, str]], state_rows: list[dict[str, str]]) -> list[dict[str, float | str | int]]:
    state_by_frame = {int(r['frame']): r for r in state_rows}
    merged = []
    for row in obs_rows:
        frame = int(row['frame'])
        state = state_by_frame.get(frame)
        if state is None:
            continue
        merged.append({
            'frame': frame,
            'time': parse_float(row, 'time', 0.0),
            'u': parse_float(row, 'ref_u'),
            'v': parse_float(row, 'ref_v'),
            # Match the baseline solver philosophy: fit the raw stable object
            # reference observation, and let optimizer smoothness handle temporal
            # regularization. Input-level smooth targets caused visible lag.
            'u_smooth': parse_float(row, 'ref_u'),
            'v_smooth': parse_float(row, 'ref_v'),
            'support_v': parse_float(row, 'support_v'),
            'support_dv': parse_float(row, 'support_dv', parse_float(row, 'support_v') - parse_float(row, 'ref_v')),
            'support_dv_smooth': parse_float(row, 'support_dv_smooth', parse_float(row, 'support_dv', parse_float(row, 'support_v') - parse_float(row, 'ref_v'))),
            'depth': parse_float(row, 'object_depth_smooth', parse_float(row, 'object_ref_depth_m')),
            'depth_raw': parse_float(row, 'object_depth_raw', parse_float(row, 'object_ref_depth_m')),
            'depth_conf': parse_float(row, 'object_depth_confidence', parse_float(row, 'depth_conf', 1.0)),
            'obs_conf': parse_float(row, 'observation_conf', 1.0),
            'proxy_conf': parse_float(row, 'proxy_conf', parse_float(row, 'observation_conf', 1.0)),
            'support_conf': parse_float(row, 'support_conf', 0.0),
            'support_gap_px': parse_float(row, 'support_gap_px', 0.0),
            'proxy_sigma_px': parse_float(row, 'proxy_sigma_px', 5.0),
            'support_sigma_px': parse_float(row, 'support_sigma_px', 8.0),
            'contact_offset': parse_float(row, 'contact_depth_offset_m', 0.0),
            'active_label': str(row.get('active_label', '') or ''),
            'contact_label': str(state.get('contact_label', '') or ''),
            'anchor_score': parse_float(state, 'anchor_score', 0.0),
            'floor_score': parse_float(state, 'floor_score', 0.0),
            'proposal_score': parse_float(state, 'proposal_score', 0.0),
            'human_contact_state': int(round(parse_float(state, 'human_contact_state', 0.0))),
            'floor_contact_state': int(round(parse_float(state, 'floor_contact_state', parse_float(state, 'plane_support_state', 0.0)))),
        })
    if not merged:
        raise RuntimeError('No merged stage1 rows')
    return merged


def initial_xyz(rows: list[dict[str, float | str | int]], K: np.ndarray) -> np.ndarray:
    depths = [float(r['depth']) for r in rows]
    finite_depths = [d for d in depths if math.isfinite(d)]
    fallback_z = float(np.median(finite_depths)) if finite_depths else 5.0
    xyz = []
    for i, row in enumerate(rows):
        d = float(row['depth'])
        z = max(d if math.isfinite(d) else fallback_z, 0.2)
        u = float(row['u']); v = float(row['v'])
        x = (u - K[i, 0, 2]) * z / K[i, 0, 0]
        y = (v - K[i, 1, 2]) * z / K[i, 1, 1]
        xyz.append([x, y, z])
    return np.asarray(xyz, dtype=np.float64)


def residuals(flat: np.ndarray, rows, K, part_z, center_w, depth_w, support_w, contact_w, vel_w, z_vel_w, acc_w, xy_acc_w):
    xyz = flat.reshape(-1, 3)
    u_proj, v_proj = project_xyz(xyz, K)
    z = xyz[:, 2]
    res = []
    for i, row in enumerate(rows):
        obs_conf = float(row['obs_conf'])
        proxy_conf = float(row['proxy_conf'])
        depth_conf = float(row['depth_conf'])
        u_obs = float(row['u_smooth'])
        v_obs = float(row['v_smooth'])
        sigma_uv = max(float(row['proxy_sigma_px']), 1.0)
        res.append(np.sqrt(center_w * max(proxy_conf, 0.05)) * ((u_proj[i] - u_obs) / sigma_uv))
        res.append(np.sqrt(center_w * max(proxy_conf, 0.05)) * ((v_proj[i] - v_obs) / sigma_uv))
        depth_prior = float(row['depth'])
        if depth_conf > 0 and math.isfinite(depth_prior):
            res.append(depth_w * depth_conf * (z[i] - depth_prior))
        support_gap_px = abs(float(row['support_gap_px']))
        support_valid = float(row['support_conf']) > 0.15 and support_gap_px < 18.0
        floor_conf = np.clip(float(row['support_conf']) * float(row['floor_score']), 0.0, 1.0) if support_valid else 0.0
        if floor_conf > 1e-6:
            pred_support_v = v_proj[i] + float(row['support_dv'])
            sigma_support = max(float(row['support_sigma_px']), 1.0)
            res.append(np.sqrt(support_w * floor_conf) * ((pred_support_v - float(row['support_v'])) / sigma_support))
        offset = float(row['contact_offset'])
        offset = max(offset, 0.0)  # negative coff pushes anchor past part_z (wrong direction)
        offset_valid = np.isfinite(offset) and offset <= 0.75
        anchor_conf = np.clip(float(row['anchor_score']), 0.0, 1.0) if offset_valid else 0.0
        if anchor_conf > 1e-6 and np.isfinite(part_z[i]):
            res.append(contact_w * (anchor_conf ** 2) * ((z[i] + offset) - part_z[i]))
        res.append(0.35 * max(0.0, 0.2 - z[i]))
    for i in range(1, len(xyz)):
        d = xyz[i] - xyz[i-1]
        res.append(vel_w * d[0])
        res.append(vel_w * d[1])
        res.append(z_vel_w * d[2])
    for i in range(1, len(xyz) - 1):
        z_acc = z[i+1] - 2.0 * z[i] + z[i-1]
        res.append(acc_w * z_acc)
        xy_acc = xyz[i+1, :2] - 2.0 * xyz[i, :2] + xyz[i-1, :2]
        res.extend((xy_acc_w * xy_acc).tolist())
    return np.asarray(res, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run radius-free object pose optimization.')
    parser.add_argument('--sample-dir', type=Path, required=True)
    parser.add_argument('--body-model-root', type=Path, default=Path('third-party/GVHMR/inputs/checkpoints/body_models'))
    parser.add_argument('--proxy-csv', type=Path, default=None)
    parser.add_argument('--state-csv', type=Path, default=None)
    parser.add_argument('--out-subdir', type=str, default='pose6d_object_proxy_da3_init')
    parser.add_argument('--center-weight', type=float, default=0.05)
    parser.add_argument('--depth-weight', type=float, default=0.3)
    parser.add_argument('--support-weight', type=float, default=10.0)
    parser.add_argument('--contact-weight', type=float, default=3.0)
    parser.add_argument('--vel-weight', type=float, default=0.0)
    parser.add_argument('--z-vel-weight', type=float, default=2.0)
    parser.add_argument('--z-acc-weight', type=float, default=0.50)
    parser.add_argument('--xy-acc-weight', type=float, default=0.08)
    args = parser.parse_args()

    sample_dir = args.sample_dir
    results_dir = sample_dir / 'results'
    proxy_csv = args.proxy_csv or (results_dir / 'object_proxy_observations' / 'object_proxy_observations.csv')
    state_csv = args.state_csv or (results_dir / 'contact_candidates_object_proxy' / 'contact_state_frames.csv')
    out_dir = results_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_rows = read_rows(proxy_csv)
    state_rows = read_rows(state_csv) if state_csv.exists() else [
        {
            'frame': r['frame'],
            'contact_label': '',
            'anchor_score': '0',
            'floor_score': '0',
            'proposal_score': '0',
            'human_contact_state': '0',
            'floor_contact_state': '0',
        }
        for r in obs_rows
    ]
    rows = merge_rows(obs_rows, state_rows)

    human = read_human_result(results_dir / 'gvhmr' / 'result.pkl')
    K = np.asarray(human['K_fullimg'], dtype=np.float64)[:len(rows)]
    joints = build_body_joints(args.body_model_root, human)[:len(rows)]
    part_centers = build_contact_part_centers(joints)

    part_z = np.full(len(rows), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        label = str(row['contact_label'] or '')
        if not label or label == 'floor':
            label = normalize_contact_label({'target': str(row['active_label'])}, default_part='hand', fallback_side='right')
        part_z[i] = choose_part_z(label, part_centers, i)

    init = initial_xyz(rows, K)
    result = least_squares(
        residuals,
        x0=init.reshape(-1),
        method='trf',
        loss='soft_l1',
        f_scale=1.0,
        max_nfev=400,
        args=(rows, K, part_z, args.center_weight, args.depth_weight, args.support_weight, args.contact_weight, args.vel_weight, args.z_vel_weight, args.z_acc_weight, args.xy_acc_weight),
    )
    opt = result.x.reshape(-1, 3)
    u_proj, v_proj = project_xyz(opt, K)
    residual_px = np.hypot(u_proj - np.asarray([float(r['u']) for r in rows]), v_proj - np.asarray([float(r['v']) for r in rows]))

    out_rows = []
    reproj_rows = []
    for i, row in enumerate(rows):
        pred_support_v = v_proj[i] + float(row['support_dv'])
        out_rows.append({
            'frame': int(row['frame']),
            'time': f"{float(row['time']):.6f}",
            'tx': f"{opt[i,0]:.6f}",
            'ty': f"{opt[i,1]:.6f}",
            'tz': f"{opt[i,2]:.6f}",
            'qw': '1.000000','qx': '0.000000','qy': '0.000000','qz': '0.000000',
            'radius_m': '',
            'coord_frame': 'gvhmr_incam',
            'u_obs': f"{float(row['u']):.3f}",'v_obs': f"{float(row['v']):.3f}",
            'radius_obs_px': '',
            'u_proj': f"{u_proj[i]:.3f}",'v_proj': f"{v_proj[i]:.3f}",
            'radius_proj_px': '',
            'bottom_proj_v': f"{pred_support_v:.3f}",
            'floor_v': f"{float(row['support_v']):.3f}",
            'u_ref_obs': f"{float(row['u']):.3f}",'v_ref_obs': f"{float(row['v']):.3f}",
            'u_ref_proj': f"{u_proj[i]:.3f}",'v_ref_proj': f"{v_proj[i]:.3f}",
            'support_v_obs': f"{float(row['support_v']):.3f}",'support_proj_v': f"{pred_support_v:.3f}",
            'residual_px': f"{residual_px[i]:.6f}",
            'proposal_score': f"{float(row['proposal_score']):.6f}",
            'anchor_score': f"{float(row['anchor_score']):.6f}",
            'floor_score': f"{float(row['floor_score']):.6f}",
            'depth_prior_m': f"{float(row['depth']):.6f}",
            'depth_prior_gap_m': f"{(opt[i,2] - float(row['depth'])):.6f}",
            'contact_depth_offset_m': f"{float(row['contact_offset']):.6f}",
            'contact_frame': int(float(row['anchor_score']) > 0.45),
            'audio_contact_frame': int(float(row['proposal_score']) > 0.25),
        })
        reproj_rows.append({
            'frame': int(row['frame']),
            'u_obs': f"{float(row['u']):.3f}",'v_obs': f"{float(row['v']):.3f}",
            'u_reproj': f"{u_proj[i]:.3f}",'v_reproj': f"{v_proj[i]:.3f}",
            'error_u': f"{(u_proj[i]-float(row['u'])):.6f}",'error_v': f"{(v_proj[i]-float(row['v'])):.6f}",'error_px': f"{residual_px[i]:.6f}",
        })

    write_csv(out_dir / 'object_pose6d_sharedcam_trajectory.csv', out_rows, list(out_rows[0].keys()))
    write_csv(out_dir / 'object_pose6d_sharedcam_reprojection_comparison.csv', reproj_rows, list(reproj_rows[0].keys()))
    plot_outputs(out_dir, np.asarray([float(r['time']) for r in rows], dtype=np.float64), init, opt, residual_px)

    floor_vals = np.asarray([float(r['support_v']) for r in rows], dtype=np.float64)
    floor_conf = np.asarray([float(r['support_conf']) * float(r['floor_score']) for r in rows], dtype=np.float64)
    good = floor_conf > 0.45
    payload = {
        'support_type': 'proxy_support',
        'floor_v': float(np.median(floor_vals[good])) if np.any(good) else float(np.median(floor_vals)),
        'source': 'radius_free_proxy_observations',
        'confidence': float(np.median(floor_conf[good])) if np.any(good) else float(np.median(floor_conf)),
    }
    with (out_dir / 'support_geometry.json').open('w') as f:
        json.dump(payload, f, indent=2)
    with (out_dir / 'object_pose6d_sharedcam_summary.txt').open('w') as f:
        f.write('Radius-free object pose optimization.\n')
        f.write('segment_mode: one_segment_full_sequence\n')
        f.write('candidate_cuts_segment: 0\n')
        f.write(f'num_frames: {len(rows)}\n')
        f.write(f'center_weight: {args.center_weight:.6f}\n')
        f.write(f'depth_weight: {args.depth_weight:.6f}\n')
        f.write(f'support_weight: {args.support_weight:.6f}\n')
        f.write(f'contact_weight: {args.contact_weight:.6f}\n')
        f.write(f'vel_weight: {args.vel_weight:.6f}\n')
        f.write(f'z_acc_weight: {args.z_acc_weight:.6f}\n')
    print(f"pose6d_csv: {out_dir / 'object_pose6d_sharedcam_trajectory.csv'}")


if __name__ == '__main__':
    main()
