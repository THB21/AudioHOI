#!/usr/bin/env python3
"""Stage 1 – Handle phase optimization.

Reads : mug body pose CSV, M12 phase prior CSV, VLM visibility CSV,
        mug_articraft_contact_points CSV.
Writes: handle_phase.csv  (frame, time, phase_rad, phase_deg)

Algorithm
---------
1. Temporal-consistency filter on VLM labels (suppress ≤3-frame islands).
2. Continuous visibility-alpha: ramps smoothly at hidden/visible boundaries.
3. Full-sequence least-squares optimizer:
   - Weak M12 prior.
   - Strong hand-handle attachment on confirmed-contact frames.
   - Keep-previous attachment on hidden/depth-misaligned frames.
   - Temporal velocity + acceleration smoothness.
   - Table-static lock once sustained support is detected.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_STAGE1 = Path(__file__).resolve().parents[2] / "scripts" / "shared" / "radius_free_proxy" / "stage1_observation"
if str(_STAGE1) not in sys.path:
    sys.path.insert(0, str(_STAGE1))

import fit_mug_articraft_keyframe_pose as base  # noqa: E402
import render_mug_articraft_rigid_mesh_vlm as rigid  # noqa: E402


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _read(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _ff(row: dict, key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, "") or default)
    except Exception:
        return default


def _truthy(v: object) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes"}


# ---------------------------------------------------------------------------
# VLM temporal filter
# ---------------------------------------------------------------------------

def _vlm_handle_visible(row: dict) -> bool:
    vis = str(row.get("visibility", "")).strip().lower()
    constraint = str(row.get("recommended_visibility_constraint", "")).strip().lower()
    part = str(row.get("hand_contact_part", "")).strip().lower()
    handle = _truthy(row.get("handle_contact", "")) or part == "handle"
    return (vis == "visible" or constraint.startswith("force_visible")) and handle


def filter_vlm(rows: dict[int, dict], max_island: int = 3) -> dict[int, dict]:
    """Suppress short visibility islands surrounded by the opposite state."""
    if not rows:
        return rows
    frames = sorted(rows)
    state = {fr: _vlm_handle_visible(rows[fr]) for fr in frames}

    changed = True
    while changed:
        changed = False
        runs: list[tuple[int, int, bool]] = []
        s, cur = frames[0], state[frames[0]]
        for fr in frames[1:]:
            if state[fr] != cur:
                runs.append((s, fr - 1, cur))
                s, cur = fr, state[fr]
        runs.append((s, frames[-1], cur))
        for i, (a, b, val) in enumerate(runs):
            if b - a + 1 > max_island or i == 0 or i + 1 >= len(runs):
                continue
            if runs[i - 1][2] == runs[i + 1][2] and runs[i - 1][2] != val:
                for fr in range(a, b + 1):
                    state[fr] = runs[i - 1][2]
                changed = True
                break

    out = {fr: dict(rows[fr]) for fr in frames}
    for fr in frames:
        raw = _vlm_handle_visible(rows[fr])
        if state[fr] == raw:
            continue
        if state[fr]:
            out[fr].update({"visibility": "visible", "handle_contact": "True",
                            "hand_contact_part": "handle",
                            "recommended_visibility_constraint": "force_visible_temporal_filter"})
        else:
            out[fr].update({"visibility": "hidden", "handle_contact": "False",
                            "hand_contact_part": "body",
                            "recommended_visibility_constraint": "force_hidden_temporal_filter"})
    return out


# ---------------------------------------------------------------------------
# Visibility alpha (smooth hidden↔visible ramp)
# ---------------------------------------------------------------------------

def visibility_alpha(frames: list[int], vlm: dict[int, dict], ramp: int = 5) -> dict[int, float]:
    visible = {fr: _vlm_handle_visible(vlm.get(fr, {})) for fr in frames}
    alpha = {fr: 1.0 if visible[fr] else 0.0 for fr in frames}

    # Build runs
    runs: list[tuple[int, int, bool]] = []
    s, cur = frames[0], visible[frames[0]]
    for fr in frames[1:]:
        if visible[fr] != cur:
            runs.append((s, fr - 1, cur))
            s, cur = fr, visible[fr]
    runs.append((s, frames[-1], cur))

    # Ramp visible-run edges
    for i, (a, b, vis) in enumerate(runs):
        if not vis:
            continue
        has_before = i > 0 and not runs[i - 1][2]
        has_after = i + 1 < len(runs) and not runs[i + 1][2]
        length = b - a + 1
        for fr in range(a, b + 1):
            v = 1.0
            if has_before:
                v = min(v, (fr - a + 1) / float(ramp + 1))
            if has_after:
                v = min(v, (b - fr + 1) / float(ramp + 1))
            if length <= 2 * ramp:
                v = min(v, 0.85)
            alpha[fr] = float(np.clip(v, 0.0, 1.0))

    # Bleed a small alpha into hidden shoulders
    for i, (a, b, vis) in enumerate(runs):
        if vis:
            continue
        if i > 0 and runs[i - 1][2]:
            for d, fr in enumerate(range(a, min(b, a + ramp - 1) + 1), 1):
                alpha[fr] = max(alpha[fr], max(0.0, (ramp - d + 1) / float(ramp + 1)) * 0.35)
        if i + 1 < len(runs) and runs[i + 1][2]:
            for d, fr in enumerate(range(b, max(a, b - ramp + 1) - 1, -1), 1):
                alpha[fr] = max(alpha[fr], max(0.0, (ramp - d + 1) / float(ramp + 1)) * 0.35)
    return alpha


# ---------------------------------------------------------------------------
# Contact state helpers
# ---------------------------------------------------------------------------

def _confirmed_handle(contact: dict) -> bool:
    return (str(contact.get("use_this_point_for_hand_attachment", "")).strip() == "1"
            and contact.get("object_contact_event") == "hand_handle_grasp"
            and contact.get("hand_mug_contact_state") == "direct_hand_grasp_point")


def _keep_previous(contact: dict) -> bool:
    if str(contact.get("use_previous_grasp_for_hand_attachment", "")).strip() == "1":
        return True
    return contact.get("object_contact_event") == "hand_handle_grasp_depth_misaligned"


# ---------------------------------------------------------------------------
# Phase optimizer
# ---------------------------------------------------------------------------

def optimize(
    poses: dict[int, np.ndarray],
    prior: dict[int, float],
    vlm: dict[int, dict],
    contacts: dict[int, dict],
    K_all: np.ndarray,
    meshes: dict,
    proxy_obs: dict[int, dict] | None = None,
    max_nfev: int = 120,
) -> dict[int, float]:
    frames = sorted(poses)
    n = len(frames)
    idx = {fr: i for i, fr in enumerate(frames)}
    theta0 = np.unwrap([prior.get(fr, 0.0) for fr in frames])

    alpha = visibility_alpha(frames, vlm)
    hverts = meshes["handle_loop"][0]
    hc_local = np.mean(hverts, axis=0, keepdims=True)

    # Build grasp attachment items
    stable_local: np.ndarray | None = None
    grasp_items: list[tuple] = []

    # Carry anchor from any confirmed frame before this segment
    for fr in sorted(k for k in contacts if k < frames[0]):
        c = contacts[fr]
        if _confirmed_handle(c):
            loc = np.array([_ff(c, "mug_local_x"), _ff(c, "mug_local_y"), _ff(c, "mug_local_z")])
            if np.all(np.isfinite(loc)):
                stable_local = loc

    for fr in frames:
        p = poses[fr]
        K = K_all[fr - 1] if K_all.ndim == 3 else K_all
        c = contacts.get(fr, {})
        use_now = _confirmed_handle(c)
        use_prev = _keep_previous(c)
        rim = str(c.get("rim_drinking_contact", "")).strip() == "1"

        if use_now:
            loc = np.array([_ff(c, "mug_local_x"), _ff(c, "mug_local_y"), _ff(c, "mug_local_z")])
            if np.all(np.isfinite(loc)):
                stable_local = loc

        if stable_local is not None and (use_now or use_prev or rim):
            hu, hv = _ff(c, "active_part_u"), _ff(c, "active_part_v")
            conf = _ff(c, "active_label_conf", 0.0)
            if np.isfinite(hu) and np.isfinite(hv) and conf >= 0.15:
                hand_uv = np.array([hu, hv])
                w = 1.0 if use_now else (0.30 if rim else 0.60)
                grasp_items.append((idx[fr], p, K, hand_uv, min(1.0, conf), stable_local.copy(), w))

    # Table-static detection
    table_idx: int | None = None
    table_items: list[tuple[int, float]] = []
    if proxy_obs:
        ok = [bool(_ff(proxy_obs.get(fr, {}), "support_conf", 0) >= 0.65
                   and _ff(proxy_obs.get(fr, {}), "object_motion_score", 1) <= 0.15)
              for fr in frames]
        for s in range(n):
            end = s
            while end < n and ok[end]:
                end += 1
            if end - s >= 12:
                table_idx = s
                break
        if table_idx is not None:
            for i in range(table_idx, n):
                fr = frames[i]
                sc = _ff(proxy_obs.get(fr, {}), "support_conf", 0)
                ms = _ff(proxy_obs.get(fr, {}), "object_motion_score", 1)
                if sc >= 0.45 and ms <= 0.25:
                    table_items.append((i, float(np.clip(sc * (1 - ms), 0, 1))))

    def residual(theta: np.ndarray) -> np.ndarray:
        res: list[float] = []
        # Weak prior
        for i in range(n):
            res.append(0.35 * base.wrap(float(theta[i] - theta0[i])) / math.radians(25))
        # Hand-handle attachment
        for i, p, K, hand_uv, conf, loc, w in grasp_items:
            pts_local = loc[None, :] @ rigid.rot_y(float(theta[i])).T
            cam = base.transform(p, pts_local)
            z = max(cam[0, 2], 1e-6)
            uv = np.array([K[0, 0] * cam[0, 0] / z + K[0, 2], K[1, 1] * cam[0, 1] / z + K[1, 2]])
            d = float(np.linalg.norm(uv - hand_uv))
            res.append((24.0 * w * max(0.35, conf)) * d / 7.0)
        # Table static
        if table_idx is not None:
            ref = theta[table_idx]
            for i, wt in table_items:
                res.append((3.5 * wt) * base.wrap(float(theta[i] - ref)) / math.radians(4))
        # Temporal smoothness
        for i in range(1, n):
            vel = float(theta[i] - theta[i - 1])
            res.append(0.8 * vel / math.radians(10))
            excess = max(0.0, abs(vel) - math.radians(8))
            res.append(2.0 * excess / math.radians(3))
        for i in range(1, n - 1):
            acc = float(theta[i + 1] - 2 * theta[i] + theta[i - 1])
            res.append(1.2 * acc / math.radians(5))
        return np.asarray(res, dtype=float)

    opt = least_squares(residual, theta0.copy(), loss="soft_l1", f_scale=1.0, max_nfev=max_nfev)
    theta = np.unwrap(opt.x)
    return {fr: base.wrap(float(theta[i])) for i, fr in enumerate(frames)}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    sample: Path,
    pose_csv: Path,
    prior_csv: Path,
    contact_csv: Path,
    vlm_csv: Path | None = None,
    out_csv: Path | None = None,
    max_nfev: int = 120,
) -> Path:
    out_csv = out_csv or (sample / "results" / "pipe" / "handle_phase.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    poses = {int(float(r["frame"])): np.array([_ff(r, k) for k in ["x", "y", "z", "yaw", "pitch", "roll", "scale"]]) for r in _read(pose_csv)}
    prior = {}
    for r in _read(prior_csv):
        fr = int(float(r["frame"]))
        prior[fr] = _ff(r, "mug_axial_phase_rad", _ff(r, "handle_phase_rad", 0.0))
    contacts = {int(float(r["frame"])): r for r in _read(contact_csv)}

    vlm_path = vlm_csv or (sample / "annotations" / "vlm_handle_visibility_full" / "qwen_handle_visibility.csv")
    raw_vlm = {int(float(r["frame"])): r for r in _read(vlm_path)} if vlm_path.exists() else {}
    vlm = filter_vlm(raw_vlm)

    K_all = base.load_K(sample)

    _STAGE1_PATH = Path(__file__).resolve().parents[2] / "scripts" / "shared" / "radius_free_proxy" / "stage1_observation"
    import importlib.util, sys as _sys
    rigid_path = _STAGE1_PATH / "render_mug_articraft_rigid_mesh_vlm.py"
    mesh_root = sample / "articraft" / "materialized_mug_mesh"
    meshes = rigid.load_articraft_meshes(mesh_root)

    proxy_obs_path = sample / "results" / "object_proxy_observations" / "object_proxy_observations.csv"
    proxy_obs = {int(float(r["frame"])): r for r in _read(proxy_obs_path)} if proxy_obs_path.exists() else {}

    phases = optimize(poses, prior, vlm, contacts, K_all, meshes, proxy_obs or None, max_nfev)

    fps = 24.0
    rows = [{"frame": fr, "time": f"{(fr-1)/fps:.6f}",
             "phase_rad": f"{phases[fr]:.9f}", "phase_deg": f"{math.degrees(phases[fr]):.4f}"}
            for fr in sorted(phases)]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[phase] {len(rows)} frames → {out_csv}")
    return out_csv


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, default=Path("samples_known_object/02_mug"))
    ap.add_argument("--pose-csv", type=Path, default=None)
    ap.add_argument("--prior-csv", type=Path, default=None)
    ap.add_argument("--contact-csv", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--max-nfev", type=int, default=120)
    args = ap.parse_args()
    s = args.sample_dir
    run(
        s,
        args.pose_csv or s / "proxy" / "mug_body_only_cylinder_pose_table_static_sequence.csv",
        args.prior_csv or s / "results" / "renders" / "M12_articraft_rigid_mesh_vlm" / "handle_phase_all.csv",
        args.contact_csv or s / "results" / "mug_articraft_contact_points" / "mug_articraft_contact_points.csv",
        out_csv=args.out_csv,
        max_nfev=args.max_nfev,
    )
