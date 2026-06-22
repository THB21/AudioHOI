#!/usr/bin/env python3
"""Stage 3 – Depth (Z) + camera-space XY anchor refinement.

Reads : mug body pose CSV, corrected phase CSV, grasp-state CSV,
        contact-state CSV, GVHMR result.pkl.
Writes: anchored_pose.csv  (frame, time, x, y, z, yaw, pitch, roll, scale)

Algorithm
---------
1. Build per-frame Z / XY anchor targets from continuous contact frames:
   - Rotate stable_grasp_local_{x,y,z} by the handle phase to get camera-
     space position of the contact point on the mug.
   - target_z  = hand_z − contact_depth_offset − z_delta_from_center
   - target_xy = hand_xy − xy_delta_from_center
2. Linearly interpolate anchor targets → reference trajectory.
3. Least-squares smooth + anchor optimization for Z.
4. Re-project mug center UV at new Z → new XY reference; optimize XY.
5. Output pose CSV with refined x, y, z.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_STAGE1 = Path(__file__).resolve().parents[2] / "scripts" / "shared" / "radius_free_proxy" / "stage1_observation"
_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _STAGE1):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fit_mug_articraft_keyframe_pose as base  # noqa: E402
import render_mug_articraft_rigid_mesh_vlm as rigid  # noqa: E402
from scripts.shared.radius_free_proxy.stage1_observation.object_proxy_observation_utils import (  # noqa: E402
    read_human_result,
    build_body_joints,
)
from scripts.shared.human_ball.contact.contact_part_utils import build_contact_part_centers  # noqa: E402


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


def _ii(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, "") or default))
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_pose(path: Path) -> list[dict]:
    rows = []
    for r in _read(path):
        fr = int(float(r["frame"]))
        rows.append({
            "frame": fr,
            "time": _ff(r, "time", (fr - 1) / 24.0),
            "x": _ff(r, "x"), "y": _ff(r, "y"), "z": _ff(r, "z"),
            "yaw": _ff(r, "yaw"), "yaw_deg": _ff(r, "yaw_deg", math.degrees(_ff(r, "yaw", 0.0))),
            "pitch": _ff(r, "pitch"), "pitch_deg": _ff(r, "pitch_deg", math.degrees(_ff(r, "pitch", 0.0))),
            "roll": _ff(r, "roll"), "roll_deg": _ff(r, "roll_deg", math.degrees(_ff(r, "roll", 0.0))),
            "scale": _ff(r, "scale", 1.0),
        })
    return rows


def _read_phase(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with path.open() as f:
        header = next(csv.reader(f))
    col = "m17_phase_rad" if "m17_phase_rad" in header else None
    for r in _read(path):
        fr = int(float(r["frame"]))
        if col:
            out[fr] = _ff(r, col, 0.0)
        elif "phase_rad" in r:
            out[fr] = _ff(r, "phase_rad", 0.0)
        else:
            v = _ff(r, "phase_deg", 0.0)
            out[fr] = math.radians(v) if abs(v) > 2 * math.pi else v
    return out


def _pose_params(r: dict) -> np.ndarray:
    return np.array([r["x"], r["y"], r["z"], r["yaw"], r["pitch"], r["roll"], r["scale"]], dtype=float)


def _center_uv(p: np.ndarray, K: np.ndarray) -> tuple[float, float]:
    z = max(float(p[2]), 1e-6)
    return float(K[0, 0] * p[0] / z + K[0, 2]), float(K[1, 1] * p[1] / z + K[1, 2])


def _xyz_from_uvz(u: np.ndarray, v: np.ndarray, z: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return x, y


# ---------------------------------------------------------------------------
# Anchor construction
# ---------------------------------------------------------------------------

def _robust_offset(v: float) -> float:
    """Clip contact depth offset to ±0.08m (proxy tracking errors can be huge)."""
    return float(np.clip(v, -0.08, 0.08)) if np.isfinite(v) else 0.0


def _choose_hand_label(g: dict, s: dict | None) -> str | None:
    for k in ("stable_grasp_hand_side", "active_label"):
        lbl = str(g.get(k, "") or "").strip()
        if lbl in {"left_hand", "right_hand"}:
            return lbl
    if s:
        lbl = str(s.get("contact_label", "") or "").strip()
        if lbl in {"left_hand", "right_hand"}:
            return lbl
    return None


def build_anchor_targets(
    pose_rows: list[dict],
    phases: dict[int, float],
    grasp_by_frame: dict[int, dict],
    state_by_frame: dict[int, dict],
    part_centers: dict[str, np.ndarray],
) -> dict[int, dict]:
    anchors: dict[int, dict] = {}
    for row in pose_rows:
        fr = int(row["frame"])
        g = grasp_by_frame.get(fr)
        if not g:
            continue
        s = state_by_frame.get(fr)
        stable = [_ff(g, "stable_grasp_local_x"), _ff(g, "stable_grasp_local_y"), _ff(g, "stable_grasp_local_z")]
        if not all(np.isfinite(stable)):
            continue
        human_state = _ii(s, "human_contact_state", 0) if s else 0
        anchor_state = _ii(s, "anchor_contact_state", 0) if s else 0
        use_now = _ii(g, "use_this_point_for_hand_attachment", 0) == 1
        use_prev = _ii(g, "use_previous_grasp_for_hand_attachment", 0) == 1
        if not (human_state == 1 or anchor_state == 1 or use_now or use_prev):
            continue
        label = _choose_hand_label(g, s)
        if not label or label not in part_centers:
            continue
        idx = fr - 1
        if idx < 0 or idx >= len(part_centers[label]):
            continue
        hand_xyz = part_centers[label][idx]
        hx, hy, hz = float(hand_xyz[0]), float(hand_xyz[1]), float(hand_xyz[2])
        if not np.isfinite(hz) or hz <= 0.2:
            continue

        p = _pose_params(row)
        phase = phases.get(fr, 0.0)
        loc = np.asarray(stable, dtype=float)[None, :] @ rigid.rot_y(phase).T
        anchor_cam = base.transform(p, loc)[0]
        dz = float(anchor_cam[2] - p[2])
        dx = float(anchor_cam[0] - p[0])
        dy = float(anchor_cam[1] - p[1])
        offset_raw = _ff(s, "contact_depth_offset_m", math.nan) if s else _ff(g, "hand_object_z_gap_m", math.nan)
        offset = _robust_offset(offset_raw)

        conf = max(_ff(g, "stable_grasp_conf", 0.25),
                   _ff(s, "anchor_score", 0.0) if s else 0.0, 0.20)
        mode = str(g.get("frame_mode", "") or "")
        if use_now:
            kind, weight = "confirmed_update", 3.0 * conf
        elif use_prev or "keep" in mode:
            kind, weight = "continuous_keep", 1.4 * conf
        else:
            kind, weight = "candidate_continuous", 1.0 * conf

        anchors[fr] = {
            "target_z": hz - offset - dz,
            "target_x": hx - dx,
            "target_y": hy - dy,
            "hand_z": hz, "hand_x": hx, "hand_y": hy,
            "local_delta_z": dz, "local_delta_x": dx, "local_delta_y": dy,
            "offset": offset, "label": label, "mode": mode,
            "weight": float(np.clip(weight, 0.25, 4.0)),
            "kind": kind,
            "human_state": human_state,
            "floor_state": _ii(s, "floor_contact_state", 0) if s else 0,
            "support_conf": _ff(s, "support_conf", 0.0) if s else 0.0,
        }
    return anchors


def _find_table_start(state_by_frame: dict[int, dict], min_run: int = 10) -> int | None:
    run: list[int] = []
    for fr in sorted(state_by_frame):
        r = state_by_frame[fr]
        if _ii(r, "floor_contact_state", 0) == 1 and _ff(r, "support_conf", 0.0) >= 0.65:
            run.append(fr)
            if len(run) >= min_run:
                return run[0]
        else:
            run = []
    return None


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

def _interp_ref(z_init: np.ndarray, anchors: dict[int, dict], frames: list[int]) -> np.ndarray:
    z_ref = z_init.copy()
    idx_map = {fr: i for i, fr in enumerate(frames)}
    aidx = [idx_map[fr] for fr in sorted(anchors) if fr in idx_map and np.isfinite(anchors[fr]["target_z"])]
    if not aidx:
        return z_ref
    z_ref[:aidx[0] + 1] = anchors[frames[aidx[0]]]["target_z"]
    for a, b in zip(aidx[:-1], aidx[1:]):
        za, zb = anchors[frames[a]]["target_z"], anchors[frames[b]]["target_z"]
        alpha = np.linspace(0.0, 1.0, b - a + 1)
        z_ref[a:b + 1] = (1 - alpha) * za + alpha * zb
    z_ref[aidx[-1]:] = anchors[frames[aidx[-1]]]["target_z"]
    return np.maximum(z_ref, 0.30)


def _optimize_z(
    z_init: np.ndarray, z_ref: np.ndarray,
    frames: list[int], anchors: dict[int, dict],
    table_start: int | None,
) -> np.ndarray:
    idx_map = {fr: i for i, fr in enumerate(frames)}
    aitems = [(idx_map[fr], anchors[fr]) for fr in sorted(anchors) if fr in idx_map]
    static_idx = idx_map.get(table_start) if table_start and table_start in idx_map else None

    def residual(z: np.ndarray) -> np.ndarray:
        out = list((z - z_ref) * 0.45)
        for i, a in aitems:
            out.append((z[i] - float(a["target_z"])) * float(a["weight"]))
        if len(z) > 1:
            out.extend((np.diff(z) * 4.0).tolist())
        if len(z) > 2:
            out.extend((np.diff(z, n=2) * 12.0).tolist())
        if static_idx is not None:
            zs = z[static_idx]
            out.extend(((z[static_idx:] - zs) * 20.0).tolist())
        return np.asarray(out, dtype=float)

    r = least_squares(residual, z_ref.copy(), loss="soft_l1", f_scale=0.03, max_nfev=300)
    return np.maximum(r.x, 0.30)


def _optimize_xy(
    xy_init: np.ndarray, xy_ref: np.ndarray,
    frames: list[int], anchors: dict[int, dict],
    table_start: int | None, key: str,
) -> np.ndarray:
    idx_map = {fr: i for i, fr in enumerate(frames)}
    aitems = [
        (idx_map[fr], anchors[fr])
        for fr in sorted(anchors)
        if fr in idx_map and key in anchors[fr] and np.isfinite(float(anchors[fr][key]))
    ]
    static_idx = idx_map.get(table_start) if table_start and table_start in idx_map else None

    def residual(xy: np.ndarray) -> np.ndarray:
        out = list((xy - xy_ref) * 0.30)
        for i, a in aitems:
            out.append((xy[i] - float(a[key])) * float(a["weight"]) * 0.75)
        if len(xy) > 1:
            out.extend((np.diff(xy) * 3.0).tolist())
        if len(xy) > 2:
            out.extend((np.diff(xy, n=2) * 8.0).tolist())
        if static_idx is not None:
            xs = xy[static_idx]
            out.extend(((xy[static_idx:] - xs) * 15.0).tolist())
        return np.asarray(out, dtype=float)

    r = least_squares(residual, xy_ref.copy(), loss="soft_l1", f_scale=0.04, max_nfev=300)
    return r.x


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    sample: Path,
    pose_csv: Path,
    phase_csv: Path,
    body_model_root: Path | None = None,
    out_csv: Path | None = None,
) -> Path:
    out_csv = out_csv or (sample / "results" / "pipe" / "anchored_pose.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    body_root = body_model_root or (_REPO / "third-party" / "GVHMR" / "inputs" / "checkpoints" / "body_models")

    pose_rows = _read_pose(pose_csv)
    phases = _read_phase(phase_csv)

    grasp_csv = sample / "results" / "mug_grasp_anchor_state" / "mug_grasp_anchor_state.csv"
    state_csv = sample / "results" / "contact_candidates_object_proxy" / "contact_state_frames.csv"
    grasp_by_frame = {int(float(r["frame"])): r for r in _read(grasp_csv)}
    state_by_frame = {int(float(r["frame"])): r for r in _read(state_csv)}

    human = read_human_result(sample / "results" / "gvhmr" / "result.pkl")
    K_all = np.asarray(human["K_fullimg"], dtype=float)
    K = K_all[0] if K_all.ndim == 3 else K_all
    joints = build_body_joints(body_root, human)
    part_centers = build_contact_part_centers(joints)

    frames = [int(r["frame"]) for r in pose_rows]
    z_init = np.array([r["z"] for r in pose_rows], dtype=float)
    x_init = np.array([r["x"] for r in pose_rows], dtype=float)
    y_init = np.array([r["y"] for r in pose_rows], dtype=float)

    anchors = build_anchor_targets(pose_rows, phases, grasp_by_frame, state_by_frame, part_centers)
    table_start = _find_table_start(state_by_frame)

    z_ref = _interp_ref(z_init, anchors, frames)
    z_final = _optimize_z(z_init, z_ref, frames, anchors, table_start)

    u0 = np.array([_center_uv(_pose_params(r), K)[0] for r in pose_rows], dtype=float)
    v0 = np.array([_center_uv(_pose_params(r), K)[1] for r in pose_rows], dtype=float)
    x_ref, y_ref = _xyz_from_uvz(u0, v0, z_final, K)
    x_final = _optimize_xy(x_init, x_ref, frames, anchors, table_start, "target_x")
    y_final = _optimize_xy(y_init, y_ref, frames, anchors, table_start, "target_y")

    fields = ["frame", "time", "x", "y", "z", "yaw", "yaw_deg", "pitch", "pitch_deg", "roll", "roll_deg", "scale"]
    out_rows = []
    for i, r in enumerate(pose_rows):
        out = dict(r)
        out["x"] = f"{x_final[i]:.9f}"
        out["y"] = f"{y_final[i]:.9f}"
        out["z"] = f"{z_final[i]:.9f}"
        out_rows.append(out)

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in fields})

    print(f"[anchor] {len(out_rows)} frames, {len(anchors)} anchors, table_start={table_start} → {out_csv}")
    return out_csv


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, default=Path("samples_known_object/02_mug"))
    ap.add_argument("--pose-csv", type=Path, default=None)
    ap.add_argument("--phase-csv", type=Path, default=None)
    ap.add_argument("--body-model-root", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()
    s = args.sample_dir
    run(
        s,
        args.pose_csv or s / "proxy" / "mug_body_only_cylinder_pose_table_static_sequence.csv",
        args.phase_csv or s / "results" / "pipe" / "corrected_phase.csv",
        body_model_root=args.body_model_root,
        out_csv=args.out_csv,
    )
