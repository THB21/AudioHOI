#!/usr/bin/env python3
"""Stage 2 – Hidden-segment far-side phase correction (M17 logic).

Reads : M14 phase CSV, mug body pose CSV, VLM visibility CSV.
Writes: corrected_phase.csv  (frame, time, m17_phase_rad, m17_phase_deg,
                               m14_phase_rad, m14_phase_deg, vlm_visibility)

Algorithm (smooth boundary interpolation)
-----------------------------------------
For each hidden segment [i, j):
  - Linearly interpolate in the unwrapped domain from the last visible frame
    before the segment to the first visible frame after.
  - Choose between the two possible arcs (shorter vs. longer) by counting
    how many hidden frames would land on the far side (sin(phase+yaw) > 0).
  - Prefer the arc that keeps more hidden frames on the far side.
Visible frames are never modified.
A light Gaussian smooth (sigma/2) removes residual sub-degree roughness.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

_STAGE1 = Path(__file__).resolve().parents[2] / "scripts" / "shared" / "radius_free_proxy" / "stage1_observation"
if str(_STAGE1) not in sys.path:
    sys.path.insert(0, str(_STAGE1))


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


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Core correction algorithm (extracted from M17, render-free)
# ---------------------------------------------------------------------------

def correct_phases(
    frames: list[int],
    m14_phases: dict[int, float],
    yaw_by_frame: dict[int, float],
    visibility: dict[int, str],
    sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (corrected_phase_wrapped, is_error_bool) arrays aligned to frames.

    is_error marks hidden frames where M14 had sin(phase+yaw) < 0 — diagnostics only.
    """
    N = len(frames)
    phase_arr = np.array([m14_phases.get(fr, 0.0) for fr in frames], dtype=float)
    yaw_arr = np.array([yaw_by_frame.get(fr, 0.0) for fr in frames], dtype=float)
    vis_list = [visibility.get(fr, "visible") for fr in frames]

    is_error = np.array([
        vis_list[i] == "hidden" and math.sin(yaw_arr[i] + phase_arr[i]) < 0
        for i in range(N)
    ], dtype=bool)

    corrected = phase_arr.copy()

    i = 0
    while i < N:
        if vis_list[i] != "hidden":
            i += 1
            continue
        j = i
        while j < N and vis_list[j] == "hidden":
            j += 1

        left = phase_arr[i - 1] if i > 0 else (phase_arr[j] if j < N else 0.0)
        right = phase_arr[j] if j < N else (phase_arr[i - 1] if i > 0 else 0.0)
        n = j - i

        delta_short = float(_wrap(np.array([right - left]))[0])
        sign = 1.0 if delta_short >= 0 else -1.0
        delta_long = delta_short - sign * 2.0 * math.pi
        right_ccw = left + delta_short
        right_cw = left + delta_long

        def _far_count(right_uw: float) -> int:
            return sum(
                1 for k in range(n)
                if math.sin(left + (k + 1) / (n + 1) * (right_uw - left) + yaw_arr[i + k]) > 0
            )

        fc_ccw, fc_cw = _far_count(right_ccw), _far_count(right_cw)
        if fc_cw > fc_ccw:
            right_uw = right_cw
        elif fc_ccw > fc_cw:
            right_uw = right_ccw
        else:
            right_uw = right_ccw if abs(delta_short) <= abs(delta_long) else right_cw

        for k in range(n):
            t = (k + 1) / (n + 1)
            corrected[i + k] = float(_wrap(np.array([left + t * (right_uw - left)]))[0])

        i = j

    # Light global smooth — does not meaningfully shift visible frames.
    m17_uw = gaussian_filter1d(np.unwrap(corrected), sigma=max(0.5, sigma / 2.0), mode="nearest")
    return _wrap(m17_uw), is_error


# ---------------------------------------------------------------------------
# VLM visibility loader (simple: just need hidden/visible string)
# ---------------------------------------------------------------------------

def _load_visibility(vis_csv: Path, frames: list[int]) -> dict[int, str]:
    if not vis_csv.exists():
        return {fr: "visible" for fr in frames}
    rows = {int(float(r["frame"])): r for r in _read(vis_csv)}
    out: dict[int, str] = {}
    for fr in frames:
        row = rows.get(fr, {})
        vis = str(row.get("visibility", "")).strip().lower()
        constraint = str(row.get("recommended_visibility_constraint", "")).strip().lower()
        part = str(row.get("hand_contact_part", "")).strip().lower()
        handle_contact = str(row.get("handle_contact", "")).strip().lower() in {"1", "true", "yes"} or part == "handle"
        if vis == "hidden" or constraint.startswith("force_hidden"):
            out[fr] = "hidden"
        elif (vis == "visible" or constraint.startswith("force_visible")) and handle_contact:
            out[fr] = "visible"
        else:
            out[fr] = "hidden"
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    sample: Path,
    phase_csv: Path,
    pose_csv: Path,
    vlm_csv: Path | None = None,
    sigma: float = 3.0,
    out_csv: Path | None = None,
) -> Path:
    out_csv = out_csv or (sample / "results" / "pipe" / "corrected_phase.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load M14 phases
    m14_phases: dict[int, float] = {}
    m14_time: dict[int, float] = {}
    fps = 24.0
    for r in _read(phase_csv):
        fr = int(float(r["frame"]))
        m14_phases[fr] = _ff(r, "phase_rad", _ff(r, "m17_phase_rad", 0.0))
        m14_time[fr] = _ff(r, "time", (fr - 1) / fps)

    # Load yaw from pose
    yaw_by_frame: dict[int, float] = {}
    for r in _read(pose_csv):
        fr = int(float(r["frame"]))
        yaw_by_frame[fr] = _ff(r, "yaw", 0.0)

    frames = sorted(m14_phases)

    vlm_path = vlm_csv or (sample / "annotations" / "vlm_handle_visibility_full" / "qwen_handle_visibility.csv")
    visibility = _load_visibility(vlm_path, frames)

    corrected, is_error = correct_phases(frames, m14_phases, yaw_by_frame, visibility, sigma)

    rows = []
    for i, fr in enumerate(frames):
        m14_rad = m14_phases[fr]
        c_rad = float(corrected[i])
        rows.append({
            "frame": fr,
            "time": f"{m14_time.get(fr, (fr-1)/fps):.6f}",
            "m17_phase_rad": f"{c_rad:.9f}",
            "m17_phase_deg": f"{math.degrees(c_rad):.4f}",
            "m14_phase_rad": f"{m14_rad:.9f}",
            "m14_phase_deg": f"{math.degrees(m14_rad):.4f}",
            "is_error_frame": int(is_error[i]),
            "phase_correction_rad": f"{(c_rad - m14_rad):.9f}",
            "vlm_visibility": visibility.get(fr, "visible"),
        })

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[correction] {len(rows)} frames → {out_csv}")
    return out_csv


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, default=Path("samples_known_object/02_mug"))
    ap.add_argument("--phase-csv", type=Path, default=None)
    ap.add_argument("--pose-csv", type=Path, default=None)
    ap.add_argument("--vlm-csv", type=Path, default=None)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()
    s = args.sample_dir
    run(
        s,
        args.phase_csv or s / "results" / "pipe" / "handle_phase.csv",
        args.pose_csv or s / "proxy" / "mug_body_only_cylinder_pose_table_static_sequence.csv",
        vlm_csv=args.vlm_csv,
        sigma=args.sigma,
        out_csv=args.out_csv,
    )
