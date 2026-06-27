#!/usr/bin/env python3
"""CSV-only M17 mug handle phase correction.

This is the pipeline-owned version of the solved M17 logic. It keeps the same
hidden-segment interpolation and smoothing behavior, but does not render videos.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d


SIGMA = 3.0

FIELDS = [
    "frame",
    "time",
    "m17_phase_rad",
    "m17_phase_deg",
    "m14_phase_rad",
    "m14_phase_deg",
    "is_error_frame",
    "phase_correction_rad",
    "phase_correction_deg",
    "vlm_visibility",
    "handle_z_sin_m14",
    "handle_z_sin_m17",
    "mug_yaw_deg",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def ff(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value != "" else default
    except Exception:
        return default


def wrap_angles(values: np.ndarray) -> np.ndarray:
    return (values + math.pi) % (2.0 * math.pi) - math.pi


def read_yaw_by_frame(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in read_csv(path):
        fr = int(float(row["frame"]))
        out[fr] = ff(row, "yaw", 0.0)
    return out


def compute_m17_phases(
    frames: list[int],
    m14_phases: dict[int, float],
    yaw_data: dict[int, float],
    visibility: dict[int, str],
    sigma: float = SIGMA,
) -> tuple[np.ndarray, np.ndarray]:
    phase_arr = np.array([m14_phases.get(fr, 0.0) for fr in frames], dtype=float)
    yaw_arr = np.array([yaw_data.get(fr, 0.0) for fr in frames], dtype=float)
    vis_list = [visibility.get(fr, "visible") for fr in frames]

    is_error = np.array(
        [vis_list[i] == "hidden" and math.sin(yaw_arr[i] + phase_arr[i]) < 0 for i in range(len(frames))],
        dtype=bool,
    )
    corrected = phase_arr.copy()

    i = 0
    while i < len(frames):
        if vis_list[i] != "hidden":
            i += 1
            continue
        j = i
        while j < len(frames) and vis_list[j] == "hidden":
            j += 1

        left = phase_arr[i - 1] if i > 0 else phase_arr[j] if j < len(frames) else 0.0
        right = phase_arr[j] if j < len(frames) else phase_arr[i - 1] if i > 0 else 0.0
        n = j - i

        delta_short = float(wrap_angles(np.array([right - left]))[0])
        sign = 1.0 if delta_short >= 0 else -1.0
        delta_long = delta_short - sign * 2.0 * math.pi
        right_ccw = left + delta_short
        right_cw = left + delta_long

        def far_count(right_uw: float) -> int:
            count = 0
            for k in range(n):
                t = (k + 1) / (n + 1)
                phase = left + t * (right_uw - left)
                if math.sin(phase + yaw_arr[i + k]) > 0:
                    count += 1
            return count

        fc_ccw = far_count(right_ccw)
        fc_cw = far_count(right_cw)
        if fc_cw > fc_ccw:
            right_uw = right_cw
        elif fc_ccw > fc_cw:
            right_uw = right_ccw
        else:
            right_uw = right_ccw if abs(delta_short) <= abs(delta_long) else right_cw

        for k in range(n):
            t = (k + 1) / (n + 1)
            corrected[i + k] = wrap_angles(np.array([left + t * (right_uw - left)]))[0]
        i = j

    m17_uw = np.unwrap(corrected)
    m17_uw = gaussian_filter1d(m17_uw, sigma=max(0.5, sigma / 2.0), mode="nearest")
    return wrap_angles(m17_uw), is_error


def build(args: argparse.Namespace) -> Path:
    sample = args.sample_dir
    m14_csv = args.m14_csv or sample / "results/renders/M14_joint_contact_handle_phase/handle_phase_joint_contact.csv"
    default_static = sample / "proxy/mug_body_only_cylinder_pose_table_static_sequence.csv"
    pose_csv = args.pose_csv or (default_static if default_static.exists() else sample / "proxy/mug_body_only_cylinder_pose_segmented_sequence.csv")
    vis_csv = args.vis_csv or sample / "annotations/vlm_handle_visibility_full/qwen_handle_visibility.csv"
    out_dir = args.out_dir or sample / "results/renders/M17_phase_corrected"
    out_dir.mkdir(parents=True, exist_ok=True)

    m14_rows = read_csv(m14_csv)
    m14_phases: dict[int, float] = {}
    m14_time: dict[int, float] = {}
    for row in m14_rows:
        fr = int(float(row["frame"]))
        m14_phases[fr] = ff(row, "mug_axial_phase_rad")
        m14_time[fr] = ff(row, "time", 0.0)

    yaw_data = read_yaw_by_frame(pose_csv)
    visibility = {int(float(row["frame"])): row["visibility"] for row in read_csv(vis_csv)}
    frames = sorted(set(m14_phases) & set(yaw_data))
    m17_wrapped, is_error = compute_m17_phases(frames, m14_phases, yaw_data, visibility, sigma=args.sigma)

    rows: list[dict[str, str]] = []
    for i, fr in enumerate(frames):
        m17_phase = float(m17_wrapped[i])
        m14_phase = m14_phases[fr]
        yaw = yaw_data.get(fr, 0.0)
        corr_rad = m17_phase - m14_phase
        sin_m14 = math.sin(yaw + m14_phase)
        sin_m17 = math.sin(yaw + m17_phase)
        rows.append(
            {
                "frame": str(fr),
                "time": f"{m14_time.get(fr, 0.0):.6f}",
                "m17_phase_rad": f"{m17_phase:.6f}",
                "m17_phase_deg": f"{math.degrees(m17_phase):.4f}",
                "m14_phase_rad": f"{m14_phase:.6f}",
                "m14_phase_deg": f"{math.degrees(m14_phase):.4f}",
                "is_error_frame": "1" if bool(is_error[i]) else "0",
                "phase_correction_rad": f"{corr_rad:.6f}",
                "phase_correction_deg": f"{math.degrees(corr_rad):.4f}",
                "vlm_visibility": visibility.get(fr, "visible"),
                "handle_z_sin_m14": f"{sin_m14:.6f}",
                "handle_z_sin_m17": f"{sin_m17:.6f}",
                "mug_yaw_deg": f"{math.degrees(yaw):.4f}",
            }
        )

    out_csv = out_dir / "corrected_handle_phase.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "csv": str(out_csv),
        "rows": len(rows),
        "m14_csv": str(m14_csv),
        "pose_csv": str(pose_csv),
        "vis_csv": str(vis_csv),
        "sigma": args.sigma,
        "mode": "csv_only_no_render",
    }
    (out_dir / "mug_handle_phase_correction_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-dir", type=Path, default=Path("samples_known_object/02_mug"))
    ap.add_argument("--m14-csv", type=Path, default=None)
    ap.add_argument("--pose-csv", type=Path, default=None)
    ap.add_argument("--vis-csv", type=Path, default=None)
    ap.add_argument("--sigma", type=float, default=SIGMA)
    ap.add_argument("--out-dir", type=Path, default=None)
    build(ap.parse_args())


if __name__ == "__main__":
    main()
