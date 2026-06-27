#!/usr/bin/env python3
"""CSV-only stable mug grasp-anchor projection.

This is the pipeline-owned version of the solved M15 diagnostic projection. It
projects the stable object-local grasp anchor through the current mug pose/phase
and writes the projection CSV only; no video is rendered.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
GENERIC_RENDER = REPO / "scripts" / "shared" / "generic_contact_pipeline" / "components" / "render"
for p in (REPO, GENERIC_RENDER):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import fit_mug_articraft_keyframe_pose as base  # noqa: E402
import render_mug_articraft_rigid_mesh_vlm as rigid  # noqa: E402


FIELDS = [
    "frame",
    "time",
    "frame_mode",
    "stable_grasp_source_frame",
    "stable_grasp_object_part",
    "stable_grasp_object_region",
    "stable_grasp_hand_side",
    "stable_grasp_primary_finger",
    "stable_grasp_type",
    "anchor_local_x",
    "anchor_local_y",
    "anchor_local_z",
    "anchor_u",
    "anchor_v",
    "anchor_z",
    "active_label",
    "active_part_u",
    "active_part_v",
    "active_label_conf",
    "hand_anchor_dist_px",
    "residual_weight",
    "use_as_attachment_residual",
    "reason",
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


def read_phase_sequence(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in read_csv(path):
        fr = int(float(row["frame"]))
        out[fr] = ff(row, "mug_axial_phase_rad", ff(row, "handle_phase_rad", 0.0))
    return out


def read_pose_sequence(path: Path) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for row in read_csv(path):
        fr = int(float(row["frame"]))
        out[fr] = np.array([ff(row, key) for key in ["x", "y", "z", "yaw", "pitch", "roll", "scale"]], dtype=float)
    return out


def project_anchor(params: np.ndarray, k: np.ndarray, local: np.ndarray, phase: float) -> tuple[np.ndarray, np.ndarray]:
    pts = local[None, :] @ rigid.rot_y(phase).T
    cam = base.transform(params, pts)
    uv = np.array([base.project(point, k) for point in cam], dtype=float)
    return cam, uv


def residual_policy(state: dict[str, str], dist_px: float, conf: float) -> tuple[float, int, str]:
    mode = state.get("frame_mode", "")
    has_anchor = state.get("stable_grasp_local_x", "") != ""
    if not has_anchor:
        return 0.0, 0, "no stable grasp anchor exists yet"
    if state.get("stable_grasp_object_part") != "handle":
        return 0.0, 0, "stable grasp is not handle semantic region"
    if not np.isfinite(dist_px):
        return 0.0, 0, "active hand uv missing"
    if mode == "direct_grasp_anchor":
        return 1.0, 1, "VISIBLE_CONFIRMED_CONTACT: update stable upper_handle grasp anchor"
    if mode == "rim_contact_keep_previous_grasp_anchor":
        return 0.45, 1, "DRINKING_OCCLUDED_HAND_CONTACT: rim/mouth visible, keep previous hand-handle grasp; do not update anchor"
    if mode == "keep_previous_grasp_anchor":
        weight = 0.85 if dist_px < 90.0 and conf >= 0.15 else 0.45
        return weight, 1, "OCCLUDED_CONTINUOUS_CONTACT: keep previous upper_handle grasp; still physical attachment"
    return 0.0, 0, "mode does not use attachment residual"


def build(args: argparse.Namespace) -> Path:
    sample = args.sample_dir
    default_static_pose = sample / "proxy/mug_body_only_cylinder_pose_table_static_sequence.csv"
    pose_csv = args.pose_csv or (default_static_pose if default_static_pose.exists() else sample / "proxy/mug_body_only_cylinder_pose_segmented_sequence.csv")
    phase_csv = args.phase_csv or sample / "results/renders/M14_joint_contact_handle_phase/handle_phase_joint_contact.csv"
    state_csv = args.state_csv or sample / "results/mug_grasp_anchor_state/mug_grasp_anchor_state.csv"
    contact_csv = args.contact_csv or sample / "results/mug_articraft_contact_points/mug_articraft_contact_points.csv"
    out_dir = args.out_dir or sample / "results/renders/M15_grasp_anchor_projection"
    out_dir.mkdir(parents=True, exist_ok=True)

    poses = read_pose_sequence(pose_csv)
    phases = read_phase_sequence(phase_csv)
    states = {int(float(row["frame"])): row for row in read_csv(state_csv)}
    contacts = {int(float(row["frame"])): row for row in read_csv(contact_csv)}
    k_all = base.load_K(sample)

    rows: list[dict[str, str]] = []
    for fr in sorted(set(poses) & set(states)):
        state = states[fr]
        contact = contacts.get(fr, {})
        phase = phases.get(fr, 0.0)
        local = np.array(
            [
                ff(state, "stable_grasp_local_x"),
                ff(state, "stable_grasp_local_y"),
                ff(state, "stable_grasp_local_z"),
            ],
            dtype=float,
        )
        anchor_cam = np.full(3, math.nan)
        anchor_uv = np.full(2, math.nan)
        if np.all(np.isfinite(local)):
            cam, uv = project_anchor(poses[fr], k_all[fr - 1], local, phase)
            anchor_cam = cam[0]
            anchor_uv = uv[0]

        hu = ff(contact, "active_part_u")
        hv = ff(contact, "active_part_v")
        conf = ff(contact, "active_label_conf", 0.0)
        if np.all(np.isfinite(anchor_uv)) and np.isfinite(hu) and np.isfinite(hv):
            dist = float(np.linalg.norm(anchor_uv - np.array([hu, hv])))
        else:
            dist = math.nan
        weight, use_res, reason = residual_policy(state, dist, conf)
        rows.append(
            {
                "frame": str(fr),
                "time": state.get("time", ""),
                "frame_mode": state.get("frame_mode", ""),
                "stable_grasp_source_frame": state.get("stable_grasp_source_frame", ""),
                "stable_grasp_object_part": state.get("stable_grasp_object_part", ""),
                "stable_grasp_object_region": state.get("stable_grasp_object_region", ""),
                "stable_grasp_hand_side": state.get("stable_grasp_hand_side", ""),
                "stable_grasp_primary_finger": state.get("stable_grasp_primary_finger", ""),
                "stable_grasp_type": state.get("stable_grasp_type", ""),
                "anchor_local_x": state.get("stable_grasp_local_x", ""),
                "anchor_local_y": state.get("stable_grasp_local_y", ""),
                "anchor_local_z": state.get("stable_grasp_local_z", ""),
                "anchor_u": f"{anchor_uv[0]:.3f}" if np.isfinite(anchor_uv[0]) else "",
                "anchor_v": f"{anchor_uv[1]:.3f}" if np.isfinite(anchor_uv[1]) else "",
                "anchor_z": f"{anchor_cam[2]:.6f}" if np.isfinite(anchor_cam[2]) else "",
                "active_label": contact.get("active_label", state.get("active_label", "")),
                "active_part_u": f"{hu:.3f}" if np.isfinite(hu) else "",
                "active_part_v": f"{hv:.3f}" if np.isfinite(hv) else "",
                "active_label_conf": f"{conf:.6f}" if np.isfinite(conf) else "",
                "hand_anchor_dist_px": f"{dist:.3f}" if np.isfinite(dist) else "",
                "residual_weight": f"{weight:.3f}",
                "use_as_attachment_residual": str(use_res),
                "reason": reason,
            }
        )

    out_csv = out_dir / "grasp_anchor_projection.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    dists = [float(row["hand_anchor_dist_px"]) for row in rows if row["hand_anchor_dist_px"]]
    used = [row for row in rows if row["use_as_attachment_residual"] == "1"]
    summary = {
        "pose_csv": str(pose_csv),
        "phase_csv": str(phase_csv),
        "state_csv": str(state_csv),
        "contact_csv": str(contact_csv),
        "csv": str(out_csv),
        "num_frames": len(rows),
        "num_residual_frames": len(used),
        "mean_hand_anchor_dist_px": float(np.mean(dists)) if dists else None,
        "median_hand_anchor_dist_px": float(np.median(dists)) if dists else None,
        "mode": "csv_only_no_render",
    }
    (out_dir / "mug_grasp_anchor_projection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-dir", type=Path, default=Path("samples_known_object/02_mug"))
    ap.add_argument("--pose-csv", type=Path, default=None)
    ap.add_argument("--phase-csv", type=Path, default=None)
    ap.add_argument("--state-csv", type=Path, default=None)
    ap.add_argument("--contact-csv", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    build(ap.parse_args())


if __name__ == "__main__":
    main()
