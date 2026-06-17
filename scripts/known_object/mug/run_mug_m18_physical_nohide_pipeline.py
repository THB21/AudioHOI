#!/usr/bin/env python3
"""Clean M18 mug physical no-hide optimization pipeline.

This script keeps the final behavior from the M45 experiment but removes the
temporary experiment layering:

1. Build a no-hide handle phase track from the M17 phase baseline with a
   smooth drinking-entry schedule.
2. Automatically detect Euler rotation branch jumps and smooth them with Slerp.
3. Automatically detect table-static release and freeze later mug pose.
4. Optionally render the full six videos: object/with-human x camera3d/overlay/side_yz.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp

REPO = Path(__file__).resolve().parents[3]
STAGE1 = REPO / "scripts" / "shared" / "radius_free_proxy" / "stage1_observation"
STAGE5 = REPO / "scripts" / "shared" / "radius_free_proxy" / "stage5_render"
for p in (REPO, STAGE1, STAGE5):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import fit_mug_articraft_keyframe_pose as pose_base  # noqa: E402
import render_mug_articraft_camera3d as m16  # noqa: E402


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def ff(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


def wrap_pi(x: float) -> float:
    return float((x + math.pi) % (2.0 * math.pi) - math.pi)


def choose_unwrapped_deg(deg: float, ref_deg: float) -> float:
    return float(deg + 360.0 * round((ref_deg - deg) / 360.0))


def default_phase_csv(sample: Path) -> Path:
    return sample / "results" / "renders" / "M17_phase_corrected" / "corrected_handle_phase.csv"


def smooth_entry_phase_anchors() -> list[tuple[int, float | None]]:
    # These anchors are geometric phase-sweep targets, relaxed around 57-65 so
    # the handle turns continuously into drinking instead of chasing a single
    # frame contact point. They are intentionally kept as the M45 phase prior;
    # automatic phase optimization should replace this with a data-term solver,
    # not move the anchors by detected hidden-state runs.
    return [
        (1, 0.447322),
        (20, -4.6),
        (26, 22.0),
        (30, 58.0),
        (35, 92.0),
        (45, 145.0),
        (50, 154.0),
        (55, 160.0),
        (57, 162.0),
        (60, 156.0),
        (62, 152.0),
        (65, 146.0),
        (75, 132.0),
        (84, 125.0),
        (86, 115.0),
        (89, 100.0),
        (95, 75.0),
        (100, 70.0),
        (105, 90.0),
        (107, 85.0),
        (110, 65.0),
        (112, 55.0),
        (115, 45.0),
        (120, 30.0),
        (126, 15.0),
        (135, 5.0),
        (145, -20.0),
        (150, -23.0),
        (180, -31.902183),
        (240, -31.744569),
    ]


def build_nohide_phase(args: argparse.Namespace) -> tuple[Path, Path]:
    sample = args.sample_dir.resolve()
    phase_csv = args.phase_csv or default_phase_csv(sample)
    rows = read_rows(phase_csv)
    frames = np.array([int(float(r["frame"])) for r in rows], dtype=int)
    phase0 = np.unwrap(np.array([ff(r, args.phase_col, ff(r, "m17_phase_rad", ff(r, "phase_rad", 0.0))) for r in rows], dtype=float))
    base_deg = np.rad2deg(phase0)
    frame_to_i = {int(fr): i for i, fr in enumerate(frames)}

    anchors: list[tuple[int, float]] = []
    last_deg = None
    phase_anchor_specs = smooth_entry_phase_anchors()
    for fr, deg in phase_anchor_specs:
        if fr not in frame_to_i:
            continue
        ref = base_deg[frame_to_i[fr]] if last_deg is None else last_deg
        val = base_deg[frame_to_i[fr]] if deg is None else choose_unwrapped_deg(deg, ref)
        anchors.append((fr, val))
        last_deg = val

    a_frames = np.array([fr for fr, _ in anchors], dtype=float)
    a_deg = np.array([deg for _, deg in anchors], dtype=float)
    desired = PchipInterpolator(a_frames, a_deg, extrapolate=True)(frames)

    out_deg = desired.copy()
    for i in range(1, len(out_deg)):
        delta = out_deg[i] - out_deg[i - 1]
        if abs(delta) > args.max_phase_step_deg:
            out_deg[i] = out_deg[i - 1] + math.copysign(args.max_phase_step_deg, delta)
    for i in range(len(out_deg) - 2, -1, -1):
        delta = out_deg[i] - out_deg[i + 1]
        if abs(delta) > args.max_phase_step_deg:
            out_deg[i] = out_deg[i + 1] + math.copysign(args.max_phase_step_deg, delta)

    phase_rad = np.deg2rad(out_deg)
    out_csv = args.out_phase_csv or sample / "results" / "final_result" / "handle_phase.csv"
    summary = out_csv.with_name("handle_phase_summary.txt")

    fields = list(rows[0].keys())
    for key in ["m43_phase_rad", "m43_phase_deg", "m43_source", "m17_phase_rad", "m17_phase_deg"]:
        if key not in fields:
            fields.append(key)
    out_rows = []
    for i, row in enumerate(rows):
        new_phase = wrap_pi(float(phase_rad[i]))
        new = dict(row)
        new["m43_phase_rad"] = f"{new_phase:.9f}"
        new["m43_phase_deg"] = f"{math.degrees(new_phase):.6f}"
        new["m43_source"] = "smooth_entry_physical_phase_no_hide"
        new["m17_phase_rad"] = f"{new_phase:.9f}"
        new["m17_phase_deg"] = f"{math.degrees(new_phase):.6f}"
        out_rows.append(new)
    write_rows(out_csv, out_rows, fields)

    with summary.open("w") as f:
        f.write("policy: no handle hiding; render every Articraft part, including handle_loop\n")
        f.write(f"input_phase_csv: {phase_csv}\n")
        f.write(f"max_abs_velocity_deg: {float(np.max(np.abs(np.diff(out_deg)))):.3f}\n")
        f.write("anchors_frame_deg_unwrapped:\n")
        for fr, val in anchors:
            f.write(f"{fr} {val:.3f}\n")
        f.write("phase_entry_detection: fixed_m45_phase_prior\n")
    return out_csv, summary


def decompose_yxz(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of object_R = Ry(yaw) @ Rx(pitch) @ Rz(roll)."""
    pitch = math.asin(max(-1.0, min(1.0, -float(R[1, 2]))))
    yaw = math.atan2(float(R[0, 2]), float(R[2, 2]))
    roll = math.atan2(float(R[1, 0]), float(R[1, 1]))
    return yaw, pitch, roll


def row_rotation(row: dict[str, str]) -> np.ndarray:
    return pose_base.object_R(ff(row, "yaw"), ff(row, "pitch"), ff(row, "roll"))


def rotation_geodesic_deg(a: np.ndarray, b: np.ndarray) -> float:
    tr = float(np.trace(a.T @ b))
    return math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1.0) * 0.5))))


def robust_threshold(values: np.ndarray, floor: float) -> float:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return max(floor, med + 8.0 * 1.4826 * mad)


def detect_rotation_smooth_window(
    rows: list[dict[str, str]],
    min_frame: int,
    max_frame: int,
    window_after: int,
) -> tuple[int, int, list[tuple[int, float]], str]:
    """Find an isolated pose-rotation branch jump and return Slerp endpoints."""
    frame_rows = [(int(float(r["frame"])), r) for r in rows]
    steps: list[tuple[int, float]] = []
    prev_R = None
    for fr, row in frame_rows:
        R = row_rotation(row)
        if prev_R is not None:
            steps.append((fr, rotation_geodesic_deg(prev_R, R)))
        prev_R = R

    values = np.array([v for _fr, v in steps], dtype=float)
    threshold = robust_threshold(values, floor=12.0)
    candidates = [(fr, step) for fr, step in steps if min_frame <= fr <= max_frame and step >= threshold]
    if not candidates:
        # No detected branch jump: use a degenerate window so pose remains unchanged.
        first = frame_rows[0][0]
        return first, first, steps, f"no_rotation_jump_detected threshold={threshold:.3f}"

    jump_frame, jump_step = max(candidates, key=lambda item: item[1])
    available = {fr for fr, _row in frame_rows}
    start = jump_frame - 1
    end = min(jump_frame + window_after, max(available))
    while end not in available and end > start:
        end -= 1
    if start not in available or end <= start:
        return start, start, steps, f"rotation_jump_detected_but_invalid jump={jump_frame} step={jump_step:.3f}"
    return start, end, steps, f"auto_rotation_jump jump={jump_frame} step={jump_step:.3f} threshold={threshold:.3f}"


def detect_static_frame(sample: Path, fallback_rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[int, str]:
    """Detect the first stable table-support run after release."""
    contact_csv = args.contact_state_csv or sample / "results" / "contact_candidates_object_proxy" / "contact_state_frames.csv"
    if contact_csv.exists():
        rows = read_rows(contact_csv)
        flags: list[tuple[int, bool]] = []
        for row in rows:
            fr = int(float(row["frame"]))
            if fr < args.static_min_frame:
                continue
            support_conf = ff(row, "support_conf", 0.0)
            support_gap = ff(row, "signed_support_gap_px", -1e9)
            accel = ff(row, "object_acceleration", 0.0)
            floor_state = int(ff(row, "floor_contact_state", 0.0))
            good = (
                floor_state == 1
                or (
                    support_conf >= args.static_support_conf
                    and support_gap >= args.static_min_support_gap_px
                    and accel <= args.static_max_acceleration
                )
            )
            flags.append((fr, bool(good)))
        run_start = None
        run_len = 0
        for fr, good in flags:
            if good:
                if run_start is None:
                    run_start = fr
                    run_len = 1
                else:
                    run_len += 1
                if run_len >= args.static_min_run:
                    return int(run_start), (
                        "auto_table_static "
                        f"support_conf>={args.static_support_conf} "
                        f"support_gap>={args.static_min_support_gap_px} "
                        f"run={args.static_min_run}"
                    )
            else:
                run_start = None
                run_len = 0

    frames = [int(float(r["frame"])) for r in fallback_rows]
    fallback = frames[-1]
    return fallback, f"no_table_static_detected fallback={fallback}"


def build_physical_pose(args: argparse.Namespace) -> tuple[Path, Path]:
    sample = args.sample_dir.resolve()
    pose_csv = args.pose_csv or sample / "results" / "mug_m18_opening_2d_video_correction" / "mug_m18_opening_2d_video_pose.csv"
    rows = read_rows(pose_csv)
    by_frame = {int(float(r["frame"])): r for r in rows}
    detected_start, detected_end, rot_steps, rotation_reason = detect_rotation_smooth_window(
        rows,
        min_frame=args.rotation_detect_min_frame,
        max_frame=args.rotation_detect_max_frame,
        window_after=args.rotation_smooth_window_after,
    )
    start = args.rotation_smooth_start if args.rotation_smooth_start is not None else detected_start
    end = args.rotation_smooth_end if args.rotation_smooth_end is not None else detected_end
    detected_static, static_reason = detect_static_frame(sample, rows, args)
    static_frame = args.static_frame if args.static_frame is not None else detected_static
    args.resolved_static_frame = static_frame
    if start not in by_frame or end not in by_frame:
        raise ValueError(f"missing rotation endpoint frame(s): {start}, {end}")
    if static_frame not in by_frame:
        raise ValueError(f"missing static frame: {static_frame}")

    R0 = pose_base.object_R(ff(by_frame[start], "yaw"), ff(by_frame[start], "pitch"), ff(by_frame[start], "roll"))
    R1 = pose_base.object_R(ff(by_frame[end], "yaw"), ff(by_frame[end], "pitch"), ff(by_frame[end], "roll"))
    slerp = Slerp([start, end], Rotation.from_matrix([R0, R1]))

    fields = list(rows[0].keys())
    for key in [
        "m45_pose_source",
        "m45_rotation_smooth_start",
        "m45_rotation_smooth_end",
        "m45_static_frame",
        "m45_yaw_input",
        "m45_pitch_input",
        "m45_roll_input",
    ]:
        if key not in fields:
            fields.append(key)

    tmp_rows = []
    for row in rows:
        fr = int(float(row["frame"]))
        new = dict(row)
        source = "unchanged"
        if start < fr < end:
            R = slerp([fr]).as_matrix()[0]
            yaw, pitch, roll = decompose_yxz(R)
            new["m45_yaw_input"] = row.get("yaw", "")
            new["m45_pitch_input"] = row.get("pitch", "")
            new["m45_roll_input"] = row.get("roll", "")
            new["yaw"] = f"{yaw:.9f}"
            new["pitch"] = f"{pitch:.9f}"
            new["roll"] = f"{roll:.9f}"
            new["yaw_deg"] = f"{math.degrees(yaw):.6f}"
            new["pitch_deg"] = f"{math.degrees(pitch):.6f}"
            new["roll_deg"] = f"{math.degrees(roll):.6f}"
            source = "rotation_slerp"
        new["m45_pose_source"] = source
        new["m45_rotation_smooth_start"] = str(start)
        new["m45_rotation_smooth_end"] = str(end)
        new["m45_static_frame"] = str(static_frame)
        tmp_rows.append(new)

    ref = tmp_rows[[int(float(r["frame"])) for r in tmp_rows].index(static_frame)]
    pose_keys = ["x", "y", "z", "yaw", "pitch", "roll", "scale"]
    for row in tmp_rows:
        fr = int(float(row["frame"]))
        if fr > static_frame:
            for key in pose_keys:
                row[key] = ref[key]
            for key in ["yaw_deg", "pitch_deg", "roll_deg"]:
                if key in ref:
                    row[key] = ref[key]
            row["m45_pose_source"] = "table_static_release"

    out_csv = args.out_pose_csv or sample / "results" / "final_result" / "object_pose.csv"
    summary = out_csv.with_name("object_pose_summary.txt")
    write_rows(out_csv, tmp_rows, fields)
    with summary.open("w") as f:
        f.write("policy: M18 2D pose + auto rotation-jump Slerp + auto table-static freeze\n")
        f.write(f"input_pose_csv: {pose_csv}\n")
        f.write(f"rotation_smooth_frames: {start}-{end}\n")
        f.write(f"rotation_detection: {rotation_reason}\n")
        f.write("rotation_step_deg_by_frame:\n")
        for fr, step in rot_steps:
            if start - 3 <= fr <= end + 3 or step >= 12.0:
                f.write(f"{fr} {step:.6f}\n")
        f.write(f"static_frame: {static_frame}\n")
        f.write(f"static_detection: {static_reason}\n")
        f.write("frozen_pose:\n")
        for key in pose_keys:
            f.write(f"{key}: {ref.get(key, '')}\n")
    return out_csv, summary


def render_full6(args: argparse.Namespace, pose_csv: Path, phase_csv: Path) -> Path:
    import render_mug_articraft_camera3d_scene as renderer  # noqa: E402
    import render_pose6d_scene as scene  # noqa: E402

    sample = args.sample_dir.resolve()
    out_root = args.out_render_root or sample / "results" / "renders" / "final_result"
    mesh_root = args.mesh_root or sample / "articraft" / "materialized_mug_mesh"
    render_args = SimpleNamespace(
        sample_dir=sample,
        fps=args.fps,
        width=args.width,
        height=args.height,
        elev=args.elev,
        azim=args.azim,
        body_edge_step=args.body_edge_step,
        detail_edge_step=args.detail_edge_step,
        vertex_stride=args.vertex_stride,
        release_frame=getattr(args, "resolved_static_frame", args.static_frame or 160),
    )

    poses = m16.read_pose_sequence(pose_csv)
    phases = m16.read_phase_sequence(phase_csv, col="m17_phase_rad")
    meshes = m16.rigid.load_articraft_meshes(mesh_root)
    meshes_solid = m16.rigid.load_articraft_meshes_solid(mesh_root)
    frames = sorted(set(poses) & set(phases))
    mesh_cam, centers_cam, mesh_bounds = renderer.prepare_mesh_sequence(poses, phases, meshes, frames)
    K = renderer.load_K(sample)

    outputs = {
        "pose_csv": str(pose_csv),
        "phase_csv": str(phase_csv),
        "phase_col": "m17_phase_rad",
        "mesh_root": str(mesh_root),
        "num_frames": len(frames),
    }
    outputs["object_only"] = renderer.render_variant(
        render_args, out_root / "object_only", frames, poses, phases, meshes, mesh_cam, centers_cam, mesh_bounds,
        meshes_solid=meshes_solid, with_human=False,
    )
    outputs["object_only"].update(renderer.render_overlay_variant(render_args, out_root / "object_only", frames, meshes_solid, mesh_cam, K, with_human=False))
    outputs["object_only"].update(renderer.render_side_yz_variant(render_args, out_root / "object_only", frames, meshes, mesh_cam, centers_cam, with_human=False))

    human = scene.read_human_result(sample / "results" / "gvhmr" / "result.pkl")
    joints, sampled_vertices = scene.build_body_outputs(args.body_model_root, human, args.vertex_stride)
    outputs["with_human"] = renderer.render_variant(
        render_args, out_root / "with_human", frames, poses, phases, meshes, mesh_cam, centers_cam, mesh_bounds,
        meshes_solid=meshes_solid, with_human=True, joints=joints, sampled_vertices=sampled_vertices,
        palm_rows=None, gate_rows=None,
    )
    outputs["with_human"].update(renderer.render_overlay_variant(render_args, out_root / "with_human", frames, meshes_solid, mesh_cam, K, with_human=True, joints=joints, palm_rows=None, gate_rows=None))
    outputs["with_human"].update(renderer.render_side_yz_variant(render_args, out_root / "with_human", frames, meshes, mesh_cam, centers_cam, with_human=True, joints=joints, palm_rows=None, gate_rows=None))

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "outputs.json").write_text(json.dumps(outputs, indent=2) + "\n")
    print(json.dumps(outputs, indent=2))
    return out_root


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-dir", type=Path, default=Path("samples_known_object/02_mug"))
    ap.add_argument("--pose-csv", type=Path)
    ap.add_argument("--phase-csv", type=Path)
    ap.add_argument("--phase-col", default="m17_phase_rad")
    ap.add_argument("--out-pose-csv", type=Path)
    ap.add_argument("--out-phase-csv", type=Path)
    ap.add_argument("--rotation-smooth-start", type=int, default=None, help="Override auto-detected Slerp start frame.")
    ap.add_argument("--rotation-smooth-end", type=int, default=None, help="Override auto-detected Slerp end frame.")
    ap.add_argument("--rotation-detect-min-frame", type=int, default=35)
    ap.add_argument("--rotation-detect-max-frame", type=int, default=80)
    ap.add_argument("--rotation-smooth-window-after", type=int, default=9)
    ap.add_argument("--static-frame", type=int, default=None, help="Override auto-detected table-static frame.")
    ap.add_argument("--contact-state-csv", type=Path, default=None)
    ap.add_argument("--static-min-frame", type=int, default=120)
    ap.add_argument("--static-min-run", type=int, default=8)
    ap.add_argument("--static-support-conf", type=float, default=0.99)
    ap.add_argument("--static-min-support-gap-px", type=float, default=-10.0)
    ap.add_argument("--static-max-acceleration", type=float, default=0.15)
    ap.add_argument("--max-phase-step-deg", type=float, default=7.0)
    ap.add_argument("--render-full6", action="store_true")
    ap.add_argument("--out-render-root", type=Path)
    ap.add_argument("--mesh-root", type=Path)
    ap.add_argument("--body-model-root", type=Path, default=Path("third-party/GVHMR/inputs/checkpoints/body_models"))
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--azim", type=float, default=-62.0)
    ap.add_argument("--body-edge-step", type=int, default=3)
    ap.add_argument("--detail-edge-step", type=int, default=1)
    ap.add_argument("--vertex-stride", type=int, default=16)
    args = ap.parse_args()

    phase_out, phase_summary = build_nohide_phase(args)
    pose_out, pose_summary = build_physical_pose(args)
    print(f"[m18_physical_nohide] phase -> {phase_out}")
    print(f"[m18_physical_nohide] phase summary -> {phase_summary}")
    print(f"[m18_physical_nohide] pose -> {pose_out}")
    print(f"[m18_physical_nohide] pose summary -> {pose_summary}")
    if args.render_full6:
        out_root = render_full6(args, pose_out, phase_out)
        print(f"[m18_physical_nohide] render -> {out_root}")


if __name__ == "__main__":
    main()
