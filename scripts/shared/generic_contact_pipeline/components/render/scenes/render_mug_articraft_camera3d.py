#!/usr/bin/env python3
"""Render the mug Articraft rigid mesh in shared-camera 3D.

This is the radius-free mug camera-3D handoff: it consumes the M14 table-static
body pose + axial phase, then renders the actual Articraft mesh parts rather
than a ball/ellipse/procedural mug proxy.
"""

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
POSE_SOLVERS = REPO / "scripts/shared/generic_contact_pipeline/components/pose/solvers"
for path in (REPO, HERE, POSE_SOLVERS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import fit_mug_articraft_keyframe_pose as base  # noqa: E402
import render_mug_articraft_rigid_mesh_vlm as rigid  # noqa: E402

PART_COLORS = {
    "body_shell": "#d9dee2",
    "rim_ring": "#ff46d1",
    "bottom_disk": "#ffd42a",
    "handle_loop": "#11b5ff",
}
PART_LW = {
    "body_shell": 0.45,
    "rim_ring": 1.25,
    "bottom_disk": 1.15,
    "handle_loop": 1.45,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def ff(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        v = row.get(key, "")
        if v == "":
            return default
        return float(v)
    except Exception:
        return default


def read_pose_sequence(path: Path) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for row in read_csv(path):
        fr = int(float(row["frame"]))
        if any(row.get(k, "") != "" for k in ("tx", "ty", "tz", "qw", "qx", "qy", "qz")):
            quat = [ff(row, "qx", 0.0), ff(row, "qy", 0.0), ff(row, "qz", 0.0), ff(row, "qw", 1.0)]
            # ``base.object_R`` composes Ry(yaw) @ Rx(pitch) @ Rz(roll).
            # SciPy's intrinsic YXZ convention is the exact inverse mapping;
            # the former zyx conversion silently changed a typed quaternion.
            yaw, pitch, roll = Rotation.from_quat(quat).as_euler("YXZ")
            out[fr] = np.asarray(
                [
                    ff(row, "tx", ff(row, "x", 0.0)),
                    ff(row, "ty", ff(row, "y", 0.0)),
                    ff(row, "tz", ff(row, "z", 3.0)),
                    yaw,
                    pitch,
                    roll,
                    ff(row, "scale", 1.0),
                ],
                dtype=float,
            )
        else:
            out[fr] = np.asarray([ff(row, k) for k in ["x", "y", "z", "yaw", "pitch", "roll", "scale"]], dtype=float)
    return out


def read_phase_sequence(path: Path, col: str | None = None) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in read_csv(path):
        fr = int(float(row["frame"]))
        if col:
            out[fr] = ff(row, col, 0.0)
        else:
            out[fr] = ff(row, "mug_axial_phase_rad", ff(row, "handle_phase_rad", 0.0))
    return out


def cam_to_worldlike(points: np.ndarray) -> np.ndarray:
    out = np.asarray(points, dtype=float).copy()
    out[..., 1] *= -1.0
    return out


def transform_mesh(params: np.ndarray, verts: np.ndarray, phase: float) -> np.ndarray:
    local = verts @ rigid.rot_y(phase).T
    return base.transform(params, local)


def figure_to_bgr(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgb = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def make_writer(path: Path, fps: float, size: tuple[int, int]):
    tmp = path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {tmp}")
    return writer, tmp


def finalize(tmp: Path, out: Path) -> None:
    candidates = [Path("/home/yang/miniconda3/bin/ffmpeg"), Path("/usr/bin/ffmpeg"), Path(shutil.which("ffmpeg") or "ffmpeg")]
    last_err = None
    for ffmpeg_path in candidates:
        cmd = [str(ffmpeg_path), "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            tmp.unlink(missing_ok=True)
            return
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Could not transcode {tmp} to H264: {last_err}")


def draw_mesh(ax, pts_world: np.ndarray, edges: list[tuple[int, int]], color: str, lw: float, alpha: float, edge_step: int = 1) -> None:
    # Worldlike plotting convention matches render_pose6d_scene: X, Z(depth), Y(up).
    use_edges = edges[:: max(1, edge_step)]
    if not use_edges:
        return
    segs = []
    for a, b in use_edges:
        seg = pts_world[[a, b]]
        if np.all(np.isfinite(seg)):
            segs.append(np.column_stack([seg[:, 0], seg[:, 2], seg[:, 1]]))
    if segs:
        ax.add_collection3d(Line3DCollection(segs, colors=color, linewidths=lw, alpha=alpha))


def wrap_angle_rad(v: float) -> float:
    return (float(v) + math.pi) % (2.0 * math.pi) - math.pi


def draw_phase_arrow(ax, params: np.ndarray, phase: float, direction: float, radius: float, y: float) -> None:
    if abs(direction) < 1e-5 or not np.isfinite(direction):
        return
    sign = 1.0 if direction >= 0.0 else -1.0
    span = 0.80
    # The arrow is a diagnostic only: an orange arc around the mug axial phase,
    # projected in the same camera/worldlike convention as the Articraft mesh.
    angles = phase + np.linspace(-0.45 * sign, 0.45 * sign, 22)
    local = np.column_stack([radius * np.cos(angles), np.full_like(angles, y), radius * np.sin(angles)])
    pts = cam_to_worldlike(base.transform(params, local))
    ax.plot(pts[:, 0], pts[:, 2], pts[:, 1], color="#ff7a00", linewidth=2.4, alpha=0.95)
    head = pts[-1]
    prev = pts[-4]
    vec = head - prev
    n = np.linalg.norm(vec)
    if n > 1e-8:
        vec = vec / n
        ax.quiver(prev[0], prev[2], prev[1], vec[0], vec[2], vec[1], length=0.055, color="#ff7a00", linewidth=2.0, arrow_length_ratio=0.55, normalize=True)


def render(args: argparse.Namespace) -> dict[str, object]:
    sample = args.sample_dir
    pose_csv = args.pose_csv or (sample / "proxy" / "mug_body_only_cylinder_pose_table_static_sequence.csv")
    if not pose_csv.exists():
        pose_csv = sample / "proxy" / "mug_body_only_cylinder_pose_segmented_sequence.csv"
    phase_csv = args.phase_csv or (sample / "results" / "renders" / "M14_joint_contact_handle_phase" / "handle_phase_joint_contact.csv")
    poses = read_pose_sequence(pose_csv)
    phases = read_phase_sequence(phase_csv, col=getattr(args, "phase_col", None))
    meshes = rigid.load_articraft_meshes(args.mesh_root)
    frames = sorted(set(poses) & set(phases))
    if not frames:
        raise RuntimeError("No overlapping pose/phase frames")
    body_verts = meshes.get("body_shell", next(iter(meshes.values())))[0]
    body_r = float(np.nanpercentile(np.linalg.norm(body_verts[:, [0, 2]], axis=1), 92))
    arrow_y = float(np.nanmax(body_verts[:, 1]) + 0.030)
    phase_dirs: dict[int, float] = {}
    for i, fr in enumerate(frames):
        if i == 0:
            d = wrap_angle_rad(phases[frames[min(1, len(frames) - 1)]] - phases[fr])
        elif i == len(frames) - 1:
            d = wrap_angle_rad(phases[fr] - phases[frames[i - 1]])
        else:
            d = wrap_angle_rad(phases[frames[i + 1]] - phases[frames[i - 1]])
        phase_dirs[fr] = d

    out_dir = args.out_dir or (sample / "results" / "renders" / "M16_articraft_camera3d")
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / "camera3d.mp4"
    preview = out_dir / "camera3d_preview.png"

    # Precompute transformed mesh points for bounds and rendering.
    mesh_cam: dict[int, dict[str, np.ndarray]] = {}
    centers = []
    all_bounds = []
    for fr in frames:
        p = poses[fr]
        ph = phases[fr]
        frame_parts = {}
        for name, (verts, _edges) in meshes.items():
            cam = transform_mesh(p, verts, ph)
            frame_parts[name] = cam
            if name in {"body_shell", "rim_ring", "bottom_disk", "handle_loop"}:
                all_bounds.append(cam[:: max(1, len(cam)//120)])
        mesh_cam[fr] = frame_parts
        centers.append(base.transform(p, np.zeros((1, 3), dtype=float))[0])
    centers_cam = np.asarray(centers)
    centers_world = cam_to_worldlike(centers_cam)
    all_world = cam_to_worldlike(np.concatenate(all_bounds + [centers_cam], axis=0))

    x_min, y_min, z_min = np.nanmin(all_world, axis=0)
    x_max, y_max, z_max = np.nanmax(all_world, axis=0)
    mx = max(0.10, 0.22 * (x_max - x_min))
    my = max(0.10, 0.22 * (y_max - y_min))
    mz = max(0.10, 0.22 * (z_max - z_min))

    writer, tmp = make_writer(mp4, args.fps, (args.width, args.height))
    preview_written = False
    for idx, fr in enumerate(frames):
        fig = plt.figure(figsize=(args.width / 100.0, args.height / 100.0), dpi=100, facecolor="#f7f8fb")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#f7f8fb")
        ax.set_xlim(float(x_min - mx), float(x_max + mx))
        ax.set_ylim(float(z_min - mz), float(z_max + mz))
        ax.set_zlim(float(y_min - my), float(y_max + my))
        ax.set_xlabel("X_cam (m)")
        ax.set_ylabel("Z_cam depth (m)")
        ax.set_zlabel("Y_worldlike up (m)")
        ax.set_title("Radius-free Mug Articraft Camera 3D")
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.grid(True, alpha=0.18)

        ax.plot(centers_world[:, 0], centers_world[:, 2], centers_world[:, 1], color="#aeb8c8", linewidth=1.4, alpha=0.45)
        ax.plot(centers_world[: idx + 1, 0], centers_world[: idx + 1, 2], centers_world[: idx + 1, 1], color="#4c72b0", linewidth=2.4, alpha=0.9)

        for name in ["body_shell", "rim_ring", "bottom_disk", "handle_loop"]:
            if name not in meshes:
                continue
            _verts, edges = meshes[name]
            pts_world = cam_to_worldlike(mesh_cam[fr][name])
            edge_step = args.body_edge_step if name == "body_shell" else args.detail_edge_step
            draw_mesh(
                ax,
                pts_world,
                edges,
                PART_COLORS.get(name, "#333333"),
                PART_LW.get(name, 1.0),
                0.92 if name != "body_shell" else 0.55,
                edge_step=edge_step,
            )
        if args.show_phase_arrow:
            draw_phase_arrow(ax, poses[fr], phases[fr], phase_dirs.get(fr, 0.0), body_r * 1.12, arrow_y)

        c = centers_world[idx]
        ax.scatter([c[0]], [c[2]], [c[1]], s=55, color="#db7a20", edgecolors="#1f1f1f", linewidths=0.8, depthshade=False)
        vis_rows_map = getattr(args, "_vis_rows", {})
        vis_lbl = vis_rows_map.get(fr, "")
        fig.text(0.04, 0.045, f"frame {fr:03d}  phase={math.degrees(phases[fr]):.1f} deg  {vis_lbl}  pose={pose_csv.name}", fontsize=10, color="#1a5fa8" if vis_lbl == "hidden" else "#333333")
        canvas = figure_to_bgr(fig)
        plt.close(fig)
        writer.write(canvas)
        if not preview_written:
            cv2.imwrite(str(preview), canvas)
            preview_written = True
    writer.release()
    finalize(tmp, mp4)

    summary = {
        "render": str(mp4),
        "preview": str(preview),
        "pose_csv": str(pose_csv),
        "phase_csv": str(phase_csv),
        "mesh_root": str(args.mesh_root),
        "num_frames": len(frames),
        "note": "Camera 3D render using the materialized Articraft mug mesh and M14 table-static pose/phase. This is the radius-free mug camera3D handoff; no ball/ellipse proxy is used.",
    }
    (out_dir / "outputs.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Render Articraft mug mesh in camera 3D from M14 pose/phase.")
    ap.add_argument("--sample-dir", type=Path, default=Path("samples_known_object/02_mug"))
    ap.add_argument("--mesh-root", type=Path, default=Path("samples_known_object/02_mug/articraft/materialized_mug_mesh"))
    ap.add_argument("--pose-csv", type=Path, default=None)
    ap.add_argument("--phase-csv", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--azim", type=float, default=-62.0)
    ap.add_argument("--body-edge-step", type=int, default=3)
    ap.add_argument("--detail-edge-step", type=int, default=1)
    ap.add_argument("--phase-col", type=str, default=None, help="CSV column name for phase (default: mug_axial_phase_rad / handle_phase_rad).")
    ap.add_argument("--show-phase-arrow", action="store_true", help="Draw an orange diagnostic arrow for axial phase rotation direction.")
    args = ap.parse_args()
    print(json.dumps(render(args), indent=2))


if __name__ == "__main__":
    main()
