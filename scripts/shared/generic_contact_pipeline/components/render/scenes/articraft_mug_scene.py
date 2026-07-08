#!/usr/bin/env python3
"""Fast mug scene renderer.

Overlay keeps the real Articraft solid mesh projection. Camera3D and side Y-Z
use a compact line model derived from the Articraft mesh bounds, so the views
keep the generated mug proportions without plotting thousands of mesh edges.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import render_mug_articraft_camera3d as m16  # noqa: E402
import render_mug_articraft_camera3d_scene as full  # noqa: E402
import render_pose6d_scene as scene  # noqa: E402


def read_human_outputs(sample: Path, body_model_root: Path, vertex_stride: int) -> tuple[np.ndarray, np.ndarray]:
    human = scene.read_human_result(sample / "results/gvhmr/result.pkl")
    return scene.build_body_outputs(body_model_root, human, vertex_stride=vertex_stride)


def circle_points(cx: float, cy: float, cz: float, radius: float, n: int = 28) -> tuple[list[list[float]], list[tuple[int, int]]]:
    pts = []
    edges = []
    for i in range(n):
        a = 2.0 * math.pi * i / n
        pts.append([cx + radius * math.cos(a), cy, cz + radius * math.sin(a)])
        edges.append((i, (i + 1) % n))
    return pts, edges


def build_articraft_ratio_model(meshes) -> tuple[np.ndarray, list[tuple[int, int, str, float]]]:
    body = meshes.get("body_shell", next(iter(meshes.values())))[0]
    mn = np.nanmin(body, axis=0)
    mx = np.nanmax(body, axis=0)
    cx = float((mn[0] + mx[0]) * 0.5)
    cy0 = float(mn[1])
    cy1 = float(mx[1])
    cz = float((mn[2] + mx[2]) * 0.5)
    radius = float(np.nanpercentile(np.linalg.norm(body[:, [0, 2]] - np.array([[cx, cz]]), axis=1), 92))

    pts: list[list[float]] = []
    edges: list[tuple[int, int, str, float]] = []

    def add_polyline(poly: list[list[float]], color: str, lw: float, closed: bool = False) -> None:
        start = len(pts)
        pts.extend(poly)
        for i in range(len(poly) - 1):
            edges.append((start + i, start + i + 1, color, lw))
        if closed and len(poly) > 2:
            edges.append((start + len(poly) - 1, start, color, lw))

    bottom, bottom_edges = circle_points(cx, cy0, cz, radius)
    top, top_edges = circle_points(cx, cy1, cz, radius)
    start = len(pts)
    pts.extend(bottom + top)
    for a, b in bottom_edges:
        edges.append((start + a, start + b, "#ffd42a", 2.0))
    off = start + len(bottom)
    for a, b in top_edges:
        edges.append((off + a, off + b, "#ff46d1", 2.0))
    for i in range(0, len(bottom), max(1, len(bottom) // 6)):
        edges.append((start + i, off + i, "#d9dee2", 1.4))

    if "handle_loop" in meshes:
        handle = meshes["handle_loop"][0]
        hmn = np.nanmin(handle, axis=0)
        hmx = np.nanmax(handle, axis=0)
        hx = float((hmn[0] + hmx[0]) * 0.5)
        hy = float((hmn[1] + hmx[1]) * 0.5)
        hz = float((hmn[2] + hmx[2]) * 0.5)
        rx = max(0.008, float((hmx[0] - hmn[0]) * 0.5))
        ry = max(0.010, float((hmx[1] - hmn[1]) * 0.5))
        # The generated handle is a loop/tube. A single ellipse is enough for
        # side/camera diagnostics and preserves the Articraft proportions.
        curve = []
        for i in range(32):
            a = 2.0 * math.pi * i / 31
            curve.append([hx + rx * math.cos(a), hy + ry * math.sin(a), hz])
        add_polyline(curve, "#11b5ff", 2.6, closed=False)
        body_side_x = cx + math.copysign(radius * 0.96, hx - cx)
        add_polyline([[body_side_x, hy + ry * 0.65, hz], [hx, hy + ry * 0.65, hz]], "#11b5ff", 2.0)
        add_polyline([[body_side_x, hy - ry * 0.65, hz], [hx, hy - ry * 0.65, hz]], "#11b5ff", 2.0)

    return np.asarray(pts, dtype=float), edges


def transform_proxy(params: np.ndarray, phase: float, local_pts: np.ndarray) -> np.ndarray:
    return m16.transform_mesh(params, local_pts, phase)


def worldlike(points: np.ndarray) -> np.ndarray:
    return m16.cam_to_worldlike(points)


def draw_proxy_3d(ax, pts_cam: np.ndarray, edges: list[tuple[int, int, str, float]], alpha: float = 0.95) -> None:
    pts = worldlike(pts_cam)
    for a, b, color, lw in edges:
        seg = pts[[a, b]]
        if np.all(np.isfinite(seg)):
            ax.plot(seg[:, 0], seg[:, 2], seg[:, 1], color=color, lw=lw, alpha=alpha)


def draw_human_3d(ax, joints: np.ndarray, idx: int) -> None:
    j = worldlike(joints[idx, :55, :])
    body = j[:22]
    for a, b in scene.BODY_EDGES:
        if a < len(body) and b < len(body):
            seg = body[[a, b]]
            ax.plot(seg[:, 0], seg[:, 2], seg[:, 1], color="#6b54d2", lw=2.0, alpha=0.9)
    ax.scatter(body[:, 0], body[:, 2], body[:, 1], s=14, color="#e4dbff", edgecolors="#6b54d2", linewidths=0.3, depthshade=False)
    for hand_edges, color in [(full.LEFT_HAND_EDGES, "#31c978"), (full.RIGHT_HAND_EDGES, "#86d7a6")]:
        ids = sorted({v for e in hand_edges for v in e if v < len(j)})
        for a, b in hand_edges:
            if a < len(j) and b < len(j):
                seg = j[[a, b]]
                ax.plot(seg[:, 0], seg[:, 2], seg[:, 1], color=color, lw=2.0, alpha=0.95)
        if ids:
            hp = j[ids]
            ax.scatter(hp[:, 0], hp[:, 2], hp[:, 1], s=18, color=color, edgecolors="#0b3b22", linewidths=0.3, depthshade=False)


def draw_human_yz(ax, joints: np.ndarray, idx: int) -> None:
    j = worldlike(joints[idx, :55, :])
    body = j[:22]
    for a, b in scene.BODY_EDGES:
        if a < len(body) and b < len(body):
            seg = body[[a, b]]
            ax.plot(seg[:, 2], seg[:, 1], color="#6b54d2", lw=1.8, alpha=0.9)
    for hand_edges, color in [(full.LEFT_HAND_EDGES, "#31c978"), (full.RIGHT_HAND_EDGES, "#86d7a6")]:
        for a, b in hand_edges:
            if a < len(j) and b < len(j):
                seg = j[[a, b]]
                ax.plot(seg[:, 2], seg[:, 1], color=color, lw=2.0, alpha=0.95)


def bounds(points_by_frame: dict[int, np.ndarray], joints: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    chunks = list(points_by_frame.values())
    if joints is not None:
        chunks.append(joints[:, :55, :].reshape(-1, 3))
    all_w = worldlike(np.concatenate(chunks, axis=0))
    mn = np.nanmin(all_w, axis=0)
    mx = np.nanmax(all_w, axis=0)
    margin = np.maximum(0.10, 0.18 * (mx - mn))
    return mn - margin, mx + margin


def make_writer(path: Path, fps: float, size: tuple[int, int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {tmp}")
    return writer, tmp


def render_camera3d(args, out_dir: Path, frames: list[int], pts_by_frame: dict[int, np.ndarray], edges, joints=None) -> dict[str, str]:
    mp4 = out_dir / "camera3d.mp4"
    preview = out_dir / "camera3d_preview.png"
    writer, tmp = make_writer(mp4, args.fps, (args.width, args.height))
    lo, hi = bounds(pts_by_frame, joints)
    preview_written = False
    for idx, fr in enumerate(frames):
        fig = plt.figure(figsize=(args.width / 100, args.height / 100), dpi=100, facecolor="#f7f8fb")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#f7f8fb")
        ax.set_xlim(float(lo[0]), float(hi[0]))
        ax.set_ylim(float(lo[2]), float(hi[2]))
        ax.set_zlim(float(lo[1]), float(hi[1]))
        ax.set_xlabel("X_cam (m)")
        ax.set_ylabel("Z_cam depth (m)")
        ax.set_zlabel("Y up (m)")
        ax.set_title("Mug Articraft-ratio camera 3D" + (" with human" if joints is not None else ""))
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.grid(True, alpha=0.18)
        draw_proxy_3d(ax, pts_by_frame[fr], edges)
        if joints is not None:
            draw_human_3d(ax, joints, min(max(fr - 1, 0), len(joints) - 1))
        fig.text(0.04, 0.045, f"frame {fr:03d}", fontsize=10, color="#333333")
        canvas = m16.figure_to_bgr(fig)
        plt.close(fig)
        writer.write(canvas)
        if not preview_written:
            cv2.imwrite(str(preview), canvas)
            preview_written = True
    writer.release()
    m16.finalize(tmp, mp4)
    return {"camera3d": str(mp4), "camera3d_preview": str(preview)}


def render_side_yz(args, out_dir: Path, frames: list[int], pts_by_frame: dict[int, np.ndarray], edges, joints=None) -> dict[str, str]:
    mp4 = out_dir / "side_yz.mp4"
    preview = out_dir / "side_yz_preview.png"
    writer, tmp = make_writer(mp4, args.fps, (args.width, args.height))
    lo, hi = bounds(pts_by_frame, joints)
    preview_written = False
    for fr in frames:
        fig, ax = plt.subplots(figsize=(args.width / 100, args.height / 100), dpi=100, facecolor="#f7f8fb")
        ax.set_facecolor("#f7f8fb")
        ax.set_xlim(float(lo[2]), float(hi[2]))
        ax.set_ylim(float(lo[1]), float(hi[1]))
        ax.set_xlabel("Z_cam depth (m)")
        ax.set_ylabel("Y up (m)")
        ax.set_title("Mug side Y-Z" + (" with human" if joints is not None else ""))
        ax.grid(True, alpha=0.25)
        pts = worldlike(pts_by_frame[fr])
        for a, b, color, lw in edges:
            seg = pts[[a, b]]
            if np.all(np.isfinite(seg)):
                ax.plot(seg[:, 2], seg[:, 1], color=color, lw=lw, alpha=0.95)
        if joints is not None:
            draw_human_yz(ax, joints, min(max(fr - 1, 0), len(joints) - 1))
        fig.text(0.04, 0.045, f"frame {fr:03d}", fontsize=10, color="#333333")
        canvas = m16.figure_to_bgr(fig)
        plt.close(fig)
        writer.write(canvas)
        if not preview_written:
            cv2.imwrite(str(preview), canvas)
            preview_written = True
    writer.release()
    m16.finalize(tmp, mp4)
    return {"side_yz": str(mp4), "side_yz_preview": str(preview)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, default=Path("samples_known_object/02_mug"))
    ap.add_argument("--mesh-root", type=Path, default=Path("samples_known_object/02_mug/articraft/materialized_mug_mesh"))
    ap.add_argument("--pose-csv", type=Path, required=True)
    ap.add_argument("--phase-csv", type=Path, required=True)
    ap.add_argument("--phase-col", type=str, default="m43_phase_rad")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--body-model-root", type=Path, default=Path("third-party/GVHMR/inputs/checkpoints/body_models"))
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--azim", type=float, default=-62.0)
    ap.add_argument("--body-edge-step", type=int, default=3)
    ap.add_argument("--detail-edge-step", type=int, default=1)
    ap.add_argument("--vertex-stride", type=int, default=16)
    ap.add_argument("--release-frame", type=int, default=150)
    args = ap.parse_args()

    poses = m16.read_pose_sequence(args.pose_csv)
    phases = m16.read_phase_sequence(args.phase_csv, col=args.phase_col)
    meshes = m16.rigid.load_articraft_meshes(args.mesh_root)
    meshes_solid = m16.rigid.load_articraft_meshes_solid(args.mesh_root)
    frames = sorted(set(poses) & set(phases))
    if not frames:
        raise RuntimeError("No overlapping pose/phase frames")

    full_mesh_cam, centers_cam, mesh_bounds = full.prepare_mesh_sequence(poses, phases, meshes, frames)
    K = full.load_K(args.sample_dir)
    local_pts, proxy_edges = build_articraft_ratio_model(meshes)
    proxy_by_frame = {fr: transform_proxy(poses[fr], phases[fr], local_pts) for fr in frames}

    palm_rows = full.read_plugin_csv(args.sample_dir / "results/hand_contact_proxy/mug_palm_proxy.csv")
    gate_rows = full.read_plugin_csv(args.sample_dir / "results/hand_contact_proxy/palm_handle_verification.csv")
    joints, sampled_vertices = read_human_outputs(args.sample_dir, args.body_model_root, args.vertex_stride)

    out = {
        "pose_csv": str(args.pose_csv),
        "phase_csv": str(args.phase_csv),
        "phase_col": args.phase_col,
        "mesh_root": str(args.mesh_root),
        "num_frames": len(frames),
        "policy": "real Articraft mesh overlay and camera3d; Articraft-ratio compact model only for side_yz",
        "object_only": {},
        "with_human": {},
    }
    object_dir = args.out_root / "object_only"
    human_dir = args.out_root / "with_human"
    out["object_only"].update(
        full.render_variant(
            args,
            object_dir,
            frames,
            poses,
            phases,
            meshes,
            full_mesh_cam,
            centers_cam,
            mesh_bounds,
            meshes_solid=meshes_solid,
            with_human=False,
        )
    )
    out["object_only"].update(full.render_overlay_variant(args, object_dir, frames, meshes_solid, full_mesh_cam, K, with_human=False))
    out["object_only"].update(render_side_yz(args, object_dir, frames, proxy_by_frame, proxy_edges, joints=None))
    out["with_human"].update(
        full.render_variant(
            args,
            human_dir,
            frames,
            poses,
            phases,
            meshes,
            full_mesh_cam,
            centers_cam,
            mesh_bounds,
            meshes_solid=meshes_solid,
            with_human=True,
            joints=joints,
            sampled_vertices=sampled_vertices,
            palm_rows=palm_rows,
            gate_rows=gate_rows,
        )
    )
    out["with_human"].update(full.render_overlay_variant(args, human_dir, frames, meshes_solid, full_mesh_cam, K, with_human=True, joints=joints, palm_rows=palm_rows, gate_rows=gate_rows))
    out["with_human"].update(render_side_yz(args, human_dir, frames, proxy_by_frame, proxy_edges, joints=joints))
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "outputs.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
