#!/usr/bin/env python3
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
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import render_chair_articraft_visual_model as vismod  # noqa: E402
import render_chair_articraft_3d_scene as scene3d  # noqa: E402

WOOD = "#b8793e"
WOOD_LIGHT = "#cf9553"
EDGE = "#4b2c14"
HUMAN = "#5242b8"
HUMAN_VERT = "#7768d8"
HAND = "#18b866"
CONTACT = "#00bcd4"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows


def finalize_video(tmp: Path, out: Path) -> None:
    ffmpeg = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    cmds = [
        [ffmpeg, "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
        [ffmpeg, "-y", "-i", str(tmp), "-c:v", "libopenh264", "-pix_fmt", "yuv420p", str(out)],
        ["/usr/bin/ffmpeg", "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
    ]
    last = ""
    for cmd in cmds:
        if "/" in cmd[0] and not Path(cmd[0]).exists():
            continue
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            tmp.unlink(missing_ok=True)
            return
        except subprocess.CalledProcessError as exc:
            last = exc.stderr[-2000:]
    raise RuntimeError(f"ffmpeg finalize failed for {tmp} -> {out}\n{last}")


def make_writer(mp4: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path]:
    mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp = mp4.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {tmp}")
    return writer, tmp


def fig_to_bgr(fig: plt.Figure) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def cam_to_worldlike(points: np.ndarray) -> np.ndarray:
    out = np.asarray(points, dtype=float).copy()
    out[..., 1] *= -1.0
    return out


def mpl_points(cam: np.ndarray) -> np.ndarray:
    w = cam_to_worldlike(cam)
    return w[:, [0, 2, 1]]


def visual_meshes_for_row(row: dict[str, str], visuals: list[dict[str, object]]) -> list[dict[str, object]]:
    return vismod.visual_cam_meshes(scene3d.pose_params(row), visuals)


def sample_bounds(rows: list[dict[str, str]], visuals: list[dict[str, object]], joints: np.ndarray | None, verts: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    all_pts: list[np.ndarray] = []
    step = max(1, len(rows) // 32)
    for row in rows[::step]:
        for mesh in visual_meshes_for_row(row, visuals):
            cam = np.asarray(mesh["cam"], dtype=float)
            all_pts.append(cam[:: max(1, len(cam) // 80)])
    if joints is not None:
        all_pts.append(joints[:, :55, :].reshape(-1, 3)[:: max(1, joints[:, :55, :].size // 9000)])
    if verts is not None:
        flat = verts.reshape(-1, 3)
        all_pts.append(flat[:: max(1, len(flat) // 5000)])
    world = cam_to_worldlike(np.concatenate(all_pts, axis=0))
    lo = np.nanmin(world, axis=0)
    hi = np.nanmax(world, axis=0)
    margin = np.maximum(0.16, 0.16 * (hi - lo))
    return lo - margin, hi + margin


def add_solid_chair(ax, row: dict[str, str], visuals: list[dict[str, object]], max_faces: int) -> np.ndarray:
    centers = []
    for mesh in visual_meshes_for_row(row, visuals):
        cam = np.asarray(mesh["cam"], dtype=float)
        faces = np.asarray(mesh["faces"], dtype=np.int32)
        if len(cam) == 0 or len(faces) == 0:
            continue
        centers.append(np.mean(cam, axis=0))
        face_step = max(1, int(math.ceil(len(faces) / max_faces)))
        tris = mpl_points(cam)[faces[::face_step]]
        name = str(mesh.get("name", ""))
        face = WOOD_LIGHT if ("backrest" in name or "seat" in name or "slat" in name) else WOOD
        coll = Poly3DCollection(tris, facecolor=face, edgecolor=EDGE, linewidths=0.18, alpha=0.98)
        ax.add_collection3d(coll)
    return np.mean(np.vstack(centers), axis=0) if centers else np.zeros(3)


def add_human(ax, joints_frame: np.ndarray, verts_frame: np.ndarray | None) -> None:
    if verts_frame is not None:
        vw = mpl_points(verts_frame)
        ax.scatter(vw[:, 0], vw[:, 1], vw[:, 2], s=5.0, color=HUMAN_VERT, alpha=0.34, linewidths=0, depthshade=False)
    jw = mpl_points(joints_frame[:55])
    for a, b in scene3d.BODY_EDGES:
        if a < len(jw) and b < len(jw):
            seg = jw[[a, b]]
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=HUMAN, lw=2.8, alpha=0.98)
    hand_edges = [
        (20, 25), (25, 26), (26, 27), (20, 28), (28, 29), (29, 30), (20, 31), (31, 32), (32, 33), (20, 34), (34, 35), (35, 36),
        (21, 40), (40, 41), (41, 42), (21, 43), (43, 44), (44, 45), (21, 46), (46, 47), (47, 48), (21, 49), (49, 50), (50, 51),
    ]
    for a, b in hand_edges:
        if a < len(jw) and b < len(jw):
            seg = jw[[a, b]]
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=HAND, lw=3.0, alpha=1.0)
    ax.scatter(jw[:22, 0], jw[:22, 1], jw[:22, 2], s=20, color="#eee8ff", edgecolors=HUMAN, linewidths=0.5, depthshade=False)


def add_contacts(ax, row: dict[str, str], contacts: list[dict[str, str]], joints_frame: np.ndarray | None) -> None:
    if joints_frame is None or not contacts:
        return
    params = scene3d.pose_params(row)
    entries = scene3d.contact_gap_entries(params, contacts, joints_frame)
    for contact_row, (side, anchor, hand, gap) in zip(contacts, entries):
        aw = mpl_points(anchor[None, :])[0]
        hw = mpl_points(hand[None, :])[0]
        mid = (aw + hw) * 0.5
        side_label = "R palm" if side == "right_hand" else "L palm" if side == "left_hand" else side.replace("_", " ")
        anchor_label = contact_row.get("chair_local_point_id") or contact_row.get("chair_local_role") or contact_row.get("chair_local_part") or "chair contact"
        ax.plot([aw[0], hw[0]], [aw[1], hw[1]], [aw[2], hw[2]], color="#111111", lw=1.6, ls="--", alpha=0.85)
        ax.scatter([aw[0]], [aw[1]], [aw[2]], s=70, color=CONTACT, edgecolors="#111", linewidths=0.5, depthshade=False)
        ax.scatter([hw[0]], [hw[1]], [hw[2]], s=70, color=HAND, edgecolors="#111", linewidths=0.5, depthshade=False)
        ax.text(aw[0], aw[1], aw[2], f"  {anchor_label}", color="#005a68", fontsize=7)
        ax.text(hw[0], hw[1], hw[2], f"  {side_label}", color="#087a35", fontsize=7)
        ax.text(mid[0], mid[1], mid[2], f"{gap:.2f}m", color="#111111", fontsize=7)


def render_camera3d(
    rows: list[dict[str, str]],
    visuals: list[dict[str, object]],
    contacts: dict[int, list[dict[str, str]]],
    out_dir: Path,
    args: argparse.Namespace,
    joints: np.ndarray | None = None,
    verts: np.ndarray | None = None,
    *,
    name: str = "camera3d_solid",
    title_suffix: str = "",
    elev: float | None = None,
    azim: float | None = None,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / f"{name}.mp4"
    preview = out_dir / f"{name}_preview.png"
    writer, tmp = make_writer(mp4, args.fps, (args.width, args.height))
    lo, hi = sample_bounds(rows, visuals, joints, verts)
    centers = []
    preview_written = False
    view_elev = args.elev if elev is None else elev
    view_azim = args.azim if azim is None else azim
    for row in rows:
        fr = int(row["frame"])
        fig = plt.figure(figsize=(args.width / 100, args.height / 100), dpi=100, facecolor="#f4f6fa")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#f4f6fa")
        ax.set_xlim(float(lo[0]), float(hi[0]))
        ax.set_ylim(float(lo[2]), float(hi[2]))
        ax.set_zlim(float(lo[1]), float(hi[1]))
        ax.set_xlabel("X_cam (m)")
        ax.set_ylabel("Z_cam depth (m)")
        ax.set_zlabel("Y up (m)")
        ax.set_title("Solid Articraft chair camera 3D" + (" with human" if joints is not None else " object only") + title_suffix)
        ax.view_init(elev=view_elev, azim=view_azim)
        ax.grid(True, alpha=0.22)
        centers.append(add_solid_chair(ax, row, visuals, args.max_faces_per_visual))
        if len(centers) > 1:
            c = mpl_points(np.vstack(centers))
            ax.plot(c[:, 0], c[:, 1], c[:, 2], color="#5f6f86", lw=1.4, alpha=0.45)
        if joints is not None:
            jidx = min(max(fr - 1, 0), len(joints) - 1)
            add_human(ax, joints[jidx], None if verts is None else verts[jidx])
            add_contacts(ax, row, contacts.get(fr, []), joints[jidx])
        fig.text(0.04, 0.045, f"frame {fr:03d}", fontsize=10, color="#222")
        canvas = fig_to_bgr(fig)
        plt.close(fig)
        writer.write(canvas)
        if not preview_written:
            cv2.imwrite(str(preview), canvas)
            preview_written = True
    writer.release()
    finalize_video(tmp, mp4)
    return {name: str(mp4), f"{name}_preview": str(preview)}



def face_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    edges = set()
    for face in np.asarray(faces, dtype=np.int32):
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    return list(edges)


def render_side_yz(rows: list[dict[str, str]], visuals: list[dict[str, object]], contacts: dict[int, list[dict[str, str]]], out_dir: Path, args: argparse.Namespace, joints: np.ndarray | None = None) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / "side_yz_solid.mp4"
    preview = out_dir / "side_yz_solid_preview.png"
    writer, tmp = make_writer(mp4, args.fps, (args.width, args.height))
    lo, hi = sample_bounds(rows, visuals, joints, None)
    full_edges = [face_edges(np.asarray(v["faces"], dtype=np.int32)) for v in visuals] if args.full_side_yz else None
    preview_written = False
    for row in rows:
        fr = int(row["frame"])
        fig, ax = plt.subplots(figsize=(args.width / 100, args.height / 100), dpi=100, facecolor="#f4f6fa")
        ax.set_facecolor("#f4f6fa")
        ax.set_xlim(float(lo[2]), float(hi[2]))
        ax.set_ylim(float(lo[1]), float(hi[1]))
        ax.set_xlabel("Z_cam depth (m)")
        ax.set_ylabel("Y up (m)")
        ax.set_title("Solid Articraft chair side Y-Z" + (" with human" if joints is not None else " object only"))
        ax.grid(True, alpha=0.24)
        for mesh_idx, mesh in enumerate(visual_meshes_for_row(row, visuals)):
            cam = np.asarray(mesh["cam"], dtype=float)
            pts = cam_to_worldlike(cam)
            if args.full_side_yz:
                edges = full_edges[mesh_idx] if full_edges is not None else []
                for a, b in edges:
                    seg = pts[[a, b]]
                    ax.plot(seg[:, 2], seg[:, 1], color=EDGE, lw=0.42, alpha=0.55)
                sample = pts[:: max(1, len(pts) // 420)]
                ax.scatter(sample[:, 2], sample[:, 1], s=4.5, color=WOOD, alpha=0.72)
            else:
                sample = pts[:: max(1, len(pts) // 260)]
                ax.scatter(sample[:, 2], sample[:, 1], s=12, color=WOOD, alpha=0.90)
                if len(sample) > 2:
                    zq = np.quantile(sample[:, 2], [0.02, 0.98])
                    yq = np.quantile(sample[:, 1], [0.02, 0.98])
                    ax.plot([zq[0], zq[1], zq[1], zq[0], zq[0]], [yq[0], yq[0], yq[1], yq[1], yq[0]], color=EDGE, lw=0.8, alpha=0.30)
        if joints is not None:
            jidx = min(max(fr - 1, 0), len(joints) - 1)
            jw = cam_to_worldlike(joints[jidx, :55])
            for a, b in scene3d.BODY_EDGES:
                if a < len(jw) and b < len(jw):
                    seg = jw[[a, b]]
                    ax.plot(seg[:, 2], seg[:, 1], color=HUMAN, lw=2.4, alpha=0.95)
            for _side, anchor, hand, _gap in scene3d.contact_gap_entries(scene3d.pose_params(row), contacts.get(fr, []), joints[jidx]):
                aw = cam_to_worldlike(anchor[None, :])[0]
                hw = cam_to_worldlike(hand[None, :])[0]
                ax.plot([aw[2], hw[2]], [aw[1], hw[1]], color="#111", lw=1.4, ls="--", alpha=0.85)
                ax.scatter([aw[2]], [aw[1]], s=70, color=CONTACT, edgecolors="#111", linewidths=0.4)
                ax.scatter([hw[2]], [hw[1]], s=70, color=HAND, edgecolors="#111", linewidths=0.4)
        fig.text(0.04, 0.045, f"frame {fr:03d}", fontsize=10, color="#222")
        canvas = fig_to_bgr(fig)
        plt.close(fig)
        writer.write(canvas)
        if not preview_written:
            cv2.imwrite(str(preview), canvas)
            preview_written = True
    writer.release()
    finalize_video(tmp, mp4)
    return {"side_yz_solid": str(mp4), "side_yz_solid_preview": str(preview)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Render solid Articraft chair 3D scene.")
    ap.add_argument("--sample-dir", type=Path, default=Path("samples_known_object/05_chair"))
    ap.add_argument("--pose-csv", type=Path, required=True)
    ap.add_argument("--contacts-csv", type=Path, default=None)
    ap.add_argument("--urdf", type=Path, default=Path("samples_known_object/05_chair/articraft/generated_record/rec_fork-6911e934-oldtopology-rearup-stretcherlevel-pass_20260619_1056/model.urdf"))
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--body-model-root", type=Path, default=Path("third-party/GVHMR/inputs/checkpoints/body_models"))
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--azim", type=float, default=-62.0)
    ap.add_argument("--vertex-stride", type=int, default=18)
    ap.add_argument("--max-faces-per-visual", type=int, default=420)
    ap.add_argument("--alpha", type=float, default=0.92)
    ap.add_argument("--semantic-aligned", action="store_true")
    ap.add_argument("--full-side-yz", action="store_true", help="Draw full mesh face edges in side Y-Z instead of simplified sampled silhouette.")
    ap.add_argument("--video-facing-diagnostic", action="store_true", help="Also render the extra video-facing 3D camera diagnostic.")
    args = ap.parse_args()

    sample = args.sample_dir
    pose_rows = read_rows(args.pose_csv)
    contacts_csv = args.contacts_csv or args.pose_csv.parent / "object_contact_points.csv"
    contacts = scene3d.load_contacts(contacts_csv)
    visuals = vismod.load_articraft_visuals(args.urdf)
    if args.semantic_aligned:
        visuals = vismod.add_semantic_aligned_leg_visuals(sample, visuals)
    human = vismod.read_human_result(sample / "results/gvhmr/result.pkl")
    K_all = np.asarray(human["K_fullimg"], dtype=float)
    joints_overlay = vismod.build_body_joints(args.body_model_root, human)
    full_human = scene3d.read_human_result(sample / "results/gvhmr/result.pkl")
    body_joints, body_verts = scene3d.build_body_outputs(args.body_model_root, full_human, args.vertex_stride)

    object_dir = args.out_root / "object_only"
    human_dir = args.out_root / "with_human"
    outputs = {
        "pose_csv": str(args.pose_csv),
        "contacts_csv": str(contacts_csv),
        "urdf": str(args.urdf),
        "num_visuals": len(visuals),
        "object_only": {},
        "with_human": {},
    }
    outputs["object_only"].update(render_camera3d(pose_rows, visuals, contacts, object_dir, args))
    outputs["object_only"].update(render_side_yz(pose_rows, visuals, contacts, object_dir, args))
    oprev, omp4 = vismod.render_overlay(sample, pose_rows, visuals, contacts, K_all, object_dir, "overlay_solid.mp4", args.fps, args.alpha, True, joints=None)
    outputs["object_only"].update({"overlay_solid": str(omp4), "overlay_solid_preview": str(oprev)})

    outputs["with_human"].update(render_camera3d(pose_rows, visuals, contacts, human_dir, args, joints=body_joints, verts=body_verts))
    if args.video_facing_diagnostic:
        outputs["with_human"].update(
            render_camera3d(
                pose_rows,
                visuals,
                contacts,
                human_dir,
                args,
                joints=body_joints,
                verts=body_verts,
                name="camera3d_video_facing_solid",
                title_suffix=" video-facing diagnostic",
                elev=10.0,
                azim=-82.0,
            )
        )
    outputs["with_human"].update(render_side_yz(pose_rows, visuals, contacts, human_dir, args, joints=body_joints))
    hprev, hmp4 = vismod.render_overlay(sample, pose_rows, visuals, contacts, K_all, human_dir, "contact_overlay_solid.mp4", args.fps, args.alpha, True, joints=joints_overlay)
    outputs["with_human"].update({"contact_overlay_solid": str(hmp4), "contact_overlay_solid_preview": str(hprev)})

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "outputs.json").write_text(json.dumps(outputs, indent=2) + chr(10))
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
