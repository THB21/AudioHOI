#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation


BODY_EDGES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
    (6, 9),
    (7, 10),
    (8, 11),
    (9, 12),
    (12, 13),
    (12, 14),
    (12, 15),
    (13, 16),
    (14, 17),
    (16, 18),
    (17, 19),
    (18, 20),
    (19, 21),
]

LEFT_HAND_EDGES = [
    (20, 25),
    (25, 26),
    (26, 27),
    (20, 28),
    (28, 29),
    (29, 30),
    (20, 31),
    (31, 32),
    (32, 33),
    (20, 34),
    (34, 35),
    (35, 36),
    (20, 37),
    (37, 38),
    (38, 39),
]

RIGHT_HAND_EDGES = [
    (21, 40),
    (40, 41),
    (41, 42),
    (21, 43),
    (43, 44),
    (44, 45),
    (21, 46),
    (46, 47),
    (47, 48),
    (21, 49),
    (49, 50),
    (50, 51),
    (21, 52),
    (52, 53),
    (53, 54),
]

LEFT_PALM_IDS = [20, 25, 28, 31, 34]
RIGHT_PALM_IDS = [21, 40, 43, 46, 49]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows


def frame_path(sample: Path, frame: int) -> Path:
    for name in (f"{frame:05d}.png", f"{frame:06d}.png", f"{frame:04d}.png", f"{frame:03d}.png"):
        p = sample / "frames" / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No extracted frame {frame} under {sample / 'frames'}")


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value in {"", None} else float(value)
    except Exception:
        return default


def parse_vec(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=float)
    return np.asarray([float(x) for x in text.strip().split()], dtype=float)


def origin_matrix(origin_el: ET.Element | None) -> np.ndarray:
    xyz = parse_vec(origin_el.get("xyz") if origin_el is not None else None, (0.0, 0.0, 0.0))
    rpy = parse_vec(origin_el.get("rpy") if origin_el is not None else None, (0.0, 0.0, 0.0))
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    mat[:3, 3] = xyz
    return mat


def transform_vertices(vertices: np.ndarray, mat: np.ndarray) -> np.ndarray:
    homog = np.column_stack([vertices, np.ones(len(vertices), dtype=float)])
    return (homog @ mat.T)[:, :3]


def visual_color_bgr(visual_el: ET.Element) -> tuple[int, int, int]:
    color_el = visual_el.find("material/color")
    if color_el is None:
        return 60, 140, 210
    rgba = parse_vec(color_el.get("rgba"), (0.24, 0.55, 0.82, 1.0))
    rgb = np.clip(rgba[:3] * 255.0, 0, 255).astype(int)
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def load_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load_mesh(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=np.int32)


def primitive_mesh(geom_el: ET.Element, root: Path) -> tuple[np.ndarray, np.ndarray]:
    box = geom_el.find("box")
    if box is not None:
        size = parse_vec(box.get("size"), (1.0, 1.0, 1.0))
        mesh = trimesh.creation.box(extents=size)
        return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=np.int32)
    cyl = geom_el.find("cylinder")
    if cyl is not None:
        radius = float(cyl.get("radius", "0.01"))
        length = float(cyl.get("length", "0.1"))
        mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=24)
        return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=np.int32)
    sphere = geom_el.find("sphere")
    if sphere is not None:
        radius = float(sphere.get("radius", "0.01"))
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=np.int32)
    mesh_el = geom_el.find("mesh")
    if mesh_el is not None:
        filename = mesh_el.get("filename")
        if not filename:
            raise RuntimeError("URDF mesh visual without filename")
        mesh_path = Path(filename)
        if not mesh_path.is_absolute():
            mesh_path = root / mesh_path
        return load_obj_mesh(mesh_path)
    raise RuntimeError(f"Unsupported visual geometry: {ET.tostring(geom_el, encoding='unicode')}")


def _fixed_joint_values(descriptor_path: Path | None) -> dict[str, float]:
    if descriptor_path is None:
        return {}
    descriptor = json.loads(descriptor_path.read_text())
    values = {
        str(name): float(value)
        for name, value in descriptor.get("fixed_assembly_joints", {}).items()
    }
    for dof in descriptor.get("state_contract", {}).get("dofs", ()):
        joint_name = dof.get("bounds_from_urdf")
        default = dof.get("default")
        if joint_name is not None and default is not None:
            values[str(joint_name)] = float(default)
    return values


def _root_from_link_transforms(
    robot: ET.Element,
    fixed_joint_values: dict[str, float],
) -> dict[str, np.ndarray]:
    by_child: dict[str, tuple[str, np.ndarray]] = {}
    for joint in robot.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_link = str(parent.get("link", ""))
        child_link = str(child.get("link", ""))
        transform = origin_matrix(joint.find("origin"))
        angle = float(fixed_joint_values.get(str(joint.get("name", "")), 0.0))
        axis_node = joint.find("axis")
        axis = parse_vec(axis_node.get("xyz") if axis_node is not None else None, (1.0, 0.0, 0.0))
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12:
            raise ValueError(f"URDF joint {joint.get('name', '')} has a zero axis")
        rotation = np.eye(4, dtype=float)
        rotation[:3, :3] = Rotation.from_rotvec(axis / norm * angle).as_matrix()
        by_child[child_link] = (parent_link, transform @ rotation)

    cache: dict[str, np.ndarray] = {}

    def resolve(link: str) -> np.ndarray:
        if link in cache:
            return cache[link]
        parent_record = by_child.get(link)
        if parent_record is None:
            result = np.eye(4, dtype=float)
        else:
            parent_link, parent_from_child = parent_record
            result = resolve(parent_link) @ parent_from_child
        cache[link] = result
        return result

    for link in robot.findall("link"):
        resolve(str(link.get("name", "")))
    return cache


def load_articraft_visuals(
    urdf_path: Path,
    descriptor_path: Path | None = None,
) -> list[dict[str, object]]:
    root_dir = urdf_path.parent
    robot = ET.parse(urdf_path).getroot()
    root_from_link = _root_from_link_transforms(robot, _fixed_joint_values(descriptor_path))
    visuals: list[dict[str, object]] = []
    for link in robot.findall("link"):
        link_name = link.get("name", "")
        for visual in link.findall("visual"):
            geom = visual.find("geometry")
            if geom is None:
                continue
            verts, faces = primitive_mesh(geom, root_dir)
            verts = transform_vertices(
                verts,
                root_from_link[link_name] @ origin_matrix(visual.find("origin")),
            )
            visuals.append(
                {
                    "link": link_name,
                    "name": visual.get("name", ""),
                    "vertices": verts,
                    "faces": faces,
                    "color": visual_color_bgr(visual),
                }
            )
    if not visuals:
        raise RuntimeError(f"No URDF visuals found in {urdf_path}")
    return visuals


def pose_transform(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    t = np.array([f(row, "tx"), f(row, "ty"), f(row, "tz", 1.0)], dtype=float)
    quat_xyzw = np.array([f(row, "qx"), f(row, "qy"), f(row, "qz"), f(row, "qw", 1.0)], dtype=float)
    if np.linalg.norm(quat_xyzw) < 1e-8:
        quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw)
    return Rotation.from_quat(quat_xyzw).as_matrix(), t


def visual_meshes(row: dict[str, str], visuals: list[dict[str, object]]) -> list[dict[str, object]]:
    rot, t = pose_transform(row)
    meshes = []
    for visual in visuals:
        vertices = np.asarray(visual["vertices"], dtype=float)
        cam = vertices @ rot.T + t
        meshes.append({**visual, "cam": cam})
    return meshes


def project(points: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points[:, 2]
    valid = z > 1e-6
    zz = np.clip(z, 1e-6, None)
    uv = np.column_stack([K[0, 0] * points[:, 0] / zz + K[0, 2], K[1, 1] * points[:, 1] / zz + K[1, 2]])
    return uv, valid


def draw_human_projection_2d(img: np.ndarray, joints_frame: np.ndarray, K: np.ndarray) -> None:
    """Reuse the chair/mug overlay convention: draw skeleton and hands over video."""
    body_uv, body_valid = project(joints_frame[:22], K)
    for a, b in BODY_EDGES:
        if a < len(body_valid) and b < len(body_valid) and body_valid[a] and body_valid[b]:
            cv2.line(
                img,
                tuple(np.round(body_uv[a]).astype(int)),
                tuple(np.round(body_uv[b]).astype(int)),
                (210, 90, 255),
                2,
                cv2.LINE_AA,
            )
    for uv, valid in zip(body_uv, body_valid):
        if valid:
            cv2.circle(img, tuple(np.round(uv).astype(int)), 4, (245, 230, 255), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(np.round(uv).astype(int)), 4, (110, 80, 210), 1, cv2.LINE_AA)

    for edges, color in ((LEFT_HAND_EDGES, (80, 245, 130)), (RIGHT_HAND_EDGES, (130, 220, 255))):
        hand_ids = sorted({v for edge in edges for v in edge if v < len(joints_frame)})
        if not hand_ids:
            continue
        hand_uv, hand_valid = project(joints_frame[hand_ids], K)
        uv_by_id = {jid: hand_uv[i] for i, jid in enumerate(hand_ids) if hand_valid[i]}
        for a, b in edges:
            if a in uv_by_id and b in uv_by_id:
                cv2.line(
                    img,
                    tuple(np.round(uv_by_id[a]).astype(int)),
                    tuple(np.round(uv_by_id[b]).astype(int)),
                    color,
                    2,
                    cv2.LINE_AA,
                )
        for uv, valid in zip(hand_uv, hand_valid):
            if valid:
                cv2.circle(img, tuple(np.round(uv).astype(int)), 3, color, -1, cv2.LINE_AA)

    for ids, color in ((LEFT_PALM_IDS, (70, 180, 70)), (RIGHT_PALM_IDS, (230, 200, 70))):
        if max(ids) >= len(joints_frame):
            continue
        palm = joints_frame[ids].mean(axis=0, keepdims=True)
        uv, valid = project(palm, K)
        if valid[0]:
            p = tuple(np.round(uv[0]).astype(int))
            cv2.circle(img, p, 11, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(img, p, 8, color, -1, cv2.LINE_AA)


def read_human_result(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        data = pickle.load(f)
    params = data.get("smpl_params_incam")
    if not isinstance(params, dict):
        return None
    return {
        "body_pose": np.asarray(params["body_pose"], dtype=np.float32),
        "betas": np.asarray(params["betas"], dtype=np.float32),
        "global_orient": np.asarray(params["global_orient"], dtype=np.float32),
        "transl": np.asarray(params["transl"], dtype=np.float32),
    }


def build_body_geometry(body_models_root: Path, human_params: dict[str, np.ndarray] | None) -> dict[str, np.ndarray] | None:
    if human_params is None:
        return None
    import smplx
    import torch

    model = smplx.create(
        str(body_models_root),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        batch_size=human_params["transl"].shape[0],
    )
    with torch.inference_mode():
        output = model(
            body_pose=torch.from_numpy(human_params["body_pose"]),
            betas=torch.from_numpy(human_params["betas"]),
            global_orient=torch.from_numpy(human_params["global_orient"]),
            transl=torch.from_numpy(human_params["transl"]),
            return_verts=True,
        )
    return {
        "joints": output.joints.detach().cpu().numpy().astype(np.float64),
        "vertices": output.vertices.detach().cpu().numpy().astype(np.float64),
        "faces": np.asarray(model.faces, dtype=np.int32),
    }


def human_occlusion_mask(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dark_clothing = (gray < 76).astype(np.uint8) * 255
    hair = ((gray < 95) & (hsv[:, :, 1] > 20)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(dark_clothing, hair)
    h, _w = gray.shape
    mask[int(h * 0.86) :, :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
    return mask


def visible_object_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color = cv2.inRange(hsv, (8, 45, 42), (38, 235, 245))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 170)
    mask = cv2.bitwise_and(color, cv2.dilate(edges, np.ones((5, 5), np.uint8)))
    h, w = mask.shape
    mask[int(h * 0.88) :, :] = 0
    mask[:, :5] = 0
    mask[:, w - 5 :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.dilate(mask, np.ones((17, 17), np.uint8), iterations=1)


def read_visible_line_tracks(sample: Path) -> dict[int, tuple[float, float, float, float]]:
    path = sample / "results" / "tracking" / "object_mesh_tracks_test.csv"
    if not path.exists():
        return {}
    grouped: dict[int, dict[str, tuple[float, float]]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            pid = row.get("point_id", "")
            if pid not in {"tip_a", "tip_b"}:
                continue
            try:
                fr = int(float(row.get("frame", "0") or 0))
                grouped.setdefault(fr, {})[pid] = (float(row["x"]), float(row["y"]))
            except Exception:
                continue
    tracks = {}
    for fr, pts in grouped.items():
        if "tip_a" in pts and "tip_b" in pts:
            ax, ay = pts["tip_a"]
            bx, by = pts["tip_b"]
            tracks[fr] = (ax, ay, bx, by)
    return tracks


def visible_line_mask(shape: tuple[int, int], track: tuple[float, float, float, float] | None) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if track is None:
        return mask
    x1, y1, x2, y2 = track
    length = float(np.hypot(x2 - x1, y2 - y1))
    if length < 25.0:
        return mask
    thickness = int(max(13, min(31, round(length * 0.055))))
    cv2.line(mask, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), 255, thickness, cv2.LINE_AA)
    return cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)


def write_quality_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finalize(tmp: Path, out: Path) -> None:
    ffmpeg = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    subprocess.run([ffmpeg, "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True, capture_output=True, text=True)
    tmp.unlink(missing_ok=True)


def writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.mp4")
    w = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not w.isOpened():
        raise RuntimeError(f"Could not open video writer for {tmp}")
    return w, tmp


def draw_overlay(
    sample: Path,
    rows: list[dict[str, str]],
    visuals: list[dict[str, object]],
    out: Path,
    fps: float,
    K: np.ndarray,
    alpha: float,
    *,
    respect_human_occlusion: bool = False,
    clip_to_visible_object: bool = False,
    line_clip_tracks: dict[int, tuple[float, float, float, float]] | None = None,
    human_joints: np.ndarray | None = None,
) -> dict[str, str]:
    first = cv2.imread(str(frame_path(sample, int(float(rows[0]["frame"])))))
    h, w = first.shape[:2]
    wr, tmp = writer(out, fps, (w, h))
    preview = out.with_name(out.stem + "_preview.png")
    quality_csv = out.with_name(out.stem + "_quality.csv")
    preview_written = False
    quality_rows: list[dict[str, object]] = []
    for row in rows:
        fr = int(float(row["frame"]))
        img = cv2.imread(str(frame_path(sample, fr)), cv2.IMREAD_COLOR)
        overlay = img.copy()
        render_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for mesh in visual_meshes(row, visuals):
            uv, valid = project(np.asarray(mesh["cam"], dtype=float), K)
            faces = np.asarray(mesh["faces"], dtype=np.int32)
            color = tuple(int(c) for c in mesh.get("color", (60, 140, 210)))
            for face in faces[:: max(1, len(faces) // 180)]:
                if not np.all(valid[face]):
                    continue
                pts = np.round(uv[face]).astype(np.int32)
                if np.any(pts[:, 0] < -50) or np.any(pts[:, 0] > w + 50) or np.any(pts[:, 1] < -50) or np.any(pts[:, 1] > h + 50):
                    continue
                cv2.fillConvexPoly(overlay, pts, color)
                cv2.fillConvexPoly(render_mask, pts, 255)
                cv2.polylines(overlay, [pts], True, (20, 40, 60), 1, cv2.LINE_AA)
        img_base = img
        object_mask = None
        line_track = None
        if clip_to_visible_object:
            line_track = (line_clip_tracks or {}).get(fr)
            track_mask = visible_line_mask(img_base.shape[:2], line_track)
            color_mask = visible_object_mask(img_base)
            object_mask = cv2.bitwise_and(color_mask, cv2.dilate(track_mask, np.ones((13, 13), np.uint8), iterations=1))
            if int(np.count_nonzero(object_mask)) < 80:
                object_mask = track_mask if int(np.count_nonzero(track_mask)) else color_mask
        used_visible_line_fallback = 0
        if object_mask is not None:
            pre_area = int(np.count_nonzero(render_mask))
            pre_overlap = int(np.count_nonzero(cv2.bitwise_and(render_mask, object_mask))) if pre_area else 0
            if pre_overlap < 80 and line_track is not None:
                x1, y1, x2, y2 = line_track
                line_len = float(np.hypot(x2 - x1, y2 - y1))
                if line_len >= 25.0:
                    thickness = int(max(8, min(18, round(line_len * 0.035))))
                    p1 = (int(round(x1)), int(round(y1)))
                    p2 = (int(round(x2)), int(round(y2)))
                    cv2.line(overlay, p1, p2, (58, 112, 188), thickness, cv2.LINE_AA)
                    cv2.line(render_mask, p1, p2, 255, thickness, cv2.LINE_AA)
                    used_visible_line_fallback = 1
        raw_render_area = int(np.count_nonzero(render_mask))
        raw_overlap = int(np.count_nonzero(cv2.bitwise_and(render_mask, object_mask))) if object_mask is not None else 0
        img = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)
        if object_mask is not None:
            img[object_mask == 0] = img_base[object_mask == 0]
        if respect_human_occlusion:
            human_mask = human_occlusion_mask(img_base)
            img[human_mask > 0] = img_base[human_mask > 0]
        if human_joints is not None:
            jidx = min(max(fr - 1, 0), len(human_joints) - 1)
            draw_human_projection_2d(img, human_joints[jidx], K)
        final_changed = np.any(np.abs(img.astype(np.int16) - img_base.astype(np.int16)) > 5, axis=2).astype(np.uint8) * 255
        final_area = int(np.count_nonzero(final_changed))
        final_overlap = int(np.count_nonzero(cv2.bitwise_and(final_changed, object_mask))) if object_mask is not None else 0
        quality_rows.append(
            {
                "frame": fr,
                "raw_render_area_px": raw_render_area,
                "visible_object_area_px": int(np.count_nonzero(object_mask)) if object_mask is not None else "",
                "raw_overlap_px": raw_overlap if object_mask is not None else "",
                "raw_overlap_precision": f"{(raw_overlap / raw_render_area):.6f}" if object_mask is not None and raw_render_area else "",
                "final_changed_area_px": final_area,
                "final_overlap_px": final_overlap if object_mask is not None else "",
                "final_overlap_precision": f"{(final_overlap / final_area):.6f}" if object_mask is not None and final_area else "",
                "clip_to_visible_object": int(clip_to_visible_object),
                "respect_human_occlusion": int(respect_human_occlusion),
                "used_visible_line_fallback": used_visible_line_fallback,
            }
        )
        cv2.putText(img, f"frame {fr:03d}", (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, f"frame {fr:03d}", (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        wr.write(img)
        if not preview_written:
            cv2.imwrite(str(preview), img)
            preview_written = True
    wr.release()
    finalize(tmp, out)
    write_quality_csv(quality_csv, quality_rows)
    return {out.stem: str(out), preview.stem: str(preview), quality_csv.stem: str(quality_csv)}


def fig_bgr(fig: plt.Figure) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def bounds(rows: list[dict[str, str]], visuals: list[dict[str, object]], body: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    pts = []
    step = max(1, len(rows) // 40)
    for row in rows[::step]:
        for mesh in visual_meshes(row, visuals):
            cam = np.asarray(mesh["cam"], dtype=float)
            pts.append(cam[:: max(1, len(cam) // 50)])
    if body is not None:
        joints = body.get("joints")
        if joints is not None and len(joints):
            flat = joints[:, :55, :].reshape(-1, 3)
            pts.append(flat[:: max(1, len(flat) // 9000)])
        vertices = body.get("vertices")
        if vertices is not None and len(vertices):
            flat_v = vertices.reshape(-1, 3)
            pts.append(flat_v[:: max(1, len(flat_v) // 12000)])
    all_pts = np.concatenate(pts, axis=0)
    lo = np.nanmin(all_pts, axis=0)
    hi = np.nanmax(all_pts, axis=0)
    margin = np.maximum(0.18, 0.20 * (hi - lo))
    return lo - margin, hi + margin


def render_3d(
    rows: list[dict[str, str]],
    visuals: list[dict[str, object]],
    out: Path,
    fps: float,
    *,
    side_yz: bool = False,
    body: dict[str, np.ndarray] | None = None,
) -> dict[str, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    wr, tmp = writer(out, fps, (1280, 720))
    preview = out.with_name(out.stem + "_preview.png")
    lo, hi = bounds(rows, visuals, body)
    trail: list[np.ndarray] = []
    preview_written = False
    for row in rows:
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#f4f6fa")
        if side_yz:
            ax = fig.add_subplot(111)
            ax.set_xlim(float(lo[2]), float(hi[2]))
            ax.set_ylim(float(hi[1]), float(lo[1]))
            ax.set_xlabel("Z depth (m)")
            ax.set_ylabel("Y image/camera (m)")
            ax.set_title("Generic URDF side Y-Z")
            for mesh in visual_meshes(row, visuals):
                cam = np.asarray(mesh["cam"], dtype=float)
                faces = np.asarray(mesh["faces"], dtype=np.int32)
                for face in faces[:: max(1, len(faces) // 140)]:
                    pts = cam[face]
                    ax.plot(pts[[0, 1, 2, 0], 2], pts[[0, 1, 2, 0], 1], color="#8a5728", lw=0.35, alpha=0.7)
            if body is not None and body.get("vertices") is not None:
                vertices = body["vertices"]
                jidx = min(max(int(float(row["frame"])) - 1, 0), len(vertices) - 1)
                vw = vertices[jidx]
                sample = vw[:: max(1, len(vw) // 950)]
                ax.scatter(sample[:, 2], sample[:, 1], s=7, color="#8f86d9", alpha=0.34, linewidths=0)
            if body is not None and body.get("joints") is not None:
                joints = body["joints"]
                jidx = min(max(int(float(row["frame"])) - 1, 0), len(joints) - 1)
                jw = joints[jidx]
                for a, b in BODY_EDGES:
                    if a < len(jw) and b < len(jw):
                        seg = jw[[a, b]]
                        ax.plot(seg[:, 2], seg[:, 1], color="#5145b8", lw=2.2, alpha=0.95)
        else:
            ax = fig.add_subplot(111, projection="3d")
            ax.set_xlim(float(lo[0]), float(hi[0]))
            ax.set_ylim(float(lo[2]), float(hi[2]))
            ax.set_zlim(float(hi[1]), float(lo[1]))
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Z depth (m)")
            ax.set_zlabel("Y image/camera (m)")
            ax.set_title("Generic URDF camera 3D")
            ax.view_init(elev=16.0, azim=-72.0)
            for mesh in visual_meshes(row, visuals):
                cam = np.asarray(mesh["cam"], dtype=float)
                faces = np.asarray(mesh["faces"], dtype=np.int32)
                tris = cam[faces[:: max(1, len(faces) // 240)]]
                tris_plot = tris[:, :, [0, 2, 1]]
                coll = Poly3DCollection(tris_plot, facecolor="#b8793e", edgecolor="#3a2412", linewidths=0.12, alpha=0.95)
                ax.add_collection3d(coll)
            trail.append(np.array([f(row, "tx"), f(row, "tz"), f(row, "ty")], dtype=float))
            if len(trail) > 1:
                tr = np.vstack(trail)
                ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color="#40566d", lw=1.4, alpha=0.5)
            if body is not None and body.get("vertices") is not None:
                vertices = body["vertices"]
                jidx = min(max(int(float(row["frame"])) - 1, 0), len(vertices) - 1)
                vw_raw = vertices[jidx]
                faces = body.get("faces")
                if faces is not None:
                    face_arr = np.asarray(faces, dtype=np.int32)
                    body_tris = vw_raw[face_arr[:: max(1, len(face_arr) // 520)]][:, :, [0, 2, 1]]
                    body_coll = Poly3DCollection(
                        body_tris,
                        facecolor="#8f86d9",
                        edgecolor="none",
                        linewidths=0.0,
                        alpha=0.16,
                    )
                    ax.add_collection3d(body_coll)
                vw = vw_raw[:, [0, 2, 1]]
                sample = vw[:: max(1, len(vw) // 1200)]
                ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=5, color="#8f86d9", alpha=0.34, depthshade=False, linewidths=0)
            if body is not None and body.get("joints") is not None:
                joints = body["joints"]
                jidx = min(max(int(float(row["frame"])) - 1, 0), len(joints) - 1)
                jw = joints[jidx]
                for a, b in BODY_EDGES:
                    if a < len(jw) and b < len(jw):
                        seg = jw[[a, b]][:, [0, 2, 1]]
                        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="#5145b8", lw=2.6, alpha=0.98)
                body_joints = jw[:22, [0, 2, 1]]
                ax.scatter(body_joints[:, 0], body_joints[:, 1], body_joints[:, 2], s=18, color="#f0ecff", edgecolors="#5145b8", linewidths=0.45, depthshade=False)
        ax.grid(True, alpha=0.25)
        fig.text(0.035, 0.045, f"frame {int(float(row['frame'])):03d}", fontsize=10, color="#222")
        img = fig_bgr(fig)
        plt.close(fig)
        wr.write(img)
        if not preview_written:
            cv2.imwrite(str(preview), img)
            preview_written = True
    wr.release()
    finalize(tmp, out)
    return {out.stem: str(out), preview.stem: str(preview)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Render generic Articraft/URDF object pose outputs.")
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--pose-csv", type=Path, required=True)
    ap.add_argument("--contacts-csv", type=Path)
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--geometry-descriptor", type=Path)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--alpha", type=float, default=0.78)
    ap.add_argument("--fx", type=float, default=1468.604736328125)
    ap.add_argument("--fy", type=float, default=1468.604736328125)
    ap.add_argument("--cx", type=float, default=640.0)
    ap.add_argument("--cy", type=float, default=360.0)
    ap.add_argument("--body-model-root", type=Path, default=Path("third-party/GVHMR/inputs/checkpoints/body_models"))
    args = ap.parse_args()

    rows = read_rows(args.pose_csv)
    visuals = load_articraft_visuals(args.urdf, args.geometry_descriptor)
    body = build_body_geometry(args.body_model_root, read_human_result(args.sample_dir / "results/gvhmr/result.pkl"))
    visible_tracks = read_visible_line_tracks(args.sample_dir)
    K = np.array([[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]], dtype=float)
    object_dir = args.out_root / "object_only"
    human_dir = args.out_root / "with_human"
    outputs = {"pose_csv": str(args.pose_csv), "urdf": str(args.urdf), "object_only": {}, "with_human": {}}
    outputs["object_only"].update(draw_overlay(args.sample_dir, rows, visuals, object_dir / "overlay.mp4", args.fps, K, args.alpha))
    outputs["object_only"].update(render_3d(rows, visuals, object_dir / "camera3d.mp4", args.fps))
    outputs["object_only"].update(render_3d(rows, visuals, object_dir / "side_yz.mp4", args.fps, side_yz=True))
    outputs["with_human"].update(
        draw_overlay(
            args.sample_dir,
            rows,
            visuals,
            human_dir / "overlay.mp4",
            args.fps,
            K,
            args.alpha,
            human_joints=body.get("joints") if body is not None else None,
        )
    )
    outputs["with_human"].update(
        draw_overlay(
            args.sample_dir,
            rows,
            visuals,
            human_dir / "overlay_vlm_masked.mp4",
            args.fps,
            K,
            args.alpha,
            respect_human_occlusion=True,
            clip_to_visible_object=True,
            line_clip_tracks=visible_tracks,
        )
    )
    outputs["with_human"].update(render_3d(rows, visuals, human_dir / "camera3d.mp4", args.fps, body=body))
    outputs["with_human"].update(render_3d(rows, visuals, human_dir / "side_yz.mp4", args.fps, side_yz=True, body=body))
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "outputs.json").write_text(json.dumps(outputs, indent=2) + "\n")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
