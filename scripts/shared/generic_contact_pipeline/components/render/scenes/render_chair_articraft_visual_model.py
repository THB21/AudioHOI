#!/usr/bin/env python3
"""Render the actual Articraft chair URDF visual model on the RGB video."""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

import smplx

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parents[5]
POSE_SOLVERS = REPO / "scripts/shared/generic_contact_pipeline/components/pose/solvers"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(POSE_SOLVERS) not in sys.path:
    sys.path.insert(0, str(POSE_SOLVERS))

from render_chair_articraft_3d_scene import (  # noqa: E402
    BODY_EDGES,
    contact_gap_entries,
    frame_path,
    load_contacts,
    palm_centers,
    pose_params,
    project_cam,
)
import fit_chair_metric_6d_pose as fit  # noqa: E402


WARM_WOOD_BGR = (42, 122, 205)
EDGE_BGR = (28, 55, 88)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows


def parse_vec(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=float)
    vals = [float(x) for x in text.strip().split()]
    return np.asarray(vals, dtype=float)


def origin_matrix(origin_el: ET.Element | None) -> np.ndarray:
    xyz = parse_vec(origin_el.get("xyz") if origin_el is not None else None, (0.0, 0.0, 0.0))
    rpy = parse_vec(origin_el.get("rpy") if origin_el is not None else None, (0.0, 0.0, 0.0))
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    mat[:3, 3] = xyz
    return mat


def visual_color_bgr(visual_el: ET.Element) -> tuple[int, int, int]:
    color_el = visual_el.find("material/color")
    if color_el is None:
        return WARM_WOOD_BGR
    rgba = parse_vec(color_el.get("rgba"), (0.78, 0.46, 0.2, 1.0))
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
        # URDF cylinders are aligned along local Z before the visual origin RPY.
        mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=18)
        return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=np.int32)

    mesh_el = geom_el.find("mesh")
    if mesh_el is not None:
        filename = mesh_el.get("filename")
        if not filename:
            raise RuntimeError("URDF mesh visual without filename")
        return load_obj_mesh(root / filename)

    raise RuntimeError(f"Unsupported visual geometry: {ET.tostring(geom_el, encoding='unicode')}")


def transform_vertices(vertices: np.ndarray, mat: np.ndarray) -> np.ndarray:
    homog = np.column_stack([vertices, np.ones(len(vertices), dtype=float)])
    return (homog @ mat.T)[:, :3]


def load_joint_origins(urdf_path: Path) -> dict[str, np.ndarray]:
    robot = ET.parse(urdf_path).getroot()
    origins: dict[str, np.ndarray] = {}
    for joint in robot.findall("joint"):
        child = joint.find("child")
        origin = joint.find("origin")
        child_link = child.get("link") if child is not None else None
        if child_link:
            origins[child_link] = parse_vec(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
    return origins


def load_articraft_visuals(urdf_path: Path) -> list[dict[str, object]]:
    root_dir = urdf_path.parent
    robot = ET.parse(urdf_path).getroot()
    joint_origins = load_joint_origins(urdf_path)
    visuals: list[dict[str, object]] = []
    for link in robot.findall("link"):
        link_name = link.get("name", "")
        for visual in link.findall("visual"):
            geom = visual.find("geometry")
            if geom is None:
                continue
            verts, faces = primitive_mesh(geom, root_dir)
            local_mat = origin_matrix(visual.find("origin"))
            verts_link = transform_vertices(verts, local_mat)
            visuals.append(
                {
                    "link": link_name,
                    "name": visual.get("name", ""),
                    "vertices": verts_link,
                    "faces": faces,
                    "color": visual_color_bgr(visual),
                    "joint_origin": joint_origins.get(link_name),
                }
            )
    if not visuals:
        raise RuntimeError(f"No URDF visuals found in {urdf_path}")
    return visuals


def cylinder_between(a: np.ndarray, b: np.ndarray, radius: float, sections: int = 18) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    axis = b - a
    length = float(np.linalg.norm(axis))
    if length <= 1e-8:
        raise ValueError("Cannot build cylinder for zero-length segment")
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    direction = axis / length
    mat = trimesh.geometry.align_vectors(np.array([0.0, 0.0, 1.0]), direction)
    mat[:3, 3] = (a + b) * 0.5
    verts = transform_vertices(np.asarray(mesh.vertices, dtype=float), mat)
    return verts, np.asarray(mesh.faces, dtype=np.int32)


def add_semantic_aligned_leg_visuals(sample: Path, visuals: list[dict[str, object]]) -> list[dict[str, object]]:
    """Replace URDF leg visuals with semantic legs used by the 2D pose fitter."""
    remove_names = {"front_top_rail_0", "front_top_rail_1", "rear_leg_0", "rear_leg_1"}
    out = [v for v in visuals if str(v.get("name", "")) not in remove_names]
    segs = fit.load_segments(sample / "results/chair_semantic_local_points/chair_semantic_local_segments.csv")
    for sid, (part, role, a, b) in segs.items():
        if sid not in {"front_leg_left", "front_leg_right", "rear_leg_left", "rear_leg_right"}:
            continue
        verts, faces = cylinder_between(a, b, radius=0.014, sections=20)
        out.append(
            {
                "link": "front_frame",
                "name": f"semantic_aligned_{sid}",
                "vertices": verts,
                "faces": faces,
                "color": WARM_WOOD_BGR,
                "semantic_segment_id": sid,
                "semantic_part": part,
            }
        )
    return out


def link_vertices_to_front_frame(link: str, vertices: np.ndarray, params: np.ndarray, joint_origin: np.ndarray | None = None) -> np.ndarray:
    if link == "front_frame":
        return vertices
    if link == "rear_support":
        origin = np.asarray(joint_origin if joint_origin is not None else [0.0, 0.040, 0.548], dtype=float)
        rot = Rotation.from_rotvec(np.array([float(params[6]), 0.0, 0.0])).as_matrix()
        return vertices @ rot.T + origin
    if link == "seat":
        origin = np.asarray(joint_origin if joint_origin is not None else [0.0, 0.090, 0.440], dtype=float)
        rot = Rotation.from_rotvec(np.array([-float(params[7]), 0.0, 0.0])).as_matrix()
        return vertices @ rot.T + origin
    return vertices


def visual_cam_meshes(params: np.ndarray, visuals: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for visual in visuals:
        verts = np.asarray(visual["vertices"], dtype=float)
        if "semantic_part" in visual:
            front = fit.articulate_points(
                str(visual.get("semantic_segment_id", "")),
                str(visual["semantic_part"]),
                params,
                verts,
            )
        else:
            joint_origin = visual.get("joint_origin")
            front = link_vertices_to_front_frame(str(visual["link"]), verts, params, joint_origin if isinstance(joint_origin, np.ndarray) else None)
        cam = fit.transform(params, front)
        out.append({**visual, "cam": cam})
    return out


def shade_color(color: tuple[int, int, int], normal_cam: np.ndarray) -> tuple[int, int, int]:
    light = np.asarray([-0.35, -0.55, -0.75], dtype=float)
    light /= np.linalg.norm(light)
    n = normal_cam / max(np.linalg.norm(normal_cam), 1e-9)
    diffuse = max(0.0, float(np.dot(n, -light)))
    scale = 0.48 + 0.52 * diffuse
    return tuple(int(np.clip(c * scale, 0, 255)) for c in color)


def draw_solid_visual_model(
    img: np.ndarray,
    params: np.ndarray,
    visuals: list[dict[str, object]],
    K: np.ndarray,
    alpha: float,
    draw_edges: bool,
) -> None:
    faces_to_draw: list[tuple[float, np.ndarray, tuple[int, int, int]]] = []
    for visual in visual_cam_meshes(params, visuals):
        cam = np.asarray(visual["cam"], dtype=float)
        faces = np.asarray(visual["faces"], dtype=np.int32)
        color = tuple(int(x) for x in visual["color"])
        uv, valid = project_cam(cam, K)
        for face in faces:
            if not np.all(valid[face]):
                continue
            tri_cam = cam[face]
            if np.any(tri_cam[:, 2] <= 1e-5):
                continue
            normal = np.cross(tri_cam[1] - tri_cam[0], tri_cam[2] - tri_cam[0])
            tri_uv = np.round(uv[face]).astype(np.int32)
            if cv2.contourArea(tri_uv) < 0.5:
                continue
            depth = float(np.mean(tri_cam[:, 2]))
            faces_to_draw.append((depth, tri_uv, shade_color(color, normal)))

    overlay = img.copy()
    for _depth, tri_uv, color in sorted(faces_to_draw, key=lambda item: item[0], reverse=True):
        cv2.fillConvexPoly(overlay, tri_uv, color, cv2.LINE_AA)
        if draw_edges:
            cv2.polylines(overlay, [tri_uv], True, EDGE_BGR, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, dst=img)


def draw_human_overlay(img: np.ndarray, joints_frame: np.ndarray, K: np.ndarray, contacts: list[dict[str, str]], params: np.ndarray) -> None:
    uv, valid = project_cam(joints_frame[:22], K)
    for a, b in BODY_EDGES:
        if valid[a] and valid[b]:
            cv2.line(img, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[b]).astype(int)), (210, 120, 255), 2, cv2.LINE_AA)
    left, right = palm_centers(joints_frame)
    for label, palm, color in (("left palm", left, (70, 190, 80)), ("right palm", right, (230, 200, 70))):
        puv, pvalid = project_cam(palm[None, :], K)
        if pvalid[0]:
            p = tuple(np.round(puv[0]).astype(int))
            cv2.circle(img, p, 8, (25, 25, 25), -1, cv2.LINE_AA)
            cv2.circle(img, p, 6, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (p[0] + 5, p[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    for _hand_side, anchor, hand, gap in contact_gap_entries(params, contacts, joints_frame):
        auv, av = project_cam(anchor[None, :], K)
        huv, hv = project_cam(hand[None, :], K)
        if av[0] and hv[0]:
            cv2.line(img, tuple(np.round(auv[0]).astype(int)), tuple(np.round(huv[0]).astype(int)), (20, 20, 20), 2, cv2.LINE_AA)
            mid = tuple(np.round((auv[0] + huv[0]) * 0.5).astype(int))
            cv2.putText(img, f"{gap:.2f}m", (mid[0] + 4, mid[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2, cv2.LINE_AA)


def read_human_result(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as f:
        data = pickle.load(f)
    params = data["smpl_params_incam"]
    return {
        "body_pose": np.asarray(params["body_pose"], dtype=np.float32),
        "betas": np.asarray(params["betas"], dtype=np.float32),
        "global_orient": np.asarray(params["global_orient"], dtype=np.float32),
        "transl": np.asarray(params["transl"], dtype=np.float32),
        "K_fullimg": np.asarray(data["K_fullimg"], dtype=np.float32),
    }


def build_body_joints(body_models_root: Path, human: dict[str, np.ndarray]) -> np.ndarray:
    model = smplx.create(
        str(body_models_root),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        batch_size=human["transl"].shape[0],
    )
    with torch.inference_mode():
        output = model(
            body_pose=torch.from_numpy(human["body_pose"]),
            betas=torch.from_numpy(human["betas"]),
            global_orient=torch.from_numpy(human["global_orient"]),
            transl=torch.from_numpy(human["transl"]),
            return_verts=False,
        )
    return output.joints.detach().cpu().numpy().astype(np.float32)


def finalize_video(tmp: Path, out: Path) -> None:
    ffmpeg = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    try:
        subprocess.run([ffmpeg, "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        subprocess.run([ffmpeg, "-y", "-i", str(tmp), "-c:v", "libopenh264", "-pix_fmt", "yuv420p", str(out)], check=True, capture_output=True, text=True)
    tmp.unlink(missing_ok=True)


def render_overlay(
    sample: Path,
    pose_rows: list[dict[str, str]],
    visuals: list[dict[str, object]],
    contacts: dict[int, list[dict[str, str]]],
    K_all: np.ndarray,
    out_dir: Path,
    name: str,
    fps: float,
    alpha: float,
    draw_edges: bool,
    joints: np.ndarray | None = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(frame_path(sample, int(pose_rows[0]["frame"]))))
    if first is None:
        raise RuntimeError("Could not read first frame")
    h, w = first.shape[:2]
    tmp = out_dir / f"{Path(name).stem}.tmp.mp4"
    mp4 = out_dir / name
    preview = out_dir / f"{Path(name).stem}_preview.png"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    preview_written = False

    for idx, row in enumerate(pose_rows):
        fr = int(row["frame"])
        img = cv2.imread(str(frame_path(sample, fr)))
        if img is None:
            raise RuntimeError(f"Could not read frame {fr}")
        k_idx = min(max(fr - 1, 0), len(K_all) - 1) if K_all.ndim == 3 else 0
        K = K_all[k_idx] if K_all.ndim == 3 else K_all
        params = pose_params(row)
        draw_solid_visual_model(img, params, visuals, K, alpha=alpha, draw_edges=draw_edges)
        if joints is not None:
            j_idx = min(max(fr - 1, 0), len(joints) - 1)
            draw_human_overlay(img, joints[j_idx], K, contacts.get(fr, []), params)
        cv2.putText(img, f"frame {fr:03d}  Articraft visual model", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(img, f"frame {fr:03d}  Articraft visual model", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 1, cv2.LINE_AA)
        writer.write(img)
        if not preview_written:
            cv2.imwrite(str(preview), img)
            preview_written = True
    writer.release()
    finalize_video(tmp, mp4)
    return preview, mp4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-dir", type=Path, default=Path("samples_known_object/05_chair"))
    ap.add_argument("--pose-csv", type=Path, default=Path("samples_known_object/05_chair/results/chair_palm_contact_refined_existing_logic/object_pose_chair_merged_6d.csv"))
    ap.add_argument("--urdf", type=Path, default=Path("samples_known_object/05_chair/articraft/generated_record/rec_continue-editing-the-current-topology-fixed-chai_20260615_113825_272909_6911e934/model.urdf"))
    ap.add_argument("--out-root", type=Path, default=Path("samples_known_object/05_chair/results/renders/chair_palm_contact_refined_existing_logic/articraft_visual_model"))
    ap.add_argument("--body-model-root", type=Path, default=Path("third-party/GVHMR/inputs/checkpoints/body_models"))
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--alpha", type=float, default=0.68)
    ap.add_argument("--no-edges", action="store_true")
    ap.add_argument("--semantic-aligned", action="store_true", help="Replace URDF leg visuals with semantic leg primitives used by the 2D fitter.")
    args = ap.parse_args()

    sample = args.sample_dir
    pose_rows = read_rows(args.pose_csv)
    visuals = load_articraft_visuals(args.urdf)
    if args.semantic_aligned:
        visuals = add_semantic_aligned_leg_visuals(sample, visuals)
    contacts = load_contacts(sample / "results/final_result/object_contact_points.csv")
    human = read_human_result(sample / "results/gvhmr/result.pkl")
    K_all = np.asarray(human["K_fullimg"], dtype=float)
    joints = build_body_joints(args.body_model_root, human)

    object_preview, object_mp4 = render_overlay(
        sample, pose_rows, visuals, contacts, K_all, args.out_root / "object_only", "articraft_model_overlay.mp4",
        args.fps, args.alpha, not args.no_edges, joints=None,
    )
    human_preview, human_mp4 = render_overlay(
        sample, pose_rows, visuals, contacts, K_all, args.out_root / "with_human", "articraft_model_contact_overlay.mp4",
        args.fps, args.alpha, not args.no_edges, joints=joints,
    )
    summary = {
        "pose_csv": str(args.pose_csv),
        "urdf": str(args.urdf),
        "num_visuals": len(visuals),
        "semantic_aligned": bool(args.semantic_aligned),
        "object_only": {"overlay": str(object_mp4), "preview": str(object_preview)},
        "with_human": {"overlay": str(human_mp4), "preview": str(human_preview)},
        "note": "URDF visual render: boxes/cylinders plus the Articraft backrest OBJ, articulated by the current rear/seat joint angles and chair 6D pose.",
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "articraft_visual_model_render_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
