#!/usr/bin/env python3
"""Bake a (mostly-fixed-joint) URDF into a single GLB at zero joint angles.

Made for the teammate's Articraft object models (mug: part OBJs, chair: box/cylinder
primitives + one OBJ) so the full-scene renderer can consume them via --object-mesh.
Joint angles are taken as zero (resting pose); revolute/prismatic joints are treated
as fixed at their origin — fine for rendering a folded-open chair or a rigid mug.

  conda run -n gvhmr python scripts/shared/rendering/bake_urdf_to_glb.py \
    --urdf <model.urdf> --out assets/object_meshes/mug_articraft.glb
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh


def _origin_matrix(el) -> np.ndarray:
    M = np.eye(4)
    if el is None:
        return M
    xyz = [float(v) for v in (el.get("xyz", "0 0 0").split())]
    rpy = [float(v) for v in (el.get("rpy", "0 0 0").split())]
    M[:3, 3] = xyz
    cr, cp, cy = np.cos(rpy)
    sr, sp, sy = np.sin(rpy)
    # URDF fixed-axis rpy: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    M[:3, :3] = Rz @ Ry @ Rx
    return M


def _geom_mesh(geom, urdf_dir: Path) -> trimesh.Trimesh | None:
    if geom is None:
        return None
    box = geom.find("box")
    if box is not None:
        return trimesh.creation.box(extents=[float(v) for v in box.get("size").split()])
    cyl = geom.find("cylinder")
    if cyl is not None:
        return trimesh.creation.cylinder(radius=float(cyl.get("radius")), height=float(cyl.get("length")), sections=24)
    sph = geom.find("sphere")
    if sph is not None:
        return trimesh.creation.uv_sphere(radius=float(sph.get("radius")), count=[16, 16])
    mesh_el = geom.find("mesh")
    if mesh_el is not None:
        path = urdf_dir / mesh_el.get("filename")
        m = trimesh.load(str(path), force="mesh")
        scale = mesh_el.get("scale")
        if scale:
            m.apply_scale([float(v) for v in scale.split()])
        return m
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--color", type=str, default="205,190,170", help="R,G,B for untextured parts")
    ap.add_argument("--keep-origin", action="store_true",
                    help="keep the URDF frame origin (required for generic-mainline SE3 poses)")
    args = ap.parse_args()

    root = ET.parse(str(args.urdf)).getroot()
    urdf_dir = args.urdf.parent
    rgb = [int(v) for v in args.color.split(",")]

    # link -> world transform via joint tree (zero angles: joint transform = origin)
    joints = root.findall("joint")
    parent_of = {j.find("child").get("link"): (j.find("parent").get("link"), _origin_matrix(j.find("origin"))) for j in joints}

    def world_T(link: str) -> np.ndarray:
        T = np.eye(4)
        while link in parent_of:
            link, T_j = parent_of[link]
            T = T_j @ T
        return T

    parts = []
    for link in root.findall("link"):
        name = link.get("name")
        T_link = world_T(name)
        for vis in link.findall("visual"):
            m = _geom_mesh(vis.find("geometry"), urdf_dir)
            if m is None:
                continue
            m = m.copy()
            m.apply_transform(T_link @ _origin_matrix(vis.find("origin")))
            if not isinstance(m.visual, trimesh.visual.ColorVisuals) or m.visual.vertex_colors is None or len(m.visual.vertex_colors) == 0:
                m.visual = trimesh.visual.ColorVisuals(m)
            m.visual.vertex_colors = np.tile([*rgb, 255], (len(m.vertices), 1)).astype(np.uint8)
            parts.append(m)
    if not parts:
        raise RuntimeError(f"No visual geometry baked from {args.urdf}")
    mesh = trimesh.util.concatenate(parts)
    if not getattr(args, "keep_origin", False):
        # SE3 poses from the generic mainline are defined w.r.t. the URDF frame —
        # recentring breaks them (chair origin is at floor level). Use --keep-origin.
        mesh.apply_translation(-mesh.bounding_box.centroid)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(args.out))
    ext = mesh.bounding_box.extents
    print(f"{args.out}  parts={len(parts)}  extents={ext[0]:.3f}x{ext[1]:.3f}x{ext[2]:.3f} m  verts={len(mesh.vertices)}")


if __name__ == "__main__":
    main()
