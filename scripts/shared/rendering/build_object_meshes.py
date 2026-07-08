#!/usr/bin/env python3
"""Procedural metric object meshes estimated from the sample prompts/images.

SAM-3D-Objects-class mesh generation needs ~32 GB VRAM (off-box), so these are
hand-parameterized primitive assemblies at real-world scale (metres), centered at
the origin, colored via vertex colors — good enough for the paper's scene renders
until generated meshes are available. Render with:
  render_full_scene_3d.py --object-mesh assets/object_meshes/<name>.glb --object-scale 1.0

Dimensions: basketball FIBA size-7 (r=0.121), football/soccer size-5 (r=0.110),
ceramic mug (r=0.042, h=0.10, torus handle), claw hammer (0.30 handle, 0.11 head),
folding wooden chair (seat 0.45 m high, back 0.80 m).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def _color(mesh: trimesh.Trimesh, rgb: tuple[int, int, int]) -> trimesh.Trimesh:
    mesh.visual.vertex_colors = np.tile([*rgb, 255], (len(mesh.vertices), 1)).astype(np.uint8)
    return mesh


def basketball() -> trimesh.Trimesh:
    return _color(trimesh.creation.uv_sphere(radius=0.121, count=[32, 32]), (222, 111, 40))


def football() -> trimesh.Trimesh:
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.110)
    colors = np.tile([245, 245, 245, 255], (len(sphere.vertices), 1)).astype(np.uint8)
    # black patches around the 12 icosahedron vertices ~ pentagon panels
    ico = trimesh.creation.icosphere(subdivisions=0, radius=0.110)
    for v in ico.vertices:
        near = np.linalg.norm(sphere.vertices - v, axis=1) < 0.045
        colors[near] = [25, 25, 25, 255]
    sphere.visual.vertex_colors = colors
    return sphere


def mug() -> trimesh.Trimesh:
    body = trimesh.creation.cylinder(radius=0.042, height=0.10, sections=48)
    body = _color(body, (240, 240, 235))
    try:
        handle = trimesh.creation.torus(major_radius=0.030, minor_radius=0.0075)
        handle.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2.0, [0, 1, 0]))
        handle.apply_translation([0.052, 0.0, 0.005])
        handle = _color(handle, (240, 240, 235))
        return trimesh.util.concatenate([body, handle])
    except BaseException:
        return body  # older trimesh without torus


def hammer() -> trimesh.Trimesh:
    handle = trimesh.creation.cylinder(radius=0.015, height=0.30, sections=24)
    handle = _color(handle, (166, 124, 54))  # wood
    head = trimesh.creation.cylinder(radius=0.026, height=0.11, sections=24)
    head.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2.0, [0, 1, 0]))
    head.apply_translation([0.0, 0.0, 0.155])
    head = _color(head, (90, 90, 96))  # steel
    return trimesh.util.concatenate([handle, head])


def chair() -> trimesh.Trimesh:
    wood = (176, 132, 60)
    parts = []
    seat = trimesh.creation.box(extents=[0.40, 0.38, 0.02])
    seat.apply_translation([0.0, 0.0, 0.45])
    parts.append(_color(seat, wood))
    back = trimesh.creation.box(extents=[0.40, 0.02, 0.32])
    back.apply_translation([0.0, -0.17, 0.66])
    parts.append(_color(back, wood))
    # folding-chair legs: front pair slanting back, rear pair slanting forward
    for sx in (-0.18, 0.18):
        for (y0, tilt) in ((0.16, -0.35), (-0.16, 0.35)):
            leg = trimesh.creation.box(extents=[0.03, 0.03, 0.50])
            leg.apply_transform(trimesh.transformations.rotation_matrix(tilt, [1, 0, 0]))
            leg.apply_translation([sx, y0, 0.22])
            parts.append(_color(leg, wood))
    mesh = trimesh.util.concatenate(parts)
    mesh.apply_translation(-mesh.bounding_box.centroid)
    return mesh


BUILDERS = {"basketball": basketball, "football": football, "mug": mug, "hammer": hammer, "chair": chair}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("assets/object_meshes"))
    ap.add_argument("--only", type=str, default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in BUILDERS.items():
        if args.only and name != args.only:
            continue
        mesh = fn()
        mesh.apply_translation(-mesh.bounding_box.centroid)
        out = args.out_dir / f"{name}.glb"
        mesh.export(str(out))
        ext = mesh.bounding_box.extents
        print(f"{name}: {out}  extents={ext[0]:.3f}x{ext[1]:.3f}x{ext[2]:.3f} m  {len(mesh.vertices)} verts")


if __name__ == "__main__":
    main()
