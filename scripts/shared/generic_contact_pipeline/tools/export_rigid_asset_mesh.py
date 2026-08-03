#!/usr/bin/env python3
"""Export a fixed-state URDF visual assembly as one provider-neutral mesh."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

from scripts.shared.generic_contact_pipeline.components.render.scenes.generic_urdf_scene import (
    load_articraft_visuals,
)
from scripts.shared.generic_contact_pipeline.core.base.io import REPO


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def export_fixed_mesh(
    *, descriptor_path: Path, output_path: Path, minimum_vertices: int = 4096
) -> dict[str, object]:
    descriptor = json.loads(descriptor_path.read_text())
    urdf_path = resolve_repo_path(str(descriptor["resource_path"]))
    fixed_joint_state = {
        str(name): float(value)
        for name, value in dict(descriptor.get("fixed_resource_joint_state", {})).items()
    }
    visuals = load_articraft_visuals(urdf_path, fixed_joint_state)
    meshes: list[trimesh.Trimesh] = []
    visual_records: list[dict[str, object]] = []
    for visual in visuals:
        vertices_m = np.asarray(visual["vertices"], dtype=float)
        faces = np.asarray(visual["faces"], dtype=np.int64)
        bgr = tuple(int(value) for value in visual["color"])
        rgba = np.asarray([bgr[2], bgr[1], bgr[0], 255], dtype=np.uint8)
        face_colors = np.repeat(rgba[None, :], len(faces), axis=0)
        meshes.append(
            trimesh.Trimesh(
                vertices=vertices_m * 1000.0,
                faces=faces,
                face_colors=face_colors,
                process=False,
            )
        )
        visual_records.append(
            {
                "link": str(visual["link"]),
                "visual": str(visual["name"]),
                "vertex_count": int(len(vertices_m)),
                "face_count": int(len(faces)),
            }
        )
    mesh = trimesh.util.concatenate(meshes)
    source_vertex_count = int(len(mesh.vertices))
    source_face_count = int(len(mesh.faces))
    subdivision_passes = 0
    while len(mesh.vertices) < minimum_vertices:
        vertices, faces = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        subdivision_passes += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)
    bounds_mm = np.asarray(mesh.bounds, dtype=float)
    record = {
        "schema_version": 1,
        "geometry_provider_contract": "fixed_rigid_mesh_mm",
        "asset_descriptor": str(descriptor_path),
        "asset_descriptor_sha256": file_sha256(descriptor_path),
        "source_urdf": str(urdf_path),
        "source_urdf_sha256": file_sha256(urdf_path),
        "fixed_joint_state": fixed_joint_state,
        "source_units": "m",
        "output_units": "mm",
        "scale": 1000.0,
        "output_mesh": str(output_path),
        "output_mesh_sha256": file_sha256(output_path),
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "source_vertex_count": source_vertex_count,
        "source_face_count": source_face_count,
        "minimum_vertices": minimum_vertices,
        "subdivision_passes": subdivision_passes,
        "bounds_mm": bounds_mm.tolist(),
        "extents_mm": (bounds_mm[1] - bounds_mm[0]).tolist(),
        "visuals": visual_records,
    }
    sidecar = output_path.with_suffix(output_path.suffix + ".json")
    sidecar.write_text(json.dumps(record, indent=2) + "\n")
    return {**record, "sidecar": str(sidecar)}


def render_preview(mesh_path: Path, preview_path: Path) -> None:
    """Write deterministic orthographic front/side/top views for asset QA."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=float)
    triangles = vertices[np.asarray(mesh.faces, dtype=np.int64)]
    views = (
        ("front (XZ)", 0.0, -90.0),
        ("side (YZ)", 0.0, 0.0),
        ("top (XY)", 90.0, -90.0),
    )
    figure = plt.figure(figsize=(15, 5), constrained_layout=True)
    extent = float(np.max(np.ptp(vertices, axis=0))) * 0.58
    center = np.mean(np.asarray(mesh.bounds), axis=0)
    for index, (title, elevation, azimuth) in enumerate(views, start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        collection = Poly3DCollection(
            triangles,
            facecolor=(0.16, 0.18, 0.20, 0.95),
            edgecolor=(0.55, 0.58, 0.60, 0.35),
            linewidth=0.15,
        )
        axis.add_collection3d(collection)
        axis.set_xlim(center[0] - extent, center[0] + extent)
        axis.set_ylim(center[1] - extent, center[1] + extent)
        axis.set_zlim(center[2] - extent, center[2] + extent)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_title(title)
        axis.set_axis_off()
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(preview_path, dpi=180, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-descriptor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--minimum-vertices", type=int, default=4096)
    args = parser.parse_args()
    descriptor = args.asset_descriptor if args.asset_descriptor.is_absolute() else REPO / args.asset_descriptor
    output = args.output if args.output.is_absolute() else REPO / args.output
    record = export_fixed_mesh(
        descriptor_path=descriptor,
        output_path=output,
        minimum_vertices=args.minimum_vertices,
    )
    if args.preview is not None:
        preview = args.preview if args.preview.is_absolute() else REPO / args.preview
        render_preview(output, preview)
        record["preview"] = str(preview)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
