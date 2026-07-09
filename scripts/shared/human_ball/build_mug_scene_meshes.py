#!/usr/bin/env python3
"""Export the Articraft mug as a per-frame 6DOF transformed mesh sequence.

The teammate's mug pipeline produces a true 6DOF object pose (x,y,z,yaw,pitch,roll,scale)
and a 4-part canonical mesh (body_shell, rim_ring, bottom_disk, handle_loop). Her renderer
places the rigid mesh with that pose; we reuse the SAME convention
(``fit_mug_articraft_keyframe_pose.transform``) so the mug matches her result, and combine
the parts into one trimesh per frame for the unified body+hands+object scene renderer.

Writes a combined canonical OBJ (object-local, scale-normalized) + a CSV trajectory carrying
the full 6DOF pose, which ``render_full_scene_3d.py`` consumes with ``--object-mesh`` and the
new ``--object-pose-csv`` 6DOF columns.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import trimesh

_STAGE1 = Path(__file__).resolve().parents[1] / "radius_free_proxy" / "stage1_observation"
sys.path.insert(0, str(_STAGE1))
import render_mug_articraft_rigid_mesh_vlm as rigid  # noqa: E402

# warm ceramic colors per part (RGB 0..1)
_PART_RGB = {
    "body_shell": (0.87, 0.80, 0.68),
    "rim_ring": (0.78, 0.72, 0.61),
    "bottom_disk": (0.74, 0.68, 0.57),
    "handle_loop": (0.87, 0.80, 0.68),
}


def build_combined_mesh(mesh_root: Path) -> trimesh.Trimesh:
    """Combine the 4 Articraft parts (with their part origins) into one object-local mesh."""
    parts = rigid.load_articraft_meshes_solid(mesh_root)
    meshes = []
    for name, (verts, faces, _edges) in parts.items():
        m = trimesh.Trimesh(vertices=np.asarray(verts, float), faces=np.asarray(faces, int), process=False)
        rgba = np.tile((np.array([*_PART_RGB.get(name, (0.85, 0.78, 0.66)), 1.0]) * 255).astype(np.uint8),
                       (m.vertices.shape[0], 1))
        m.visual.vertex_colors = rgba
        meshes.append(m)
    return trimesh.util.concatenate(meshes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, default=Path("samples_known_object/02_mug"))
    ap.add_argument("--mesh-root", type=Path,
                    default=Path("samples_known_object/02_mug/articraft/materialized_mug_mesh"))
    ap.add_argument("--object-pose-csv", type=Path,
                    default=Path("samples_known_object/02_mug/results/final_result/object_pose.csv"))
    ap.add_argument("--out-obj", type=Path,
                    default=Path("samples_known_object/02_mug/results/final_result/mug_combined.obj"))
    ap.add_argument("--out-pose-csv", type=Path,
                    default=Path("samples_known_object/02_mug/results/final_result/mug_pose6d_traj.csv"))
    args = ap.parse_args()

    mesh = build_combined_mesh(args.mesh_root)
    args.out_obj.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(args.out_obj))

    rows = list(csv.DictReader(args.object_pose_csv.open()))
    with args.out_pose_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "time", "tx", "ty", "tz", "yaw", "pitch", "roll", "scale", "radius_m"])
        for r in rows:
            w.writerow([r["frame"], r.get("time", ""), r["x"], r["y"], r["z"],
                        r["yaw"], r["pitch"], r["roll"], r["scale"], 0.05])
    print(f"combined mug mesh: {args.out_obj} ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")
    print(f"6DOF pose trajectory: {args.out_pose_csv} ({len(rows)} frames)")


if __name__ == "__main__":
    main()
