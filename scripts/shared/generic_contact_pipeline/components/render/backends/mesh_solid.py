from __future__ import annotations

import subprocess
from pathlib import Path

from ....core.base.config import CaseProfile
from ....core.base.io import copy_file, repo_path, write_json
from ....core.base.runtime import runtime_python
from ....core.base.schema import REQUIRED_RENDER_FILES, stage_paths


def render(profile: CaseProfile) -> dict[str, object]:
    """Render the real object mesh (assets/object_meshes/*.glb) over the trajectory using the
    generic URDF scene's --object-mesh path (metric-scaled, per-frame SE(3) placement). Falls
    back to the proxy sphere when no object_mesh is configured or the file is missing."""
    mesh = profile.object_mesh
    if mesh is None or not mesh.exists():
        from . import proxy_sphere
        return proxy_sphere.render(profile)

    python_bin = runtime_python("audiohoi", override_env="AUDIOHOI_PYTHON")
    script = repo_path(
        "scripts/shared/generic_contact_pipeline/components/render/scenes/generic_urdf_scene.py"
    )
    paths = stage_paths(profile)
    dst = profile.render_dir
    cmd = [
        python_bin,
        str(script),
        "--sample-dir", str(profile.sample_dir),
        "--pose-csv", str(paths["object_pose"]),
        "--contacts-csv", str(paths["object_contact_points"]),
        "--object-mesh", str(mesh),
        "--object-mesh-metric", profile.object_mesh_metric,
        "--out-root", str(dst),
        "--fx", str(profile.camera["fx"]),
        "--fy", str(profile.camera["fy"]),
        "--cx", str(profile.camera["cx"]),
        "--cy", str(profile.camera["cy"]),
    ]
    subprocess.run(cmd, cwd=repo_path("."), check=True)

    mapping = {dst / rel: rel for rel in REQUIRED_RENDER_FILES}
    for rel in ["object_only/overlay_quality.csv", "with_human/overlay_quality.csv"]:
        src = dst / rel
        if src.exists():
            mapping[src] = rel
    outputs = {}
    for srel, drel in mapping.items():
        target = dst / drel
        if repo_path(srel).resolve() == repo_path(target).resolve():
            outputs[drel] = str(target)
        else:
            outputs[drel] = str(copy_file(srel, target))

    manifest = {
        "backend": "mesh_solid",
        "outputs": outputs,
        "required": REQUIRED_RENDER_FILES,
        "policy": "render the real object .glb mesh (metric-scaled, per-frame SE(3)) via generic_urdf_scene --object-mesh",
        "renderer": {"python": python_bin, "script": str(script), "object_mesh": str(mesh),
                     "object_mesh_metric": profile.object_mesh_metric},
    }
    write_json(paths["render_manifest"], manifest)
    write_json(paths["stage5_metrics"], manifest)
    return manifest
