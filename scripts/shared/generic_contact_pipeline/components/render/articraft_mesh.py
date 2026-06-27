from __future__ import annotations

import subprocess

from ...core.config import CaseProfile
from ...core.io import copy_file, repo_path, write_json
from ...core.runtime import runtime_python
from ...core.schema import REQUIRED_RENDER_FILES, stage_paths


def render(profile: CaseProfile) -> dict[str, object]:
    python_bin = runtime_python("audiohoi", override_env="AUDIOHOI_PYTHON")
    script = repo_path("scripts/shared/generic_contact_pipeline/components/render/articraft_mug_scene.py")
    paths = stage_paths(profile)
    dst = profile.render_dir
    cmd = [
        python_bin,
        str(script),
        "--sample-dir",
        str(profile.sample_dir),
        "--pose-csv",
        str(paths["object_pose"]),
        "--phase-csv",
        str(paths["object_phase"]),
        "--phase-col",
        "m43_phase_rad",
        "--out-root",
        str(dst),
    ]
    subprocess.run(cmd, cwd=repo_path("."), check=True)
    mapping = {
        dst / "object_only/overlay.mp4": "object_only/overlay.mp4",
        dst / "object_only/camera3d.mp4": "object_only/camera3d.mp4",
        dst / "object_only/side_yz.mp4": "object_only/side_yz.mp4",
        dst / "with_human/overlay.mp4": "with_human/overlay.mp4",
        dst / "with_human/camera3d.mp4": "with_human/camera3d.mp4",
        dst / "with_human/side_yz.mp4": "with_human/side_yz.mp4",
    }
    outputs = {}
    for srel, drel in mapping.items():
        target = dst / drel
        if repo_path(srel).resolve() == repo_path(target).resolve():
            outputs[drel] = str(target)
        else:
            outputs[drel] = str(copy_file(srel, target))
    manifest = {
        "backend": "articraft_mesh",
        "outputs": outputs,
        "required": REQUIRED_RENDER_FILES,
        "policy": "rerender real Articraft mug mesh with generic articraft_mug_scene; compact Articraft-ratio side_yz",
        "renderer": {"python": python_bin, "script": str(script)},
    }
    write_json(paths["render_manifest"], manifest)
    write_json(paths["stage5_metrics"], manifest)
    return manifest
