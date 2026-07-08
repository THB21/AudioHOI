from __future__ import annotations

import subprocess

from ....core.base.config import CaseProfile
from ....core.base.io import copy_file, repo_path, write_json
from ....core.base.runtime import runtime_python
from ....core.base.schema import REQUIRED_RENDER_FILES, stage_paths


def render(profile: CaseProfile) -> dict[str, object]:
    python_bin = runtime_python("audiohoi", override_env="AUDIOHOI_PYTHON")
    script = repo_path("scripts/shared/generic_contact_pipeline/components/render/scenes/proxy_sphere_scene.py")
    paths = stage_paths(profile)
    cmd = [
        python_bin,
        str(script),
        "--sample-dir",
        str(profile.sample_dir),
        "--ball-csv",
        str(paths["object_pose"]),
        "--render-tag",
        profile.result_name,
        "--with-human",
        "--video-codec",
        "h264",
        "--h264-encoder",
        "auto",
    ]
    subprocess.run(cmd, cwd=repo_path("."), check=True)

    dst = profile.render_dir
    mapping = {
        dst / "ball/overlay.mp4": "object_only/overlay.mp4",
        dst / "ball/camera3d.mp4": "object_only/camera3d.mp4",
        dst / "ball/side_yz.mp4": "object_only/side_yz.mp4",
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
        "backend": "proxy_sphere",
        "outputs": outputs,
        "required": REQUIRED_RENDER_FILES,
        "policy": "rerendered with generic proxy_sphere_scene using generic object_pose.csv and automatic h264 finalize",
        "renderer": {"python": python_bin, "script": str(script)},
    }
    write_json(paths["render_manifest"], manifest)
    write_json(paths["stage5_metrics"], manifest)
    return manifest
