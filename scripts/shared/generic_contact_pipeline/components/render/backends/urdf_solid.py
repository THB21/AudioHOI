from __future__ import annotations

import subprocess

from ....core.base.config import CaseProfile
from ....core.base.io import copy_file, repo_path, write_json
from ....core.base.runtime import runtime_python
from ....core.base.schema import REQUIRED_RENDER_FILES, stage_paths


def render(profile: CaseProfile) -> dict[str, object]:
    python_bin = runtime_python("audiohoi", override_env="AUDIOHOI_PYTHON")
    script = repo_path("scripts/shared/generic_contact_pipeline/components/render/scenes/urdf_solid_scene.py")
    paths = stage_paths(profile)
    dst = profile.render_dir
    urdf = (
        profile.sample_dir
        / "articraft/generated_record/rec_fork-6911e934-oldtopology-rearup-stretcherlevel-pass_20260619_1056/model.urdf"
    )
    if not urdf.exists():
        urdf = profile.sample_dir / "results/mainline_0425/articraft_urdf/model.urdf"
    cmd = [
        python_bin,
        str(script),
        "--sample-dir",
        str(profile.sample_dir),
        "--pose-csv",
        str(paths["object_pose"]),
        "--contacts-csv",
        str(paths["object_contact_points"]),
        "--urdf",
        str(urdf),
        "--out-root",
        str(dst),
    ]
    subprocess.run(cmd, cwd=repo_path("."), check=True)
    mapping = {
        dst / "object_only/overlay_solid.mp4": "object_only/overlay.mp4",
        dst / "object_only/camera3d_solid.mp4": "object_only/camera3d.mp4",
        dst / "object_only/side_yz_solid.mp4": "object_only/side_yz.mp4",
        dst / "with_human/contact_overlay_solid.mp4": "with_human/overlay.mp4",
        dst / "with_human/camera3d_solid.mp4": "with_human/camera3d.mp4",
        dst / "with_human/side_yz_solid.mp4": "with_human/side_yz.mp4",
    }
    outputs = {}
    for srel, drel in mapping.items():
        target = dst / drel
        if repo_path(srel).resolve() == repo_path(target).resolve():
            outputs[drel] = str(target)
        else:
            outputs[drel] = str(copy_file(srel, target))
    for extra in [
        dst / "object_only/overlay_solid.mp4",
        dst / "object_only/camera3d_solid.mp4",
        dst / "object_only/side_yz_solid.mp4",
        dst / "with_human/contact_overlay_solid.mp4",
        dst / "with_human/camera3d_solid.mp4",
        dst / "with_human/camera3d_video_facing_solid.mp4",
        dst / "with_human/side_yz_solid.mp4",
    ]:
        extra.unlink(missing_ok=True)
    manifest = {
        "backend": "urdf_solid",
        "outputs": outputs,
        "required": REQUIRED_RENDER_FILES,
        "policy": "rerender Articraft URDF solid mesh with generic urdf_solid_scene from generic pose/contact",
        "renderer": {"python": python_bin, "script": str(script), "urdf": str(urdf)},
    }
    write_json(paths["render_manifest"], manifest)
    write_json(paths["stage5_metrics"], manifest)
    return manifest
