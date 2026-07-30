from __future__ import annotations

import subprocess
from pathlib import Path

from ....core.base.config import CaseProfile
from ....core.base.io import copy_file, repo_path, write_json
from ....core.base.runtime import runtime_python
from ....core.base.schema import REQUIRED_RENDER_FILES, stage_paths


DEFAULT_CHAIR_URDF = (
    "articraft/generated_record/rec_fork-6911e934-oldtopology-rearup-stretcherlevel-pass_20260619_1056/model.urdf"
)


def resolve_urdf_path(profile: CaseProfile) -> Path:
    configured = profile.data.get("articraft_urdf") or profile.data.get("urdf")
    if configured:
        return repo_path(str(configured)).resolve()
    urdf = profile.sample_dir / DEFAULT_CHAIR_URDF
    if urdf.exists():
        return urdf.resolve()
    return (profile.sample_dir / "results/mainline_0425/articraft_urdf/model.urdf").resolve()


def render(profile: CaseProfile) -> dict[str, object]:
    python_bin = runtime_python("audiohoi", override_env="AUDIOHOI_PYTHON")
    if profile.data.get("render_scene") == "generic_urdf":
        script = repo_path("scripts/shared/generic_contact_pipeline/components/render/scenes/generic_urdf_scene.py")
    else:
        script = repo_path("scripts/shared/generic_contact_pipeline/components/render/scenes/urdf_solid_scene.py")
    paths = stage_paths(profile)
    dst = profile.render_dir
    urdf = resolve_urdf_path(profile)
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
    if profile.data.get("render_scene") == "generic_urdf":
        cmd.extend(
            [
                "--geometry-descriptor",
                str(repo_path(str(profile.data["geometry_asset_descriptor"]))),
                "--fx",
                str(profile.camera["fx"]),
                "--fy",
                str(profile.camera["fy"]),
                "--cx",
                str(profile.camera["cx"]),
                "--cy",
                str(profile.camera["cy"]),
            ]
        )
    subprocess.run(cmd, cwd=repo_path("."), check=True)
    if profile.data.get("render_scene") == "generic_urdf":
        mapping = {dst / rel: rel for rel in REQUIRED_RENDER_FILES}
        for rel in ["object_only/overlay_quality.csv", "with_human/overlay_quality.csv"]:
            src = dst / rel
            if src.exists():
                mapping[src] = rel
    else:
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
    if profile.data.get("render_scene") != "generic_urdf":
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
        "policy": "rerender configured Articraft URDF solid mesh with generic urdf_solid_scene from generic pose/contact",
        "renderer": {"python": python_bin, "script": str(script), "urdf": str(urdf)},
    }
    write_json(paths["render_manifest"], manifest)
    write_json(paths["stage5_metrics"], manifest)
    return manifest
