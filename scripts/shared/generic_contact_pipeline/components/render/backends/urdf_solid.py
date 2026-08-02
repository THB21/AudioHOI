from __future__ import annotations

import subprocess
import json
from pathlib import Path

import numpy as np

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
    descriptor_path = profile.data.get("geometry_asset_descriptor")
    if descriptor_path:
        descriptor = json.loads(repo_path(str(descriptor_path)).read_text())
        resource_path = descriptor.get("resource_path")
        if resource_path:
            return repo_path(str(resource_path)).resolve()
    urdf = profile.sample_dir / DEFAULT_CHAIR_URDF
    if urdf.exists():
        return urdf.resolve()
    return (profile.sample_dir / "results/mainline_0425/articraft_urdf/model.urdf").resolve()


def render_candidate_overlay_evidence(
    profile: CaseProfile,
    pose_csv: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Render the current Stage-4 candidate for VLM review, without Stage-5 outputs."""

    if profile.data.get("render_scene") != "generic_urdf":
        return {"status": "not_applicable", "reason": "candidate evidence requires generic_urdf"}
    from ..scenes import generic_urdf_scene as scene

    descriptor_path = profile.data.get("geometry_asset_descriptor")
    descriptor = (
        json.loads(repo_path(str(descriptor_path)).read_text())
        if descriptor_path
        else {}
    )
    joint_positions = {
        str(name): float(value)
        for name, value in descriptor.get("fixed_resource_joint_state", {}).items()
    }
    rows = scene.read_rows(pose_csv)
    visuals = scene.load_articraft_visuals(resolve_urdf_path(profile), joint_positions)
    camera = profile.camera
    intrinsic = np.asarray(
        (
            (float(camera["fx"]), 0.0, float(camera["cx"])),
            (0.0, float(camera["fy"]), float(camera["cy"])),
            (0.0, 0.0, 1.0),
        ),
        dtype=float,
    )
    outputs = scene.draw_overlay(
        profile.sample_dir,
        rows,
        visuals,
        output_dir / "object_only" / "overlay.mp4",
        float(profile.data.get("preprocess", {}).get("fps", 24.0)),
        intrinsic,
        0.78,
    )
    return {
        "status": "rendered",
        "scope": "stage4_candidate_object_only",
        "pose_csv": str(pose_csv),
        "outputs": outputs,
        "accepted_outputs_written": False,
    }


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
        descriptor = profile.data.get("geometry_asset_descriptor")
        if descriptor:
            cmd.extend(["--asset-descriptor", str(repo_path(str(descriptor)))])
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
