from __future__ import annotations

from ...core.base.config import CaseProfile
from ...core.base.io import copy_file, repo_path, write_json
from ...core.base.schema import stage_paths


def build(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    src = repo_path(profile.baseline["final_pose_csv"])
    out = copy_file(src, paths["object_pose_init"])
    metrics = {"component": "rigid6", "source": str(src), "object_pose_init": str(out)}
    write_json(paths["stage3_metrics"], metrics)
    return metrics
