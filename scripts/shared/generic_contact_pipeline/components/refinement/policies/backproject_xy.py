from __future__ import annotations

from ....core.base.config import CaseProfile
from ....core.base.io import copy_file, write_json
from ....core.base.schema import stage_paths


def apply(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    copied_pose = ""
    copied_contacts = ""
    if not paths["object_pose"].exists() and paths["object_pose_init"].exists():
        copied_pose = str(copy_file(paths["object_pose_init"], paths["object_pose"]))
    if paths["contact_candidates"].exists() and not paths["object_contact_points"].exists():
        copied_contacts = str(copy_file(paths["contact_candidates"], paths["object_contact_points"]))
    metrics = {
        "component": "backproject_xy",
        "case_name": profile.case_name,
        "object_pose": copied_pose or (str(paths["object_pose"]) if paths["object_pose"].exists() else ""),
        "object_contact_points": copied_contacts or (str(paths["object_contact_points"]) if paths["object_contact_points"].exists() else ""),
        "policy": "generic pass-through from Stage3 backprojected pose when no stronger refinement has produced object_pose",
    }
    write_json(paths["stage4_metrics"], metrics)
    return metrics
