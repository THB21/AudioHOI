"""Generic SE(3) placeholder seed from a rigid mask center and metric depth."""
from __future__ import annotations

from ....core.base.camera import backproject_uvz
from ....core.base.config import CaseProfile
from ....core.base.io import read_csv, write_csv, write_json
from ....core.base.schema import stage_paths


def build(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    rows = []
    for observation in read_csv(paths["object_observations"]):
        u = float(observation["center_x"])
        v = float(observation["center_y"])
        depth = float(observation["object_ref_depth_m"])
        tx, ty, tz = backproject_uvz(u, v, depth, profile.camera)
        rows.append({
            "frame": observation["frame"], "time": observation["time"],
            "tx": tx, "ty": ty, "tz": tz,
            "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
            "coord_frame": "camera_meters",
            "source": "rigid_mask_center_depth_placeholder_before_capability_initializer",
        })
    out = write_csv(paths["object_pose_init"], rows)
    metrics = {
        "component": "rigid_mask_seed",
        "object_pose_init": str(out),
        "rows": len(rows),
        "baseline_pose_read": False,
        "case_dispatch_used": False,
        "policy": "Stage3 publishes only an observation-derived placeholder; Stage4 capability initializer estimates orientation.",
    }
    write_json(paths["stage3_metrics"], metrics)
    return metrics
