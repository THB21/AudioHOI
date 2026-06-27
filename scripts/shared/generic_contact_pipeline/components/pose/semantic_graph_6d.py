from __future__ import annotations

import subprocess

from ...core.base.config import CaseProfile
from ...core.base.io import REPO, copy_file, repo_path, write_json
from ...core.base.runtime import runtime_python
from ...core.base.schema import stage_paths


def build(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    seed_dir = profile.result_dir / "physical6d_seed"
    script = REPO / "scripts/shared/generic_contact_pipeline/components/render/fit_chair_metric_6d_pose.py"
    cmd = [
        runtime_python("audiohoi", override_env="AUDIOHOI_PYTHON"),
        str(script),
        "--sample-dir",
        str(profile.sample_dir),
        "--observations-csv",
        str(repo_path(paths["object_observations"])),
        "--leg-candidates-csv",
        str(repo_path(profile.result_dir / "chair_leg_candidates.csv")),
        "--front-frame-tracks-csv",
        str(repo_path(profile.result_dir / "chair_front_frame_cotracker_observations.csv")),
        "--grasp-state-csv",
        str(profile.result_dir / "chair_grasp_anchor_state/chair_grasp_anchor_state.csv"),
        "--local-points-csv",
        str(repo_path(paths["object_local_points"])),
        "--local-segments-csv",
        str(repo_path(paths["object_local_segments"])),
        "--result-dir",
        str(seed_dir),
        "--render-dir",
        str(profile.render_dir / "stage3_physical6d_seed"),
        "--fx",
        str(profile.camera["fx"]),
        "--fy",
        str(profile.camera["fy"]),
        "--cx",
        str(profile.camera["cx"]),
        "--cy",
        str(profile.camera["cy"]),
        "--no-render",
    ]
    subprocess.run(cmd, cwd=REPO, check=True)
    out = copy_file(seed_dir / "object_pose.csv", paths["object_pose_init"])
    metrics = {
        "component": "semantic_graph_6d",
        "solver": str(script),
        "seed_dir": str(seed_dir),
        "seed_render_dir": "",
        "render_skipped": True,
        "object_pose_init": str(out),
        "semantic_graph": "top_rail+hole+seat+front_rear_legs+stretchers+feet",
    }
    write_json(paths["stage3_metrics"], metrics)
    return metrics
