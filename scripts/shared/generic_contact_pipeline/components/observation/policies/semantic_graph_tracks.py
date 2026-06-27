from __future__ import annotations

import subprocess

from ....core.base.config import CaseProfile
from ....core.base.io import copy_file, repo_path, write_json
from ....core.base.runtime import runtime_python
from ....core.base.schema import stage_paths


def build(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    semantic_dir = profile.result_dir / "chair_semantic_observations_internal"
    semantic_builder = repo_path("scripts/shared/generic_contact_pipeline/components/observation/build_chair_semantic_observations.py")
    tracks_csv = profile.sample_dir / "results/tracking/object_mesh_tracks.csv"
    semantic_cmd = [
        runtime_python("audiohoi", override_env="AUDIOHOI_PYTHON"),
        str(semantic_builder),
        "--sample-dir",
        str(profile.sample_dir),
        "--tracks-csv",
        str(tracks_csv),
        "--out-dir",
        str(semantic_dir),
    ]
    subprocess.run(semantic_cmd, cwd=repo_path("."), check=True)
    obs = copy_file(semantic_dir / "chair_semantic_observations.csv", paths["object_observations"])
    leg_candidates = copy_file(semantic_dir / "chair_leg_candidates.csv", profile.result_dir / "chair_leg_candidates.csv")
    front_frame_dir = profile.result_dir / "chair_front_frame_cotracker_internal"
    front_frame_builder = repo_path("scripts/shared/generic_contact_pipeline/components/observation/build_chair_front_frame_cotracker_observations.py")
    front_frame_cmd = [
        runtime_python("audiohoi", override_env="AUDIOHOI_PYTHON"),
        str(front_frame_builder),
        "--sample-dir",
        str(profile.sample_dir),
        "--tracks-csv",
        str(tracks_csv),
        "--semantic-csv",
        str(paths["object_observations"]),
        "--out-dir",
        str(front_frame_dir),
    ]
    subprocess.run(front_frame_cmd, cwd=repo_path("."), check=True)
    front_frame = copy_file(front_frame_dir / "chair_front_frame_cotracker_observations.csv", profile.result_dir / "chair_front_frame_cotracker_observations.csv")
    model_py = repo_path(profile.data["articraft_model_py"])
    generator = repo_path("scripts/shared/generic_contact_pipeline/components/observation/sample_chair_semantic_local_points.py")
    generated_dir = profile.result_dir / "chair_semantic_local_points"
    cmd = [
        runtime_python("audiohoi", override_env="AUDIOHOI_PYTHON"),
        str(generator),
        "--sample-dir",
        str(profile.sample_dir),
        "--model-py",
        str(model_py),
        "--out-dir",
        str(generated_dir),
    ]
    subprocess.run(cmd, cwd=repo_path("."), check=True)
    pts = copy_file(generated_dir / "chair_semantic_local_points.csv", paths["object_local_points"])
    segs = copy_file(generated_dir / "chair_semantic_local_segments.csv", paths["object_local_segments"])
    metrics = {
        "component": "semantic_graph_tracks",
        "object_observations": str(obs),
        "chair_leg_candidates": str(leg_candidates),
        "chair_front_frame_cotracker_observations": str(front_frame),
        "object_local_points": str(pts),
        "object_local_segments": str(segs),
        "semantic_observations_source": str(semantic_dir / "chair_semantic_observations.csv"),
        "semantic_leg_candidates_source": str(semantic_dir / "chair_leg_candidates.csv"),
        "semantic_builder": str(semantic_builder),
        "front_frame_builder": str(front_frame_builder),
        "front_frame_source": str(front_frame_dir / "chair_front_frame_cotracker_observations.csv"),
        "tracks_csv": str(tracks_csv),
        "semantic_local_points_source": str(generated_dir / "chair_semantic_local_points.csv"),
        "semantic_local_segments_source": str(generated_dir / "chair_semantic_local_segments.csv"),
        "articraft_model_py": str(model_py),
        "generator": str(generator),
        "semantic_parts": ["top_rail", "backrest_hole", "seat", "front_leg", "rear_leg", "stretchers", "feet"],
    }
    write_json(paths["stage1_metrics"], metrics)
    return metrics
