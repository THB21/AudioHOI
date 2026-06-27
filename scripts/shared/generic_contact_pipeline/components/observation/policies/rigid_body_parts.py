from __future__ import annotations

import subprocess

from ....core.base.config import CaseProfile
from ....core.base.io import REPO, copy_file, read_csv, write_csv, write_json
from ....core.semantics.provenance import resolve_mug_m17_phase, write_mug_m17_reconstruction_report
from ....core.base.runtime import runtime_python
from ....core.base.schema import stage_paths


def build(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    observation_csv = profile.sample_dir / "results/object_observations/object_observations.csv"
    phase_source, phase_info = resolve_mug_m17_phase(profile)
    write_mug_m17_reconstruction_report(profile, phase_info)
    obs_rows = read_csv(observation_csv)
    phase_by_frame = {}
    phase_csv = phase_source
    if phase_csv:
        for row in read_csv(phase_csv):
            phase_by_frame[str(int(float(row["frame"])))] = row
    rows = []
    for row in obs_rows:
        frame_key = str(int(float(row["frame"])))
        phase = phase_by_frame.get(frame_key, {})
        merged = dict(row)
        for key in ["m43_phase_rad", "m43_phase_deg", "m17_phase_rad", "m17_phase_deg", "vlm_visibility"]:
            if key in phase:
                merged[key] = phase[key]
        merged["observation_model"] = "rigid_body_plus_parts"
        rows.append(merged)
    out_obs = write_csv(paths["object_observations"], rows)
    proxy_dir = profile.result_dir / "object_proxy_observations_internal"
    proxy_builder = REPO / "scripts/shared/generic_contact_pipeline/components/observation/build_object_proxy_observations.py"
    proxy_cmd = [
        runtime_python("audiohoi", override_env="AUDIOHOI_PYTHON"),
        str(proxy_builder),
        "--sample-dir",
        str(profile.sample_dir),
        "--object-observation-csv",
        str(observation_csv),
        "--out-subdir",
        f"{profile.result_name}/object_proxy_observations_internal",
    ]
    subprocess.run(proxy_cmd, cwd=REPO, check=True)
    proxy_csv = proxy_dir / "object_proxy_observations.csv"
    export_dir = profile.result_dir / "mug_articraft_contact_points_internal"
    exporter = REPO / "scripts/shared/generic_contact_pipeline/components/observation/export_mug_articraft_contact_points.py"
    cmd = [
        runtime_python("audiohoi", override_env="AUDIOHOI_PYTHON"),
        str(exporter),
        "--sample-dir",
        str(profile.sample_dir),
        "--object-proxy-csv",
        str(proxy_csv),
        "--out-dir",
        str(export_dir),
    ]
    subprocess.run(cmd, cwd=REPO, check=True)
    out_pts = copy_file(export_dir / "mug_articraft_contact_points.csv", paths["object_local_points"])
    metrics = {
        "component": "rigid_body_plus_parts",
        "observation_rows": len(obs_rows),
        "object_observations": str(out_obs),
        "object_local_points": str(out_pts),
        "object_local_points_source": str(export_dir / "mug_articraft_contact_points.csv"),
        "object_local_points_exporter": str(exporter),
        "object_proxy_observations": str(proxy_csv),
        "object_proxy_builder": str(proxy_builder),
        "observation_source": str(observation_csv),
        "phase_source": str(phase_source),
        "phase_reconstruction": phase_info,
        "handle_visibility_from_phase": bool(phase_by_frame),
    }
    write_json(paths["stage1_metrics"], metrics)
    return metrics
