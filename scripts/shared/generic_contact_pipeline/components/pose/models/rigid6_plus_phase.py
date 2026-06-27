from __future__ import annotations

import subprocess

from ....core.base.config import CaseProfile
from ....core.base.io import REPO, read_csv, write_csv, write_json
from ....core.semantics.provenance import resolve_mug_m17_phase, write_mug_m17_reconstruction_report
from ....core.base.runtime import runtime_python
from ....core.base.schema import stage_paths


def build(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    phase_source, phase_info = resolve_mug_m17_phase(profile)
    write_mug_m17_reconstruction_report(profile, phase_info)
    m18_dir = profile.result_dir / "m18_opening_2d_video_correction_internal"
    m18_solver = REPO / "scripts/shared/generic_contact_pipeline/components/pose/mug_opening_2d_pose_correction.py"
    m18_cmd = [
        runtime_python("audiohoi", override_env="AUDIOHOI_PYTHON"),
        str(m18_solver),
        "--sample-dir",
        str(profile.sample_dir),
        "--phase-csv",
        str(phase_source),
        "--object-observation-csv",
        str(paths["object_observations"]),
        "--contact-csv",
        str(paths["object_local_points"]),
        "--out-dir",
        str(m18_dir),
    ]
    subprocess.run(m18_cmd, cwd=REPO, check=True)
    m18_pose_csv = m18_dir / "mug_m18_opening_2d_video_pose.csv"
    pose_rows = read_csv(m18_pose_csv)
    phase_by_frame = {}
    for row in read_csv(phase_source):
        phase_by_frame[int(float(row["frame"]))] = row
    out_rows = []
    for row in pose_rows:
        fr = int(float(row["frame"]))
        phase = phase_by_frame.get(fr, {})
        new = dict(row)
        new["handle_phase_rad"] = phase.get("m17_phase_rad", "")
        new["handle_phase_deg"] = phase.get("m17_phase_deg", "")
        new["vlm_visibility"] = phase.get("vlm_visibility", "")
        new["source"] = "m18_body_pose_plus_m17_handle_phase"
        out_rows.append(new)
    pose = write_csv(paths["object_pose_init"], out_rows)
    metrics = {
        "component": "rigid6_plus_phase",
        "object_pose_init": str(pose),
        "body_pose_source": str(m18_pose_csv),
        "body_pose_solver": str(m18_solver),
        "body_pose_report": str(m18_dir / "mug_m18_opening_2d_video_report.csv"),
        "body_pose_trajectory": str(m18_dir / "mug_m18_opening_2d_video_trajectory.csv"),
        "phase_source": str(phase_source),
        "phase_reconstruction": phase_info,
        "rows": len(out_rows),
    }
    write_json(paths["stage3_metrics"], metrics)
    return metrics
