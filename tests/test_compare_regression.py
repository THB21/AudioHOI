from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.shared.generic_contact_pipeline.core.base.io import write_csv
from scripts.shared.generic_contact_pipeline.core.base.schema import stage_paths
from scripts.shared.generic_contact_pipeline.core.evaluation.compare import compare_case


def test_declared_missing_phase_baseline_is_explicit_non_evaluable_failure(tmp_path: Path) -> None:
    result_dir = tmp_path / "sample/results/fresh"
    profile = SimpleNamespace(
        case_name="mug",
        result_dir=result_dir,
        render_dir=tmp_path / "sample/results/renders/fresh",
        baseline={
            "final_pose_csv": str(tmp_path / "baseline/object_pose.csv"),
            "final_phase_csv": str(tmp_path / "baseline/missing_phase.csv"),
        },
    )
    paths = stage_paths(profile)
    pose_rows = [
        {
            "frame": "1", "time": "0.0", "x": "0", "y": "0", "z": "3",
            "yaw": "0", "pitch": "0", "roll": "0", "scale": "1",
        }
    ]
    write_csv(Path(profile.baseline["final_pose_csv"]), pose_rows)
    write_csv(paths["object_pose"], pose_rows)
    write_csv(paths["object_pose_init"], pose_rows)
    write_csv(paths["object_phase"], [{"frame": "1", "m17_phase_rad": "0"}])
    write_csv(paths["object_observations"], [{"frame": "1", "time": "0"}])
    write_csv(paths["contact_candidates"], [{"frame": "1", "time": "0"}])
    write_csv(paths["object_contact_points"], [{"frame": "1", "time": "0"}])

    report = compare_case(profile)

    assert report["checks"]["phase_delta"] == {
        "comparable": False,
        "reason": "declared_phase_baseline_missing",
        "baseline_path": profile.baseline["final_phase_csv"],
        "new_phase_exists": True,
    }
    assert report["checks"]["phase_delta_pass"] is False
    assert report["checks"]["overall_pass"] is False
