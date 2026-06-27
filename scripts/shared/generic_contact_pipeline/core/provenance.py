from __future__ import annotations

import subprocess
from pathlib import Path

from .config import CaseProfile
from .io import REPO, repo_path, write_json
from .runtime import runtime_python


def resolve_mug_m17_phase(profile: CaseProfile) -> tuple[Path, dict[str, object]]:
    """Return the best available M17 handle phase source for the mug pipeline.

    The preferred path rebuilds M17 from the preserved recovered M15 phase
    branch. It is not a render artifact and can be regenerated CSV-only. The
    historical M17 snapshot is only a fallback when the recovered M15 input is
    absent.
    """
    rebuild_dir = profile.result_dir / "m17_rebuild_from_recovered_m15"
    rebuilt = rebuild_dir / "corrected_handle_phase.csv"
    recovered_m15 = profile.baseline.get("m15_recovered_phase_csv", "")
    recovered_m15_path = repo_path(recovered_m15) if recovered_m15 else Path("")
    solver = REPO / "scripts/shared/generic_contact_pipeline/components/pose/mug_handle_phase_correction.py"
    if recovered_m15_path.exists() and not rebuilt.exists():
        cmd = [
            runtime_python("audiohoi", override_env="AUDIOHOI_PYTHON"),
            str(solver),
            "--sample-dir",
            str(profile.sample_dir),
            "--m14-csv",
            str(recovered_m15_path),
            "--out-dir",
            str(rebuild_dir),
        ]
        subprocess.run(cmd, cwd=REPO, check=True)
    if rebuilt.exists():
        return rebuilt, {
            "policy": "rebuilt_from_recovered_m15",
            "phase_source": str(rebuilt),
            "recovered_m15_csv": str(recovered_m15_path),
            "solver": str(solver),
            "snapshot_fallback_used": False,
        }

    snapshot = profile.result_dir / "provenance_snapshots" / "mug_m17_handle_phase.csv"
    if snapshot.exists():
        return snapshot, {
            "policy": "fallback_preserved_m17_snapshot",
            "phase_source": str(snapshot),
            "recovered_m15_csv": str(recovered_m15_path),
            "solver": str(solver),
            "snapshot_fallback_used": True,
        }

    baseline = repo_path(profile.baseline["m17_phase_csv"])
    return baseline, {
        "policy": "fallback_solved_m17_baseline",
        "phase_source": str(baseline),
        "recovered_m15_csv": str(recovered_m15_path),
        "solver": str(solver),
        "snapshot_fallback_used": True,
    }


def write_mug_m17_reconstruction_report(profile: CaseProfile, info: dict[str, object]) -> Path:
    report = profile.result_dir / "m17_rebuild_from_recovered_m15" / "reconstruction_report.json"
    return write_json(report, info)


def resolve_chair_physical6d_seed(profile: CaseProfile) -> tuple[Path, dict[str, object]]:
    """Return a reproducible chair physical6d seed when possible.

    The accepted mainline retained the upstream 2D inputs, semantic local
    segments, and accepted pose snapshot. We can rebuild the physical6d seed
    from those retained inputs CSV-only. If those retained inputs disappear,
    fall back to the explicit seed snapshot.
    """
    rebuild_dir = profile.result_dir / "physical6d_rebuild_from_mainline_saved2d"
    rebuilt = rebuild_dir / "physical6d_pose.csv"
    mainline_pose = repo_path(profile.baseline["final_pose_csv"])
    mainline_segments = profile.sample_dir / "results/mainline_0425/semantic_local_points/chair_semantic_local_segments.csv"
    mainline_obs = profile.sample_dir / "results/mainline_0425/inputs_2d/chair_semantic_observations.csv"
    target_segments = profile.result_dir / "object_local_segments.csv"
    solver = REPO / "scripts/shared/generic_contact_pipeline/components/pose/chair_physical6d_from_baseline_constraints.py"
    can_rebuild = mainline_pose.exists() and mainline_segments.exists() and mainline_obs.exists() and target_segments.exists()
    if can_rebuild and not rebuilt.exists():
        cmd = [
            runtime_python("audiohoi", override_env="AUDIOHOI_PYTHON"),
            str(solver),
            "--sample-dir",
            str(profile.sample_dir),
            "--baseline-pose-csv",
            str(mainline_pose),
            "--baseline-segments-csv",
            str(mainline_segments),
            "--target-segments-csv",
            str(target_segments),
            "--semantic-observations-csv",
            str(mainline_obs),
            "--out-dir",
            str(rebuild_dir),
            "--fit-fx",
            str(profile.camera["fx"]),
            "--fit-fy",
            str(profile.camera["fy"]),
            "--fit-cx",
            str(profile.camera["cx"]),
            "--fit-cy",
            str(profile.camera["cy"]),
            "--leg-line-weight",
            "1.35",
            "--leg-endpoint-weight",
            "0.08",
            "--side-weight",
            "18.0",
            "--top-rail-weight",
            "3.2",
            "--bottom-rail-weight",
            "0.65",
            "--seat-front-weight",
            "0.25",
            "--pose-prior-weight",
            "0.2",
            "--joint-prior-weight",
            "0.45",
            "--no-render",
        ]
        subprocess.run(cmd, cwd=REPO, check=True)
    if rebuilt.exists():
        return rebuilt, {
            "policy": "rebuilt_from_mainline_saved2d",
            "seed_source": str(rebuilt),
            "mainline_pose_csv": str(mainline_pose),
            "mainline_segments_csv": str(mainline_segments),
            "mainline_observations_csv": str(mainline_obs),
            "target_segments_csv": str(target_segments),
            "solver": str(solver),
            "snapshot_fallback_used": False,
        }

    snapshot = profile.result_dir / "provenance_snapshots" / "chair_physical6d_seed.csv"
    if snapshot.exists():
        return snapshot, {
            "policy": "fallback_preserved_physical6d_seed_snapshot",
            "seed_source": str(snapshot),
            "mainline_pose_csv": str(mainline_pose),
            "mainline_segments_csv": str(mainline_segments),
            "mainline_observations_csv": str(mainline_obs),
            "target_segments_csv": str(target_segments),
            "solver": str(solver),
            "snapshot_fallback_used": True,
        }

    baseline = repo_path(profile.baseline["physical6d_seed_csv"])
    return baseline, {
        "policy": "fallback_solved_physical6d_seed_baseline",
        "seed_source": str(baseline),
        "mainline_pose_csv": str(mainline_pose),
        "mainline_segments_csv": str(mainline_segments),
        "mainline_observations_csv": str(mainline_obs),
        "target_segments_csv": str(target_segments),
        "solver": str(solver),
        "snapshot_fallback_used": True,
    }
