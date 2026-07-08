from __future__ import annotations

import subprocess
from pathlib import Path

from ..base.config import CaseProfile
from ..base.io import REPO, read_csv, repo_path, write_csv, write_json
from ..base.runtime import runtime_python


def resolve_mug_m17_phase(profile: CaseProfile) -> tuple[Path, dict[str, object]]:
    """Best available M17 handle phase source for the mug pipeline.

    Prefer rebuilding M17 from the recovered M15 phase branch (CSV-only, no
    render artifact). The old M17 snapshot is only a fallback for when the
    recovered M15 input is missing.
    """
    rebuild_dir = profile.result_dir / "m17_rebuild_from_recovered_m15"
    rebuilt = rebuild_dir / "corrected_handle_phase.csv"
    recovered_m15 = profile.baseline.get("m15_recovered_phase_csv", "")
    recovered_m15_path = repo_path(recovered_m15) if recovered_m15 else Path("")
    solver = REPO / "scripts/shared/generic_contact_pipeline/components/pose/solvers/mug_handle_phase_correction.py"
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

    baseline_raw = profile.baseline.get("m17_phase_csv", "")
    baseline = repo_path(baseline_raw) if baseline_raw else Path("")
    if baseline.exists():
        return baseline, {
            "policy": "fallback_solved_m17_baseline",
            "phase_source": str(baseline),
            "recovered_m15_csv": str(recovered_m15_path),
            "solver": str(solver),
            "snapshot_fallback_used": True,
        }

    final_raw = profile.baseline.get("final_phase_csv", "")
    final_phase = repo_path(final_raw) if final_raw else Path("")
    if final_phase.exists():
        return final_phase, {
            "policy": "fallback_final_phase_csv",
            "phase_source": str(final_phase),
            "recovered_m15_csv": str(recovered_m15_path),
            "solver": str(solver),
            "snapshot_fallback_used": True,
            "missing_preferred_phase_sources": [str(p) for p in [recovered_m15_path, snapshot, baseline] if str(p)],
        }

    identity = rebuild_dir / "identity_handle_phase.csv"
    observation_csv = profile.sample_dir / "results/object_observations/object_observations.csv"
    rows = []
    if observation_csv.exists():
        for row in read_csv(observation_csv):
            rows.append(
                {
                    "frame": row.get("frame", ""),
                    "time": row.get("time", ""),
                    "m17_phase_rad": "0.000000",
                    "m17_phase_deg": "0.000000",
                    "m43_phase_rad": "0.000000",
                    "m43_phase_deg": "0.000000",
                    "vlm_visibility": "unclear",
                    "source": "identity_phase_fallback_missing_mug_phase_sources",
                }
            )
    write_csv(identity, rows, ["frame", "time", "m17_phase_rad", "m17_phase_deg", "m43_phase_rad", "m43_phase_deg", "vlm_visibility", "source"])
    return identity, {
        "policy": "identity_phase_fallback_missing_all_phase_sources",
        "phase_source": str(identity),
        "recovered_m15_csv": str(recovered_m15_path),
        "solver": str(solver),
        "snapshot_fallback_used": True,
        "missing_preferred_phase_sources": [str(p) for p in [recovered_m15_path, snapshot, baseline, final_phase] if str(p)],
    }


def write_mug_m17_reconstruction_report(profile: CaseProfile, info: dict[str, object]) -> Path:
    report = profile.result_dir / "m17_rebuild_from_recovered_m15" / "reconstruction_report.json"
    return write_json(report, info)


def resolve_chair_physical6d_seed(profile: CaseProfile) -> tuple[Path, dict[str, object]]:
    """Reproducible chair physical6d seed when possible.

    The accepted mainline kept the upstream 2D inputs, semantic local segments,
    and pose snapshot, so we can rebuild the physical6d seed from those CSV-only.
    Fall back to the explicit seed snapshot if any of them disappear.
    """
    rebuild_dir = profile.result_dir / "physical6d_rebuild_from_mainline_saved2d"
    rebuilt = rebuild_dir / "physical6d_pose.csv"
    mainline_pose = repo_path(profile.baseline["final_pose_csv"])
    mainline_segments = profile.sample_dir / "results/mainline_0425/semantic_local_points/chair_semantic_local_segments.csv"
    mainline_obs = profile.sample_dir / "results/mainline_0425/inputs_2d/chair_semantic_observations.csv"
    target_segments = profile.result_dir / "object_local_segments.csv"
    solver = REPO / "scripts/shared/generic_contact_pipeline/components/pose/solvers/chair_physical6d_from_baseline_constraints.py"
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
