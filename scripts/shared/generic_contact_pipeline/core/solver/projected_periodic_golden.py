from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..base.io import REPO


DEFAULT_PROJECTED_PERIODIC_GOLDEN = REPO / "tests/golden/mug_projected_periodic_migration_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_projected_periodic_regression_summary(attempt_path: Path) -> dict[str, object]:
    attempt_path = attempt_path.resolve()
    attempt = json.loads(attempt_path.read_text())
    candidate_dir = attempt_path.parent
    result_dir = candidate_dir.parent
    body_path = candidate_dir / str(attempt["body_candidate_artifact"])
    phase_path = candidate_dir / str(attempt["phase_candidate_artifact"])
    reference_body = result_dir / "observation_seed/body_pose.csv"
    reference_phase = result_dir / "observation_seed/axial_phase.csv"
    return {
        "schema_version": 1,
        "attempt_id": attempt["attempt_id"],
        "solver_executed": attempt["solver_executed"],
        "accepted_outputs_written": attempt["accepted_outputs_written"],
        "baseline_pose_read": attempt["baseline_pose_read"],
        "historical_phase_read": attempt["historical_phase_read"],
        "state_spec": attempt["state_spec"],
        "geometry_provider": attempt["geometry_provider"],
        "kinematic_contract": attempt["kinematic_contract"],
        "parameters": attempt["parameters"],
        "input_sha256": {name: item["sha256"] for name, item in attempt["inputs"].items()},
        "body_candidate_sha256": _sha256(body_path),
        "phase_candidate_sha256": _sha256(phase_path),
        "body_reference_sha256": _sha256(reference_body),
        "phase_reference_sha256": _sha256(reference_phase),
        "body_equals_observation_seed": body_path.read_bytes() == reference_body.read_bytes(),
        "phase_equals_observation_seed": phase_path.read_bytes() == reference_phase.read_bytes(),
        "frames": attempt["frames"],
        "body_fit_success_frames": attempt["body_fit_success_frames"],
        "body_residual_rms_median": attempt["body_residual_rms_median"],
        "body_residual_rms_p90": attempt["body_residual_rms_p90"],
        "phase": attempt["phase"],
    }


def verify_projected_periodic_regression(
    attempt_path: Path,
    expected_path: Path = DEFAULT_PROJECTED_PERIODIC_GOLDEN,
) -> list[str]:
    try:
        actual = build_projected_periodic_regression_summary(attempt_path)
        expected = json.loads(expected_path.read_text())
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return [f"projected periodic regression artifacts are incomplete: {exc}"]
    if actual == expected:
        return []
    return [
        f"{key}: expected {expected.get(key)!r}, got {actual.get(key)!r}"
        for key in sorted(set(expected) | set(actual))
        if expected.get(key) != actual.get(key)
    ]
