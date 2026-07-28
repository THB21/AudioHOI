from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.shared.generic_contact_pipeline.core.base.config import load_case_profile
from scripts.shared.generic_contact_pipeline.core.solver import (
    CHAIR_FACTOR_ATTEMPT_NAME,
    CHAIR_FACTOR_RESIDUALS_NAME,
    prepare_chair_factor_executor_candidate,
    validate_chair_factor_executor_candidate,
)


REPO = Path(__file__).resolve().parents[1]


def test_chair_factor_candidate_attempt_writes_only_safe_manifest(tmp_path: Path) -> None:
    profile = load_case_profile("chair")
    candidate_dir = tmp_path / "chair_candidate"

    attempt = prepare_chair_factor_executor_candidate(
        profile,
        REPO / "samples_known_object/05_chair/results/benchmark_vlm_qwen",
        candidate_dir,
    )

    assert attempt["mode"] == "chair_generic_factor_executor_candidate_attempt"
    assert attempt["status"] == "blocked_by_private_solver_gap"
    assert attempt["solver_executed"] is False
    assert attempt["accepted_outputs_written"] is False
    assert attempt["baseline_pose_read"] is False
    assert attempt["missing_required_factor_kinds"] == []
    assert {"point_reprojection", "contact_distance", "joint_limit", "gauge_constraint"}.issubset(
        attempt["supported_residual_blocks"]
    )
    assert {path.name for path in candidate_dir.iterdir()} == {CHAIR_FACTOR_ATTEMPT_NAME, CHAIR_FACTOR_RESIDUALS_NAME}
    residuals = json.loads((candidate_dir / CHAIR_FACTOR_RESIDUALS_NAME).read_text())
    assert residuals["mode"] == "chair_generic_factor_residual_coverage"
    assert residuals["residual_evaluator_executed"] is True
    assert residuals["solver_executed"] is False
    assert residuals["required_factor_kinds_present"] is True
    assert not (candidate_dir / "object_pose.csv").exists()
    assert validate_chair_factor_executor_candidate(attempt) == []


def test_chair_factor_candidate_cli_writes_reviewable_attempt(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "chair_candidate"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/prepare_chair_factor_candidate.py",
            "--candidate-dir",
            str(candidate_dir),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "blocked_by_private_solver_gap" in completed.stdout
    payload = json.loads((candidate_dir / CHAIR_FACTOR_ATTEMPT_NAME).read_text())
    assert payload["mode"] == "chair_generic_factor_executor_candidate_attempt"
