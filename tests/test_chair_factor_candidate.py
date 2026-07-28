from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.shared.generic_contact_pipeline.core.base.config import load_case_profile
from scripts.shared.generic_contact_pipeline.core.solver import (
    CHAIR_FACTOR_CANDIDATE_NAME,
    CHAIR_FACTOR_ATTEMPT_NAME,
    CHAIR_FACTOR_RESIDUAL_TABLE_NAME,
    CHAIR_FACTOR_RESIDUALS_NAME,
    build_chair_factor_residual_coverage,
    prepare_chair_factor_executor_candidate,
    validate_chair_factor_executor_candidate,
    validate_chair_factor_residual_coverage,
    verify_materialized_chair_factor_candidate,
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
    assert attempt["status"] == "ready_for_candidate_executor"
    assert attempt["solver_executed"] is False
    assert attempt["accepted_outputs_written"] is False
    assert attempt["baseline_pose_read"] is False
    assert attempt["missing_required_factor_kinds"] == []
    assert {"point_reprojection", "contact_distance", "joint_limit", "gauge_constraint"}.issubset(
        attempt["supported_residual_blocks"]
    )
    assert {path.name for path in candidate_dir.iterdir()} == {
        CHAIR_FACTOR_ATTEMPT_NAME,
        CHAIR_FACTOR_RESIDUALS_NAME,
        CHAIR_FACTOR_CANDIDATE_NAME,
        CHAIR_FACTOR_RESIDUAL_TABLE_NAME,
    }
    assert attempt["isolated_candidate_materialized"] is True
    assert attempt["candidate_pose"]["rows"] == 192
    assert attempt["residual_table"]["rows"] == 125
    residuals = json.loads((candidate_dir / CHAIR_FACTOR_RESIDUALS_NAME).read_text())
    assert residuals["mode"] == "chair_generic_factor_residual_coverage"
    assert residuals["residual_evaluator_executed"] is True
    assert residuals["solver_executed"] is False
    assert residuals["required_factor_kinds_present"] is True
    assert validate_chair_factor_residual_coverage(residuals) == []
    assert not (candidate_dir / "object_pose.csv").exists()
    assert (candidate_dir / CHAIR_FACTOR_CANDIDATE_NAME).exists()
    assert (candidate_dir / CHAIR_FACTOR_RESIDUAL_TABLE_NAME).exists()
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

    assert "ready_for_candidate_executor" in completed.stdout
    payload = json.loads((candidate_dir / CHAIR_FACTOR_ATTEMPT_NAME).read_text())
    assert payload["mode"] == "chair_generic_factor_executor_candidate_attempt"


def test_materialized_chair_factor_candidate_verifier_accepts_safe_artifacts(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "chair_candidate"
    profile = load_case_profile("chair")
    prepare_chair_factor_executor_candidate(
        profile,
        REPO / "samples_known_object/05_chair/results/benchmark_vlm_qwen",
        candidate_dir,
    )

    assert verify_materialized_chair_factor_candidate(
        profile,
        REPO / "samples_known_object/05_chair/results/benchmark_vlm_qwen",
        candidate_dir,
    ) == []


def test_materialized_chair_factor_candidate_verifier_rejects_missing_or_accepted_outputs(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "chair_candidate"
    profile = load_case_profile("chair")
    prepare_chair_factor_executor_candidate(
        profile,
        REPO / "samples_known_object/05_chair/results/benchmark_vlm_qwen",
        candidate_dir,
    )
    (candidate_dir / CHAIR_FACTOR_CANDIDATE_NAME).unlink()
    (candidate_dir / "object_pose.csv").write_text("frame\n1\n")

    errors = verify_materialized_chair_factor_candidate(
        profile,
        REPO / "samples_known_object/05_chair/results/benchmark_vlm_qwen",
        candidate_dir,
    )

    assert any("missing planned output generic_chair_factor_candidate.csv" in error for error in errors)
    assert any("accepted output artifact present" in error for error in errors)


def test_materialized_chair_factor_candidate_verifier_cli_reports_candidate(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "chair_candidate"
    prepare_chair_factor_executor_candidate(
        load_case_profile("chair"),
        REPO / "samples_known_object/05_chair/results/benchmark_vlm_qwen",
        candidate_dir,
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/verify_chair_factor_candidate.py",
            "--candidate-dir",
            str(candidate_dir),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "chair: factor_candidate materialized=True" in completed.stdout


def test_chair_factor_candidate_cli_execute_solver_writes_isolated_solution(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "chair_candidate"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/prepare_chair_factor_candidate.py",
            "--candidate-dir",
            str(candidate_dir),
            "--execute-solver",
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    attempt = json.loads((candidate_dir / CHAIR_FACTOR_ATTEMPT_NAME).read_text())
    assert json.loads(completed.stdout) == attempt
    assert attempt["solver_executed"] is True
    assert attempt["accepted_outputs_written"] is False
    assert attempt["baseline_pose_read"] is False
    assert attempt["executor_scope"] == "isolated_candidate_dir"
    assert attempt["candidate_pose"]["source"] == "isolated_chair_factor_executor"
    assert (candidate_dir / CHAIR_FACTOR_CANDIDATE_NAME).exists()
    assert (candidate_dir / CHAIR_FACTOR_RESIDUAL_TABLE_NAME).exists()
    assert not (candidate_dir / "object_pose.csv").exists()


def test_chair_factor_residual_coverage_validator_rejects_false_coverage() -> None:
    payload = build_chair_factor_residual_coverage(
        load_case_profile("chair"),
        REPO / "samples_known_object/05_chair/results/benchmark_vlm_qwen",
    )
    payload["required_factor_kinds_present"] = False
    payload["solver_executed"] = True
    payload["accepted_outputs_written"] = True

    errors = validate_chair_factor_residual_coverage(payload)

    assert any("required factor kinds" in error for error in errors)
    assert any("must not execute solver" in error for error in errors)
    assert any("must not write accepted outputs" in error for error in errors)
