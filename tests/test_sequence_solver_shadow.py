from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.shared.generic_contact_pipeline.core.base.config import load_case_profile
from scripts.shared.generic_contact_pipeline.core.solver import (
    SANDBOX_MANIFEST_NAME,
    SPHERE_SANDBOX_ARTIFACTS,
    build_candidate_sandbox_manifest,
    build_canonical_candidate_sandbox_summary,
    build_canonical_sequence_problem_summary,
    build_canonical_sequence_solver_diagnostics_summary,
    build_sequence_problem_shadow,
    build_sequence_solver_shadow_diagnostics,
    validate_candidate_sandbox_manifest,
    validate_sequence_problem_shadow,
    verify_candidate_sandbox_summary,
    verify_sequence_problem_summary,
    verify_sequence_solver_diagnostics_summary,
    write_candidate_sandbox_manifest,
)


REPO = Path(__file__).resolve().parents[1]
CASE_DIRECTORIES = {
    "basketball": "01_basketball",
    "football": "10_football",
    "mug": "02_mug",
    "chair": "05_chair",
    "stick": "11_stick",
}


def test_sequence_problem_shadow_is_plan_only_and_never_consumes_legacy_pose() -> None:
    result_dir = REPO / "samples_known_object/01_basketball/results/benchmark_vlm_qwen"
    problem = build_sequence_problem_shadow(load_case_profile("basketball"), result_dir)
    assert problem["mode"] == "generic_sequence_solver_shadow"
    assert problem["solver_executed"] is False
    assert problem["accepted_outputs_written"] is False
    assert problem["baseline_pose_read"] is False
    assert problem["state_contract"]["baseline_pose_read"] is False
    assert problem["attempt_plan"]["writes"] == []
    assert problem["attempt_plan"]["initializer_status"] == "not_executed"
    assert validate_sequence_problem_shadow(problem) == []


def test_sequence_problem_uses_profile_state_contract_not_object_pose_init() -> None:
    result_dir = REPO / "samples_known_object/05_chair/results/benchmark_vlm_qwen"
    problem = build_sequence_problem_shadow(load_case_profile("chair"), result_dir)
    payload = json.dumps(problem, sort_keys=True)
    assert "object_pose_init.csv" not in payload
    assert "physical6d_seed" not in payload
    assert problem["state_contract"]["state_model"] == "semantic_graph_6d"
    assert problem["state_contract"]["geometry_kind"] == "articulated_urdf"


def test_sequence_problem_hash_is_independent_of_absolute_worktree_prefix() -> None:
    relative = Path("samples_known_object/02_mug/results/benchmark_vlm_qwen")
    absolute = REPO / relative
    relative_problem = build_sequence_problem_shadow(load_case_profile("mug"), relative)
    absolute_problem = build_sequence_problem_shadow(load_case_profile("mug"), absolute)
    assert relative_problem["canonical_sha256"] == absolute_problem["canonical_sha256"]
    assert relative_problem["attempt_plan"]["attempt_id"] == absolute_problem["attempt_plan"]["attempt_id"]


def test_five_case_sequence_problem_matches_frozen_manifest() -> None:
    expected = json.loads((REPO / "tests/golden/sequence_problem_shadow_v1.json").read_text())
    actual = build_canonical_sequence_problem_summary()
    assert actual == expected
    assert verify_sequence_problem_summary() == []


def test_sequence_problem_validation_rejects_accepted_output_writes() -> None:
    result_dir = REPO / "samples_known_object/11_stick/results/benchmark_vlm_qwen"
    problem = build_sequence_problem_shadow(load_case_profile("stick"), result_dir)
    problem["accepted_outputs_written"] = True
    problem["attempt_plan"]["writes"] = ["object_pose_init.csv"]
    errors = validate_sequence_problem_shadow(problem)
    assert any("accepted outputs" in error for error in errors)
    assert any("accepted writes" in error for error in errors)


def test_sequence_problem_export_cli_writes_reviewable_manifest(tmp_path: Path) -> None:
    out = tmp_path / "mug_sequence_problem.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/export_sequence_problem_shadow.py",
            "--case",
            "mug",
            "--result-dir",
            "samples_known_object/02_mug/results/benchmark_vlm_qwen",
            "--out",
            str(out),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(out.read_text())
    assert json.loads(completed.stdout) == payload
    assert payload["mode"] == "generic_sequence_solver_shadow"
    assert payload["attempt_plan"]["attempt_id"].startswith("shadow-")


def test_sequence_problem_verifier_cli_reports_all_cases() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/verify_sequence_problem_shadow.py",
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )
    lines = completed.stdout.strip().splitlines()
    assert len(lines) == len(CASE_DIRECTORIES)
    assert lines[0].startswith("basketball: state=translation3")
    assert any(line.startswith("chair: state=semantic_graph_6d") and "gaps=[]" in line for line in lines)


def test_sequence_solver_shadow_diagnostics_never_execute_or_write() -> None:
    result_dir = REPO / "samples_known_object/10_football/results/benchmark_vlm_qwen"
    diagnostics = build_sequence_solver_shadow_diagnostics(load_case_profile("football"), result_dir)
    assert diagnostics["status"] == "ready_for_future_shadow_solve"
    assert diagnostics["solver_executed"] is False
    assert diagnostics["accepted_outputs_written"] is False
    assert diagnostics["baseline_pose_read"] is False
    phase_statuses = {phase["phase_id"]: phase["status"] for phase in diagnostics["phases"]}
    assert phase_statuses == {
        "assemble_problem": "pass",
        "initialize_state": "not_executed",
        "solve_sequence": "not_executed",
        "evaluate_candidate": "not_executed",
    }


def test_sequence_solver_shadow_diagnostics_block_only_required_unmigrated_mechanisms() -> None:
    mug = build_sequence_solver_shadow_diagnostics(
        load_case_profile("mug"),
        REPO / "samples_known_object/02_mug/results/benchmark_vlm_qwen",
    )
    chair = build_sequence_solver_shadow_diagnostics(
        load_case_profile("chair"),
        REPO / "samples_known_object/05_chair/results/benchmark_vlm_qwen",
    )
    stick = build_sequence_solver_shadow_diagnostics(
        load_case_profile("stick"),
        REPO / "samples_known_object/11_stick/results/benchmark_vlm_qwen",
    )
    assert mug["blocking_gap_ids"] == []
    assert chair["blocking_gap_ids"] == []
    assert chair["nonblocking_gap_ids"] == []
    assert stick["blocking_gap_ids"] == []
    assert stick["nonblocking_gap_ids"] == ["line_contact_lock_special_refinement"]
    assert mug["status"] == "ready_for_future_shadow_solve"
    assert chair["status"] == "ready_for_future_shadow_solve"
    assert stick["status"] == "ready_for_future_shadow_solve"


def test_five_case_sequence_solver_diagnostics_matches_frozen_manifest() -> None:
    expected = json.loads((REPO / "tests/golden/sequence_solver_diagnostics_v1.json").read_text())
    actual = build_canonical_sequence_solver_diagnostics_summary()
    assert actual == expected
    assert verify_sequence_solver_diagnostics_summary() == []


def test_sequence_solver_diagnostics_export_cli_writes_reviewable_manifest(tmp_path: Path) -> None:
    out = tmp_path / "chair_sequence_solver_diagnostics.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/export_sequence_solver_diagnostics.py",
            "--case",
            "chair",
            "--result-dir",
            "samples_known_object/05_chair/results/benchmark_vlm_qwen",
            "--out",
            str(out),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(out.read_text())
    assert json.loads(completed.stdout) == payload
    assert payload["mode"] == "generic_sequence_solver_shadow_diagnostics"
    assert payload["blocking_gap_ids"] == []
    assert payload["nonblocking_gap_ids"] == []


def test_sequence_solver_diagnostics_verifier_cli_reports_all_cases() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/verify_sequence_solver_diagnostics.py",
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )
    lines = completed.stdout.strip().splitlines()
    assert len(lines) == len(CASE_DIRECTORIES)
    assert lines[0].startswith("basketball: status=ready_for_future_shadow_solve")
    assert any(
        line.startswith("chair: status=ready_for_future_shadow_solve")
        and "blocking_gaps=[] nonblocking_gaps=[]" in line
        for line in lines
    )


def test_candidate_sandbox_manifest_allows_ready_ball_and_chair_cases() -> None:
    basketball = build_candidate_sandbox_manifest(
        load_case_profile("basketball"),
        REPO / "samples_known_object/01_basketball/results/benchmark_vlm_qwen",
    )
    mug = build_candidate_sandbox_manifest(
        load_case_profile("mug"),
        REPO / "samples_known_object/02_mug/results/benchmark_vlm_qwen",
    )
    chair = build_candidate_sandbox_manifest(
        load_case_profile("chair"),
        REPO / "samples_known_object/05_chair/results/benchmark_vlm_qwen",
    )
    assert basketball["eligible_for_candidate_sandbox"] is True
    assert basketball["planned_artifacts"] == SPHERE_SANDBOX_ARTIFACTS
    assert basketball["accepted_outputs_written"] is False
    assert mug["eligible_for_candidate_sandbox"] is True
    assert mug["planned_artifacts"] == [
        SANDBOX_MANIFEST_NAME,
        "generic_periodic_body_candidate.csv",
        "generic_periodic_phase_candidate.csv",
        "generic_projected_periodic_attempt.json",
    ]
    assert chair["eligible_for_candidate_sandbox"] is True
    assert chair["planned_artifacts"] == [
        SANDBOX_MANIFEST_NAME,
        "generic_chair_factor_candidate.csv",
        "generic_chair_factor_residuals.csv",
        "chair_generic_factor_executor_attempt.json",
        "chair_generic_factor_residuals.json",
    ]
    assert chair["blocking_gap_ids"] == []
    assert chair["nonblocking_gap_ids"] == []
    assert validate_candidate_sandbox_manifest(basketball) == []
    assert validate_candidate_sandbox_manifest(mug) == []
    assert validate_candidate_sandbox_manifest(chair) == []


def test_candidate_sandbox_manifest_allows_line_contact_as_nonblocking_compatibility() -> None:
    stick = build_candidate_sandbox_manifest(
        load_case_profile("stick"),
        REPO / "samples_known_object/11_stick/results/benchmark_vlm_qwen",
    )
    assert stick["eligible_for_candidate_sandbox"] is True
    assert stick["status"] == "sandbox_ready"
    assert stick["blocking_gap_ids"] == []
    assert stick["nonblocking_gap_ids"] == ["line_contact_lock_special_refinement"]
    assert stick["planned_artifacts"] == [SANDBOX_MANIFEST_NAME]


def test_candidate_sandbox_validation_rejects_accepted_output_names() -> None:
    manifest = build_candidate_sandbox_manifest(
        load_case_profile("football"),
        REPO / "samples_known_object/10_football/results/benchmark_vlm_qwen",
    )
    manifest["planned_artifacts"] = ["object_pose_init.csv"]
    manifest["accepted_outputs_written"] = True
    errors = validate_candidate_sandbox_manifest(manifest)
    assert any("accepted outputs" in error for error in errors)
    assert any("accepted output names" in error for error in errors)


def test_candidate_sandbox_validation_rejects_inconsistent_gap_eligibility() -> None:
    blocked = build_candidate_sandbox_manifest(
        load_case_profile("basketball"),
        REPO / "samples_known_object/01_basketball/results/benchmark_vlm_qwen",
    )
    blocked["eligible_for_candidate_sandbox"] = True
    blocked["status"] = "sandbox_ready"
    blocked["planned_artifacts"] = blocked["planned_artifacts"] or ["generic_sequence_solver_shadow_candidate.json"]
    blocked["blocking_gap_ids"] = ["synthetic_blocking_gap"]
    errors = validate_candidate_sandbox_manifest(blocked)
    assert any("eligible sandbox must not carry blocking gaps" in error for error in errors)

    ready = build_candidate_sandbox_manifest(
        load_case_profile("basketball"),
        REPO / "samples_known_object/01_basketball/results/benchmark_vlm_qwen",
    )
    ready["eligible_for_candidate_sandbox"] = False
    ready["status"] = "blocked_by_known_gaps"
    ready["planned_artifacts"] = []
    errors = validate_candidate_sandbox_manifest(ready)
    assert any("blocked sandbox must record at least one blocking gap" in error for error in errors)


@pytest.mark.parametrize("case_name", ["basketball", "football"])
def test_candidate_sandbox_materialize_writes_sphere_safe_candidate_artifacts(case_name: str, tmp_path: Path) -> None:
    candidate_dir = tmp_path / f"{case_name}_candidate"
    manifest = write_candidate_sandbox_manifest(
        load_case_profile(case_name),
        REPO / f"samples_known_object/{CASE_DIRECTORIES[case_name]}/results/benchmark_vlm_qwen",
        candidate_dir,
    )
    written = candidate_dir / SANDBOX_MANIFEST_NAME
    assert written.exists()
    assert json.loads(written.read_text()) == manifest
    assert {path.name for path in candidate_dir.iterdir()} == {
        SANDBOX_MANIFEST_NAME,
        "generic_sphere_sequence_candidate.csv",
        "generic_sphere_sequence_residuals.csv",
        "generic_sphere_sequence_attempt.json",
    }
    attempt = json.loads((candidate_dir / "generic_sphere_sequence_attempt.json").read_text())
    assert attempt["solver_executed"] is True
    assert attempt["accepted_outputs_written"] is False
    assert attempt["baseline_pose_read"] is False
    assert attempt["executor_scope"] == "isolated_candidate_dir"
    assert not (candidate_dir / "object_pose_init.csv").exists()
    assert not (candidate_dir / "object_pose.csv").exists()


def test_candidate_sandbox_materialize_writes_chair_safe_candidate_artifacts(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "chair_candidate"
    manifest = write_candidate_sandbox_manifest(
        load_case_profile("chair"),
        REPO / "samples_known_object/05_chair/results/benchmark_vlm_qwen",
        candidate_dir,
    )

    assert manifest["eligible_for_candidate_sandbox"] is True
    assert {path.name for path in candidate_dir.iterdir()} == {
        SANDBOX_MANIFEST_NAME,
        "generic_chair_factor_candidate.csv",
        "generic_chair_factor_residuals.csv",
        "chair_generic_factor_executor_attempt.json",
        "chair_generic_factor_residuals.json",
    }
    assert not (candidate_dir / "object_pose.csv").exists()
    assert not (candidate_dir / "object_contact_points.csv").exists()


def test_candidate_sandbox_materialize_writes_mug_safe_candidate_artifacts(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "mug_candidate"
    manifest = write_candidate_sandbox_manifest(
        load_case_profile("mug"),
        REPO / "samples_known_object/02_mug/results/benchmark_vlm_qwen",
        candidate_dir,
    )

    assert manifest["eligible_for_candidate_sandbox"] is True
    assert {path.name for path in candidate_dir.iterdir()} == {
        SANDBOX_MANIFEST_NAME,
        "generic_periodic_body_candidate.csv",
        "generic_periodic_phase_candidate.csv",
        "generic_projected_periodic_attempt.json",
    }
    attempt = json.loads((candidate_dir / "generic_projected_periodic_attempt.json").read_text())
    assert attempt["solver_executed"] is True
    assert attempt["accepted_outputs_written"] is False
    assert attempt["baseline_pose_read"] is False
    assert attempt["historical_phase_read"] is False
    assert attempt["executor_scope"] == "isolated_candidate_dir"
    assert not (candidate_dir / "object_pose.csv").exists()
    assert not (candidate_dir / "object_phase.csv").exists()


def test_five_case_candidate_sandbox_matches_frozen_manifest() -> None:
    expected = json.loads((REPO / "tests/golden/sequence_candidate_sandbox_v1.json").read_text())
    actual = build_canonical_candidate_sandbox_summary()
    assert actual == expected
    assert verify_candidate_sandbox_summary() == []


def test_candidate_sandbox_export_cli_writes_reviewable_manifest(tmp_path: Path) -> None:
    out = tmp_path / "football_candidate_sandbox.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/export_candidate_sandbox.py",
            "--case",
            "football",
            "--result-dir",
            "samples_known_object/10_football/results/benchmark_vlm_qwen",
            "--out",
            str(out),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(out.read_text())
    assert json.loads(completed.stdout) == payload
    assert payload["mode"] == "generic_sequence_solver_candidate_sandbox"
    assert payload["planned_artifacts"] == SPHERE_SANDBOX_ARTIFACTS


def test_candidate_sandbox_verifier_cli_reports_all_cases() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/verify_candidate_sandbox.py",
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )
    lines = completed.stdout.strip().splitlines()
    assert len(lines) == len(CASE_DIRECTORIES)
    assert lines[0].startswith("basketball: status=sandbox_ready eligible=True")
    assert any("chair: status=sandbox_ready eligible=True" in line for line in lines)


def test_candidate_sandbox_verifier_cli_materializes_and_verifies_chair_candidate(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidate_root"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/verify_candidate_sandbox.py",
            "--materialize-chair-candidates",
            "--candidate-root",
            str(candidate_root),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    chair_dir = candidate_root / "benchmark_vlm_qwen_chair"
    assert "chair_materialized=True" in completed.stdout
    assert (chair_dir / "generic_chair_factor_candidate.csv").exists()
    assert (chair_dir / "generic_chair_factor_residuals.csv").exists()
    assert not (chair_dir / "object_pose.csv").exists()


def test_candidate_sandbox_verifier_cli_materializes_and_verifies_mug_candidate(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidate_root"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/verify_candidate_sandbox.py",
            "--materialize-mug-candidates",
            "--candidate-root",
            str(candidate_root),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    mug_dir = candidate_root / "benchmark_vlm_qwen_mug"
    assert "mug_materialized=True" in completed.stdout
    assert (mug_dir / "generic_periodic_body_candidate.csv").exists()
    assert (mug_dir / "generic_periodic_phase_candidate.csv").exists()
    assert not (mug_dir / "object_pose.csv").exists()
    assert not (mug_dir / "object_phase.csv").exists()
