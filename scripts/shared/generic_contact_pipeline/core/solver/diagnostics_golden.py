from __future__ import annotations

import json
from pathlib import Path

from ..base.config import load_case_profile
from ..base.io import REPO
from ..state.golden import CANONICAL_CASE_DIRECTORIES
from .diagnostics import build_sequence_solver_shadow_diagnostics


DEFAULT_SEQUENCE_DIAGNOSTICS_GOLDEN = REPO / "tests/golden/sequence_solver_diagnostics_v1.json"


def sequence_solver_diagnostics_summary(case_name: str, result_dir: Path) -> dict[str, object]:
    diagnostics = build_sequence_solver_shadow_diagnostics(load_case_profile(case_name), result_dir)
    return {
        "problem_sha256": diagnostics["problem_sha256"],
        "attempt_id": diagnostics["attempt_id"],
        "status": diagnostics["status"],
        "blocking_gap_ids": diagnostics["blocking_gap_ids"],
        "nonblocking_gap_ids": diagnostics["nonblocking_gap_ids"],
        "phase_statuses": {phase["phase_id"]: phase["status"] for phase in diagnostics["phases"]},
        "solver_executed": diagnostics["solver_executed"],
        "accepted_outputs_written": diagnostics["accepted_outputs_written"],
        "baseline_pose_read": diagnostics["baseline_pose_read"],
        "validation_errors": diagnostics["validation_errors"],
        "canonical_sha256": diagnostics["canonical_sha256"],
    }


def build_canonical_sequence_solver_diagnostics_summary(result_name: str = "benchmark_vlm_qwen") -> dict[str, object]:
    return {
        "schema_version": 1,
        "cases": {
            case_name: sequence_solver_diagnostics_summary(
                case_name,
                REPO / "samples_known_object" / directory / "results" / result_name,
            )
            for case_name, directory in CANONICAL_CASE_DIRECTORIES.items()
        },
    }


def verify_sequence_solver_diagnostics_summary(
    expected_path: Path = DEFAULT_SEQUENCE_DIAGNOSTICS_GOLDEN,
    *,
    result_name: str = "benchmark_vlm_qwen",
) -> list[str]:
    expected = json.loads(expected_path.read_text())
    actual = build_canonical_sequence_solver_diagnostics_summary(result_name=result_name)
    errors: list[str] = []
    for case_name, expected_case in expected.get("cases", {}).items():
        actual_case = actual.get("cases", {}).get(case_name)
        if actual_case is None:
            errors.append(f"{case_name}: missing actual sequence diagnostics summary")
            continue
        for key, expected_value in expected_case.items():
            actual_value = actual_case.get(key)
            if actual_value != expected_value:
                errors.append(f"{case_name}:{key}: expected {expected_value!r}, got {actual_value!r}")
    for case_name in set(actual.get("cases", {})) - set(expected.get("cases", {})):
        errors.append(f"{case_name}: unexpected actual sequence diagnostics summary")
    return errors
