from __future__ import annotations

import json
from pathlib import Path

from ..base.config import load_case_profile
from ..base.io import REPO
from ..state.golden import CANONICAL_CASE_DIRECTORIES
from .problem import build_sequence_problem_shadow
from .validation import validate_sequence_problem_shadow


DEFAULT_SEQUENCE_PROBLEM_GOLDEN = REPO / "tests/golden/sequence_problem_shadow_v1.json"


def sequence_problem_summary(case_name: str, result_dir: Path) -> dict[str, object]:
    problem = build_sequence_problem_shadow(load_case_profile(case_name), result_dir)
    validation_errors = validate_sequence_problem_shadow(problem)
    return {
        "state_model": problem["state_contract"]["state_model"],
        "geometry_kind": problem["state_contract"]["geometry_kind"],
        "measurement_count": problem["inputs"]["measurement_shadow"]["count"],
        "measurement_frames": problem["inputs"]["measurement_shadow"]["frames"],
        "contact_count": problem["inputs"]["contact_constraint_shadow"]["count"],
        "factor_count": problem["inputs"]["factor_shadow"]["factor_count"],
        "factor_kinds": problem["inputs"]["factor_shadow"]["factor_kinds"],
        "gap_ids": problem["inputs"]["factor_shadow"]["gap_ids"],
        "attempt_id": problem["attempt_plan"]["attempt_id"],
        "validation_errors": validation_errors,
        "canonical_sha256": problem["canonical_sha256"],
    }


def build_canonical_sequence_problem_summary(result_name: str = "benchmark_vlm_qwen") -> dict[str, object]:
    return {
        "schema_version": 1,
        "cases": {
            case_name: sequence_problem_summary(
                case_name,
                REPO / "samples_known_object" / directory / "results" / result_name,
            )
            for case_name, directory in CANONICAL_CASE_DIRECTORIES.items()
        },
    }


def verify_sequence_problem_summary(
    expected_path: Path = DEFAULT_SEQUENCE_PROBLEM_GOLDEN,
    *,
    result_name: str = "benchmark_vlm_qwen",
) -> list[str]:
    expected = json.loads(expected_path.read_text())
    actual = build_canonical_sequence_problem_summary(result_name=result_name)
    errors: list[str] = []
    for case_name, expected_case in expected.get("cases", {}).items():
        actual_case = actual.get("cases", {}).get(case_name)
        if actual_case is None:
            errors.append(f"{case_name}: missing actual sequence problem summary")
            continue
        for key, expected_value in expected_case.items():
            actual_value = actual_case.get(key)
            if actual_value != expected_value:
                errors.append(f"{case_name}:{key}: expected {expected_value!r}, got {actual_value!r}")
    for case_name in set(actual.get("cases", {})) - set(expected.get("cases", {})):
        errors.append(f"{case_name}: unexpected actual sequence problem summary")
    return errors
