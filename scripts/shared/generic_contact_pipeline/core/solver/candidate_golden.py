from __future__ import annotations

import json
from pathlib import Path

from ..base.config import load_case_profile
from ..base.io import REPO
from ..state.golden import CANONICAL_CASE_DIRECTORIES
from .candidate import build_candidate_sandbox_manifest


DEFAULT_CANDIDATE_SANDBOX_GOLDEN = REPO / "tests/golden/sequence_candidate_sandbox_v1.json"


def candidate_sandbox_summary(case_name: str, result_dir: Path) -> dict[str, object]:
    manifest = build_candidate_sandbox_manifest(load_case_profile(case_name), result_dir)
    return {
        "status": manifest["status"],
        "eligible_for_candidate_sandbox": manifest["eligible_for_candidate_sandbox"],
        "candidate_dir": manifest["candidate_dir"],
        "attempt_id": manifest["attempt_id"],
        "problem_sha256": manifest["problem_sha256"],
        "diagnostics_sha256": manifest["diagnostics_sha256"],
        "blocking_gap_ids": manifest["blocking_gap_ids"],
        "planned_artifacts": manifest["planned_artifacts"],
        "canonical_sha256": manifest["canonical_sha256"],
    }


def build_canonical_candidate_sandbox_summary(result_name: str = "benchmark_vlm_qwen") -> dict[str, object]:
    return {
        "schema_version": 1,
        "cases": {
            case_name: candidate_sandbox_summary(
                case_name,
                REPO / "samples_known_object" / directory / "results" / result_name,
            )
            for case_name, directory in CANONICAL_CASE_DIRECTORIES.items()
        },
    }


def verify_candidate_sandbox_summary(
    expected_path: Path = DEFAULT_CANDIDATE_SANDBOX_GOLDEN,
    *,
    result_name: str = "benchmark_vlm_qwen",
) -> list[str]:
    expected = json.loads(expected_path.read_text())
    actual = build_canonical_candidate_sandbox_summary(result_name=result_name)
    errors: list[str] = []
    for case_name, expected_case in expected.get("cases", {}).items():
        actual_case = actual.get("cases", {}).get(case_name)
        if actual_case is None:
            errors.append(f"{case_name}: missing actual candidate sandbox summary")
            continue
        for key, expected_value in expected_case.items():
            actual_value = actual_case.get(key)
            if actual_value != expected_value:
                errors.append(f"{case_name}:{key}: expected {expected_value!r}, got {actual_value!r}")
    for case_name in set(actual.get("cases", {})) - set(expected.get("cases", {})):
        errors.append(f"{case_name}: unexpected actual candidate sandbox summary")
    return errors
