from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from ..base.config import load_case_profile
from ..base.io import REPO
from ..state.golden import CANONICAL_CASE_DIRECTORIES
from .candidate import build_candidate_sandbox_manifest


DEFAULT_CANDIDATE_SANDBOX_GOLDEN = REPO / "tests/golden/sequence_candidate_sandbox_v1.json"
DEFAULT_MATERIALIZED_CANDIDATE_GOLDEN = REPO / "tests/golden/sequence_candidate_materialized_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv_rows(path: Path) -> int:
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


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
        "nonblocking_gap_ids": manifest["nonblocking_gap_ids"],
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


def _attempt_summary(path: Path) -> dict[str, object]:
    attempt = json.loads(path.read_text())
    stable_keys = (
        "mode",
        "solver_executed",
        "isolated_candidate_materialized",
        "accepted_outputs_written",
        "baseline_pose_read",
        "historical_phase_read",
        "executor_scope",
        "frames",
        "residual_rows",
        "candidate_artifact",
        "residual_artifact",
        "body_candidate_artifact",
        "phase_candidate_artifact",
        "candidate_sha256",
        "residual_sha256",
        "body_candidate_sha256",
        "phase_candidate_sha256",
        "compatibility_gap_id",
        "compatibility_gap_status",
        "mechanism",
    )
    return {key: attempt[key] for key in stable_keys if key in attempt}


def materialized_candidate_case_summary(case_name: str, candidate_dir: Path, *, result_name: str) -> dict[str, object]:
    manifest = build_candidate_sandbox_manifest(
        load_case_profile(case_name),
        REPO / "samples_known_object" / CANONICAL_CASE_DIRECTORIES[case_name] / "results" / result_name,
        candidate_dir,
    )
    artifacts: dict[str, object] = {}
    for artifact_name in manifest["planned_artifacts"]:
        if artifact_name == "generic_sequence_solver_shadow_candidate.json":
            continue
        artifact_path = candidate_dir / str(artifact_name)
        record: dict[str, object] = {}
        if artifact_path.suffix == ".csv":
            record["sha256"] = _sha256(artifact_path)
            record["rows"] = _csv_rows(artifact_path)
        if artifact_path.name == "generic_object_publication.json":
            payload = json.loads(artifact_path.read_text())
            record["publication"] = {
                "status": payload.get("status"),
                "hard_gate": payload.get("hard_gate"),
                "case_dispatch_used": payload.get("case_dispatch_used"),
                "human_state_optimized": payload.get("human_state_optimized"),
                "accepted_output_written": payload.get("accepted_path") is not None,
            }
        elif artifact_path.name == "generic_problem_preparation.json":
            payload = json.loads(artifact_path.read_text())
            problem = payload.get("problem", {})
            record["preparation"] = {
                "initializer_kind": payload.get("initializer_kind"),
                "state_spec_id": payload.get("state_spec_id"),
                "selected_factor_ids": problem.get("selected_factor_ids"),
                "case_dispatch_used": payload.get("case_dispatch_used"),
                "human_state_optimized": payload.get("human_state_optimized"),
                "accepted_outputs_written": payload.get("accepted_outputs_written"),
            }
        elif artifact_path.name == "generic_sequence_solver_attempts":
            statuses = sorted(artifact_path.glob("*/status.json"))
            if len(statuses) != 1:
                raise ValueError("generic candidate requires exactly one attempt status")
            payload = json.loads(statuses[0].read_text())
            record["attempt"] = {
                "factor_ids": payload.get("factor_ids"),
                "function_evaluations": payload.get("function_evaluations"),
                "solver_executed": True,
                "case_dispatch_used": payload.get("case_dispatch_used"),
                "accepted_outputs_written": payload.get("accepted_outputs_written"),
            }
        elif artifact_path.name.endswith("attempt.json"):
            record["attempt"] = _attempt_summary(artifact_path)
        elif artifact_path.suffix == ".json":
            record["sha256"] = _sha256(artifact_path)
        artifacts[str(artifact_name)] = record
    return {
        "status": manifest["status"],
        "eligible_for_candidate_sandbox": manifest["eligible_for_candidate_sandbox"],
        "nonblocking_gap_ids": manifest["nonblocking_gap_ids"],
        "planned_artifacts": manifest["planned_artifacts"],
        "artifacts": artifacts,
    }


def build_materialized_candidate_summary(
    candidate_root: Path,
    *,
    result_name: str = "benchmark_vlm_qwen",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "result_name": result_name,
        "cases": {
            case_name: materialized_candidate_case_summary(
                case_name,
                candidate_root / f"{result_name}_{case_name}",
                result_name=result_name,
            )
            for case_name in CANONICAL_CASE_DIRECTORIES
        },
    }


def verify_materialized_candidate_summary(
    expected_path: Path = DEFAULT_MATERIALIZED_CANDIDATE_GOLDEN,
    *,
    candidate_root: Path,
    result_name: str = "benchmark_vlm_qwen",
) -> list[str]:
    expected = json.loads(expected_path.read_text())
    try:
        actual = build_materialized_candidate_summary(candidate_root, result_name=result_name)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return [f"materialized candidate artifacts are incomplete: {exc}"]
    if actual == expected:
        return []
    errors: list[str] = []
    for case_name, expected_case in expected.get("cases", {}).items():
        actual_case = actual.get("cases", {}).get(case_name)
        if actual_case is None:
            errors.append(f"{case_name}: missing actual materialized candidate summary")
            continue
        for key, expected_value in expected_case.items():
            actual_value = actual_case.get(key)
            if actual_value != expected_value:
                errors.append(f"{case_name}:{key}: expected {expected_value!r}, got {actual_value!r}")
    for case_name in set(actual.get("cases", {})) - set(expected.get("cases", {})):
        errors.append(f"{case_name}: unexpected actual materialized candidate summary")
    if actual.get("result_name") != expected.get("result_name"):
        errors.append(f"result_name: expected {expected.get('result_name')!r}, got {actual.get('result_name')!r}")
    return errors
