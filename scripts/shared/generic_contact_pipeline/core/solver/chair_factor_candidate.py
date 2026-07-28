from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..base.config import CaseProfile
from ..base.io import repo_relative_value, write_json
from ..factors import build_chair_factor_executor_bundle, validate_chair_factor_executor_bundle
from .candidate import ACCEPTED_OUTPUT_NAMES
from .chair_diagnostics import build_chair_contact_diagnostics, validate_chair_contact_diagnostics


CHAIR_FACTOR_ATTEMPT_NAME = "chair_generic_factor_executor_attempt.json"


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_chair_factor_executor_candidate(profile: CaseProfile, result_dir: Path, candidate_dir: Path) -> dict[str, object]:
    if profile.case_name != "chair":
        raise ValueError("chair factor executor candidate only supports the chair case")
    bundle = build_chair_factor_executor_bundle(profile, result_dir)
    diagnostics = build_chair_contact_diagnostics(result_dir)
    bundle_errors = validate_chair_factor_executor_bundle(bundle)
    diagnostics_errors = validate_chair_contact_diagnostics(diagnostics)
    status = str(bundle["status"])
    solver_executed = status == "ready_for_candidate_executor"
    # This branch deliberately has no chair generic optimizer yet. If a future
    # branch marks the bundle ready, it must replace this guard with a real
    # isolated executor before solver_executed can become true.
    if solver_executed:
        status = "blocked_by_missing_executor_implementation"
        solver_executed = False
    core = {
        "bundle_sha256": bundle["canonical_sha256"],
        "contact_diagnostics_sha256": diagnostics["canonical_sha256"],
        "status": status,
        "missing_required_factor_kinds": bundle["missing_required_factor_kinds"],
        "compatibility_gap_id": bundle["compatibility_gap_id"],
        "compatibility_gap_status": bundle["compatibility_gap_status"],
        "validation_error_count": len(bundle_errors) + len(diagnostics_errors),
        "candidate_dir": str(repo_relative_value(candidate_dir)),
    }
    canonical = _canonical_hash(core)
    return {
        "schema_version": 1,
        "mode": "chair_generic_factor_executor_candidate_attempt",
        "attempt_id": f"chair-factor-{canonical[:12]}",
        "sample_id": "chair",
        "canonical_result_dir": str(repo_relative_value(result_dir)),
        "candidate_dir": str(repo_relative_value(candidate_dir)),
        "status": status,
        "solver_executed": solver_executed,
        "accepted_outputs_written": False,
        "baseline_pose_read": False,
        "missing_required_factor_kinds": bundle["missing_required_factor_kinds"],
        "bundle": {
            "canonical_sha256": bundle["canonical_sha256"],
            "status": bundle["status"],
            "available_factor_kinds": bundle["available_factor_kinds"],
            "missing_required_factor_kinds": bundle["missing_required_factor_kinds"],
        },
        "contact_diagnostics": {
            "canonical_sha256": diagnostics["canonical_sha256"],
            "seed_policy": diagnostics["summary"]["seed_policy"],
            "compatibility_gap_status": diagnostics["compatibility_gap_status"],
        },
        "blocking_reasons": list(bundle.get("blocking_reasons", [])) + ["generic chair factor executor not implemented in this branch"],
        "validation_errors": bundle_errors + diagnostics_errors,
        "planned_outputs": [CHAIR_FACTOR_ATTEMPT_NAME],
        "forbidden_artifact_names": sorted(ACCEPTED_OUTPUT_NAMES),
        "canonical_sha256": canonical,
    }


def validate_chair_factor_executor_candidate(attempt: dict[str, object]) -> list[str]:
    errors: list[str] = []
    if attempt.get("mode") != "chair_generic_factor_executor_candidate_attempt":
        errors.append("chair factor candidate mode must be chair_generic_factor_executor_candidate_attempt")
    if attempt.get("accepted_outputs_written") is not False:
        errors.append("chair factor candidate must not write accepted outputs")
    if attempt.get("baseline_pose_read") is not False:
        errors.append("chair factor candidate must not read baseline pose")
    planned = attempt.get("planned_outputs", [])
    if not isinstance(planned, list) or planned != [CHAIR_FACTOR_ATTEMPT_NAME]:
        errors.append("chair factor candidate must only plan the safe attempt manifest")
    forbidden = set(ACCEPTED_OUTPUT_NAMES)
    overlap = sorted(str(item) for item in planned if Path(str(item)).name in forbidden) if isinstance(planned, list) else []
    if overlap:
        errors.append(f"chair factor candidate planned accepted output names: {overlap}")
    if attempt.get("status") != "ready_for_candidate_executor" and attempt.get("solver_executed") is not False:
        errors.append("blocked chair factor candidate must not execute solver")
    if attempt.get("candidate_dir") == attempt.get("canonical_result_dir"):
        errors.append("chair factor candidate dir must not equal canonical result dir")
    return errors


def prepare_chair_factor_executor_candidate(profile: CaseProfile, result_dir: Path, candidate_dir: Path) -> dict[str, object]:
    attempt = build_chair_factor_executor_candidate(profile, result_dir, candidate_dir)
    errors = validate_chair_factor_executor_candidate(attempt)
    if errors:
        raise ValueError("; ".join(errors))
    candidate_dir.mkdir(parents=True, exist_ok=True)
    write_json(candidate_dir / CHAIR_FACTOR_ATTEMPT_NAME, attempt)
    return attempt
