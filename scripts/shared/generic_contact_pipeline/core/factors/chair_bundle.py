from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..base.config import CaseProfile
from ..base.io import repo_relative_value
from .shadow import build_factor_shadow


CHAIR_FACTOR_BUNDLE_MODE = "chair_generic_factor_executor_bundle_shadow"
CHAIR_REQUIRED_EXECUTOR_FACTOR_KINDS = (
    "point_reprojection",
    "contact_distance",
    "joint_limit",
    "gauge_constraint",
)


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_chair_factor_executor_bundle(profile: CaseProfile, result_dir: Path) -> dict[str, object]:
    """Build a read-only readiness contract for chair generic factor execution.

    This does not run optimization. It records whether the existing factor
    shadow has enough typed factor coverage to replace the chair-private
    semantic graph solver in a later candidate branch.
    """
    if profile.case_name != "chair":
        raise ValueError("chair factor executor bundle only supports the chair case")
    factor_shadow = build_factor_shadow(profile, result_dir)
    by_kind = factor_shadow.get("factors", {}).get("by_kind", {})
    available = sorted(str(kind) for kind, count in by_kind.items() if int(count or 0) > 0) if isinstance(by_kind, dict) else []
    missing = sorted(kind for kind in CHAIR_REQUIRED_EXECUTOR_FACTOR_KINDS if kind not in available)
    gap_ids = [str(gap.get("gap_id")) for gap in factor_shadow.get("gaps", []) if isinstance(gap, dict)]
    private_gap_present = "semantic_graph_solver_private" in gap_ids
    ready = not missing and not private_gap_present
    if ready:
        status = "ready_for_candidate_executor"
    elif missing:
        status = "blocked_by_missing_factor_contracts"
    else:
        status = "blocked_by_private_solver_gap"
    gap_status = "nonblocking" if ready else "blocking"
    payload = {
        "schema_version": 1,
        "mode": CHAIR_FACTOR_BUNDLE_MODE,
        "sample_id": "chair",
        "result_dir": str(repo_relative_value(result_dir)),
        "solver_executed": False,
        "accepted_outputs_written": False,
        "baseline_pose_read": False,
        "status": status,
        "compatibility_gap_id": "semantic_graph_solver_private",
        "compatibility_gap_status": gap_status,
        "required_factor_kinds": list(CHAIR_REQUIRED_EXECUTOR_FACTOR_KINDS),
        "available_factor_kinds": available,
        "missing_required_factor_kinds": missing,
        "factor_shadow_sha256": factor_shadow["canonical_sha256"],
        "blocking_reasons": [
            reason
            for reason in (
                "missing required factors: " + ",".join(missing) if missing else "",
                "semantic_graph_solver_private still present in factor gap ledger" if private_gap_present else "",
            )
            if reason
        ],
        "policy": "read-only factor executor readiness contract; no accepted output writes and no solver execution",
    }
    payload["canonical_sha256"] = _canonical_hash(
        {
            "mode": payload["mode"],
            "sample_id": payload["sample_id"],
            "status": payload["status"],
            "compatibility_gap_status": payload["compatibility_gap_status"],
            "required_factor_kinds": payload["required_factor_kinds"],
            "available_factor_kinds": payload["available_factor_kinds"],
            "missing_required_factor_kinds": payload["missing_required_factor_kinds"],
            "factor_shadow_sha256": payload["factor_shadow_sha256"],
        }
    )
    return payload


def validate_chair_factor_executor_bundle(bundle: dict[str, object]) -> list[str]:
    errors: list[str] = []
    if bundle.get("mode") != CHAIR_FACTOR_BUNDLE_MODE:
        errors.append(f"mode must be {CHAIR_FACTOR_BUNDLE_MODE}")
    if bundle.get("solver_executed") is not False:
        errors.append("chair factor bundle must not execute the solver")
    if bundle.get("accepted_outputs_written") is not False:
        errors.append("chair factor bundle must not write accepted outputs")
    if bundle.get("baseline_pose_read") is not False:
        errors.append("chair factor bundle must not read baseline pose")
    if bundle.get("compatibility_gap_id") != "semantic_graph_solver_private":
        errors.append("chair factor bundle must track semantic_graph_solver_private")

    missing = bundle.get("missing_required_factor_kinds", [])
    if not isinstance(missing, list):
        errors.append("missing required factors must be a list")
        missing = []
    status = bundle.get("status")
    gap_status = bundle.get("compatibility_gap_status")
    if missing and status == "ready_for_candidate_executor":
        errors.append("chair factor bundle cannot be ready with missing required factors")
    if not missing and status == "blocked_by_missing_factor_contracts":
        errors.append("chair factor bundle cannot report missing-factor status when no required factors are missing")
    if missing and gap_status != "blocking":
        errors.append("chair factor bundle with missing required factors must remain blocking")
    blocking_reasons = bundle.get("blocking_reasons", [])
    private_gap_present = isinstance(blocking_reasons, list) and any("semantic_graph_solver_private" in str(reason) for reason in blocking_reasons)
    if private_gap_present and gap_status != "blocking":
        errors.append("chair factor bundle with semantic_graph_solver_private must remain blocking")
    if gap_status == "nonblocking" and status != "ready_for_candidate_executor":
        errors.append("nonblocking gap requires ready_for_candidate_executor status")
    required = bundle.get("required_factor_kinds", [])
    if list(required) != list(CHAIR_REQUIRED_EXECUTOR_FACTOR_KINDS):
        errors.append("chair factor bundle required factor contract changed unexpectedly")
    return errors
