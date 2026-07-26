from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from ..base.io import REPO
from ..state.golden import CANONICAL_CASE_DIRECTORIES
from .sphere_sequence import SPHERE_ATTEMPT_NAME, SPHERE_CANDIDATE_NAME, SPHERE_RESIDUAL_NAME


DEFAULT_SPHERE_SEQUENCE_GOLDEN = REPO / "tests/golden/sphere_sequence_migration_v1.json"
SPHERE_CASES = ("basketball", "football")
PRIMARY_RENDER_ARTIFACTS = (
    "ball/overlay.mp4",
    "ball/camera3d.mp4",
    "ball/side_yz.mp4",
    "with_human/overlay.mp4",
    "with_human/camera3d.mp4",
    "with_human/side_yz.mp4",
)
LOSS_TERMS = ("E_total", "E_visual", "E_mask", "E_contact", "E_support", "E_smooth", "E_reg")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv_rows(path: Path) -> int:
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _loss_term_summary(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        term: {
            "sum": round(sum(float(row.get(term) or 0.0) for row in rows), 12),
            "max": round(max(float(row.get(term) or 0.0) for row in rows), 12),
        }
        for term in LOSS_TERMS
    }


def sphere_sequence_case_summary(case_name: str, result_name: str) -> dict[str, object]:
    directory = CANONICAL_CASE_DIRECTORIES[case_name]
    result_dir = REPO / "samples_known_object" / directory / "results" / result_name
    canonical_dir = REPO / "samples_known_object" / directory / "results/benchmark_vlm_qwen"
    candidate_dir = result_dir / "generic_sphere_sequence_candidate"
    candidate_path = candidate_dir / SPHERE_CANDIDATE_NAME
    residual_path = candidate_dir / SPHERE_RESIDUAL_NAME
    attempt = json.loads((candidate_dir / SPHERE_ATTEMPT_NAME).read_text())
    pre_smooth_path = result_dir / "object_pose_pre_smooth.csv"
    final_path = result_dir / "object_pose.csv"
    contact_path = result_dir / "object_contact_points.csv"
    reference_path = canonical_dir / "object_pose_pre_smooth.csv"
    render_dir = result_dir.parent / "renders" / result_name
    stage4_audit = json.loads((result_dir / "stage_audit/stage4/stage_audit_decision.json").read_text())
    stage5_audit = json.loads((result_dir / "stage_audit/stage5/stage_audit_decision.json").read_text())
    stage6_compare = json.loads((result_dir / "stage6_compare_report.json").read_text())
    pose_delta = stage6_compare["checks"]["pose_delta"]
    return {
        "attempt_id": attempt["attempt_id"],
        "solver_executed": attempt["solver_executed"],
        "baseline_pose_read": attempt["baseline_pose_read"],
        "candidate_accepted_outputs_written": attempt["accepted_outputs_written"],
        "frames": attempt["frames"],
        "human_contact_events": attempt["human_contact_events"],
        "support_events": attempt["support_events"],
        "input_sha256": {name: record["sha256"] for name, record in attempt["inputs"].items()},
        "candidate_sha256": _sha256(candidate_path),
        "candidate_rows": _csv_rows(candidate_path),
        "residual_sha256": _sha256(residual_path),
        "residual_rows": _csv_rows(residual_path),
        "legacy_exact_reference_sha256": _sha256(reference_path),
        "candidate_equals_legacy_exact_reference": candidate_path.read_bytes() == reference_path.read_bytes(),
        "promoted_pre_smooth_sha256": _sha256(pre_smooth_path),
        "candidate_equals_promoted_pre_smooth": candidate_path.read_bytes() == pre_smooth_path.read_bytes(),
        "final_pose_sha256": _sha256(final_path),
        "final_pose_rows": _csv_rows(final_path),
        "contact_sha256": _sha256(contact_path),
        "contact_rows": _csv_rows(contact_path),
        "stage4_audit": {
            "decision": stage4_audit["decision"],
            "blocking_count": stage4_audit["blocking_count"],
            "rerun_stage": stage4_audit["rerun_stage"],
        },
        "stage5_audit": {
            "decision": stage5_audit["decision"],
            "blocking_count": stage5_audit["blocking_count"],
            "rerun_stage": stage5_audit["rerun_stage"],
        },
        "stage6_profile_baseline_semantic_gap": {
            "comparison": "smoothed_final_pose_vs_profile_exact_seed",
            "overall_pass": stage6_compare["checks"]["overall_pass"],
            "pose_delta_pass": stage6_compare["checks"]["pose_delta_pass"],
            "p95_abs_delta": pose_delta["p95_abs_delta"],
            "max_abs_delta": pose_delta["max_abs_delta"],
            "candidate_equals_compared_exact_seed": candidate_path.read_bytes() == reference_path.read_bytes(),
        },
        "stage7_loss_terms": _loss_term_summary(result_dir / "loss_analysis/per_frame_residuals.csv"),
        "render_sha256": {name: _sha256(render_dir / name) for name in PRIMARY_RENDER_ARTIFACTS},
    }


def build_sphere_sequence_regression_summary(
    result_name: str = "generic_sphere_migration_switched_v1",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "result_name": result_name,
        "cases": {case_name: sphere_sequence_case_summary(case_name, result_name) for case_name in SPHERE_CASES},
    }


def verify_sphere_sequence_regression(
    expected_path: Path = DEFAULT_SPHERE_SEQUENCE_GOLDEN,
    *,
    result_name: str | None = None,
) -> list[str]:
    expected = json.loads(expected_path.read_text())
    selected_result = result_name or str(expected["result_name"])
    try:
        actual = build_sphere_sequence_regression_summary(selected_result)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return [f"sphere sequence regression artifacts are incomplete: {exc}"]
    errors: list[str] = []
    if actual != expected:
        for case_name, expected_case in expected.get("cases", {}).items():
            actual_case = actual.get("cases", {}).get(case_name, {})
            for key, expected_value in expected_case.items():
                actual_value = actual_case.get(key)
                if actual_value != expected_value:
                    errors.append(f"{case_name}:{key}: expected {expected_value!r}, got {actual_value!r}")
        if actual.get("result_name") != expected.get("result_name"):
            errors.append(f"result_name: expected {expected.get('result_name')!r}, got {actual.get('result_name')!r}")
    return errors
