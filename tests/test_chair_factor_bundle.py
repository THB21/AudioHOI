from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

from scripts.shared.generic_contact_pipeline.core.base.config import load_case_profile
from scripts.shared.generic_contact_pipeline.core.factors import (
    build_chair_factor_executor_bundle,
    validate_chair_factor_executor_bundle,
)


REPO = Path(__file__).resolve().parents[1]


def test_chair_factor_bundle_keeps_solver_gap_blocking_until_private_solver_gap_is_removed() -> None:
    profile = load_case_profile("chair")
    bundle = build_chair_factor_executor_bundle(profile, REPO / "samples_known_object/05_chair/results/benchmark_vlm_qwen")

    assert bundle["mode"] == "chair_generic_factor_executor_bundle_shadow"
    assert bundle["solver_executed"] is False
    assert bundle["accepted_outputs_written"] is False
    assert bundle["baseline_pose_read"] is False
    assert bundle["status"] == "blocked_by_private_solver_gap"
    assert bundle["compatibility_gap_id"] == "semantic_graph_solver_private"
    assert bundle["missing_required_factor_kinds"] == []
    assert "joint_limit" in bundle["available_factor_kinds"]
    assert "gauge_constraint" in bundle["available_factor_kinds"]
    assert validate_chair_factor_executor_bundle(bundle) == []


def test_chair_factor_bundle_validator_rejects_false_gap_closure() -> None:
    profile = load_case_profile("chair")
    bundle = build_chair_factor_executor_bundle(profile, REPO / "samples_known_object/05_chair/results/benchmark_vlm_qwen")
    bundle["status"] = "ready_for_candidate_executor"
    bundle["compatibility_gap_status"] = "nonblocking"
    bundle["solver_executed"] = True

    errors = validate_chair_factor_executor_bundle(bundle)

    assert any("must not execute" in error for error in errors)
    assert any("semantic_graph_solver_private must remain blocking" in error for error in errors)


def test_chair_factor_bundle_cli_reports_private_solver_gap(tmp_path: Path) -> None:
    out = tmp_path / "chair_factor_bundle.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/verify_chair_factor_bundle.py",
            "--out",
            str(out),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "status=blocked_by_private_solver_gap" in completed.stdout
    assert "missing=none" in completed.stdout
    payload = json.loads(out.read_text())
    assert payload["mode"] == "chair_generic_factor_executor_bundle_shadow"
