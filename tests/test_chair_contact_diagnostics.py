from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.shared.generic_contact_pipeline.core.base.config import load_case_profile, with_runtime_overrides
from scripts.shared.generic_contact_pipeline.core.solver import (
    build_chair_contact_diagnostics,
    validate_chair_contact_diagnostics,
)


REPO = Path(__file__).resolve().parents[1]


def _diagnostics() -> dict[str, object]:
    profile = with_runtime_overrides(load_case_profile("chair"), result_name="benchmark_vlm_qwen")
    return build_chair_contact_diagnostics(profile.result_dir)


def test_chair_contact_diagnostics_marks_current_run_seed_gap_nonblocking() -> None:
    diagnostics = _diagnostics()
    assert diagnostics["mode"] == "generic_chair_contact_diagnostics"
    assert diagnostics["solver_executed"] is False
    assert diagnostics["accepted_outputs_written"] is False
    assert diagnostics["baseline_pose_read"] is False
    assert diagnostics["compatibility_gap_id"] == "semantic_graph_solver_private"
    assert diagnostics["compatibility_gap_status"] == "nonblocking"
    summary = diagnostics["summary"]
    assert summary["seed_policy"] == "current_stage3_observation_fit"
    assert summary["seed_is_current_run"] is True
    assert summary["historical_seed_reference_fields"] == []
    assert validate_chair_contact_diagnostics(diagnostics) == []


def test_chair_contact_diagnostics_capture_pairprop_contact_quality() -> None:
    summary = _diagnostics()["summary"]
    assert summary["pairprop_pose_accepted"] is True
    assert summary["active_frames"] == 125
    assert summary["metrics_rows"] == 125
    assert summary["median_optimized_contact_gap_m"] >= 0.0
    assert summary["p90_optimized_contact_gap_m"] >= summary["median_optimized_contact_gap_m"]


def test_chair_contact_diagnostics_validator_rejects_false_gap_closure() -> None:
    diagnostics = _diagnostics()
    diagnostics["compatibility_gap_status"] = "nonblocking"
    diagnostics["summary"]["seed_is_current_run"] = False
    diagnostics["summary"]["median_optimized_contact_gap_m"] = -1.0
    diagnostics["solver_executed"] = True
    errors = validate_chair_contact_diagnostics(diagnostics)
    assert any("must not execute" in error for error in errors)
    assert any("must keep the compatibility gap blocking" in error for error in errors)
    assert any("nonnegative contact gap" in error for error in errors)


def test_chair_contact_diagnostics_cli_reports_chair(tmp_path: Path) -> None:
    out = tmp_path / "chair_diagnostics.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/verify_chair_contact_diagnostics.py",
            "--out",
            str(out),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )
    assert completed.stdout.startswith("chair: contact_diagnostics active=125")
    payload = json.loads(out.read_text())
    assert payload["mode"] == "generic_chair_contact_diagnostics"
