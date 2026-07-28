from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.shared.generic_contact_pipeline.core.base.config import load_case_profile, with_runtime_overrides
from scripts.shared.generic_contact_pipeline.core.solver import (
    build_line_contact_diagnostics,
    validate_line_contact_diagnostics,
)


REPO = Path(__file__).resolve().parents[1]


def _diagnostics() -> dict[str, object]:
    profile = with_runtime_overrides(load_case_profile("stick"), result_name="benchmark_vlm_qwen")
    return build_line_contact_diagnostics(profile.result_dir)


def test_line_contact_diagnostics_are_read_only_and_nonblocking() -> None:
    diagnostics = _diagnostics()
    assert diagnostics["mode"] == "generic_line_contact_diagnostics"
    assert diagnostics["solver_executed"] is False
    assert diagnostics["accepted_outputs_written"] is False
    assert diagnostics["baseline_pose_read"] is False
    assert diagnostics["compatibility_gap_id"] == "line_contact_lock_special_refinement"
    assert diagnostics["compatibility_gap_status"] == "nonblocking"
    assert validate_line_contact_diagnostics(diagnostics) == []


def test_line_contact_diagnostics_capture_stick_residual_acceptance_summary() -> None:
    summary = _diagnostics()["summary"]
    assert summary["pose_rows"] == 240
    assert summary["contact_rows"] == 480
    assert summary["blend_rows"] == 240
    assert summary["solved_frames"] == 240
    assert summary["skipped_frames"] == 0
    assert summary["temporal_filled_frames"] == 0
    assert summary["se3_refined_frames"] == 240
    assert summary["overlay_err_px"]["max"] > 0
    assert summary["contact_gap_m"]["max"] > 0
    assert summary["line_s"]["source_counts"]


def test_line_contact_diagnostics_validator_rejects_gap_closure_and_bad_line_s() -> None:
    diagnostics = _diagnostics()
    diagnostics["compatibility_gap_status"] = "closed"
    diagnostics["solver_executed"] = True
    diagnostics["summary"]["line_s"]["object_local_s"]["max"] = 42.0
    errors = validate_line_contact_diagnostics(diagnostics)
    assert any("must not execute" in error for error in errors)
    assert any("nonblocking compatibility gap" in error for error in errors)
    assert any("object_local_s must stay normalized" in error for error in errors)


def test_line_contact_diagnostics_cli_reports_stick(tmp_path: Path) -> None:
    out = tmp_path / "line_diagnostics.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/verify_line_contact_diagnostics.py",
            "--out",
            str(out),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )
    assert completed.stdout.startswith("stick: line_contact_diagnostics rows=240")
    payload = json.loads(out.read_text())
    assert payload["mode"] == "generic_line_contact_diagnostics"
