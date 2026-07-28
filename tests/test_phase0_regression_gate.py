from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_phase0_regression_gate_runs_core_golden_and_materialized_candidate_checks(tmp_path: Path) -> None:
    candidate_root = tmp_path / "phase0_candidates"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/verify_phase0_regression.py",
            "--candidate-root",
            str(candidate_root),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    stdout = completed.stdout
    assert "gate=pytest_sequence_solver_shadow status=pass" in stdout
    assert "gate=golden_manifest status=pass" in stdout
    assert "gate=candidate_sandbox_summary status=pass" in stdout
    assert "gate=materialized_candidate_golden status=pass" in stdout
    assert "phase0_regression_verified=True" in stdout
