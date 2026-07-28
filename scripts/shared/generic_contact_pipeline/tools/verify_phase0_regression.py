#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[4]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def _run_gate(name: str, command: list[str]) -> None:
    completed = subprocess.run(
        command,
        cwd=REPO,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        print(f"gate={name} status=fail returncode={completed.returncode}", file=sys.stderr)
        if completed.stdout:
            print(completed.stdout, file=sys.stderr)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        raise SystemExit(completed.returncode)
    print(f"gate={name} status=pass")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 0 no-regression gate bundle.")
    parser.add_argument(
        "--candidate-root",
        type=Path,
        required=True,
        help="Directory where isolated candidate artifacts are materialized.",
    )
    parser.add_argument(
        "--materialized-golden",
        type=Path,
        default=Path("tests/golden/sequence_candidate_materialized_v1.json"),
        help="Frozen materialized candidate output manifest.",
    )
    args = parser.parse_args()

    args.candidate_root.mkdir(parents=True, exist_ok=True)
    _run_gate(
        "pytest_sequence_solver_shadow",
        [sys.executable, "-m", "pytest", "-q", "tests/test_sequence_solver_shadow.py"],
    )
    _run_gate(
        "golden_manifest",
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/manage_golden_manifest.py",
            "--verify",
            "--skip-decoded-renders",
        ],
    )
    _run_gate(
        "candidate_sandbox_summary",
        [sys.executable, "scripts/shared/generic_contact_pipeline/tools/verify_candidate_sandbox.py"],
    )
    _run_gate(
        "materialized_candidate_golden",
        [
            sys.executable,
            "scripts/shared/generic_contact_pipeline/tools/verify_candidate_sandbox.py",
            "--materialize-all-candidates",
            "--candidate-root",
            str(args.candidate_root),
            "--materialized-golden",
            str(args.materialized_golden),
        ],
    )
    print("phase0_regression_verified=True")


if __name__ == "__main__":
    main()
