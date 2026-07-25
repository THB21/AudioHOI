#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[4]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.solver import (  # noqa: E402
    DEFAULT_SEQUENCE_DIAGNOSTICS_GOLDEN,
    build_canonical_sequence_solver_diagnostics_summary,
    verify_sequence_solver_diagnostics_summary,
)
from scripts.shared.generic_contact_pipeline.core.state.golden import CANONICAL_CASE_DIRECTORIES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify five-case sequence-solver shadow diagnostics.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_SEQUENCE_DIAGNOSTICS_GOLDEN)
    parser.add_argument("--result-name", default="benchmark_vlm_qwen")
    args = parser.parse_args()

    actual = build_canonical_sequence_solver_diagnostics_summary(result_name=args.result_name)
    errors = verify_sequence_solver_diagnostics_summary(args.manifest, result_name=args.result_name)
    for case_name in CANONICAL_CASE_DIRECTORIES:
        summary = actual["cases"][case_name]
        gaps = ",".join(summary["blocking_gap_ids"])
        nonblocking = ",".join(summary["nonblocking_gap_ids"])
        print(
            f"{case_name}: status={summary['status']} attempt={summary['attempt_id']} "
            f"blocking_gaps=[{gaps}] nonblocking_gaps=[{nonblocking}] {summary['canonical_sha256']}"
        )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
