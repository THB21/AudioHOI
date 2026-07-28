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
    DEFAULT_CANDIDATE_SANDBOX_GOLDEN,
    build_canonical_candidate_sandbox_summary,
    verify_candidate_sandbox_summary,
    write_candidate_sandbox_manifest,
    verify_materialized_chair_factor_candidate,
)
from scripts.shared.generic_contact_pipeline.core.base.config import load_case_profile, with_runtime_overrides  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.base.io import REPO  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.state.golden import CANONICAL_CASE_DIRECTORIES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify five-case candidate sandbox summaries.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CANDIDATE_SANDBOX_GOLDEN)
    parser.add_argument("--result-name", default="benchmark_vlm_qwen")
    parser.add_argument(
        "--materialize-chair-candidates",
        action="store_true",
        help="Materialize and verify chair safe candidate artifacts in candidate-root.",
    )
    parser.add_argument("--candidate-root", type=Path, help="Root for materialized candidate artifacts.")
    args = parser.parse_args()
    if args.materialize_chair_candidates and args.candidate_root is None:
        raise SystemExit("--materialize-chair-candidates requires --candidate-root")

    actual = build_canonical_candidate_sandbox_summary(result_name=args.result_name)
    errors = verify_candidate_sandbox_summary(args.manifest, result_name=args.result_name)
    for case_name in CANONICAL_CASE_DIRECTORIES:
        summary = actual["cases"][case_name]
        materialized_note = ""
        if args.materialize_chair_candidates and case_name == "chair" and summary["eligible_for_candidate_sandbox"] is True:
            profile = with_runtime_overrides(load_case_profile(case_name), result_name=args.result_name)
            result_dir = REPO / "samples_known_object" / CANONICAL_CASE_DIRECTORIES[case_name] / "results" / args.result_name
            candidate_dir = args.candidate_root / f"{args.result_name}_{case_name}"
            write_candidate_sandbox_manifest(profile, result_dir, candidate_dir)
            candidate_errors = verify_materialized_chair_factor_candidate(profile, result_dir, candidate_dir)
            errors.extend(f"{case_name}: {error}" for error in candidate_errors)
            materialized_note = f" chair_materialized={not candidate_errors}"
        nonblocking = ",".join(summary["nonblocking_gap_ids"])
        print(
            f"{case_name}: status={summary['status']} eligible={summary['eligible_for_candidate_sandbox']} "
            f"candidate_dir={summary['candidate_dir']} nonblocking_gaps=[{nonblocking}] "
            f"{summary['canonical_sha256']}{materialized_note}"
        )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
