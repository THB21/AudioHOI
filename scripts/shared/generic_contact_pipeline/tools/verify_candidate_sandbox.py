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
    verify_materialized_candidate_summary,
    write_candidate_sandbox_manifest,
    verify_materialized_chair_factor_candidate,
    verify_materialized_line_contact_candidate,
    verify_materialized_sphere_sequence_candidate,
    verify_materialized_projected_periodic_candidate,
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
    parser.add_argument(
        "--materialize-mug-candidates",
        action="store_true",
        help="Materialize and verify mug projected-periodic safe candidate artifacts in candidate-root.",
    )
    parser.add_argument(
        "--materialize-sphere-candidates",
        action="store_true",
        help="Materialize and verify basketball/football sphere safe candidate artifacts in candidate-root.",
    )
    parser.add_argument(
        "--materialize-all-candidates",
        action="store_true",
        help="Materialize and verify all supported safe candidate artifacts in candidate-root.",
    )
    parser.add_argument("--candidate-root", type=Path, help="Root for materialized candidate artifacts.")
    parser.add_argument(
        "--materialized-golden",
        type=Path,
        default=None,
        help="Verify materialized candidate artifact hashes against this frozen manifest.",
    )
    args = parser.parse_args()
    if (
        args.materialize_chair_candidates
        or args.materialize_mug_candidates
        or args.materialize_sphere_candidates
        or args.materialize_all_candidates
        or args.materialized_golden is not None
    ) and args.candidate_root is None:
        raise SystemExit("--materialize-* candidates requires --candidate-root")

    actual = build_canonical_candidate_sandbox_summary(result_name=args.result_name)
    errors = verify_candidate_sandbox_summary(args.manifest, result_name=args.result_name)
    for case_name in CANONICAL_CASE_DIRECTORIES:
        summary = actual["cases"][case_name]
        materialized_note = ""
        materialize_chair = args.materialize_chair_candidates or args.materialize_all_candidates
        materialize_mug = args.materialize_mug_candidates or args.materialize_all_candidates
        materialize_sphere = args.materialize_sphere_candidates or args.materialize_all_candidates
        materialize_line_contact = args.materialize_all_candidates
        if materialize_chair and case_name == "chair" and summary["eligible_for_candidate_sandbox"] is True:
            profile = with_runtime_overrides(load_case_profile(case_name), result_name=args.result_name)
            result_dir = REPO / "samples_known_object" / CANONICAL_CASE_DIRECTORIES[case_name] / "results" / args.result_name
            candidate_dir = args.candidate_root / f"{args.result_name}_{case_name}"
            write_candidate_sandbox_manifest(profile, result_dir, candidate_dir)
            candidate_errors = verify_materialized_chair_factor_candidate(profile, result_dir, candidate_dir)
            errors.extend(f"{case_name}: {error}" for error in candidate_errors)
            materialized_note = f" chair_materialized={not candidate_errors}"
        if materialize_mug and case_name == "mug" and summary["eligible_for_candidate_sandbox"] is True:
            profile = with_runtime_overrides(load_case_profile(case_name), result_name=args.result_name)
            result_dir = REPO / "samples_known_object" / CANONICAL_CASE_DIRECTORIES[case_name] / "results" / args.result_name
            candidate_dir = args.candidate_root / f"{args.result_name}_{case_name}"
            write_candidate_sandbox_manifest(profile, result_dir, candidate_dir)
            candidate_errors = verify_materialized_projected_periodic_candidate(candidate_dir)
            errors.extend(f"{case_name}: {error}" for error in candidate_errors)
            materialized_note = f" mug_materialized={not candidate_errors}"
        if (
            materialize_sphere
            and case_name in {"basketball", "football"}
            and summary["eligible_for_candidate_sandbox"] is True
        ):
            profile = with_runtime_overrides(load_case_profile(case_name), result_name=args.result_name)
            result_dir = REPO / "samples_known_object" / CANONICAL_CASE_DIRECTORIES[case_name] / "results" / args.result_name
            candidate_dir = args.candidate_root / f"{args.result_name}_{case_name}"
            write_candidate_sandbox_manifest(profile, result_dir, candidate_dir)
            candidate_errors = verify_materialized_sphere_sequence_candidate(candidate_dir)
            errors.extend(f"{case_name}: {error}" for error in candidate_errors)
            materialized_note = f" {case_name}_materialized={not candidate_errors}"
        if materialize_line_contact and case_name == "stick" and summary["eligible_for_candidate_sandbox"] is True:
            profile = with_runtime_overrides(load_case_profile(case_name), result_name=args.result_name)
            result_dir = REPO / "samples_known_object" / CANONICAL_CASE_DIRECTORIES[case_name] / "results" / args.result_name
            candidate_dir = args.candidate_root / f"{args.result_name}_{case_name}"
            write_candidate_sandbox_manifest(profile, result_dir, candidate_dir)
            candidate_errors = verify_materialized_line_contact_candidate(candidate_dir)
            errors.extend(f"{case_name}: {error}" for error in candidate_errors)
            materialized_note = f" stick_materialized={not candidate_errors}"
        nonblocking = ",".join(summary["nonblocking_gap_ids"])
        print(
            f"{case_name}: status={summary['status']} eligible={summary['eligible_for_candidate_sandbox']} "
            f"candidate_dir={summary['candidate_dir']} nonblocking_gaps=[{nonblocking}] "
            f"{summary['canonical_sha256']}{materialized_note}"
        )
    if args.materialized_golden is not None:
        materialized_errors = verify_materialized_candidate_summary(
            args.materialized_golden,
            candidate_root=args.candidate_root,
            result_name=args.result_name,
        )
        errors.extend(materialized_errors)
        if not materialized_errors:
            print("materialized_golden_verified=True")
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
