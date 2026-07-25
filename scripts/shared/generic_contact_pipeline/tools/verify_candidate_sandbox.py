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
)
from scripts.shared.generic_contact_pipeline.core.state.golden import CANONICAL_CASE_DIRECTORIES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify five-case candidate sandbox summaries.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CANDIDATE_SANDBOX_GOLDEN)
    parser.add_argument("--result-name", default="benchmark_vlm_qwen")
    args = parser.parse_args()

    actual = build_canonical_candidate_sandbox_summary(result_name=args.result_name)
    errors = verify_candidate_sandbox_summary(args.manifest, result_name=args.result_name)
    for case_name in CANONICAL_CASE_DIRECTORIES:
        summary = actual["cases"][case_name]
        print(
            f"{case_name}: status={summary['status']} eligible={summary['eligible_for_candidate_sandbox']} "
            f"candidate_dir={summary['candidate_dir']} {summary['canonical_sha256']}"
        )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
