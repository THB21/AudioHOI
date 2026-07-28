#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[4]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.base.config import load_case_profile, with_runtime_overrides  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.solver import (  # noqa: E402
    CHAIR_FACTOR_ATTEMPT_NAME,
    default_candidate_dir,
    verify_materialized_chair_factor_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify materialized isolated chair factor candidate artifacts.")
    parser.add_argument("--case", default="chair")
    parser.add_argument("--result-name", default="benchmark_vlm_qwen")
    parser.add_argument("--candidate-dir", type=Path)
    args = parser.parse_args()
    if args.case != "chair":
        raise SystemExit("chair factor candidate verifier currently supports --case chair")
    profile = with_runtime_overrides(load_case_profile(args.case), result_name=args.result_name)
    candidate_dir = args.candidate_dir or default_candidate_dir(profile.result_dir, profile.case_name)
    errors = verify_materialized_chair_factor_candidate(profile, profile.result_dir, candidate_dir)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)
    attempt = json.loads((candidate_dir / CHAIR_FACTOR_ATTEMPT_NAME).read_text())
    print(
        "chair: factor_candidate "
        f"materialized={attempt['isolated_candidate_materialized']} "
        f"pose_rows={attempt['candidate_pose']['rows']} "
        f"residual_rows={attempt['residual_table']['rows']} "
        f"{attempt['canonical_sha256']}"
    )


if __name__ == "__main__":
    main()
