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
from scripts.shared.generic_contact_pipeline.core.solver import prepare_chair_factor_executor_candidate  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.solver.candidate import default_candidate_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an isolated chair generic factor executor candidate attempt.")
    parser.add_argument("--case", default="chair")
    parser.add_argument("--result-name", default="benchmark_vlm_qwen")
    parser.add_argument("--candidate-dir", type=Path)
    args = parser.parse_args()
    if args.case != "chair":
        raise SystemExit("chair factor candidate currently supports --case chair")
    profile = with_runtime_overrides(load_case_profile(args.case), result_name=args.result_name)
    candidate_dir = args.candidate_dir or default_candidate_dir(profile.result_dir, profile.case_name)
    attempt = prepare_chair_factor_executor_candidate(profile, profile.result_dir, candidate_dir)
    print(json.dumps(attempt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
