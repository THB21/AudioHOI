#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[4]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.base.config import (  # noqa: E402
    load_case_profile,
    with_runtime_overrides,
)
from scripts.shared.generic_contact_pipeline.core.contracts.stage_artifacts import (  # noqa: E402
    validate_stage_contracts,
)


CASES = ("basketball", "football", "mug", "chair", "stick")
STAGES = ("stage1", "stage2", "stage3", "stage4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify typed Stage 1-4 artifact contracts.")
    parser.add_argument("--case", choices=("all", *CASES), default="all")
    parser.add_argument("--result-name", default="benchmark_vlm_qwen")
    args = parser.parse_args()

    cases = CASES if args.case == "all" else (args.case,)
    errors: list[str] = []
    for case_name in cases:
        profile = with_runtime_overrides(
            load_case_profile(case_name), result_name=args.result_name
        )
        for stage_name in STAGES:
            audit = validate_stage_contracts(profile, stage_name)
            print(f"{case_name}:{stage_name}: {audit['status']}")
            errors.extend(
                f"{case_name}:{stage_name}: {error}" for error in audit["errors"]
            )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
