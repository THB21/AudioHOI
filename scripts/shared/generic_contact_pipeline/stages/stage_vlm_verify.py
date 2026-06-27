from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[4]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.config import available_cases, load_case_profile, with_runtime_overrides  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.vlm import STAGE_QUERY_TYPES, write_stage_verification  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.vlm_gates import write_stage_gates  # noqa: E402


def run(profile, stage: str) -> dict[str, object]:
    if stage not in STAGE_QUERY_TYPES:
        raise ValueError(f"Unknown VLM stage {stage}; expected one of {sorted(STAGE_QUERY_TYPES)}")
    decision = write_stage_verification(profile, stage)
    gate_decision = write_stage_gates(profile, stage)
    return {**decision, "gate_decision": gate_decision}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate conservative VLM verification queries/results in dry-run mode.")
    ap.add_argument("--case", default="all", help="Case name or all")
    ap.add_argument("--stage", default="all", help="Pipeline stage or all")
    ap.add_argument("--result-name", default="", help="Override case profile result_name.")
    ap.add_argument("--dry-run", action="store_true", help="Reserved flag; current implementation is always dry-run.")
    args = ap.parse_args()

    cases = available_cases() if args.case == "all" else [args.case]
    stages = list(STAGE_QUERY_TYPES) if args.stage == "all" else [args.stage]
    failed = False
    for case in cases:
        profile = with_runtime_overrides(load_case_profile(case), result_name=args.result_name or None)
        for stage in stages:
            try:
                decision = run(profile, stage)
                print(f"{case} {stage}: {decision['decision']} ({decision['query_count']} queries)")
            except Exception as exc:
                failed = True
                print(f"{case} {stage}: FAILED: {exc}", file=sys.stderr)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
