from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[5]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.base.config import available_cases, load_case_profile, with_runtime_overrides  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.evaluation.llm_csv_audit import run_llm_csv_audit  # noqa: E402


def run(profile, mode: str = "seed"):
    return run_llm_csv_audit(profile, mode)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run LLM-style CSV/data audit over generic pipeline outputs.")
    ap.add_argument("--case", default="all", help="Case name or all")
    ap.add_argument("--result-name", default="", help="Override case profile result_name.")
    ap.add_argument("--llm-mode", default="seed", choices=["none", "seed", "qwen", "mistral"])
    args = ap.parse_args()
    cases = available_cases() if args.case == "all" else [args.case]
    failed = False
    for case in cases:
        try:
            profile = with_runtime_overrides(load_case_profile(case), result_name=args.result_name or None)
            result = run(profile, args.llm_mode)
            print(f"{case}: {result['summary']}")
        except Exception as exc:
            failed = True
            print(f"{case}: FAILED: {exc}", file=sys.stderr)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
