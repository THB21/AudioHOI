#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[4]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.base.config import load_case_profile, with_runtime_overrides  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.evaluation.final_hoi.summary_writer import (  # noqa: E402
    run_unified_final_evaluation,
    write_unified_final_summary,
)


DEFAULT_CASES = ["basketball", "football", "mug", "chair", "stick"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run unified final HOI hard-metric evaluator.")
    ap.add_argument("--case", default="", help="Single case. If omitted, --cases or default five cases are used.")
    ap.add_argument("--cases", nargs="+", default=None)
    ap.add_argument("--result-name", default="benchmark_vlm_qwen")
    ap.add_argument("--output-dir", default="samples_known_object/final_result_evaluation")
    args = ap.parse_args()

    case_names = [args.case] if args.case else (args.cases or DEFAULT_CASES)
    profiles = [with_runtime_overrides(load_case_profile(case), result_name=args.result_name) for case in case_names]
    if len(profiles) == 1:
        result = run_unified_final_evaluation(profiles[0])
        print(f"[final-hoi] case={result['case']} summary={profiles[0].result_dir / 'evaluation' / 'final_evaluation_summary.json'}")
    else:
        result = write_unified_final_summary(profiles, output_dir=Path(args.output_dir))
        print(f"[final-hoi] rows={result['rows']} table={result['table']}")


if __name__ == "__main__":
    main()
