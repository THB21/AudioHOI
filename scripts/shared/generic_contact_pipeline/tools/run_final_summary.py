#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[4]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.base.config import load_case_profile  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.evaluation.final_summary import write_final_summary  # noqa: E402


DEFAULT_CASES = ["basketball", "football", "mug", "chair", "stick"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a final-only evaluation summary table for current generic pipeline results.")
    ap.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    ap.add_argument("--result-name", default="benchmark_vlm_qwen")
    ap.add_argument("--method", default="final_vlm_gated")
    ap.add_argument("--llm-mode", default="seed", choices=["none", "seed", "mistral"])
    ap.add_argument("--output-dir", default="final_result/evaluation")
    ap.add_argument("--no-rerun-evaluator", action="store_true", help="Reuse existing evaluation_summary.json when present.")
    args = ap.parse_args()

    profiles = [load_case_profile(case) for case in args.cases]
    result = write_final_summary(
        profiles,
        output_dir=Path(args.output_dir),
        result_name=args.result_name,
        method=args.method,
        llm_mode=args.llm_mode,
        rerun_evaluator=not args.no_rerun_evaluator,
    )
    print(f"[final-summary] rows={result['rows']} table={result['table']} html={result['html']}")


if __name__ == "__main__":
    main()
