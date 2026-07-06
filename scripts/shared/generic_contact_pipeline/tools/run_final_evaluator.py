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
from scripts.shared.generic_contact_pipeline.core.evaluation.final_evaluator import run_final_evaluator  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Run final hard-metric + VLM visual + LLM CSV evaluator for one generic pipeline result.")
    ap.add_argument("--case", required=True)
    ap.add_argument("--result-name", default="")
    ap.add_argument("--method", default="vlm_gated")
    ap.add_argument("--llm-mode", default="seed", choices=["none", "seed", "mistral"])
    args = ap.parse_args()

    profile = with_runtime_overrides(load_case_profile(args.case), result_name=args.result_name or None)
    result = run_final_evaluator(profile, method=args.method, llm_mode=args.llm_mode)
    print(f"[{profile.case_name}] final_evaluator pass={result['final_pass']} summary={profile.result_dir / 'vlm_trace' / '06_evaluation' / 'evaluation_summary.json'}")


if __name__ == "__main__":
    main()
