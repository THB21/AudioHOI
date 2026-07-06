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
from scripts.shared.generic_contact_pipeline.core.evaluation.benchmark import run_benchmark  # noqa: E402


DEFAULT_METHODS = ["baseline_no_vlm", "vlm_gated", "no_anchor", "no_smooth", "no_static_tail"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run benchmark aggregation for generic pipeline results.")
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--result-name", default="", help="Use the same result name for all cases.")
    ap.add_argument(
        "--method-result",
        action="append",
        default=[],
        metavar="METHOD=RESULT_NAME",
        help="Evaluate a method label against a result directory. Supports METHOD=RESULT or CASE:METHOD=RESULT.",
    )
    ap.add_argument("--llm-mode", default="seed", choices=["none", "seed", "mistral"])
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--allow-same-result-debug", action="store_true", help="Allow multiple method labels to evaluate the same result directory for diagnostics only.")
    args = ap.parse_args()

    profiles = [with_runtime_overrides(load_case_profile(case), result_name=args.result_name or None) for case in args.cases]
    method_result_names: dict[str, str] = {}
    for item in args.method_result:
        if "=" not in item:
            raise SystemExit(f"--method-result must be METHOD=RESULT_NAME, got {item!r}")
        method, result_name = item.split("=", 1)
        method_result_names[method] = result_name
    output_dir = Path(args.output_dir) if args.output_dir else None
    result = run_benchmark(
        profiles,
        methods=args.methods,
        output_dir=output_dir,
        method_result_names=method_result_names,
        llm_mode=args.llm_mode,
        allow_same_result_debug=args.allow_same_result_debug,
    )
    print(f"[benchmark] rows={result['rows']} table={result['table']}")


if __name__ == "__main__":
    main()
