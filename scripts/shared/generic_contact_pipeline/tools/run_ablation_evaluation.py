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
from scripts.shared.generic_contact_pipeline.core.evaluation.final_hoi.ablation_registry import (  # noqa: E402
    DEFAULT_VARIANTS,
    MATERIALIZED_DEFAULT_METHODS,
    MethodVariant,
)
from scripts.shared.generic_contact_pipeline.core.evaluation.final_hoi.ablation_runner import run_ablation_evaluation  # noqa: E402


DEFAULT_CASES = ["basketball", "football"]
DEFAULT_METHODS = list(MATERIALIZED_DEFAULT_METHODS)


def _parse_method_result(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--method-result must be METHOD=RESULT_NAME, got {item!r}")
        method, result_name = item.split("=", 1)
        out[method] = result_name
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate and summarize final HOI ablation result directories.")
    ap.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--method-result", action="append", default=[], metavar="METHOD=RESULT_NAME")
    ap.add_argument("--output-dir", default="final_result/evaluation/ablation")
    ap.add_argument("--baseline-method", default="full_audio_vlm_llm")
    ap.add_argument("--allow-same-result-debug", action="store_true")
    ap.add_argument("--require-existing", action="store_true")
    args = ap.parse_args()

    overrides = _parse_method_result(args.method_result)
    default_by_name = {variant.method: variant for variant in DEFAULT_VARIANTS}
    variants: list[MethodVariant] = []
    for method in args.methods:
        base = default_by_name.get(method, MethodVariant(method, f"eval_{method}", []))
        variants.append(
            MethodVariant(
                method=base.method,
                result_name=overrides.get(method, base.result_name),
                ablation_flags=base.ablation_flags,
                audio=base.audio,
                vlm=base.vlm,
                llm=base.llm,
                required=base.required,
            )
        )
    profiles = [load_case_profile(case) for case in args.cases]
    if args.require_existing:
        # Use the registry's strict mode through the runner's validation plus an
        # explicit existence check so missing required variants fail loudly.
        from scripts.shared.generic_contact_pipeline.core.evaluation.final_hoi.ablation_registry import validate_method_result_mapping  # noqa: E402

        validate_method_result_mapping(profiles, variants, require_existing=True)
    result = run_ablation_evaluation(
        profiles,
        variants=variants,
        output_dir=Path(args.output_dir),
        allow_same_result_debug=args.allow_same_result_debug,
        require_existing=args.require_existing,
        baseline_method=args.baseline_method,
    )
    print(f"[ablation-eval] rows={result['rows']} missing={result['missing_results']} table={result['table']}")


if __name__ == "__main__":
    main()
