from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[4]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.config import available_cases, load_case_profile, with_runtime_overrides  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.hoi_profile import resolve_hoi_profile, write_resolved_profile  # noqa: E402


def run(profile, mode: str = "seed") -> dict[str, object]:
    resolved, trace = resolve_hoi_profile(profile, mode)
    written = write_resolved_profile(profile, resolved, trace)
    return {
        "stage": "stage_minus1_llm_prior",
        "case_name": profile.case_name,
        "llm_mode": mode,
        "hoi_profile": str(written["hoi_profile"]),
        "hoi_profile_resolved": str(written["hoi_profile_resolved"]),
        "llm_prior_trace": str(written["llm_prior_trace"]),
        "prompt_context": str(written["prompt_context"]),
        "object_parts": [item.get("name") for item in resolved.get("object_parts", []) if isinstance(item, dict)],
        "human_parts": resolved.get("human_parts", []),
        "interaction_edges": resolved.get("interaction_edges", []),
        "note": "LLM/seed prior defines discrete HOI semantics only; it does not output pose, coordinates, or loss weights.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Resolve deterministic HOI semantic profile for generic pipeline v2.")
    ap.add_argument("--case", default="all", help="Case name or all")
    ap.add_argument("--llm-mode", default="seed", choices=["none", "seed", "qwen", "mistral"])
    ap.add_argument("--result-name", default="", help="Override case profile result_name.")
    args = ap.parse_args()
    cases = available_cases() if args.case == "all" else [args.case]
    failed = False
    for case in cases:
        try:
            profile = with_runtime_overrides(load_case_profile(case), result_name=args.result_name or None)
            result = run(profile, args.llm_mode)
            print(f"{case}: {result['hoi_profile']}")
        except Exception as exc:
            failed = True
            print(f"{case}: FAILED: {exc}", file=sys.stderr)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
