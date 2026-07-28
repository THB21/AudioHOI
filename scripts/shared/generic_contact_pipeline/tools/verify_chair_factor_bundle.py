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
from scripts.shared.generic_contact_pipeline.core.factors import (  # noqa: E402
    build_chair_factor_executor_bundle,
    validate_chair_factor_executor_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify read-only chair generic factor executor readiness bundle.")
    parser.add_argument("--case", default="chair")
    parser.add_argument("--result-name", default="benchmark_vlm_qwen")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.case != "chair":
        raise SystemExit("chair factor bundle currently supports --case chair")
    profile = with_runtime_overrides(load_case_profile(args.case), result_name=args.result_name)
    bundle = build_chair_factor_executor_bundle(profile, profile.result_dir)
    errors = validate_chair_factor_executor_bundle(bundle)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(bundle, indent=2, ensure_ascii=False))
    print(
        f"chair: factor_bundle status={bundle['status']} "
        f"missing={','.join(bundle['missing_required_factor_kinds']) or 'none'} "
        f"gap={bundle['compatibility_gap_id']}:{bundle['compatibility_gap_status']} "
        f"{bundle['canonical_sha256']}"
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
