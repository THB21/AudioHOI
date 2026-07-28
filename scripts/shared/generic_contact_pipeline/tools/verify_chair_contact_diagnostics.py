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
from scripts.shared.generic_contact_pipeline.core.solver import (  # noqa: E402
    build_chair_contact_diagnostics,
    validate_chair_contact_diagnostics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify read-only chair pairprop/contact diagnostics.")
    parser.add_argument("--case", default="chair")
    parser.add_argument("--result-name", default="benchmark_vlm_qwen")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.case != "chair":
        raise SystemExit("chair contact diagnostics currently supports --case chair")
    profile = with_runtime_overrides(load_case_profile(args.case), result_name=args.result_name)
    diagnostics = build_chair_contact_diagnostics(profile.result_dir)
    errors = validate_chair_contact_diagnostics(diagnostics)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    summary = diagnostics["summary"]
    print(
        f"chair: contact_diagnostics active={summary['active_frames']} "
        f"seed={summary['seed_policy']} "
        f"gap={diagnostics['compatibility_gap_id']}:{diagnostics['compatibility_gap_status']} "
        f"{diagnostics['canonical_sha256']}"
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
