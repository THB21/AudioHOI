#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[4]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.factors.golden import (  # noqa: E402
    DEFAULT_FACTOR_SHADOW_GOLDEN,
    build_canonical_factor_shadow_summary,
    verify_factor_shadow_summary,
)
from scripts.shared.generic_contact_pipeline.core.state.golden import CANONICAL_CASE_DIRECTORIES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify five-case Factor IR shadow summaries.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_FACTOR_SHADOW_GOLDEN)
    parser.add_argument("--result-name", default="benchmark_vlm_qwen")
    args = parser.parse_args()

    actual = build_canonical_factor_shadow_summary(result_name=args.result_name)
    errors = verify_factor_shadow_summary(args.manifest, result_name=args.result_name)
    for case_name in CANONICAL_CASE_DIRECTORIES:
        summary = actual["cases"][case_name]
        kinds = ",".join(f"{key}={value}" for key, value in sorted(summary["factor_kinds"].items()))
        gaps = ",".join(summary["gap_ids"])
        print(f"{case_name}: factors={summary['factor_count']} kinds=[{kinds}] gaps=[{gaps}] {summary['canonical_sha256']}")
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
