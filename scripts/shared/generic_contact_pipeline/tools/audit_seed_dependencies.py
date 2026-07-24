#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.base.config import load_case_profile, with_runtime_overrides
from scripts.shared.generic_contact_pipeline.core.provenance.seed_dependencies import audit_seed_dependencies


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit mug/chair solved-seed selection without writing pipeline data.")
    parser.add_argument("--cases", nargs="+", choices=("mug", "chair"), default=["mug", "chair"])
    parser.add_argument("--result-name", default="benchmark_vlm_qwen")
    parser.add_argument("--fresh-result-name", default="__seed_audit_fresh_result__")
    args = parser.parse_args()

    payload = {
        "schema_version": 1,
        "purpose": "Read-only comparison of existing-result reruns and fresh-result seed selection.",
        "contexts": {},
    }
    for context, result_name in (("existing", args.result_name), ("fresh", args.fresh_result_name)):
        payload["contexts"][context] = {
            case: audit_seed_dependencies(
                with_runtime_overrides(load_case_profile(case), result_name=result_name)
            )
            for case in args.cases
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
