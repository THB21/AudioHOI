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

from scripts.shared.generic_contact_pipeline.core.base.config import load_case_profile  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.base.io import write_json  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.solver import (  # noqa: E402
    build_candidate_sandbox_manifest,
    validate_candidate_sandbox_manifest,
    write_candidate_sandbox_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export or materialize a generic sequence-solver candidate sandbox manifest.")
    parser.add_argument("--case", required=True)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--materialize", action="store_true", help="Write the sandbox manifest into candidate-dir when eligible.")
    args = parser.parse_args()

    profile = load_case_profile(args.case)
    if args.materialize:
        payload = write_candidate_sandbox_manifest(profile, args.result_dir, args.candidate_dir)
    else:
        payload = build_candidate_sandbox_manifest(profile, args.result_dir, args.candidate_dir)
        errors = validate_candidate_sandbox_manifest(payload)
        if errors:
            raise SystemExit("; ".join(errors))
        if args.out:
            write_json(args.out, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
