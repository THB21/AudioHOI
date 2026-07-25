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
from scripts.shared.generic_contact_pipeline.core.solver import build_sequence_solver_shadow_diagnostics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export generic sequence-solver shadow diagnostics.")
    parser.add_argument("--case", required=True)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    payload = build_sequence_solver_shadow_diagnostics(load_case_profile(args.case), args.result_dir)
    if args.out:
        write_json(args.out, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
