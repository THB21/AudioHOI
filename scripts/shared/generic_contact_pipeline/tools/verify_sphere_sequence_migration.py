#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[4]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.solver import (  # noqa: E402
    DEFAULT_SPHERE_SEQUENCE_GOLDEN,
    verify_sphere_sequence_regression,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the isolated basketball/football sphere-solver migration.")
    parser.add_argument("--golden", type=Path, default=DEFAULT_SPHERE_SEQUENCE_GOLDEN)
    parser.add_argument("--result-name")
    args = parser.parse_args()
    errors = verify_sphere_sequence_regression(args.golden, result_name=args.result_name)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)
    print(f"verified basketball and football sphere migration from {args.golden}")


if __name__ == "__main__":
    main()
