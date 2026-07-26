#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.solver.projected_periodic_golden import (  # noqa: E402
    DEFAULT_PROJECTED_PERIODIC_GOLDEN,
    verify_projected_periodic_regression,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the isolated mug projected-periodic migration.")
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--golden", type=Path, default=DEFAULT_PROJECTED_PERIODIC_GOLDEN)
    args = parser.parse_args()
    errors = verify_projected_periodic_regression(args.attempt, args.golden)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("mug: projected-periodic candidate matches frozen observation-derived baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
