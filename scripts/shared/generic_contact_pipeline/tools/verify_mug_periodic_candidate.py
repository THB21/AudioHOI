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

from scripts.shared.generic_contact_pipeline.core.solver.projected_periodic_golden import (  # noqa: E402
    BODY_CANDIDATE_NAME,
    PHASE_CANDIDATE_NAME,
    PROJECTED_PERIODIC_ATTEMPT_NAME,
    verify_materialized_projected_periodic_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify materialized isolated mug projected-periodic candidate artifacts.")
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args()

    errors = verify_materialized_projected_periodic_candidate(args.candidate_dir)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)

    attempt = json.loads((args.candidate_dir / PROJECTED_PERIODIC_ATTEMPT_NAME).read_text())
    print(
        "mug: periodic_candidate "
        "materialized=True "
        f"body_rows={sum(1 for _ in (args.candidate_dir / BODY_CANDIDATE_NAME).open()) - 1} "
        f"phase_rows={sum(1 for _ in (args.candidate_dir / PHASE_CANDIDATE_NAME).open()) - 1} "
        f"{attempt['canonical_sha256']}"
    )


if __name__ == "__main__":
    main()
