#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[4]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.provenance.artifact_store import (  # noqa: E402
    verify_attempt_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify immutable blobs referenced by stage-attempt records.")
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()

    errors = verify_attempt_artifacts(args.result_dir)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)
    print(f"artifact store verified: {args.result_dir}")


if __name__ == "__main__":
    main()
