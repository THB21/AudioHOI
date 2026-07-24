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

from scripts.shared.generic_contact_pipeline.core.provenance.golden import (  # noqa: E402
    DEFAULT_GOLDEN_MANIFEST,
    sync_golden_inputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hydrate missing canonical inputs into a new worktree without overwriting existing data."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, default=REPO)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_GOLDEN_MANIFEST)
    parser.add_argument("--apply", action="store_true", help="Copy verified inputs; default is a dry run.")
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text())
    report = sync_golden_inputs(
        payload,
        source_root=args.source_root,
        destination_root=args.destination_root,
        apply=args.apply,
    )
    print(
        f"verified={len(report['verified'])} would_copy={len(report['would_copy'])} "
        f"copied={len(report['copied'])} errors={len(report['errors'])}"
    )
    for error in report["errors"]:
        print(error, file=sys.stderr)
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
