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
    capture_golden_manifest,
    verify_golden_manifest,
)
from scripts.shared.generic_contact_pipeline.core.base.io import write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture or verify the five-case Phase 0 golden manifest.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--capture", action="store_true", help="Capture current canonical artifacts as the golden baseline.")
    action.add_argument("--verify", action="store_true", help="Verify current artifacts against the golden baseline.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_GOLDEN_MANIFEST)
    parser.add_argument(
        "--input-root",
        type=Path,
        help="Read-only root containing generated/ignored canonical inputs, e.g. /mnt/hdd/AudioHOI.",
    )
    parser.add_argument("--skip-decoded-renders", action="store_true", help="Skip expensive decoded RGB24 render verification.")
    args = parser.parse_args()

    if args.capture:
        payload = capture_golden_manifest(input_root=args.input_root)
        write_json(args.manifest, payload)
        print(f"captured {len(payload['cases'])} cases in {args.manifest}")
        return

    payload = json.loads(args.manifest.read_text())
    errors = verify_golden_manifest(
        payload,
        verify_decoded_renders=not args.skip_decoded_renders,
        input_root=args.input_root,
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)
    print(f"verified {len(payload.get('cases', {}))} cases from {args.manifest}")


if __name__ == "__main__":
    main()
