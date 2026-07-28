#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[4]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.interaction import (  # noqa: E402
    build_interaction_timeline,
    write_interaction_timeline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export production InteractionStateIR timeline artifacts.")
    parser.add_argument("--case", required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    timeline = build_interaction_timeline(args.case, args.result_dir)
    write_interaction_timeline(timeline, args.out_dir)
    print(
        f"{args.case}: interaction_timeline frames={timeline.metrics['frame_count']} "
        f"active_contact_frames={timeline.metrics['active_contact_frames']} "
        f"audio_event_frames={timeline.metrics['audio_event_frames']}"
    )


if __name__ == "__main__":
    main()
