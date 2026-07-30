#!/usr/bin/env python3
"""Generate solver-compatible generic audio events for one ingested case."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.audio.extract import detect_fps
from src.audio.pipeline import run_sample


FIELDS = (
    "event",
    "audio_time",
    "audio_frame",
    "peak",
    "prominence",
    "rms_rise",
    "sharpness",
    "audio_score",
    "detector",
    "source",
)


def _write_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def generate_audio_events(sample_dir: Path, *, detector: str, classifier: str) -> dict[str, object]:
    sample_dir = sample_dir.resolve()
    fps = detect_fps(sample_dir)
    records = run_sample(
        sample_dir,
        detector=detector,
        classifier=classifier,
        fps=fps,
        write=False,
    )
    rows: list[dict[str, object]] = []
    for index, (event, features, _visual) in enumerate(records, start=1):
        numeric = {
            "audio_time": float(event.time),
            "audio_frame": int(event.frame),
            "peak": float(features.rms),
            "prominence": float(features.attack),
            "rms_rise": float(features.attack),
            "sharpness": float(features.hf_ratio),
            "audio_score": float(event.audio_score),
        }
        if not all(math.isfinite(float(value)) for value in numeric.values()):
            raise ValueError(f"non-finite generic audio event at row {index}")
        rows.append(
            {
                "event": f"audio event {index}",
                **numeric,
                "detector": event.detector,
                "source": event.source,
            }
        )
    output = sample_dir / "results/events/audio_events.csv"
    _write_atomic(output, rows)
    return {
        "schema_version": 1,
        "status": "generated",
        "sample_dir": str(sample_dir),
        "output": str(output),
        "event_count": len(rows),
        "fps": fps,
        "detector": detector,
        "classifier": classifier,
        "empty_event_stream_valid": not rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--detector", default="combined")
    parser.add_argument("--classifier", default="rule")
    args = parser.parse_args()
    print(
        json.dumps(
            generate_audio_events(
                args.sample_dir,
                detector=args.detector,
                classifier=args.classifier,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
