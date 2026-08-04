#!/usr/bin/env python3
"""Generate solver-compatible generic audio events for one ingested case."""
from __future__ import annotations

import argparse
import csv
import hashlib
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
from scripts.shared.generic_contact_pipeline.core.audio_events.envelope import (  # noqa: E402
    AudioEnvelopeConfig,
    extract_audio_evidence,
)


FIELDS = (
    "event",
    "event_type",
    "audio_time",
    "audio_frame",
    "start_time_s",
    "end_time_s",
    "start_frame",
    "end_frame",
    "peak",
    "prominence",
    "rms_rise",
    "sharpness",
    "audio_score",
    "snr",
    "band_profile",
    "detector",
    "source",
)

ENVELOPE_FIELDS = (
    "frame", "time_s", "rms_z", "flux_z", "hf_ratio", "motion_probability", "source",
)


def _write_atomic(path: Path, rows: list[dict[str, object]], fields=FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def generate_audio_events(
    sample_dir: Path,
    *,
    detector: str,
    classifier: str,
    envelope_config: AudioEnvelopeConfig | None = None,
) -> dict[str, object]:
    sample_dir = sample_dir.resolve()
    fps = detect_fps(sample_dir)
    records = run_sample(
        sample_dir,
        detector=detector,
        classifier=classifier,
        fps=fps,
        write=False,
    )
    audio_path = sample_dir / "audio.wav"
    config = envelope_config or AudioEnvelopeConfig(hop_ms=1000.0 / fps)
    rows, envelope_rows = extract_audio_evidence(audio_path, fps, config)
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
                "event": f"legacy_peak_{index:04d}",
                "event_type": "unknown",
                **numeric,
                "start_time_s": float(event.time),
                "end_time_s": float(event.time),
                "start_frame": int(event.frame),
                "end_frame": int(event.frame),
                "snr": float(features.attack),
                "band_profile": "legacy_peak",
                "detector": event.detector,
                "source": event.source,
            }
        )
    rows.sort(key=lambda row: (int(row["start_frame"]), str(row["event_type"])))
    output = sample_dir / "results/events/audio_events.csv"
    _write_atomic(output, rows)
    envelope_output = sample_dir / "results/events/audio_envelope.csv"
    _write_atomic(envelope_output, envelope_rows, ENVELOPE_FIELDS)
    manifest_output = sample_dir / "results/events/audio_event_manifest.json"
    manifest = {
        "schema_version": 2,
        "status": "generated",
        "sample_dir": str(sample_dir),
        "output": str(output),
        "envelope_output": str(envelope_output),
        "event_count": len(rows),
        "envelope_row_count": len(envelope_rows),
        "fps": fps,
        "detector": detector,
        "classifier": classifier,
        "empty_event_stream_valid": not rows,
        "audio_sha256": _sha256(audio_path),
        "events_sha256": _sha256(output),
        "envelope_sha256": _sha256(envelope_output),
        "envelope_config": {
            "window_ms": config.window_ms,
            "hop_ms": config.hop_ms,
            "motion_on_z": config.motion_on_z,
            "motion_off_z": config.motion_off_z,
            "min_motion_ms": config.min_motion_ms,
            "min_silence_ms": config.min_silence_ms,
            "impulse_z": config.impulse_z,
        },
    }
    _write_json_atomic(manifest_output, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--detector", default="combined")
    parser.add_argument("--classifier", default="rule")
    parser.add_argument("--window-ms", type=float, default=80.0)
    parser.add_argument("--motion-on-z", type=float, default=0.55)
    parser.add_argument("--motion-off-z", type=float, default=-0.10)
    parser.add_argument("--min-motion-ms", type=float, default=150.0)
    parser.add_argument("--min-silence-ms", type=float, default=250.0)
    parser.add_argument("--impulse-z", type=float, default=3.0)
    args = parser.parse_args()
    print(
        json.dumps(
            generate_audio_events(
                args.sample_dir,
                detector=args.detector,
                classifier=args.classifier,
                envelope_config=AudioEnvelopeConfig(
                    window_ms=args.window_ms,
                    hop_ms=1000.0 / detect_fps(args.sample_dir),
                    motion_on_z=args.motion_on_z,
                    motion_off_z=args.motion_off_z,
                    min_motion_ms=args.min_motion_ms,
                    min_silence_ms=args.min_silence_ms,
                    impulse_z=args.impulse_z,
                ),
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
