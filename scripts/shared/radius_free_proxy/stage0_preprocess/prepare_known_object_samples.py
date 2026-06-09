#!/usr/bin/env python3
"""Prepare known-object samples from video_sample/prompts.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.shared.radius_free_proxy.stage0_preprocess.prepare_sample_inputs import prepare_sample


def sample_dir_name(item: dict[str, object]) -> str:
    return f"{int(item['id']):02d}_{str(item['name'])}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("video_sample/prompts.json"))
    parser.add_argument("--root-dir", type=Path, default=Path("samples_known_object"))
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--audio-sr", type=int, default=16000)
    parser.add_argument("--overwrite-frames", action="store_true")
    parser.add_argument("--skip-audio-events", action="store_true")
    parser.add_argument("--audio-min-gap-s", type=float, default=0.25)
    parser.add_argument("--audio-top-k", type=int, default=None)
    parser.add_argument("--min-audio-score", type=float, default=0.12)
    args = parser.parse_args()

    items = json.loads(args.prompts.read_text())
    root_dir = args.root_dir
    root_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        sample_dir = root_dir / sample_dir_name(item)
        metadata_path = sample_dir / "metadata.json"
        source_video = Path(str(item["video"]))

        summary = prepare_sample(
            source_video=source_video,
            sample_dir=sample_dir,
            fps=args.fps,
            audio_sr=args.audio_sr,
            overwrite_frames=args.overwrite_frames,
            build_audio_events=not args.skip_audio_events,
            min_gap_s=args.audio_min_gap_s,
            audio_top_k=args.audio_top_k,
            min_audio_score=args.min_audio_score,
        )

        metadata = {
            "id": item["id"],
            "name": item["name"],
            "image": item["image"],
            "video": item["video"],
            "prompt": item["prompt"],
            "detection_text": f"{item['name']}.",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))

        print(
            f"{sample_dir.name}: frames={summary['frames']}, "
            f"audio={summary['audio_status']}, audio_events={summary['audio_events']}"
        )


if __name__ == "__main__":
    main()
