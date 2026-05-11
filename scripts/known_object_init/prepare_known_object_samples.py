#!/usr/bin/env python3
"""Prepare known-object samples from video_sample/prompts.json."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from scripts.manual_init.prepare_basketball_sample import extract_frames, run_ffmpeg


def sample_dir_name(item: dict[str, object]) -> str:
    return f"{int(item['id']):02d}_{str(item['name'])}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("video_sample/prompts.json"))
    parser.add_argument("--root-dir", type=Path, default=Path("samples_known_object"))
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--audio-sr", type=int, default=16000)
    parser.add_argument("--overwrite-frames", action="store_true")
    args = parser.parse_args()

    items = json.loads(args.prompts.read_text())
    root_dir = args.root_dir
    root_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        sample_dir = root_dir / sample_dir_name(item)
        frames_dir = sample_dir / "frames"
        results_dir = sample_dir / "results"
        video_path = sample_dir / "video.mp4"
        audio_path = sample_dir / "audio.wav"
        metadata_path = sample_dir / "metadata.json"

        sample_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        source_video = Path(str(item["video"]))
        if source_video.resolve() != video_path.resolve():
            shutil.copy2(source_video, video_path)

        metadata = {
            "id": item["id"],
            "name": item["name"],
            "image": item["image"],
            "video": item["video"],
            "prompt": item["prompt"],
            "detection_text": f"{item['name']}.",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))

        if args.overwrite_frames:
            for frame_path in frames_dir.glob("*.png"):
                frame_path.unlink()

        existing_frames = sorted(frames_dir.glob("*.png"))
        if existing_frames:
            frame_count = len(existing_frames)
        else:
            frame_count = extract_frames(video_path, frames_dir, args.fps)

        audio_status = "exists"
        if not audio_path.exists():
            audio_status = "missing_ffmpeg"
            if run_ffmpeg(video_path, audio_path, args.audio_sr):
                audio_status = "created"

        print(f"{sample_dir.name}: frames={frame_count}, audio={audio_status}")


if __name__ == "__main__":
    main()
