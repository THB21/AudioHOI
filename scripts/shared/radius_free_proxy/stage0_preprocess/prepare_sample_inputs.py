#!/usr/bin/env python3
"""Prepare sample video inputs for the radius-free pipeline.

This stage copies/normalizes the input video, extracts frames, extracts mono audio,
and optionally generates audio event proposals used downstream.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2

from scripts.shared.radius_free_proxy.stage0_preprocess.align_audio_events import detect_audio_events, write_csv


def run_ffmpeg(video_path: Path, audio_path: Path, sample_rate: int) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(audio_path),
    ]
    subprocess.run(cmd, check=True)
    return True


def extract_frames(video_path: Path, frames_dir: Path, fps: float) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    step = src_fps / fps
    next_src_idx = 0.0
    src_idx = 0
    out_idx = 1

    frames_dir.mkdir(parents=True, exist_ok=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if src_idx + 1e-6 >= next_src_idx:
            cv2.imwrite(str(frames_dir / f"{out_idx:05d}.png"), frame)
            out_idx += 1
            next_src_idx += step
        src_idx += 1

    cap.release()
    return out_idx - 1


def write_audio_event_tables(sample_dir: Path, fps: float, min_gap_s: float, top_k: int | None, min_audio_score: float) -> int:
    audio_path = sample_dir / "audio.wav"
    if not audio_path.exists():
        return 0
    events_dir = sample_dir / "results" / "events"
    audio_rows = detect_audio_events(audio_path, fps, min_gap_s, top_k, min_audio_score)
    audio_fields = ["event", "audio_time", "audio_frame", "peak", "prominence", "rms_rise", "sharpness", "audio_score"]
    write_csv(events_dir / "audio_events.csv", audio_rows, audio_fields)
    return len(audio_rows)


def prepare_sample(
    source_video: Path,
    sample_dir: Path,
    fps: float,
    audio_sr: int,
    overwrite_frames: bool,
    build_audio_events: bool,
    min_gap_s: float,
    audio_top_k: int | None,
    min_audio_score: float,
) -> dict[str, object]:
    frames_dir = sample_dir / "frames"
    results_dir = sample_dir / "results"
    video_path = sample_dir / "video.mp4"
    audio_path = sample_dir / "audio.wav"

    sample_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if source_video.resolve() != video_path.resolve():
        shutil.copy2(source_video, video_path)

    if overwrite_frames:
        for frame_path in frames_dir.glob("*.png"):
            frame_path.unlink()

    existing_frames = sorted(frames_dir.glob("*.png"))
    frame_count = len(existing_frames) if existing_frames else extract_frames(video_path, frames_dir, fps)

    audio_status = "exists"
    if not audio_path.exists():
        audio_status = "missing_ffmpeg"
        if run_ffmpeg(video_path, audio_path, audio_sr):
            audio_status = "created"

    audio_event_count = 0
    if build_audio_events and audio_path.exists():
        audio_event_count = write_audio_event_tables(sample_dir, fps, min_gap_s, audio_top_k, min_audio_score)

    return {
        "sample_dir": str(sample_dir),
        "video": str(video_path),
        "frames": frame_count,
        "audio": str(audio_path),
        "audio_status": audio_status,
        "audio_events": audio_event_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one sample for the radius-free pipeline.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--audio-sr", type=int, default=16000)
    parser.add_argument("--overwrite-frames", action="store_true")
    parser.add_argument("--skip-audio-events", action="store_true")
    parser.add_argument("--min-gap-s", type=float, default=0.25)
    parser.add_argument("--audio-top-k", type=int, default=None)
    parser.add_argument("--min-audio-score", type=float, default=0.12)
    args = parser.parse_args()

    summary = prepare_sample(
        source_video=args.source,
        sample_dir=args.sample_dir,
        fps=args.fps,
        audio_sr=args.audio_sr,
        overwrite_frames=args.overwrite_frames,
        build_audio_events=not args.skip_audio_events,
        min_gap_s=args.min_gap_s,
        audio_top_k=args.audio_top_k,
        min_audio_score=args.min_audio_score,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
