#!/usr/bin/env python3
"""Prepare the basketball sample folder and extract 24 fps frames/audio."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2


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
            out_path = frames_dir / f"{out_idx:05d}.png"
            cv2.imwrite(str(out_path), frame)
            out_idx += 1
            next_src_idx += step
        src_idx += 1

    cap.release()
    return out_idx - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("video_sample/1_basketball_video.mp4"),
        help="Source basketball video.",
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=Path("samples/basketball_01"),
        help="Output sample directory.",
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--audio-sr", type=int, default=16000)
    parser.add_argument("--overwrite-frames", action="store_true")
    args = parser.parse_args()

    sample_dir = args.sample_dir
    frames_dir = sample_dir / "frames"
    results_dir = sample_dir / "results"
    video_path = sample_dir / "video.mp4"
    audio_path = sample_dir / "audio.wav"

    sample_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.source.resolve() != video_path.resolve():
        shutil.copy2(args.source, video_path)

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

    print(f"sample_dir: {sample_dir}")
    print(f"video: {video_path}")
    print(f"frames: {frame_count} files at {args.fps:g} fps")
    print(f"audio: {audio_path} ({audio_status})")
    if audio_status == "missing_ffmpeg":
        print("audio.wav was not created because ffmpeg is not available in PATH.")
        print("Run: ffmpeg -i video.mp4 -vn -ac 1 -ar 16000 audio.wav")


if __name__ == "__main__":
    main()
