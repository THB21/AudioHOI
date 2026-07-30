#!/usr/bin/env python3
"""Atomically extract deterministic frame or audio inputs from a case video."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _run(command: list[str]) -> None:
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "ffmpeg failed")[-4000:])


def _extract_frames(sample_dir: Path) -> dict[str, object]:
    video = sample_dir / "video.mp4"
    if not video.is_file():
        raise FileNotFoundError(video)
    sample_dir.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".frames.", dir=sample_dir))
    try:
        _run(["ffmpeg", "-v", "error", "-i", str(video), str(staged / "%05d.png")])
        frames = sorted(staged.glob("*.png"))
        if not frames:
            raise RuntimeError("ffmpeg decoded no video frames")
        target = sample_dir / "frames"
        backup = sample_dir / ".frames.previous"
        shutil.rmtree(backup, ignore_errors=True)
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(staged, target)
        except Exception:
            if backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        shutil.rmtree(backup, ignore_errors=True)
        return {"kind": "frames", "output": str(target), "frame_count": len(frames)}
    finally:
        shutil.rmtree(staged, ignore_errors=True)


def _extract_audio(sample_dir: Path) -> dict[str, object]:
    video = sample_dir / "video.mp4"
    if not video.is_file():
        raise FileNotFoundError(video)
    descriptor, temporary = tempfile.mkstemp(prefix=".audio.", suffix=".wav", dir=sample_dir)
    os.close(descriptor)
    temp_path = Path(temporary)
    try:
        _run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(video),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(temp_path),
            ]
        )
        if temp_path.stat().st_size <= 44:
            raise RuntimeError("ffmpeg produced an empty audio stream")
        target = sample_dir / "audio.wav"
        os.replace(temp_path, target)
        return {"kind": "audio", "output": str(target), "sample_rate": 16000}
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=("frames", "audio"), required=True)
    args = parser.parse_args()
    result = _extract_frames(args.sample_dir.resolve()) if args.kind == "frames" else _extract_audio(args.sample_dir.resolve())
    print(json.dumps({"schema_version": 1, "status": "generated", **result}, sort_keys=True))


if __name__ == "__main__":
    main()
