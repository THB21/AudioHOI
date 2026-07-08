"""Stage 0 — extract a mono 16 kHz waveform from a video (or reuse audio.wav).

Object-agnostic and generated-video friendly: any video with an audio track works.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

TARGET_SR = 16000


def extract_wav(sample_dir: Path, *, force: bool = False) -> Path:
    """Return path to a 16 kHz mono ``audio.wav`` in ``sample_dir``.

    Uses the existing ``audio.wav`` unless ``force``; otherwise demuxes ``video.mp4``
    with ffmpeg. Raises if neither audio nor a decodable video track is present.
    """
    sample_dir = Path(sample_dir)
    wav = sample_dir / "audio.wav"
    if wav.exists() and not force:
        return wav
    video = sample_dir / "video.mp4"
    if not video.exists():
        raise FileNotFoundError(f"no audio.wav and no video.mp4 in {sample_dir}")
    cmd = [
        "ffmpeg", "-y", "-i", str(video),
        "-vn", "-ac", "1", "-ar", str(TARGET_SR), "-f", "wav", str(wav),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not wav.exists():
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{res.stderr[-2000:]}")
    return wav


def detect_fps(sample_dir: Path, fallback: float = 24.0) -> float:
    """Frame rate of ``video.mp4``, or ``fallback`` when unavailable.

    Onset times are converted to frame indices with this — a hardcoded rate puts
    every event on the wrong frame for clips at a different fps (a 30 fps clip with
    fps=24 lands events at 0.8x their true frame).
    """
    import cv2

    video = Path(sample_dir) / "video.mp4"
    if not video.exists():
        return fallback
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
    cap.release()
    return float(fps) if fps and fps > 1.0 else fallback


def load_wav(path: Path, sr: int = TARGET_SR) -> tuple[int, np.ndarray]:
    """Load + resample to ``sr``, mono, peak-normalized float64 in [-1, 1]."""
    import librosa

    x, file_sr = librosa.load(str(path), sr=sr, mono=True)
    x = x.astype(np.float64)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0:
        x = x / peak
    return int(sr), x
