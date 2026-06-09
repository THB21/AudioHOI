#!/usr/bin/env python3
"""Detect audio impact/event proposals for the radius-free pipeline.

This is intentionally audio-only: no HOT scores and no visual-contact model.
It writes the minimal audio tables consumed downstream by object proxy stages.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import find_peaks


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    vmax = float(np.max(values))
    if vmax <= 1e-8:
        return np.zeros_like(values)
    return values / vmax


def detect_audio_events(audio_path: Path, fps: float, min_gap_s: float, top_k: int | None, min_audio_score: float) -> list[dict[str, object]]:
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)

    distance = max(1, int(round(min_gap_s * sr / hop_length)))
    prominence = max(1e-6, float(np.std(onset_env)) * 0.45)
    peaks, props = find_peaks(onset_env, distance=distance, prominence=prominence)
    if len(peaks) == 0:
        return []

    onset_peak = onset_env[peaks]
    onset_prom = props.get("prominences", np.zeros(len(peaks), dtype=np.float64))
    rms_prev = np.empty_like(rms)
    rms_prev[0] = rms[0]
    rms_prev[1:] = rms[:-1]
    rms_rise = np.clip(rms - rms_prev, 0.0, None)[peaks]

    sharpness = []
    for peak in peaks:
        lo = max(0, peak - 2)
        hi = min(len(onset_env), peak + 3)
        local = onset_env[lo:hi]
        local_idx = peak - lo
        neigh = np.delete(local, local_idx) if len(local) > 1 else np.asarray([], dtype=np.float64)
        base = float(np.mean(neigh)) if len(neigh) else 0.0
        sharpness.append(float(max(0.0, onset_env[peak] - base)))
    sharpness = np.asarray(sharpness, dtype=np.float64)

    score = (
        0.35 * _normalize(onset_peak)
        + 0.30 * _normalize(onset_prom)
        + 0.20 * _normalize(rms_rise)
        + 0.15 * _normalize(sharpness)
    )

    rows = []
    for idx, peak in enumerate(peaks):
        audio_score = float(score[idx])
        if audio_score < min_audio_score:
            continue
        t = float(times[peak])
        rows.append({
            "event": "audio_event",
            "audio_time": f"{t:.6f}",
            "audio_frame": int(round(t * fps)),
            "peak": f"{float(onset_peak[idx]):.6f}",
            "prominence": f"{float(onset_prom[idx]):.6f}",
            "rms_rise": f"{float(rms_rise[idx]):.6f}",
            "sharpness": f"{float(sharpness[idx]):.6f}",
            "audio_score": f"{audio_score:.6f}",
        })

    rows.sort(key=lambda r: (-float(r["audio_score"]), int(r["audio_frame"])))
    if top_k is not None:
        rows = rows[:top_k]
    rows.sort(key=lambda r: int(r["audio_frame"]))
    for idx, row in enumerate(rows, start=1):
        row["event"] = f"audio event {idx}"
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect audio event proposals for radius-free object contact stages.")
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--min-gap-s", type=float, default=0.25)
    parser.add_argument("--audio-top-k", type=int, default=None)
    parser.add_argument("--min-audio-score", type=float, default=0.12)
    args = parser.parse_args()

    sample_dir = args.sample_dir
    audio_path = sample_dir / "audio.wav"
    if not audio_path.exists():
        raise RuntimeError(f"Missing audio file: {audio_path}")

    events_dir = sample_dir / "results" / "events"
    audio_rows = detect_audio_events(audio_path, args.fps, args.min_gap_s, args.audio_top_k, args.min_audio_score)

    audio_fields = ["event", "audio_time", "audio_frame", "peak", "prominence", "rms_rise", "sharpness", "audio_score"]
    write_csv(events_dir / "audio_events.csv", audio_rows, audio_fields)

    print(f"audio_events: {len(audio_rows)}")
    print(f"audio_events_csv: {events_dir / 'audio_events.csv'}")


if __name__ == "__main__":
    main()
