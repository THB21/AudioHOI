#!/usr/bin/env python3
"""Detect basketball audio impacts and align them to visual bounce frames."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, savgol_filter


def read_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frames: list[int] = []
    ys: list[float] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            y = row.get("ball_center_y", "")
            if y == "":
                continue
            frames.append(int(row["frame"]))
            ys.append(float(y))
    if not frames:
        raise RuntimeError(f"No valid trajectory rows in {path}")
    return np.array(frames), np.array(ys)


def detect_visual_events(
    trajectory_csv: Path, fps: float, min_gap_s: float
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    frames, ys = read_trajectory(trajectory_csv)
    if len(ys) >= 9:
        win = min(len(ys) // 2 * 2 - 1, 11)
        ys_smooth = savgol_filter(ys, max(win, 5), 2)
    else:
        ys_smooth = ys

    distance = max(1, int(round(min_gap_s * fps)))
    prominence = max(8.0, float(np.std(ys_smooth) * 0.12))
    peaks, props = find_peaks(ys_smooth, distance=distance, prominence=prominence)

    rows: list[dict[str, object]] = []
    for idx, peak_idx in enumerate(peaks, start=1):
        frame = int(frames[peak_idx])
        rows.append(
            {
                "event": f"visual bounce {idx}",
                "visual_event": "ball lowest point",
                "visual_frame": frame,
                "visual_time": f"{(frame - 1) / fps:.6f}",
                "ball_center_y": f"{ys[peak_idx]:.3f}",
                "prominence": f"{props['prominences'][idx - 1]:.3f}",
            }
        )
    return rows, frames, ys_smooth


def detect_audio_events(
    audio_path: Path, fps: float, min_gap_s: float, top_k: int | None
) -> list[dict[str, object]]:
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=256)
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=256)

    distance = max(1, int(round(min_gap_s / (256 / sr))))
    height = float(np.percentile(onset_env, 70))
    prominence = max(0.05, float(np.std(onset_env) * 0.5))
    peaks, props = find_peaks(onset_env, distance=distance, height=height, prominence=prominence)

    ranked = sorted(peaks, key=lambda idx: onset_env[idx], reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]
    ranked = sorted(ranked)

    rows: list[dict[str, object]] = []
    for idx, peak_idx in enumerate(ranked, start=1):
        t = float(times[peak_idx])
        # Convert the audio timestamp in seconds to the corresponding video frame.
        # Since frames were extracted at 24 fps, frame ~= timestamp_seconds * fps.
        rows.append(
            {
                "event": f"bounce {idx}",
                "audio_time": f"{t:.6f}",
                "audio_frame": int(round(t * fps)),
                "peak": f"{onset_env[peak_idx]:.6f}",
                "prominence": f"{props['prominences'][list(peaks).index(peak_idx)]:.6f}",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def align(audio_rows: list[dict[str, object]], visual_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    remaining = visual_rows[:]
    rows: list[dict[str, object]] = []
    for audio_row in audio_rows:
        audio_frame = int(audio_row["audio_frame"])
        if remaining:
            visual_row = min(remaining, key=lambda row: abs(int(row["visual_frame"]) - audio_frame))
            remaining.remove(visual_row)
            visual_frame = int(visual_row["visual_frame"])
            visual_event = visual_row["visual_event"]
            diff = visual_frame - audio_frame
        else:
            visual_frame = ""
            visual_event = ""
            diff = ""
        rows.append(
            {
                "event": audio_row["event"],
                "audio_time": audio_row["audio_time"],
                "audio_frame": audio_frame,
                "visual_event": visual_event,
                "visual_frame": visual_frame,
                "difference_frames": diff,
            }
        )
    return rows


def plot(results_dir: Path, frames: np.ndarray, ys: np.ndarray, visual_rows: list[dict[str, object]]) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(frames, ys, linewidth=1.8)
    for row in visual_rows:
        frame = int(row["visual_frame"])
        idx = np.where(frames == frame)[0]
        if len(idx):
            plt.scatter([frame], [ys[idx[0]]], color="red", s=28)
    plt.xlabel("frame index")
    plt.ylabel("ball center y-position")
    plt.title("Basketball visual bounce candidates")
    plt.tight_layout()
    plt.savefig(results_dir / "ball_y_trajectory.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, default=Path("samples/basketball_01"))
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--min-gap-s", type=float, default=0.35)
    parser.add_argument("--audio-top-k", type=int, default=None)
    args = parser.parse_args()

    results_dir = args.sample_dir / "results"
    tracking_dir = results_dir / "tracking"
    events_dir = results_dir / "events"
    diagnostics_dir = results_dir / "diagnostics"
    events_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    visual_rows, frames, ys_smooth = detect_visual_events(
        tracking_dir / "ball_trajectory.csv", args.fps, args.min_gap_s
    )
    audio_rows = detect_audio_events(args.sample_dir / "audio.wav", args.fps, args.min_gap_s, args.audio_top_k)
    if args.audio_top_k is None and visual_rows:
        audio_rows = sorted(audio_rows, key=lambda row: float(row["peak"]), reverse=True)[: len(visual_rows)]
        audio_rows = sorted(audio_rows, key=lambda row: float(row["audio_time"]))
        for idx, row in enumerate(audio_rows, start=1):
            row["event"] = f"bounce {idx}"

    alignment_rows = align(audio_rows, visual_rows)
    write_csv(
        events_dir / "visual_events.csv",
        visual_rows,
        ["event", "visual_event", "visual_frame", "visual_time", "ball_center_y", "prominence"],
    )
    write_csv(
        events_dir / "audio_events.csv",
        audio_rows,
        ["event", "audio_time", "audio_frame", "peak", "prominence"],
    )
    write_csv(
        events_dir / "audio_visual_alignment.csv",
        alignment_rows,
        ["event", "audio_time", "audio_frame", "visual_event", "visual_frame", "difference_frames"],
    )
    plot(diagnostics_dir, frames, ys_smooth, visual_rows)
    print(f"visual_events: {len(visual_rows)}")
    print(f"audio_events: {len(audio_rows)}")
    print(f"alignment: {events_dir / 'audio_visual_alignment.csv'}")


if __name__ == "__main__":
    main()
