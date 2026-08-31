#!/usr/bin/env python3
"""Build nine synchronized WITHOUT AUDIO vs WITH AUDIO three-view videos."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "deliverables/nine_audio_comparisons"
VIEWS = ("overlay", "camera3d", "side_yz")
CASES = [
    ("01_basketball", REPO / "samples_known_object/01_basketball"),
    ("10_football", REPO / "samples_known_object/10_football"),
    ("02_mug", REPO / "samples_known_object/02_mug"),
    ("05_chair", REPO / "samples_known_object/05_chair"),
    ("11_stick", REPO / "samples_known_object/11_stick"),
    ("12_back_view_basketball", REPO / "samples_known_object/12_back_view_basketball"),
    ("13_volleyball", REPO / "samples_known_object/13_volleyball"),
    ("14_pingpong_wall", REPO / "samples_known_object/14_pingpong_wall"),
    ("15_suitcase_drag", REPO / "samples_known_object/15_suitcase_drag"),
]


def sources(sample: Path, method: str) -> list[Path]:
    root = sample / "results/renders" / f"perceptual_{method}_v2" / "with_human"
    return [root / f"{view}.mp4" for view in VIEWS]


def build(case: str, sample: Path) -> dict[str, object]:
    paths = sources(sample, "ground") + sources(sample, "full")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{case}: missing render inputs: {missing}")
    caps = [cv2.VideoCapture(str(path)) for path in paths]
    if not all(cap.isOpened() for cap in caps):
        raise RuntimeError(f"{case}: cannot decode all six inputs")
    counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    fps_values = [float(cap.get(cv2.CAP_PROP_FPS) or 24.0) for cap in caps]
    n = min(counts)
    source_fps = min(fps_values)
    output_fps = 12.0
    stride = max(1, round(source_fps / output_fps))
    out_dir = OUT / case
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "without_audio_vs_with_audio.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (960, 408))
    written = 0
    for frame_idx in range(n):
        frames = []
        for cap in caps:
            ok, frame = cap.read()
            if not ok:
                frames = []
                break
            frames.append(frame)
        if not frames:
            break
        if frame_idx % stride:
            continue
        panels = []
        for idx, frame in enumerate(frames):
            panel = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            cv2.rectangle(panel, (0, 0), (320, 24), (10, 10, 10), -1)
            cv2.putText(panel, VIEWS[idx % 3], (8, 17), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(panel)
        canvas = np.zeros((408, 960, 3), dtype=np.uint8)
        canvas[24:204] = np.hstack(panels[:3])
        canvas[228:408] = np.hstack(panels[3:])
        cv2.putText(canvas, "WITHOUT AUDIO", (12, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "WITH AUDIO", (12, 222), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(canvas)
        written += 1
    writer.release()
    for cap in caps:
        cap.release()
    if written == 0:
        raise RuntimeError(f"{case}: produced no comparison frames")
    return {
        "case": case,
        "without_audio_source_method": "perceptual_ground_v2",
        "with_audio_source_method": "perceptual_full_v2",
        "source_paths": [str(path) for path in paths],
        "source_frame_counts": counts,
        "comparison_frames": written,
        "fps": output_fps,
        "video": str(out),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case, sample in CASES:
        row = build(case, sample)
        rows.append(row)
        print(f"[{case}] {row['video']}", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
