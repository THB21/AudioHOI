#!/usr/bin/env python3
"""Build chronological Full-HD contact sheets from three HOI diagnostic views."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


VIEWS = ("view_1_side.mp4", "view_2_corner_left.mp4", "view_3_corner_right.mp4")


def sampled_frames(path: Path, sample_fps: float) -> list[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot decode {path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    stride = max(1, int(round(source_fps / sample_fps)))
    rows = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            rows.append((idx / source_fps, frame))
        idx += 1
    cap.release()
    return rows


def make_sheet(items: list[tuple[float, np.ndarray]], view_label: str, part: int) -> np.ndarray:
    tile_w, tile_h = 384, 270
    image_h = 216
    sheet = np.full((1080, 1920, 3), 245, dtype=np.uint8)
    for slot, (timestamp, frame) in enumerate(items):
        row, col = divmod(slot, 5)
        x, y = col * tile_w, row * tile_h
        resized = cv2.resize(frame, (tile_w, image_h), interpolation=cv2.INTER_AREA)
        sheet[y + 30:y + 30 + image_h, x:x + tile_w] = resized
        cv2.rectangle(sheet, (x, y), (x + tile_w - 1, y + tile_h - 1), (40, 40, 40), 1)
        cv2.putText(sheet, f"{view_label}  part {part}  t={timestamp:05.2f}s",
                    (x + 7, y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(sheet, f"frame {slot + 1 + (part - 1) * 20}",
                    (x + 7, y + 263), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                    (20, 20, 20), 1, cv2.LINE_AA)
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=4.0)
    args = ap.parse_args()
    labels = ("side 90 deg", "front corner -45 deg", "front corner +45 deg")
    for scene in ("scene_A", "scene_B"):
        source = args.root / scene
        out = source / "chronological_frames_4fps"
        out.mkdir(parents=True, exist_ok=True)
        for view_index, (name, label) in enumerate(zip(VIEWS, labels), start=1):
            frames = sampled_frames(source / name, args.fps)
            for part, start in enumerate(range(0, len(frames), 20), start=1):
                sheet = make_sheet(frames[start:start + 20], label, part)
                path = out / f"view_{view_index}_part_{part}.png"
                if not cv2.imwrite(str(path), sheet):
                    raise RuntimeError(f"cannot write {path}")
        print(out)


if __name__ == "__main__":
    main()
