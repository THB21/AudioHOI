#!/usr/bin/env python3
"""Extract exactly 40 individual Full-HD frames, grouped chronologically by view."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


SPECS = (
    ("side", "view_1_side.mp4", 14),
    ("corner_left", "view_2_corner_left.mp4", 13),
    ("corner_right", "view_3_corner_right.mp4", 13),
)


def extract(video: Path, out: Path, prefix: str, count: int) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot decode {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    indices = np.linspace(0, max(total - 1, 0), count, dtype=int)
    for order, index in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {index} from {video}")
        timestamp = index / fps
        path = out / f"{prefix}_{order:02d}_t{timestamp:05.2f}s.jpg"
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"cannot write {path}")
    cap.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    for scene in ("scene_A", "scene_B"):
        scene_dir = args.root / scene
        out = scene_dir / "individual_40_frames"
        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob("*.jpg"):
            old.unlink()
        for label, filename, count in SPECS:
            extract(scene_dir / filename, out, label, count)
        files = sorted(out.glob("*.jpg"))
        if len(files) != 40:
            raise RuntimeError(f"expected 40 images, got {len(files)} in {out}")
        print(f"{out}: {len(files)} images")


if __name__ == "__main__":
    main()
