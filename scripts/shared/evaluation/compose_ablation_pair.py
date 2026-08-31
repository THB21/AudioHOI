#!/usr/bin/env python3
"""Compose two same-length ablation renders with fixed labels."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--left-label", default="VLM ONLY")
    parser.add_argument("--right-label", default="VLM + AUDIO")
    args = parser.parse_args()

    left = cv2.VideoCapture(str(args.left))
    right = cv2.VideoCapture(str(args.right))
    if not left.isOpened() or not right.isOpened():
        raise RuntimeError("could not open both input videos")
    lw = int(left.get(cv2.CAP_PROP_FRAME_WIDTH))
    lh = int(left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rw = int(right.get(cv2.CAP_PROP_FRAME_WIDTH))
    rh = int(right.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = left.get(cv2.CAP_PROP_FPS) or 24.0
    height = min(lh, rh)
    left_width = round(lw * height / lh)
    right_width = round(rw * height / rh)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (left_width + right_width, height),
    )
    frames = 0
    while True:
        lok, limg = left.read()
        rok, rimg = right.read()
        if not lok or not rok:
            break
        limg = cv2.resize(limg, (left_width, height))
        rimg = cv2.resize(rimg, (right_width, height))
        for image, label in ((limg, args.left_label), (rimg, args.right_label)):
            cv2.rectangle(image, (22, 20), (360, 72), (15, 15, 15), -1)
            cv2.putText(image, label, (38, 57), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(cv2.hconcat([limg, rimg]))
        frames += 1
    left.release()
    right.release()
    writer.release()
    if frames == 0:
        raise RuntimeError("no paired frames were written")
    print(f"wrote {args.out} ({frames} paired frames at {fps:.3f} fps)")


if __name__ == "__main__":
    main()
