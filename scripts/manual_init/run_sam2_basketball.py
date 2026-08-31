#!/usr/bin/env python3
"""Run SAM2 video segmentation for the basketball sample."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor


def parse_box(value: str) -> np.ndarray:
    vals = [float(v.strip()) for v in value.split(",")]
    if len(vals) != 4:
        raise ValueError("--box must be x1,y1,x2,y2")
    return np.array(vals, dtype=np.float32)


def parse_points(value: str | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not value:
        return None, None
    points: list[list[float]] = []
    labels: list[int] = []
    for item in value.split(";"):
        x, y, label = item.split(",")
        points.append([float(x), float(y)])
        labels.append(int(label))
    return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)


def prepare_jpg_frames(frames_dir: Path, jpg_dir: Path) -> list[Path]:
    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        raise RuntimeError(f"No PNG frames found in {frames_dir}")

    jpg_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(jpg_dir.glob("*.jpg"))
    if len(existing) == len(frame_paths):
        return existing

    for old in existing:
        old.unlink()
    for idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise RuntimeError(f"Could not read frame {frame_path}")
        cv2.imwrite(str(jpg_dir / f"{idx:05d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return sorted(jpg_dir.glob("*.jpg"))


def mask_stats(frame_idx_1based: int, mask: np.ndarray, fps: float) -> dict[str, object]:
    binary = (mask > 0).astype(np.uint8) * 255
    moments = cv2.moments(binary)
    area = moments["m00"] / 255.0
    if area <= 0:
        return {
            "frame": frame_idx_1based,
            "time": f"{(frame_idx_1based - 1) / fps:.6f}",
            "ball_center_x": "",
            "ball_center_y": "",
            "radius": "",
            "mask_area": 0,
        }
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    radius = float(np.sqrt(area / np.pi))
    return {
        "frame": frame_idx_1based,
        "time": f"{(frame_idx_1based - 1) / fps:.6f}",
        "ball_center_x": f"{cx:.3f}",
        "ball_center_y": f"{cy:.3f}",
        "radius": f"{radius:.3f}",
        "mask_area": f"{area:.1f}",
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, default=Path("samples/basketball_01"))
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--model", default="facebook/sam2.1-hiera-large")
    parser.add_argument("--box", default="470,320,570,430", help="First-frame basketball box x1,y1,x2,y2")
    parser.add_argument(
        "--points",
        default="520,380,1;520,325,0",
        help="Optional semicolon-separated x,y,label prompts. Label 1=positive, 0=negative.",
    )
    args = parser.parse_args()

    sample_dir = args.sample_dir
    results_dir = sample_dir / "results"
    masks_dir = results_dir / "masks"
    sam2_jpg_dir = results_dir / "sam2_jpg_frames"
    masks_dir.mkdir(parents=True, exist_ok=True)

    jpg_paths = prepare_jpg_frames(sample_dir / "frames", sam2_jpg_dir)
    box = parse_box(args.box)
    points, labels = parse_points(args.points)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = SAM2VideoPredictor.from_pretrained(args.model).to(device)

    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        autocast_enabled = device == "cuda"
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast_enabled):
            state = predictor.init_state(str(sam2_jpg_dir))
            predictor.add_new_points_or_box(
                state,
                frame_idx=0,
                obj_id=1,
                points=points,
                labels=labels,
                box=box,
            )
            for out_frame_idx, _object_ids, out_mask_logits in predictor.propagate_in_video(state):
                mask = (out_mask_logits[0] > 0.0).detach().cpu().numpy().squeeze()
                mask_u8 = mask.astype(np.uint8) * 255
                frame_idx_1based = int(out_frame_idx) + 1
                cv2.imwrite(str(masks_dir / f"{frame_idx_1based:05d}_mask.png"), mask_u8)
                rows.append(mask_stats(frame_idx_1based, mask_u8, args.fps))

    rows.sort(key=lambda row: int(row["frame"]))
    write_csv(
        results_dir / "ball_trajectory.csv",
        rows,
        ["frame", "time", "ball_center_x", "ball_center_y", "radius", "mask_area"],
    )
    print(f"sam2_model: {args.model}")
    print(f"frames: {len(jpg_paths)}")
    print(f"masks: {len(rows)}")
    print(f"trajectory: {results_dir / 'ball_trajectory.csv'}")


if __name__ == "__main__":
    main()
