#!/usr/bin/env python3
"""Run SAM2 segmentation for one radius-free object sample."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sam2.sam2_video_predictor import SAM2VideoPredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


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




def detect_box(
    frame_path: Path,
    text: str,
    model_id: str,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> tuple[list[float], dict[str, object]]:
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    image = Image.open(frame_path).convert("RGB")

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    if len(results["boxes"]) == 0:
        raise RuntimeError(f"No detection found for text={text!r} on {frame_path}")

    scores = results["scores"].detach().cpu().tolist()
    boxes = results["boxes"].detach().cpu().tolist()
    text_labels = results.get("text_labels", [])
    best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
    return boxes[best_idx], {
        "text": text,
        "model_id": model_id,
        "score": scores[best_idx],
        "box": boxes[best_idx],
        "label": text_labels[best_idx] if text_labels else text,
    }


def save_detection_preview(frame_path: Path, out_path: Path, box: list[float], text: str, score: float) -> None:
    image = cv2.imread(str(frame_path))
    if image is None:
        return
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        image,
        f"{text} {score:.2f}",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def run_sam2_propagation(sample_dir: Path, fps: float, model: str, box: np.ndarray, points: np.ndarray | None, labels: np.ndarray | None) -> tuple[int, Path]:
    results_dir = sample_dir / "results"
    segmentation_dir = results_dir / "segmentation"
    masks_dir = segmentation_dir / "masks"
    sam2_jpg_dir = segmentation_dir / "sam2_jpg_frames"
    masks_dir.mkdir(parents=True, exist_ok=True)

    jpg_paths = prepare_jpg_frames(sample_dir / "frames", sam2_jpg_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = SAM2VideoPredictor.from_pretrained(model).to(device)

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
                rows.append(mask_stats(frame_idx_1based, mask_u8, fps))

    rows.sort(key=lambda row: int(row["frame"]))
    trajectory_csv = results_dir / "ball_trajectory.csv"
    write_csv(
        trajectory_csv,
        rows,
        ["frame", "time", "ball_center_x", "ball_center_y", "radius", "mask_area"],
    )
    print(f"sam2_model: {model}")
    print(f"frames: {len(jpg_paths)}")
    print(f"masks: {len(rows)}")
    print(f"masks_dir: {masks_dir}")
    print(f"trajectory: {trajectory_csv}")
    return len(rows), trajectory_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM2 segmentation with either auto DINO box or manual first-frame box.")
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--sam2-model", "--model", dest="sam2_model", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--box", default="", help="Manual first-frame object box x1,y1,x2,y2. If omitted, GroundingDINO detects it.")
    parser.add_argument("--points", default="", help="Optional semicolon-separated x,y,label prompts. Label 1=positive, 0=negative.")
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--detector-device", default="cpu")
    args = parser.parse_args()

    sample_dir = args.sample_dir
    results_dir = sample_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    frame_path = sample_dir / "frames" / "00001.png"

    if args.box:
        box = parse_box(args.box)
        info = {
            "text": "manual_box",
            "model_id": "manual",
            "score": 1.0,
            "box": box.tolist(),
            "label": "manual_box",
        }
    else:
        metadata = json.loads((sample_dir / "metadata.json").read_text())
        detection_text = metadata.get("detection_text") or str(metadata["name"]) + "."
        detected_box, info = detect_box(
            frame_path,
            detection_text,
            args.model_id,
            args.box_threshold,
            args.text_threshold,
            args.detector_device,
        )
        box = np.asarray(detected_box, dtype=np.float32)

    (results_dir / "init_detection.json").write_text(json.dumps(info, indent=2))
    save_detection_preview(
        frame_path,
        results_dir / "init_detection_preview.png",
        [float(v) for v in box.tolist()],
        str(info["label"]),
        float(info["score"]),
    )
    points, labels = parse_points(args.points)
    run_sam2_propagation(sample_dir, args.fps, args.sam2_model, box, points, labels)


if __name__ == "__main__":
    main()
