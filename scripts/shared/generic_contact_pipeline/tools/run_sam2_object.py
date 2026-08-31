#!/usr/bin/env python3
"""Run first-frame Grounding DINO + SAM2 video segmentation for a sample.

This is the generic version of the original basketball SAM2 preprocessing
script: DINO supplies the first-frame box, SAM2 propagates masks, and the
output contract is the existing `results/segmentation/masks` plus
`results/tracking/object_trajectory.csv`.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_DINO_MODEL = "IDEA-Research/grounding-dino-base"
DEFAULT_SAM2_MODEL = "facebook/sam2.1-hiera-large"


def read_case_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    import yaml

    with path.open() as f:
        return yaml.safe_load(f) or {}


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
        frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No image frames found in {frames_dir}")

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
            "source": "sam2_video_propagation",
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
        "source": "sam2_video_propagation",
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def detect_first_frame_box(
    image_path: Path,
    text: str,
    *,
    model_id: str = DEFAULT_DINO_MODEL,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    device: str | None = None,
) -> dict[str, Any]:
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    image = Image.open(image_path).convert("RGB")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]], device=device)
    post_process = processor.post_process_grounded_object_detection
    kwargs: dict[str, Any] = {
        "outputs": outputs,
        "input_ids": inputs.input_ids,
        "text_threshold": text_threshold,
        "target_sizes": target_sizes,
    }
    if "box_threshold" in inspect.signature(post_process).parameters:
        kwargs["box_threshold"] = box_threshold
    else:
        kwargs["threshold"] = box_threshold
    results = post_process(**kwargs)[0]
    scores = results.get("scores")
    boxes = results.get("boxes")
    labels = results.get("labels", [])
    if scores is None or boxes is None or len(scores) == 0:
        raise RuntimeError(f"Grounding DINO found no box for text={text!r} in {image_path}")
    best_idx = int(torch.argmax(scores).item())
    label = labels[best_idx] if labels else text
    return {
        "text": text,
        "model_id": model_id,
        "score": float(scores[best_idx].detach().cpu().item()),
        "box": [float(v) for v in boxes[best_idx].detach().cpu().tolist()],
        "label": str(label),
    }


def run_sam2(
    *,
    sample_dir: Path,
    fps: float,
    box: np.ndarray,
    points: np.ndarray | None,
    labels: np.ndarray | None,
    model_id: str,
) -> dict[str, object]:
    import torch
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    segmentation_dir = sample_dir / "results" / "segmentation"
    masks_dir = segmentation_dir / "masks"
    sam2_jpg_dir = segmentation_dir / "sam2_jpg_frames"
    tracking_dir = sample_dir / "results" / "tracking"
    masks_dir.mkdir(parents=True, exist_ok=True)
    tracking_dir.mkdir(parents=True, exist_ok=True)

    jpg_paths = prepare_jpg_frames(sample_dir / "frames", sam2_jpg_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = SAM2VideoPredictor.from_pretrained(model_id).to(device)

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
                box=box.astype(np.float32),
            )
            for out_frame_idx, _object_ids, out_mask_logits in predictor.propagate_in_video(state):
                mask = (out_mask_logits[0] > 0.0).detach().cpu().numpy().squeeze()
                mask_u8 = mask.astype(np.uint8) * 255
                frame_idx_1based = int(out_frame_idx) + 1
                cv2.imwrite(str(masks_dir / f"{frame_idx_1based:05d}_mask.png"), mask_u8)
                rows.append(mask_stats(frame_idx_1based, mask_u8, fps))

    rows.sort(key=lambda row: int(row["frame"]))
    trajectory = tracking_dir / "object_trajectory.csv"
    write_csv(
        trajectory,
        rows,
        ["frame", "time", "ball_center_x", "ball_center_y", "radius", "mask_area", "source"],
    )
    return {
        "sam2_model": model_id,
        "frames": len(jpg_paths),
        "masks": len(rows),
        "masks_dir": str(masks_dir),
        "trajectory": str(trajectory),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-config", type=Path, default=None)
    parser.add_argument("--sample-dir", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--sam2-model", default=None)
    parser.add_argument("--dino-model", default=None)
    parser.add_argument("--dino-text", default=None, help="Grounding DINO text prompt, e.g. 'wooden staff.'")
    parser.add_argument("--box", default=None, help="Optional first-frame box x1,y1,x2,y2. If omitted, --dino-text is required.")
    parser.add_argument("--points", default=None, help="Optional semicolon-separated x,y,label SAM2 prompts.")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    args = parser.parse_args()

    case_config = read_case_config(args.case_config)
    preprocess = case_config.get("preprocess", {}) if isinstance(case_config, dict) else {}
    sample_dir = args.sample_dir or Path(case_config.get("sample_dir", ""))
    if not str(sample_dir):
        raise ValueError("--sample-dir or case_config.sample_dir is required")
    sam2_model = args.sam2_model or preprocess.get("segmenter_model") or DEFAULT_SAM2_MODEL
    dino_model = args.dino_model or preprocess.get("detector_model") or DEFAULT_DINO_MODEL
    dino_text = args.dino_text or preprocess.get("detector_text")
    first_frame = sorted((sample_dir / "frames").glob("*.png"))[0]
    if args.box:
        box = parse_box(args.box)
        detection = {
            "text": dino_text or "manual_box",
            "model_id": "manual_box",
            "score": 1.0,
            "box": [float(v) for v in box.tolist()],
            "label": dino_text or "manual_box",
        }
    else:
        if not dino_text:
            raise ValueError("--dino-text is required when --box is not provided")
        detection = detect_first_frame_box(
            first_frame,
            dino_text,
            model_id=dino_model,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
        box = np.array(detection["box"], dtype=np.float32)

    (sample_dir / "results").mkdir(parents=True, exist_ok=True)
    with (sample_dir / "results" / "init_detection.json").open("w") as f:
        json.dump(detection, f, indent=2)

    points, labels = parse_points(args.points)
    summary = run_sam2(
        sample_dir=sample_dir,
        fps=args.fps,
        box=box,
        points=points,
        labels=labels,
        model_id=sam2_model,
    )
    print(json.dumps({"init_detection": detection, **summary}, indent=2))


if __name__ == "__main__":
    main()
