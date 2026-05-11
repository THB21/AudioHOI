#!/usr/bin/env python3
"""Use known object labels to auto-initialize SAM2 with Grounding DINO."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

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
        threshold=box_threshold,
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
    cv2.imwrite(str(out_path), image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--detector-device", default="cpu")
    parser.add_argument("--sam2-model", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--fps", type=float, default=24.0)
    args = parser.parse_args()

    sample_dir = args.sample_dir
    metadata = json.loads((sample_dir / "metadata.json").read_text())
    frame_path = sample_dir / "frames" / "00001.png"
    detection_text = metadata.get("detection_text") or f"{metadata['name']}."

    box, info = detect_box(
        frame_path,
        detection_text,
        args.model_id,
        args.box_threshold,
        args.text_threshold,
        args.detector_device,
    )

    results_dir = sample_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "init_detection.json").write_text(json.dumps(info, indent=2))
    save_detection_preview(
        frame_path,
        results_dir / "init_detection_preview.png",
        box,
        str(info["label"]),
        float(info["score"]),
    )

    box_arg = ",".join(f"{v:.3f}" for v in box)
    cmd = [
        sys.executable,
        "-m",
        "scripts.manual_init.run_sam2_basketball",
        "--sample-dir",
        str(sample_dir),
        "--fps",
        str(args.fps),
        "--model",
        args.sam2_model,
        "--box",
        box_arg,
        "--points",
        "",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
