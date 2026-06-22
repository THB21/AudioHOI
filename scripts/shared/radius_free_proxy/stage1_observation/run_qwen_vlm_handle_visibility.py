#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

PROMPT = """You are inspecting a single video frame crop around a mug.
Task: produce visual evidence for mug handle visibility and contact reasoning.
Important definitions:
- The handle is the C-shaped side loop attached to the mug.
- A hand touching the mug body/rim is NOT handle contact unless the fingers are clearly on the handle loop.
- If the handle should exist but is behind the mug body/hand or on the far side, mark handle_visible=false and occlusion_reason accordingly.
- Ignore rendered overlays, colored debug circles/curves, skeletons, and text if present. Judge the real image content.
- "far_side" means the handle is on the back side of the mug relative to the camera and is hidden by the mug body itself.
- "handle_contact" should only be true when the visible hand/fingers are clearly on the C-shaped handle loop, not merely on the cup body.
- "yaw_anchor_quality" is high only when the handle/rim/body silhouette gives useful evidence for mug rotation; it is low if the handle is hidden and only the cup body is visible.
Return ONLY compact JSON with keys:
{
  "handle_visible": true/false,
  "visibility": "visible" | "hidden" | "uncertain",
  "visible_side": "left" | "right" | "front" | "back" | "unknown",
  "occlusion_reason": "none" | "hand" | "mug_body" | "far_side" | "out_of_crop" | "uncertain",
  "handle_shape_visible": "clear_c_loop" | "partial_arc" | "small_attachment" | "not_visible" | "uncertain",
  "handle_body_relation": "attached_and_consistent" | "appears_detached" | "not_visible" | "uncertain",
  "hand_contact_visible": true/false,
  "hand_contact_part": "handle" | "body" | "rim" | "unknown" | "none",
  "handle_contact": true/false,
  "body_contact": true/false,
  "rim_contact": true/false,
  "hand_occludes_handle": true/false,
  "body_self_occludes_handle": true/false,
  "yaw_anchor_quality": "high" | "medium" | "low" | "none",
  "recommended_visibility_constraint": "force_visible" | "force_hidden_far_side" | "force_hidden_by_hand" | "weak_unknown",
  "confidence": 0.0-1.0,
  "short_reason": "one short sentence"
}
"""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def fval(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except Exception:
        return default


def crop_mug(frame_path: Path, row: dict[str, str], pad: float = 0.65) -> tuple[Image.Image, list[int]]:
    img = cv2.imread(str(frame_path))
    if img is None:
        raise FileNotFoundError(frame_path)
    h, w = img.shape[:2]
    x1 = fval(row, "bbox_x1", fval(row, "body_bbox_x1"))
    y1 = fval(row, "bbox_y1", fval(row, "body_bbox_y1"))
    x2 = fval(row, "bbox_x2", fval(row, "body_bbox_x2"))
    y2 = fval(row, "bbox_y2", fval(row, "body_bbox_y2"))
    if x2 <= x1 or y2 <= y1:
        cx = fval(row, "center_x", fval(row, "body_center_x", w / 2))
        cy = fval(row, "center_y", fval(row, "body_center_y", h / 2))
        bw = fval(row, "bbox_w_px", 80.0)
        bh = fval(row, "bbox_h_px", 80.0)
        x1, x2 = cx - bw / 2, cx + bw / 2
        y1, y2 = cy - bh / 2, cy + bh / 2
    bw = x2 - x1
    bh = y2 - y1
    # Give VLM enough context: handle is often outside the body bbox and hand can occlude it.
    cx1 = max(0, int(round(x1 - pad * bw)))
    cx2 = min(w, int(round(x2 + pad * bw)))
    cy1 = max(0, int(round(y1 - 0.35 * bh)))
    cy2 = min(h, int(round(y2 + 0.45 * bh)))
    crop = img[cy1:cy2, cx1:cx2]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb), [cx1, cy1, cx2, cy2]


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"visibility": "uncertain", "handle_visible": False, "confidence": 0.0, "raw_text": text}
    try:
        data = json.loads(m.group(0))
    except Exception:
        return {"visibility": "uncertain", "handle_visible": False, "confidence": 0.0, "raw_text": text}
    return data


def load_model(model_id: str, local_dir: str | None, device_map: str, load_4bit: bool):
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as ModelCls
    except Exception:
        from transformers import AutoModelForVision2Seq as ModelCls

    model_path = local_dir or model_id
    kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": device_map}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
    else:
        kwargs["torch_dtype"] = "auto"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = ModelCls.from_pretrained(model_path, **kwargs)
    return model, processor


def ask_vlm(model, processor, image: Image.Image, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        except Exception:
            inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    except Exception:
        inputs = processor(text=[PROMPT], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[:, inputs.input_ids.shape[-1]:]
    return processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Qwen VLM to label mug handle visibility/contact semantics.")
    ap.add_argument("--sample-dir", type=Path, default=Path("samples_known_object/02_mug"))
    ap.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--local-dir", default=None, help="Optional local snapshot path from ModelScope/HF.")
    ap.add_argument("--frames", default="", help="Comma list or ranges, e.g. 30-90,100,118. Empty = all observation frames.")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--save-crops", action="store_true")
    args = ap.parse_args()

    rows = read_csv(args.sample_dir / "results" / "object_observations" / "object_observations.csv")
    by_frame = {int(float(r["frame"])): r for r in rows}
    if args.frames:
        frames: list[int] = []
        for chunk in args.frames.split(','):
            chunk = chunk.strip()
            if not chunk:
                continue
            if '-' in chunk:
                a, b = [int(x) for x in chunk.split('-', 1)]
                frames.extend(range(a, b + 1, args.stride))
            else:
                frames.append(int(chunk))
        frames = [f for f in frames if f in by_frame]
    else:
        frames = sorted(by_frame)[:: max(1, args.stride)]

    out_dir = args.out_dir or (args.sample_dir / "annotations" / "vlm_handle_visibility")
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = out_dir / "crops"
    if args.save_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(args.model_id, args.local_dir, args.device_map, args.load_4bit)
    results = []
    for fr in frames:
        row = by_frame[fr]
        print(f"[qwen-vlm] frame {fr}", file=sys.stderr, flush=True)
        img_path = args.sample_dir / "frames" / f"{fr:05d}.png"
        crop, bbox = crop_mug(img_path, row)
        if args.save_crops:
            crop.save(crop_dir / f"{fr:05d}_mug_crop.png")
        raw = ask_vlm(model, processor, crop, args.max_new_tokens)
        data = extract_json(raw)
        item = {"frame": fr, "time": row.get("time", ""), "crop_bbox_xyxy": bbox, "raw_text": raw, **data}
        results.append(item)
        print(json.dumps(item, ensure_ascii=False))

    json_path = out_dir / "qwen_handle_visibility.json"
    json_path.write_text(json.dumps({"model_id": args.model_id, "local_dir": args.local_dir, "prompt": PROMPT, "frames": results}, indent=2, ensure_ascii=False))
    csv_path = out_dir / "qwen_handle_visibility.csv"
    keys = [
        "frame", "time", "handle_visible", "visibility", "visible_side", "occlusion_reason",
        "handle_shape_visible", "handle_body_relation",
        "hand_contact_visible", "hand_contact_part", "handle_contact", "body_contact", "rim_contact",
        "hand_occludes_handle", "body_self_occludes_handle", "yaw_anchor_quality",
        "recommended_visibility_constraint", "confidence", "short_reason",
    ]
    with csv_path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in results:
            wr.writerow({k: r.get(k, "") for k in keys})
    print("json:", json_path)
    print("csv:", csv_path)


if __name__ == "__main__":
    main()
