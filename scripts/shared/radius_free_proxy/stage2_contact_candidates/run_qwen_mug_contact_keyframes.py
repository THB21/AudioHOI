#!/usr/bin/env python3
"""Ask a VLM for fine-grained mug hand-contact labels on confirmed keyframes.

This is deliberately keyframe-only. It should not relabel hidden/drinking frames
or create new per-frame anchors. The output is meant to refine stable
object-local grasp regions such as:

    right/left hand grips mug.handle_middle_outer with thumb/index/palm.

The exact 3D anchor is still computed from the Articraft handle mesh; VLM only
provides semantic region and visible 2D evidence on clear frames.
"""

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


PROMPT = """Inspect this mug-hand crop. Ignore overlays/text if any. Use tracker_candidate_hand_side from context as hand_side unless clearly impossible. Label only the visible hand-mug grasp in this confirmed keyframe. Do not treat mouth/rim drinking as hand-handle grasp. Return ONLY one-line valid JSON:
{"contact_visible":true/false,"hand_side":"left_hand|right_hand|unknown","contact_fingers":["thumb|index|middle|ring|pinky|palm|unknown"],"primary_contact_finger":"thumb|index|middle|ring|pinky|palm|unknown","object_part":"handle|body|rim|bottom|unknown","object_region":"upper_handle|middle_handle|lower_handle|handle_inner|handle_outer|body_side|rim_front|rim_side|unknown","handle_grasp_type":"pinch_handle|hook_handle|palm_support|body_grasp|not_handle_grasp|unknown","use_as_stable_grasp_keyframe":true/false,"confidence":0.0-1.0,"reason":"short"}
"""


CSV_FIELDS = [
    "frame",
    "time",
    "contact_visible",
    "hand_side",
    "contact_fingers",
    "primary_contact_finger",
    "object_part",
    "object_region",
    "handle_grasp_type",
    "contact_point_2d_u",
    "contact_point_2d_v",
    "visible_evidence",
    "use_as_stable_grasp_keyframe",
    "confidence",
    "reason",
    "crop_bbox_x1",
    "crop_bbox_y1",
    "crop_bbox_x2",
    "crop_bbox_y2",
    "state_active_label",
    "state_stable_grasp_local_x",
    "state_stable_grasp_local_y",
    "state_stable_grasp_local_z",
    "state_source_frame",
    "raw_text",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def f(row: dict[str, str] | None, key: str, default: float = 0.0) -> float:
    if row is None:
        return default
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_frames(spec: str) -> list[int]:
    frames: list[int] = []
    if not spec:
        return frames
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = [int(x) for x in chunk.split("-", 1)]
            frames.extend(range(a, b + 1))
        else:
            frames.append(int(chunk))
    return sorted(dict.fromkeys(frames))


def representative_anchor_frames(state_rows: list[dict[str, str]]) -> list[int]:
    """Pick a few clear frames from each direct-anchor run."""
    frames = [int(r["frame"]) for r in state_rows if r.get("frame_mode") == "direct_grasp_anchor"]
    if not frames:
        return []
    runs: list[list[int]] = []
    cur = [frames[0]]
    for fr in frames[1:]:
        if fr == cur[-1] + 1:
            cur.append(fr)
        else:
            runs.append(cur)
            cur = [fr]
    runs.append(cur)
    selected: list[int] = []
    for run in runs:
        selected.extend([run[0], run[len(run) // 2], run[-1]])
    return sorted(dict.fromkeys(selected))


def crop_contact_context(
    frame_path: Path,
    obs: dict[str, str],
    contact: dict[str, str] | None,
    margin: int = 150,
    min_size: int = 420,
) -> tuple[Image.Image, list[int]]:
    img = cv2.imread(str(frame_path))
    if img is None:
        raise FileNotFoundError(frame_path)
    h, w = img.shape[:2]

    xs = [
        f(obs, "bbox_x1", w / 2),
        f(obs, "bbox_x2", w / 2),
        f(obs, "body_bbox_x1", w / 2),
        f(obs, "body_bbox_x2", w / 2),
    ]
    ys = [
        f(obs, "bbox_y1", h / 2),
        f(obs, "bbox_y2", h / 2),
        f(obs, "body_bbox_y1", h / 2),
        f(obs, "body_bbox_y2", h / 2),
    ]
    for key_x, key_y in [
        ("contact_u", "contact_v"),
        ("active_part_u", "active_part_v"),
        ("hand_contact_u", "hand_contact_v"),
    ]:
        x = f(contact, key_x, -1)
        y = f(contact, key_y, -1)
        if x >= 0 and y >= 0:
            xs.append(x)
            ys.append(y)

    x1 = int(min(xs) - margin)
    y1 = int(min(ys) - margin)
    x2 = int(max(xs) + margin)
    y2 = int(max(ys) + margin)

    # Make the crop large enough for finger-level inspection. Keep it square-ish
    # so VLM sees wrist, fingers, mug body, and handle in one context window.
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    side = max(min_size, x2 - x1, y2 - y1)
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    x1 = max(0, x2 - side)
    y1 = max(0, y2 - side)

    crop = img[y1:y2, x1:x2]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb), [x1, y1, x2, y2]


def extract_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text.strip(), re.S)
    if not m:
        return {"raw_text": text, "contact_visible": False, "confidence": 0.0}
    try:
        out = json.loads(m.group(0))
    except Exception:
        return {"raw_text": text, "contact_visible": False, "confidence": 0.0}
    out["raw_text"] = text
    return out


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

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
        )
    else:
        kwargs["torch_dtype"] = "auto"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = ModelCls.from_pretrained(model_path, **kwargs)
    return model, processor


def ask_vlm(model, processor, image: Image.Image, max_new_tokens: int, context: str = "") -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT + ("\n\nContext hint:\n" + context if context else "")},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    except Exception:
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[:, inputs.input_ids.shape[-1] :]
    return processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def flatten_result(
    frame: int,
    time: str,
    crop_bbox: list[int],
    state: dict[str, str],
    result: dict[str, Any],
) -> dict[str, str]:
    cp = result.get("contact_point_2d")
    if isinstance(cp, list) and len(cp) >= 2:
        u, v = cp[0], cp[1]
    else:
        u, v = "", ""
    return {
        "frame": str(frame),
        "time": time,
        "contact_visible": str(result.get("contact_visible", "")),
        "hand_side": str(result.get("hand_side", "")),
        "contact_fingers": json.dumps(result.get("contact_fingers", []), ensure_ascii=False),
        "primary_contact_finger": str(result.get("primary_contact_finger", "")),
        "object_part": str(result.get("object_part", "")),
        "object_region": str(result.get("object_region", "")),
        "handle_grasp_type": str(result.get("handle_grasp_type", "")),
        "contact_point_2d_u": str(u),
        "contact_point_2d_v": str(v),
        "visible_evidence": str(result.get("visible_evidence", "")),
        "use_as_stable_grasp_keyframe": str(result.get("use_as_stable_grasp_keyframe", "")),
        "confidence": str(result.get("confidence", "")),
        "reason": str(result.get("reason", "")),
        "crop_bbox_x1": str(crop_bbox[0]),
        "crop_bbox_y1": str(crop_bbox[1]),
        "crop_bbox_x2": str(crop_bbox[2]),
        "crop_bbox_y2": str(crop_bbox[3]),
        "state_active_label": state.get("active_label", ""),
        "state_stable_grasp_local_x": state.get("stable_grasp_local_x", ""),
        "state_stable_grasp_local_y": state.get("stable_grasp_local_y", ""),
        "state_stable_grasp_local_z": state.get("stable_grasp_local_z", ""),
        "state_source_frame": state.get("stable_grasp_source_frame", ""),
        "raw_text": str(result.get("raw_text", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, default=Path("samples_known_object/02_mug"))
    parser.add_argument(
        "--state-csv",
        type=Path,
        default=Path("samples_known_object/02_mug/results/mug_grasp_anchor_state/mug_grasp_anchor_state.csv"),
    )
    parser.add_argument(
        "--contact-csv",
        type=Path,
        default=Path("samples_known_object/02_mug/results/mug_articraft_contact_points/mug_articraft_contact_points.csv"),
    )
    parser.add_argument("--frames", default="", help="Optional comma/range list. Empty = representative direct-anchor frames.")
    parser.add_argument("--out-dir", type=Path, default=Path("samples_known_object/02_mug/annotations/vlm_mug_contact_keyframes"))
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=260)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--crop-margin", type=int, default=150)
    parser.add_argument("--crop-min-size", type=int, default=420)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = args.out_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    obs_rows = read_csv(args.sample_dir / "results" / "object_observations" / "object_observations.csv")
    state_rows = read_csv(args.state_csv)
    contact_rows = read_csv(args.contact_csv)
    obs_by_frame = {int(float(r["frame"])): r for r in obs_rows}
    state_by_frame = {int(float(r["frame"])): r for r in state_rows}
    contact_by_frame = {int(float(r["frame"])): r for r in contact_rows}

    frames = parse_frames(args.frames) or representative_anchor_frames(state_rows)
    frames = [fr for fr in frames if fr in obs_by_frame and fr in state_by_frame]
    if not frames:
        raise RuntimeError("No keyframes selected.")

    model = processor = None
    if not args.prepare_only:
        model, processor = load_model(args.model_id, args.local_dir, args.device_map, args.load_4bit)

    rows_out: list[dict[str, str]] = []
    json_items: list[dict[str, Any]] = []
    for fr in frames:
        frame_path = args.sample_dir / "frames" / f"{fr:05d}.png"
        crop, bbox = crop_contact_context(
            frame_path,
            obs_by_frame[fr],
            contact_by_frame.get(fr),
            margin=args.crop_margin,
            min_size=args.crop_min_size,
        )
        crop_path = crop_dir / f"{fr:05d}_contact_crop.png"
        crop.save(crop_path)

        state = state_by_frame[fr]
        if args.prepare_only:
            result = {
                "contact_visible": "",
                "hand_side": state.get("active_label", ""),
                "contact_fingers": [],
                "primary_contact_finger": "",
                "object_part": "",
                "object_region": "",
                "handle_grasp_type": "",
                "contact_point_2d": None,
                "visible_evidence": "",
                "use_as_stable_grasp_keyframe": "",
                "confidence": "",
                "reason": "prepared crop only",
                "raw_text": "",
            }
        else:
            print(f"[qwen-contact-keyframe] frame {fr}", file=sys.stderr, flush=True)
            context = (
                f"tracker_candidate_hand_side={state.get('active_label', '')}; "
                f"crop_width={crop.size[0]}; crop_height={crop.size[1]}; "
                f"state_frame_mode={state.get('frame_mode', '')}; "
                f"stable_grasp_source_frame={state.get('stable_grasp_source_frame', '')}"
            )
            raw = ask_vlm(model, processor, crop, args.max_new_tokens, context=context)
            result = extract_json(raw)

        item = {
            "frame": fr,
            "time": state.get("time", ""),
            "crop_path": str(crop_path),
            "crop_bbox_xyxy": bbox,
            "state": state,
            "vlm": result,
        }
        json_items.append(item)
        rows_out.append(flatten_result(fr, state.get("time", ""), bbox, state, result))

    with (args.out_dir / "mug_contact_keyframe_annotations.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)

    (args.out_dir / "mug_contact_keyframe_annotations.json").write_text(
        json.dumps({"prompt": PROMPT, "frames": json_items}, indent=2, ensure_ascii=False) + "\n"
    )
    (args.out_dir / "prompt.txt").write_text(PROMPT)

    print(f"frames: {frames}")
    print(f"crops: {crop_dir}")
    print(f"csv: {args.out_dir / 'mug_contact_keyframe_annotations.csv'}")
    print(f"json: {args.out_dir / 'mug_contact_keyframe_annotations.json'}")


if __name__ == "__main__":
    main()
