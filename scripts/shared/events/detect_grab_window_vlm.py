#!/usr/bin/env python3
"""VLM grab->release window detector (Qwen3-VL).

The audio persistence machine fragments a sustained grasp into separate touches when the
hand moves during the hold (no impulse, no_contact events between) — so L_rest gets short
intervals instead of one grab->release window. This asks a VLM per frame "is the hand
holding the object?" and merges the contiguous YES run into ONE sustained keep_grasp
interval, written in the contact_records schema so L_rest (object-side or Stage C) can
consume it directly.

Runs in the `qwen-vl` env (Qwen3-VL-8B, 4-bit). Example:
  conda run -n qwen-vl python scripts/shared/events/detect_grab_window_vlm.py \
    --frames-dir samples_known_object/05_chair/frames --object chair \
    --model-dir models/modelscope/Qwen/Qwen3-VL-8B-Instruct \
    --out-csv samples_known_object/05_chair/results/human_audio_semantics/vlm_grab_intervals.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

_PROMPT = (
    "Look at this single video frame. Is the person's HAND actively grasping or holding the "
    "{obj} (fingers on it, supporting/moving it) — not just near it? Reply with ONLY compact JSON:\n"
    '{{"holding": true|false, "hand": "left|right|none"}}'
)


def _parse(ans: str) -> dict:
    m = re.search(r"\{.*\}", ans, re.DOTALL)
    if not m:
        low = ans.lower()
        return {"holding": low.startswith("yes") or '"holding": true' in low,
                "hand": next((h for h in ("left", "right") if h in low), "none")}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"holding": False, "hand": "none"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--object", type=str, default="object")
    ap.add_argument("--model-dir", type=str, default="models/modelscope/Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--stride", type=int, default=3, help="query every Nth frame (interpolate between)")
    ap.add_argument("--resize-max", type=int, default=768)
    ap.add_argument("--gap-fill", type=int, default=6, help="bridge YES gaps up to this many frames into one interval")
    ap.add_argument("--min-len", type=int, default=4, help="drop grab windows shorter than this many frames")
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    frames = sorted(args.frames_dir.glob("*.png")) or sorted(args.frames_dir.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"no frames in {args.frames_dir}")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    proc = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, quantization_config=bnb, device_map="cuda", trust_remote_code=True).eval()
    prompt = _PROMPT.format(obj=args.object)

    holding = {}   # frame_index(1-based) -> (bool, hand)
    for k in range(0, len(frames), args.stride):
        img = Image.open(frames[k]).convert("RGB")
        if max(img.size) > args.resize_max:
            s = args.resize_max / max(img.size)
            img = img.resize((int(img.size[0] * s), int(img.size[1] * s)))
        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            gen = model.generate(**inp, max_new_tokens=40, do_sample=False)
        ans = proc.batch_decode(gen[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        j = _parse(ans)
        holding[k + 1] = (bool(j.get("holding", False)), str(j.get("hand", "none")).lower())
        print(f"  frame {k+1:4d}: {ans[:60]}")

    # densify: hold value of the nearest queried frame
    queried = sorted(holding)
    dense = {}
    for f in range(1, len(frames) + 1):
        nearest = min(queried, key=lambda q: abs(q - f))
        dense[f] = holding[nearest]

    # contiguous YES runs (with gap fill) -> intervals
    intervals = []
    f = 1
    N = len(frames)
    while f <= N:
        if not dense[f][0]:
            f += 1
            continue
        lo = f
        hand_votes = {}
        last_yes = f
        g = f
        while g <= N:
            if dense[g][0]:
                last_yes = g
                h = dense[g][1]
                if h in ("left", "right"):
                    hand_votes[h] = hand_votes.get(h, 0) + 1
            elif g - last_yes > args.gap_fill:
                break
            g += 1
        hi = last_yes
        if hi - lo + 1 >= args.min_len and hand_votes:
            hand = max(hand_votes, key=hand_votes.get)
            intervals.append((lo, hi, f"{hand}_hand"))
        f = hi + 1

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["frame", "refined_frame", "time", "contact_state", "interval_id",
            "stable_entity", "target_entity", "contact_target", "relevant", "audio_support", "source"]
    with args.out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for iid, (lo, hi, part) in enumerate(intervals):
            for fr in range(lo, hi + 1):
                w.writerow({"frame": fr, "refined_frame": fr, "time": "",
                            "contact_state": "keep_grasp", "interval_id": iid,
                            "stable_entity": part, "target_entity": part, "contact_target": "part",
                            "relevant": 1, "audio_support": 0.0, "source": "vlm_grab"})
    print(f"\nVLM grab windows: {[(lo, hi, p) for lo, hi, p in intervals]}")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
