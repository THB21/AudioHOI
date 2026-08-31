#!/usr/bin/env python3
"""Counterbalanced VLM binary perceptual study for all nine AudioHOI ablations.

Each method is shown as a synchronized three-view video (overlay, camera3d, side_yz).
The judge sees A over B and then B over A.  A method wins a rubric only when both
orders map to the same method; inconsistent answers become ties.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "samples_known_object/hoi_interaction_evaluation/perceptual_binary"
MODEL_DIR = REPO / "models/modelscope/Qwen/Qwen3-VL-8B-Instruct"
VIEWS = ("overlay", "camera3d", "side_yz")


CASES = [
    ("basketball", REPO / "samples_known_object/01_basketball"),
    ("football", REPO / "samples_known_object/10_football"),
    ("mug", REPO / "samples_known_object/02_mug"),
    ("chair", REPO / "samples_known_object/05_chair"),
    ("stick", REPO / "samples_known_object/11_stick"),
    ("back_view_basketball", REPO / "samples_known_object/12_back_view_basketball"),
    ("volleyball", REPO / "samples_known_object/13_volleyball"),
    ("pingpong", REPO / "samples_known_object/14_pingpong_wall"),
    ("suitcase", REPO / "samples_known_object/15_suitcase_drag"),
]


def prompt_for(case: str, sample: Path) -> str:
    meta = sample / "metadata.json"
    if meta.exists():
        data = json.loads(meta.read_text())
        return str(data.get("prompt") or data.get("video_prompt") or "")
    prompts = json.loads((REPO / "video_sample/prompts.json").read_text())
    aliases = {"stick": "broom"}
    name = aliases.get(case, case)
    for row in prompts:
        if row.get("name") == name:
            return str(row.get("prompt", ""))
    return case.replace("_", " ")


def view_paths(sample: Path, method: str) -> list[Path]:
    root = sample / "results/renders" / f"perceptual_{method}_v2" / "with_human"
    return [root / f"{view}.mp4" for view in VIEWS]


def _open_all(paths: list[Path]) -> list[cv2.VideoCapture]:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing views: {missing}")
    caps = [cv2.VideoCapture(str(path)) for path in paths]
    if not all(cap.isOpened() for cap in caps):
        raise RuntimeError(f"cannot decode one of: {paths}")
    return caps


def build_pair_video(case: str, sample: Path, top: str, bottom: str, out: Path,
                     output_fps: float = 6.0) -> dict:
    paths = view_paths(sample, top) + view_paths(sample, bottom)
    caps = _open_all(paths)
    counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    source_fps = [float(cap.get(cv2.CAP_PROP_FPS) or 24.0) for cap in caps]
    n = min(counts)
    stride = max(1, int(round(min(source_fps) / output_fps)))
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (960, 360))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {out}")
    written = 0
    for frame_idx in range(n):
        frames = []
        ok_all = True
        for cap in caps:
            ok, frame = cap.read()
            if not ok:
                ok_all = False
                break
            frames.append(frame)
        if not ok_all:
            break
        if frame_idx % stride:
            continue
        panels = []
        for idx, frame in enumerate(frames):
            panel = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            row = "A" if idx < 3 else "B"
            view = VIEWS[idx % 3]
            cv2.rectangle(panel, (0, 0), (320, 24), (0, 0, 0), -1)
            cv2.putText(panel, f"{row} | {view}", (8, 17), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(panel)
        writer.write(np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])]))
        written += 1
    writer.release()
    for cap in caps:
        cap.release()
    return {"case": case, "top_A": top, "bottom_B": bottom, "source_frames": n,
            "output_frames": written, "fps": output_fps, "video": str(out)}


def load_model():
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    import torch

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL_DIR), trust_remote_code=True, device_map="auto", quantization_config=quant
    ).eval()
    return model, processor


def parse_answer(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.S)
    data = {}
    if match:
        try:
            data = json.loads(match.group(0))
        except Exception:
            data = {}
    valid = {"A", "B", "TIE"}
    return {
        "realism": str(data.get("realism", "TIE")).upper()
        if str(data.get("realism", "")).upper() in valid else "TIE",
        "text_alignment": str(data.get("text_alignment", "TIE")).upper()
        if str(data.get("text_alignment", "")).upper() in valid else "TIE",
        "reason": str(data.get("reason", ""))[:300],
        "raw": raw,
    }


def judge(model, processor, video: Path, interaction_prompt: str) -> dict:
    from qwen_vl_utils import process_vision_info
    import torch

    question = f"""You are evaluating two reconstructions of the SAME 4D human-object
interaction. Row A and row B each show three synchronized views: input-camera overlay,
3D scene, and Y-Z side view. The method names are hidden.

Original interaction prompt:
{interaction_prompt}

Watch the complete motion across all three views. Judge separately:
1. REALISM: contact, physical plausibility, motion continuity, and lack of penetration.
2. TEXT ALIGNMENT: which reconstruction better matches the requested interaction.

Choose A, B, or TIE for each. Do not prefer a screen position. Return JSON only:
{{"realism":"A|B|TIE","text_alignment":"A|B|TIE","reason":"brief"}}"""
    messages = [{"role": "user", "content": [
        {"type": "video", "video": str(video.resolve()), "fps": 1.0,
         "resized_height": 252, "resized_width": 672},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, _video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True,
        return_tensors="pt", fps=1.0
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    trimmed = generated[:, inputs.input_ids.shape[1]:]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0].strip()
    return parse_answer(raw)


def method_winner(label: str, top: str, bottom: str) -> str:
    return top if label == "A" else (bottom if label == "B" else "TIE")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    selected = set(filter(None, args.only.split(",")))
    cases = [(case, sample) for case, sample in CASES if not selected or case in selected]
    OUT.mkdir(parents=True, exist_ok=True)
    manifests = []
    for case, sample in cases:
        for top, bottom, suffix in (("ground", "full", "gf"), ("full", "ground", "fg")):
            out = OUT / "videos" / f"{case}_{suffix}.mp4"
            manifests.append(build_pair_video(case, sample, top, bottom, out))
            print(f"[video] {case} {top}/{bottom} -> {out}", flush=True)
    (OUT / "video_manifest.json").write_text(json.dumps(manifests, indent=2))
    if args.build_only:
        return

    model, processor = load_model()
    records = []
    by_case = {case: sample for case, sample in cases}
    for case, sample in cases:
        prompt = prompt_for(case, sample)
        order_results = []
        for top, bottom, suffix in (("ground", "full", "gf"), ("full", "ground", "fg")):
            video = OUT / "videos" / f"{case}_{suffix}.mp4"
            result = judge(model, processor, video, prompt)
            mapped = {
                rubric: method_winner(result[rubric], top, bottom)
                for rubric in ("realism", "text_alignment")
            }
            record = {"case": case, "top_A": top, "bottom_B": bottom, **result,
                      "mapped_realism": mapped["realism"],
                      "mapped_text_alignment": mapped["text_alignment"]}
            records.append(record)
            order_results.append(record)
            print(f"[judge] {case} {top}/{bottom}: realism={result['realism']} "
                  f"alignment={result['text_alignment']} -> {mapped}", flush=True)
        for rubric in ("realism", "text_alignment"):
            winners = [row[f"mapped_{rubric}"] for row in order_results]
            winner = winners[0] if winners[0] == winners[1] else "TIE"
            records[-2][f"counterbalanced_{rubric}"] = winner
            records[-1][f"counterbalanced_{rubric}"] = winner

    (OUT / "records.json").write_text(json.dumps(records, indent=2))
    summary_rows = []
    for case, _ in cases:
        pair = [row for row in records if row["case"] == case]
        summary_rows.append({
            "case": case,
            "realism_winner": pair[0]["counterbalanced_realism"],
            "text_alignment_winner": pair[0]["counterbalanced_text_alignment"],
            "ground_first_realism": pair[0]["realism"],
            "full_first_realism": pair[1]["realism"],
            "ground_first_alignment": pair[0]["text_alignment"],
            "full_first_alignment": pair[1]["text_alignment"],
        })
    counts = {}
    for rubric in ("realism_winner", "text_alignment_winner"):
        counts[rubric] = {method: sum(row[rubric] == method for row in summary_rows)
                          for method in ("ground", "full", "TIE")}
    payload = {"protocol": "three_view_video_binary_counterbalanced",
               "model": str(MODEL_DIR), "cases": summary_rows, "counts": counts}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2))
    with (OUT / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
