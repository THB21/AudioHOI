#!/usr/bin/env python3
"""Blind unary 1--5 VLM quality study for the nine AudioHOI ablation pairs.

Each reconstruction is scored independently from synchronized multi-view video.
The judge never receives the method name or whether audio was used. Pairwise
audio deltas are computed only after all unary judgments have been written.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np

from run_multiview_binary_perceptual import CASES, MODEL_DIR, prompt_for


REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "samples_known_object/hoi_interaction_evaluation/perceptual_unary_audio_ablation"
MODES = ("visual_vlm_only", "visual_vlm_plus_audio")
VIEWS = ("overlay", "camera3d", "side_yz")
SCORE_FIELDS = (
    "contact_timing",
    "contact_location",
    "object_motion",
    "physical_plausibility",
    "temporal_smoothness",
    "interaction_realism",
    "overall_quality",
)
INTERACTION_OVERRIDES = {
    "back_view_basketball": (
        "A person performs repeated backward-view basketball dribbles and crossovers. "
        "The ball should alternate between hand contact and floor contact, move causally "
        "after each hand or floor impact, remain near the appropriate hand during control, "
        "and follow continuous physically plausible arcs between contacts."
    ),
}


SYSTEM_PROMPT = """You are a rigorous evaluator of reconstructed 4D human-object
interaction scenes. You receive one reconstruction rendered as synchronized views.
Score only visible reconstruction quality. Do not infer the method, training setup,
or whether audio was used. Do not reward visual polish, camera angle, labels, or
background appearance. Concentrate on the reconstructed human-object interaction.
The render is intentionally an abstract scientific visualization with an untextured
body and simple object geometry. Never penalize missing texture, facial detail,
photorealism, simplified anatomy, neutral lighting, or visualization style. A score
of 1 requires an interaction or motion failure, not merely an abstract render.

Use the full integer scale relative to current scientific 4D-HOI reconstructions,
not relative to real footage or a production-quality animation:
1 = technical/reconstruction failure: the video is missing/corrupt, the human or
    object cannot be tracked, or no usable 4D scene is present
2 = the criterion is observable, but major or repeated errors strongly reduce quality
3 = the criterion is broadly plausible and readable, with clear local errors
4 = convincing and stable, with only minor visible errors
5 = exceptionally consistent for a reconstructed 4D-HOI sequence

Reserve 1 for missing or unusable reconstruction output. The mere presence of a gap,
detachment, penetration, simplified mesh, or imperfect event is not automatically a
1. If human and object tracks are visible and temporally coherent, distinguish 2 from
3 according to severity and duration. Use 4 and 5 when supported rather than avoiding
the top of the scale.

Evaluate the complete sequence, including approach, initial contact, sustained
contact, release, impacts, and periods without contact. A short correct event must
not compensate for long incorrect intervals. Judge uncertainty conservatively.
You must still assign every 1--5 score from the visible motion when evidence is
limited. Never replace a score with null, invalid, or not-assessable. Use confidence
only to express uncertainty and do not set all scores to 1 merely because the render
is abstract, simplified, single-view, or less detailed than a real video.
Score criteria independently. For example, incorrect contact can coexist with smooth
motion, and smooth motion can coexist with physically implausible object behavior.
Assigning 1 to all seven criteria is allowed only when every criterion independently
shows catastrophic failure through most of the sequence.
"""


def detailed_prompt(view_description: str) -> str:
    return f"""The video shows one reconstructed 4D HOI sequence using {view_description}.

Watch the whole sequence before scoring. Apply every criterion independently.
Infer the interaction directly from the rendered human and object motion. Base every
score only on visible spatial and temporal evidence in the supplied views. Do not use
an external text prompt, expected action label, audio, filename, or method identity.

contact_timing
Score whether contact starts, persists, switches, and ends at the visually correct
moments. Penalize early attraction, late response, missing brief impacts, a grasp
that is not maintained while the object is carried, and contact that continues
after release. Do not assume that proximity means contact.

contact_location
Score whether the correct body part touches the correct object region and stays
there without sliding or detaching. Penalize hand-object gaps, attachment to the
wrong hand or body part, contact on the wrong object side, and unstable grasp points.

object_motion
Score whether the object's translation, rotation, and articulation agree with the
human action. Penalize delayed motion after impact, motion before force is applied,
incorrect direction or speed, frozen objects, teleportation, and implausible spins.

physical_plausibility
Score non-penetration, support, gravity, and causal coupling. Penalize intersections,
floating, sinking, unexplained motion, loss of support, and a hand passing through
the object. Inspect the side and diagonal views specifically for the object passing
through the torso, limbs, or hands. Distinguish intended surface contact from visible
body-object interpenetration.

temporal_smoothness
Score continuity of human and object motion. Penalize jitter, frame-to-frame jumps,
abrupt pose changes without an impact, and drifting during a stable grasp. A real
impact may create a sharp velocity change and should not automatically be penalized.
Inspect whether the object covers an implausibly large distance between adjacent
frames, moves much faster than the human action can cause, or effectively teleports.
Repeated excessive speed must lower object_motion, temporal_smoothness, physical
plausibility, and overall_quality according to its severity and duration.

interaction_realism
Score whether the full human-object action reads as one coherent causal interaction
rather than independent human and object tracks.

overall_quality
Give a holistic 1--5 score, weighted toward interaction realism, contact timing,
contact location, and physical plausibility. Do not simply average mechanically.

Return exactly one JSON object and no Markdown:
{{"contact_timing":1,"contact_location":1,"object_motion":1,
"physical_plausibility":1,"temporal_smoothness":1,"interaction_realism":1,
"overall_quality":1,"body_interpenetration":"none|minor|major|persistent",
"excessive_object_speed":"none|minor|major|persistent","confidence":0.0,
"evidence":"brief sequence-specific reason naming the most important visible errors"}}
All seven scores must be integers from 1 to 5. Confidence must be between 0 and 1.
"""


def render_root(sample: Path, mode: str) -> Path:
    return sample / "results/renders/scientific_audio_ablation" / mode / "with_human"


def build_unary_video(paths: list[Path], out: Path, output_fps: float = 6.0,
                      view_names: tuple[str, ...] = VIEWS) -> dict:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing synchronized render views: {missing}")
    caps = [cv2.VideoCapture(str(p)) for p in paths]
    if not all(c.isOpened() for c in caps):
        raise RuntimeError(f"cannot decode one of {paths}")
    counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
    rates = [float(c.get(cv2.CAP_PROP_FPS) or 24.0) for c in caps]
    stride = max(1, int(round(min(rates) / output_fps)))
    out.parent.mkdir(parents=True, exist_ok=True)
    panel_width = 960 // len(paths)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), output_fps,
                             (panel_width * len(paths), 320))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {out}")
    written = 0
    for frame_idx in range(min(counts)):
        decoded = [c.read() for c in caps]
        if not all(ok for ok, _ in decoded):
            break
        if frame_idx % stride:
            continue
        panels = []
        for view, (_, frame) in zip(view_names, decoded):
            panel = cv2.resize(frame, (panel_width, 320), interpolation=cv2.INTER_AREA)
            cv2.rectangle(panel, (0, 0), (panel_width, 25), (0, 0, 0), -1)
            cv2.putText(panel, view, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(panel)
        writer.write(np.hstack(panels))
        written += 1
    writer.release()
    for cap in caps:
        cap.release()
    return {"sources": [str(p) for p in paths], "frames": written, "fps": output_fps, "video": str(out)}


def load_model():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL_DIR), trust_remote_code=True, device_map="auto", quantization_config=quant
    ).eval()
    return model, processor


def parse_score(raw: str, valid_render: bool = True) -> dict:
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError(f"VLM returned no JSON object: {raw[:300]}")
    data = json.loads(match.group(0))
    parsed = {}
    for field in SCORE_FIELDS:
        value = data.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or not 1 <= int(value) <= 5:
            raise ValueError(f"invalid {field}={value!r}")
        raw_value = int(value)
        parsed[f"raw_{field}"] = raw_value
        # A decoded video containing a visible reconstructed human and object is
        # not equivalent to a missing/corrupt pipeline output. Keep score 1 as a
        # technical-failure sentinel and calibrate severe visible HOI errors to 2.
        parsed[field] = max(2, raw_value) if valid_render else raw_value
    confidence = float(data.get("confidence", 0.0))
    parsed["confidence"] = max(0.0, min(1.0, confidence))
    severity = {"none", "minor", "major", "persistent"}
    for field in ("body_interpenetration", "excessive_object_speed"):
        value = str(data.get(field, "none")).lower()
        parsed[field] = value if value in severity else "none"
    parsed["evidence"] = str(data.get("evidence", ""))[:500]
    parsed["raw"] = raw
    return parsed


def judge(model, processor, video: Path, view_description: str) -> dict:
    import torch
    from qwen_vl_utils import process_vision_info
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "video", "video": str(video.resolve()), "fps": 6.0,
             "resized_height": 224, "resized_width": 672},
            {"type": "text", "text": detailed_prompt(view_description)},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, video_pairs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True, return_video_metadata=True
    )
    videos = [pair[0] for pair in video_pairs] if video_pairs else None
    video_metadata = [pair[1] for pair in video_pairs] if video_pairs else None
    inputs = processor(text=[text], images=images, videos=videos,
                       video_metadata=video_metadata, padding=True,
                       return_tensors="pt", **video_kwargs).to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    raw = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:],
                                 skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0].strip()
    return parse_score(raw, valid_render=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated case names")
    ap.add_argument("--output-fps", type=float, default=6.0)
    ap.add_argument("--backward-pilot", action="store_true",
                    help="score the strict back-view basketball pair using its moving-camera renders")
    args = ap.parse_args()
    selected = set(filter(None, (x.strip() for x in args.only.split(","))))
    cases = [(name, path) for name, path in CASES if not selected or name in selected]
    OUT.mkdir(parents=True, exist_ok=True)

    items = []
    missing = []
    for case, sample in cases:
        for mode in MODES:
            if args.backward_pilot and case == "back_view_basketball":
                short_mode = "vlm_only" if mode == "visual_vlm_only" else "vlm_plus_audio"
                diagnostic = (REPO / "deliverables/backward_basketball_vlm_audio_increment_ablation" /
                              short_mode / "diagnostic_views")
                paths = [diagnostic / f"{name}_world.mp4" for name in
                         ("side_90", "front_minus45", "front_plus45")]
                view_names = ("fixed side 90 degrees", "front corner -45 degrees",
                              "front corner +45 degrees")
                view_description = (
                    "three synchronized fixed diagnostic views: side at 90 degrees, "
                    "front-left corner at -45 degrees, and front-right corner at +45 degrees"
                )
            else:
                paths = [render_root(sample, mode) / f"{view}.mp4" for view in VIEWS]
                view_names = VIEWS
                view_description = "three synchronized input-camera, moving 3D-camera, and side views"
            out = OUT / "blind_videos" / f"{case}_{mode}.mp4"
            try:
                video_meta = build_unary_video(paths, out, args.output_fps, view_names)
                items.append({"case": case, "sample": str(sample), "mode": mode,
                              "blind_id": f"scene_{len(items)+1:03d}",
                              "view_description": view_description, **video_meta})
            except FileNotFoundError as exc:
                missing.append({"case": case, "mode": mode, "reason": str(exc)})
    (OUT / "video_manifest.json").write_text(json.dumps({"items": items, "missing": missing}, indent=2) + "\n")
    if args.build_only:
        print(json.dumps({"built": len(items), "missing": missing}, indent=2))
        return

    model, processor = load_model()
    records = []
    sample_by_case = dict(cases)
    # Method identity remains outside judge() so it cannot leak into the prompt.
    for item in items:
        case = item["case"]
        result = judge(model, processor, Path(item["video"]), item["view_description"])
        record = {"case": case, "mode": item["mode"], "blind_id": item["blind_id"], **result}
        records.append(record)
        print(f"[unary] {case} {item['mode']} overall={result['overall_quality']}", flush=True)

    with (OUT / "unary_scores.csv").open("w", newline="") as handle:
        fields = (["case", "mode", "blind_id", *SCORE_FIELDS] +
                  [f"raw_{field}" for field in SCORE_FIELDS] +
                  ["body_interpenetration", "excessive_object_speed",
                   "confidence", "evidence", "raw"])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (OUT / "unary_scores.json").write_text(json.dumps(records, indent=2) + "\n")

    by_key = {(r["case"], r["mode"]): r for r in records}
    deltas = []
    for case, _ in cases:
        left = by_key.get((case, "visual_vlm_only"))
        right = by_key.get((case, "visual_vlm_plus_audio"))
        if not left or not right:
            continue
        row = {"case": case}
        for field in SCORE_FIELDS:
            row[f"without_audio_{field}"] = left[field]
            row[f"with_audio_{field}"] = right[field]
            row[f"audio_delta_{field}"] = right[field] - left[field]
        deltas.append(row)
    (OUT / "paired_audio_deltas.json").write_text(json.dumps(deltas, indent=2) + "\n")
    if deltas:
        with (OUT / "paired_audio_deltas.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(deltas[0]))
            writer.writeheader()
            writer.writerows(deltas)
    print(json.dumps({"scored": len(records), "complete_pairs": len(deltas), "missing": missing}, indent=2))


if __name__ == "__main__":
    main()
