#!/usr/bin/env python3
"""Blind unary three-view HOI evaluation with Gemini 3.1 Pro Preview."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from google import genai
from google.genai import types


REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO / "deliverables/backward_basketball_vlm_audio_increment_ablation/gemini_unary"
DEFAULT_PROMPT = REPO / "deliverables/backward_basketball_vlm_audio_increment_ablation/GEMINI_UNARY_PROMPT.md"
MODEL = "gemini-3.1-pro-preview"
SCENES = ("scene_A", "scene_B")
VIEWS = ("view_1_side.mp4", "view_2_corner_left.mp4", "view_3_corner_right.mp4")


SCHEMA = {
    "type": "object",
    "properties": {
        **{name: {"type": "integer", "minimum": 1, "maximum": 5} for name in (
            "contact_timing", "contact_location", "object_motion", "physical_plausibility",
            "temporal_smoothness", "interaction_realism", "overall_quality"
        )},
        "body_interpenetration": {"type": "string", "enum": ["none", "minor", "major", "persistent"]},
        "excessive_object_speed": {"type": "string", "enum": ["none", "minor", "major", "persistent"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string"},
    },
    "required": [
        "contact_timing", "contact_location", "object_motion", "physical_plausibility",
        "temporal_smoothness", "interaction_realism", "overall_quality",
        "body_interpenetration", "excessive_object_speed", "confidence", "evidence",
    ],
    "additionalProperties": False,
}


def prompt_text(path: Path) -> str:
    text = path.read_text()
    marker = "## Prompt"
    return text.split(marker, 1)[1].strip() if marker in text else text


def wait_active(client: genai.Client, uploaded, timeout_s: int = 900):
    deadline = time.time() + timeout_s
    current = uploaded
    while time.time() < deadline:
        current = client.files.get(name=current.name)
        state = getattr(getattr(current, "state", None), "name", "")
        if state == "ACTIVE":
            return current
        if state == "FAILED":
            raise RuntimeError(f"Gemini file processing failed for {current.name}")
        time.sleep(3)
    raise TimeoutError(f"Gemini file processing timed out for {uploaded.name}")


def validate(payload: dict) -> None:
    for name in (
        "contact_timing", "contact_location", "object_motion", "physical_plausibility",
        "temporal_smoothness", "interaction_realism", "overall_quality",
    ):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"invalid {name}: {value!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--frames", action="store_true",
                    help="use six chronological 4-fps contact sheets instead of videos")
    ap.add_argument("--images40", action="store_true",
                    help="use exactly 40 individual images with explicit view/time context")
    args = ap.parse_args()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    out = args.out or args.root / "gemini_3_1_pro_unary_results.json"
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta"))
    prompt = prompt_text(args.prompt)
    if args.images40:
        prompt = prompt.replace(
            "The three attached videos are synchronized fixed renderings of",
            "The 40 attached individual images are chronological samples from"
        ).replace(
            "the same reconstruction at full resolution:",
            "the same reconstruction. Each image is preceded by its exact view and time in seconds:"
        ).replace("view_1_side.mp4", "side-view image group").replace(
            "view_2_corner_left.mp4", "front-left image group"
        ).replace("view_3_corner_right.mp4", "front-right image group")
    elif args.frames:
        prompt = prompt.replace(
            "The three attached videos are synchronized fixed renderings of",
            "The six attached chronological contact sheets are sampled at 4 frames per second from"
        ).replace(
            "the same reconstruction at full resolution:",
            "the same reconstruction. For each view, part 1 precedes part 2 in time:"
        ).replace("view_1_side.mp4", "view 1 sheets").replace(
            "view_2_corner_left.mp4", "view 2 sheets"
        ).replace("view_3_corner_right.mp4", "view 3 sheets")
    results = []
    for scene in SCENES:
        if args.images40:
            frame_dir = args.root / scene / "individual_40_frames"
            view_order = {"side": 0, "corner_left": 1, "corner_right": 2}
            local_paths = sorted(
                frame_dir.glob("*.jpg"),
                key=lambda path: (view_order[next(k for k in view_order if path.name.startswith(k))], path.name),
            )
            if len(local_paths) != 40:
                raise RuntimeError(f"expected exactly 40 images for {scene}, got {len(local_paths)}")
        elif args.frames:
            frame_dir = args.root / scene / "chronological_frames_4fps"
            local_paths = sorted(frame_dir.glob("view_*_part_*.png"))
        else:
            local_paths = [args.root / scene / name for name in VIEWS]
        missing = [str(path) for path in local_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing scene inputs: {missing}")
        uploaded = []
        try:
            if args.images40:
                media_contents = []
                view_labels = {
                    "side": "side view at 90 degrees",
                    "corner_left": "front-left corner view at -45 degrees",
                    "corner_right": "front-right corner view at +45 degrees",
                }
                for path in local_paths:
                    match = re.match(r"(side|corner_left|corner_right)_\d+_t([0-9.]+)s\.jpg", path.name)
                    if not match:
                        raise ValueError(f"cannot parse frame context from {path.name}")
                    view_key, timestamp = match.groups()
                    media_contents.append(types.Part.from_text(
                        text=f"View: {view_labels[view_key]}. Time: {timestamp} seconds."
                    ))
                    media_contents.append(types.Part.from_bytes(data=path.read_bytes(), mime_type="image/jpeg"))
            elif args.frames:
                media_contents = [
                    types.Part.from_bytes(data=path.read_bytes(), mime_type="image/png")
                    for path in local_paths
                ]
            else:
                for path in local_paths:
                    remote = client.files.upload(file=str(path), config={"mime_type": "video/mp4"})
                    uploaded.append(wait_active(client, remote))
                media_contents = uploaded
            response = client.models.generate_content(
                model=args.model,
                contents=[*media_contents, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_json_schema=SCHEMA,
                ),
            )
            payload = json.loads(response.text)
            validate(payload)
            results.append({"scene": scene, "model": args.model, **payload})
            print(f"[{scene}] overall={payload['overall_quality']}/5", flush=True)
        finally:
            for remote in uploaded:
                try:
                    client.files.delete(name=remote.name)
                except Exception:
                    pass
    protocol = (
        "blind_unary_exactly_40_individual_images_with_view_and_time_context"
        if args.images40 else
        ("blind_unary_three_views_chronological_4fps_sheets" if args.frames
         else "blind_unary_three_separate_fullhd_views")
    )
    out.write_text(json.dumps({"protocol": protocol, "results": results}, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
