#!/usr/bin/env python3
"""Blind unary Gemini evaluation of the 18 final nine-case world renders."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

from google import genai
from google.genai import types


REPO = Path(__file__).resolve().parents[3]
WORLD_ROOT = REPO / "deliverables/nine_case_visual_vlm_audio_ablation/world_results"
OUT_ROOT = REPO / "deliverables/nine_case_visual_vlm_audio_ablation/unary_vlm_evaluation"
MODEL = "gemini-3.6-flash"
CASES = (
    "basketball", "football", "mug", "chair", "stick",
    "back_view_basketball", "volleyball", "pingpong", "suitcase",
)
ARMS = ("vlm", "vlm_audio")
FIELDS = (
    "contact_timing", "contact_location", "object_motion",
    "physical_plausibility", "temporal_smoothness", "interaction_realism",
    "overall_quality",
)
SCHEMA = {
    "type": "object",
    "properties": {
        **{field: {"type": "integer", "minimum": 1, "maximum": 5} for field in FIELDS},
        "body_interpenetration": {
            "type": "string", "enum": ["none", "minor", "major", "persistent"]
        },
        "excessive_object_speed": {
            "type": "string", "enum": ["none", "minor", "major", "persistent"]
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string"},
    },
    "required": [
        *FIELDS, "body_interpenetration", "excessive_object_speed",
        "confidence", "evidence",
    ],
    "additionalProperties": False,
}

PROMPT = """You are evaluating one reconstructed 4D human-object interaction scene.
The attached video is a clean scientific world render. It contains an animated
SMPL-X human with reconstructed HaMeR hands and a reconstructed object. The camera
orbits around the scene so that spatial errors become visible from different angles.

Judge only what is visible in this one video. Do not infer the method. Do not infer
whether audio was used. Do not reward texture, lighting, camera style, facial detail,
or photorealism. The abstract appearance is intentional.

Watch the complete sequence before scoring. Use the full integer scale.

1 means a technical or interaction failure through most of the sequence
2 means major or repeated visible errors
3 means a readable and broadly plausible interaction with clear local errors
4 means a convincing and stable reconstruction with only minor errors
5 means exceptional consistency for reconstructed 4D HOI

Score contact timing by whether contact starts, persists, changes, and ends at
plausible moments. Score contact location by whether the correct body region meets
the correct object region without a visible gap or unstable sliding. Score object
motion by whether translation and rotation agree with the human action. Score
physical plausibility by checking support, gravity, causal coupling, penetration,
and unexplained motion. Score temporal smoothness by checking jitter, jumps, drift,
and excessive speed while allowing genuine impacts. Score interaction realism by
judging whether human and object motion form one causal interaction. Give an overall
quality score that emphasizes contact and physical plausibility.

A visible human and object must not receive 1 merely because the reconstruction is
abstract. Score each criterion independently. Return only the requested JSON.
"""


def wait_active(client: genai.Client, uploaded, timeout_s: int = 900):
    deadline = time.time() + timeout_s
    current = uploaded
    while time.time() < deadline:
        current = client.files.get(name=current.name)
        state = getattr(getattr(current, "state", None), "name", "")
        if state == "ACTIVE":
            return current
        if state == "FAILED":
            raise RuntimeError(f"file processing failed for {current.name}")
        time.sleep(3)
    raise TimeoutError(f"file processing timed out for {uploaded.name}")


def validate(payload: dict) -> None:
    for field in FIELDS:
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"invalid {field} value {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    records_path = OUT_ROOT / "unary_scores.json"
    records = json.loads(records_path.read_text()) if args.resume and records_path.exists() else []
    complete = {(row["case"], row["arm"]) for row in records}
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta"))

    blind_items = []
    for case in CASES:
        for arm in ARMS:
            blind_items.append((f"scene_{len(blind_items) + 1:03d}", case, arm,
                                WORLD_ROOT / case / arm / "world.mp4"))

    for blind_id, case, arm, video in blind_items:
        if (case, arm) in complete:
            continue
        if not video.exists():
            raise FileNotFoundError(video)
        remote = None
        try:
            remote = client.files.upload(file=str(video), config={"mime_type": "video/mp4"})
            remote = wait_active(client, remote)
            response = client.models.generate_content(
                model=args.model,
                contents=[remote, PROMPT],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_json_schema=SCHEMA,
                ),
            )
            payload = json.loads(response.text)
            validate(payload)
            records.append({
                "blind_id": blind_id,
                "case": case,
                "arm": arm,
                "model": args.model,
                **payload,
            })
            records_path.write_text(json.dumps(records, indent=2) + "\n")
            print(f"{blind_id} overall {payload['overall_quality']}/5", flush=True)
        finally:
            if remote is not None:
                try:
                    client.files.delete(name=remote.name)
                except Exception:
                    pass

    by_key = {(row["case"], row["arm"]): row for row in records}
    paired = []
    for case in CASES:
        without = by_key[(case, "vlm")]
        with_audio = by_key[(case, "vlm_audio")]
        row = {"case": case}
        for field in FIELDS:
            row[f"vlm_{field}"] = without[field]
            row[f"vlm_audio_{field}"] = with_audio[field]
            row[f"audio_delta_{field}"] = with_audio[field] - without[field]
        paired.append(row)
    (OUT_ROOT / "paired_scores.json").write_text(json.dumps(paired, indent=2) + "\n")
    with (OUT_ROOT / "unary_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (OUT_ROOT / "paired_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    summary = {}
    for arm in ARMS:
        subset = [row for row in records if row["arm"] == arm]
        summary[arm] = {
            field: sum(row[field] for row in subset) / len(subset) for field in FIELDS
        }
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
