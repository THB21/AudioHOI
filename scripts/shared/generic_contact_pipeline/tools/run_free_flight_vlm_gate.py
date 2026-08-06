#!/usr/bin/env python3
"""Ask a forced-choice VLM whether a non-contact sphere loop is usable evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[4]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.gates.vlm_provider import load_vlm_provider  # noqa: E402
from scripts.shared.generic_contact_pipeline.stages.gates.stage_vlm_qwen import (  # noqa: E402
    ask_qwen,
    extract_json,
    load_model,
    load_query_image,
)


CHOICES = [
    "paddle_contact",
    "practice_wall_contact",
    "floor_contact",
    "no_visible_contact",
    "unclear",
]


def _read_frame(capture: cv2.VideoCapture, frame: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
    ok, image = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame}")
    return image


def _evidence(
    video: Path,
    pose: pd.DataFrame,
    candidate: dict[str, object],
    output: Path,
    camera: tuple[float, float, float, float],
) -> list[int]:
    start = int(candidate["start_frame"])
    stop = int(candidate["stop_frame"])
    peak = int(candidate.get("peak_turn_frame_before", (start + stop) // 2))
    frames = sorted(
        {
            start,
            int(round(start + 0.25 * (stop - start))),
            max(start, peak - 1),
            peak,
            min(stop, peak + 1),
            int(round(start + 0.75 * (stop - start))),
            stop,
        }
    )
    capture = cv2.VideoCapture(str(video))
    panels: list[np.ndarray] = []
    fx, fy, cx, cy = camera
    indexed = pose.set_index("frame")
    for frame in frames:
        image = _read_frame(capture, frame)
        row = indexed.loc[frame]
        u = int(round(fx * float(row.tx) / float(row.tz) + cx))
        v = int(round(fy * float(row.ty) / float(row.tz) + cy))
        cv2.circle(image, (u, v), 22, (0, 0, 255), 3, cv2.LINE_AA)
        role = "CONTACT ENDPOINT" if frame in {start, stop} else "NO CONTACT"
        cv2.rectangle(image, (0, 0), (image.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(
            image,
            f"frame {frame} | {role}",
            (18, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA))
    capture.release()
    while len(panels) < 8:
        panels.append(np.zeros_like(panels[0]))
    sheet = np.vstack((np.hstack(panels[:4]), np.hstack(panels[4:8])))
    banner = np.zeros((90, sheet.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        banner,
        "TARGET = WHITE BALL INSIDE RED CIRCLE. IGNORE HUMAN AND PADDLE MOTION.",
        (35, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        (0, 255, 255),
        3,
        cv2.LINE_AA,
    )
    sheet = np.vstack((banner, sheet))
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"Could not write {output}")
    return frames


def main() -> None:
    provider = load_vlm_provider()
    provider.apply_env_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--sphere-pose", type=Path, required=True)
    parser.add_argument("--repair-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--model-id", default=provider.model_id)
    parser.add_argument("--local-dir", default=str(provider.local_dir) if provider.local_dir else "")
    parser.add_argument("--device-map", default=provider.device_map)
    parser.add_argument("--load-4bit", action="store_true", default=provider.load_4bit)
    parser.add_argument("--resize-max", type=int, default=provider.resize_max)
    parser.add_argument("--max-new-tokens", type=int, default=provider.max_new_tokens)
    args = parser.parse_args()

    manifest = json.loads(args.repair_manifest.read_text())
    candidates = [
        row
        for row in manifest.get("free_flight_repairs", [])
        if row.get("reason") == "unsupported_free_flight_direction_reversal"
    ]
    pose = pd.read_csv(args.sphere_pose)
    model, processor = load_model(args.model_id, args.local_dir or None, args.device_map, args.load_4bit)
    decisions: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates, start=1):
        start, stop = int(candidate["start_frame"]), int(candidate["stop_frame"])
        evidence = args.output_dir / "evidence" / f"free_flight_{start:05d}_{stop:05d}.jpg"
        frames = _evidence(
            args.video,
            pose,
            candidate,
            evidence,
            (args.fx, args.fy, args.cx, args.cy),
        )
        question = (
            "Panels are chronological observations of one tracked white sphere. Judge ONLY the "
            "white ball inside each red circle and ignore human motion. The numerical trajectory "
            "detector found a direction reversal at the central peak-turn panels. What visible "
            "contact, if any, physically explains that reversal at the red-circled ball? Select "
            "floor_contact when it reaches or bounces from the floor. Select no_visible_contact "
            "when it reverses while separated from paddle, practice wall, and floor. Do not judge "
            "trajectory smoothness and do not output pose."
        )
        print(f"[qwen-free-flight] {index}/{len(candidates)} interval={start}-{stop}", flush=True)
        raw_text = ask_qwen(
            model,
            processor,
            load_query_image(evidence, args.resize_max),
            question,
            CHOICES,
            args.max_new_tokens,
            query_type="free_flight_consistency_check",
        )
        parsed = extract_json(raw_text, CHOICES)
        label = str(parsed["label"])
        confidence = float(parsed["confidence"])
        approved = bool(label in {"floor_contact", "no_visible_contact"} and confidence >= 0.55)
        decisions.append(
            {
                "start_frame": start,
                "stop_frame": stop,
                "evidence_frames": frames,
                "evidence_path": str(evidence.resolve()),
                "label": label,
                "confidence": confidence,
                "short_reason": parsed["short_reason"],
                "declared_allowed_contacts": ["paddle_contact", "practice_wall_contact"],
                "approved_repair": approved,
                "repair_action": "replace_with_physical_flight_arc" if approved else "keep_visual_observation",
            }
        )
    payload = {
        "schema_version": 1,
        "query_type": "free_flight_consistency_check",
        "choices": CHOICES,
        "continuous_pose_fields_allowed": False,
        "model": args.model_id,
        "decisions": decisions,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "free_flight_vlm_decisions.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(output), "decisions": decisions}, indent=2))


if __name__ == "__main__":
    main()
