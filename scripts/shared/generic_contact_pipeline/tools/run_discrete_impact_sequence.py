#!/usr/bin/env python3
"""Convert generic audio peaks into discrete impact-state evidence.

The output changes interaction-state activation only. It deliberately emits an
empty metric contact-constraint artifact because a click does not provide XYZ.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


EVENT_FIELDS = [
    "event", "event_type", "audio_time", "audio_frame", "start_time_s", "end_time_s",
    "start_frame", "end_frame", "peak", "prominence", "rms_rise", "sharpness",
    "audio_score", "snr", "band_profile", "detector", "source",
]
STATE_FIELDS = [
    "frame", "time", "contact_active", "human_contact_state", "contact_label",
    "contact_part", "contact_conf", "anchor_score", "anchor_update", "visibility", "source",
]
CONSTRAINT_FIELDS = [
    "frame", "time", "contact_active", "human_part", "human_side", "object_part",
    "object_local_id", "contact_u", "contact_v", "contact_depth_offset_m", "contact_conf",
    "anchor_score", "source",
]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _motion_turn_score(
    coordinates: dict[int, tuple[float, float]], frame: int, search_radius: int
) -> tuple[float, int]:
    """Return the strongest local 2-D direction-change evidence near an audio peak."""

    best = (0.0, frame)
    for center in range(frame - search_radius, frame + search_radius + 1):
        if center - 1 not in coordinates or center not in coordinates or center + 1 not in coordinates:
            continue
        previous = coordinates[center - 1]
        current = coordinates[center]
        following = coordinates[center + 1]
        incoming = (current[0] - previous[0], current[1] - previous[1])
        outgoing = (following[0] - current[0], following[1] - current[1])
        incoming_norm = math.hypot(*incoming)
        outgoing_norm = math.hypot(*outgoing)
        if incoming_norm <= 1e-6 or outgoing_norm <= 1e-6:
            continue
        cosine = max(
            -1.0,
            min(1.0, (incoming[0] * outgoing[0] + incoming[1] * outgoing[1]) / (incoming_norm * outgoing_norm)),
        )
        angle = math.acos(cosine) / math.pi
        delta_velocity = math.hypot(outgoing[0] - incoming[0], outgoing[1] - incoming[1])
        score = angle * min(incoming_norm, outgoing_norm) + 0.25 * delta_velocity
        if score > best[0]:
            best = (score, center)
    return best


def run(
    sample_dir: Path,
    result_dir: Path,
    minimum_score: float,
    disable_audio: bool = False,
    expected_impact_count: int | None = None,
    evidence_window_frames: int = 3,
    activation_radius_frames: int = 0,
) -> dict[str, object]:
    raw_events = read(sample_dir / "results/events/audio_events.csv")
    audio_candidates = [] if disable_audio else [
        row for row in raw_events
        if row.get("event", "").startswith("legacy_peak_")
        and int(float(row["audio_frame"])) > 2
        and float(row.get("audio_score", 0.0) or 0.0) >= minimum_score
    ]
    identity_path = result_dir / "sphere_identity_selection.csv"
    if identity_path.is_file():
        identity = read(identity_path)
    else:
        identity = [
            {"frame": row["frame"], "time": row.get("time", ""), "u": row.get("ref_u", ""), "v": row.get("ref_v", "")}
            for row in read(result_dir / "object_observations.csv")
        ]
    coordinates = {
        int(float(row["frame"])): (float(row["u"]), float(row["v"]))
        for row in identity
        if row.get("u", "") != "" and row.get("v", "") != ""
    }
    ranked: list[tuple[float, int, dict[str, str]]] = []
    for row in audio_candidates:
        audio_frame = int(float(row["audio_frame"]))
        motion_score, motion_frame = _motion_turn_score(coordinates, audio_frame, evidence_window_frames)
        audio_score = float(row.get("audio_score", 0.0) or 0.0)
        ranked.append((audio_score * max(motion_score, 1e-6), motion_frame, row))
    if expected_impact_count is not None and expected_impact_count > 0:
        ranked = sorted(ranked, key=lambda item: item[0], reverse=True)[:expected_impact_count]
    selected_with_motion = sorted(ranked, key=lambda item: int(float(item[2]["audio_frame"])))
    classified: list[dict[str, object]] = []
    event_by_frame: dict[int, dict[str, str]] = {}
    event_windows: dict[int, tuple[dict[str, str], str]] = {}
    matched_motion_frames: list[int] = []
    raw_audio_frames: list[int] = []
    for index, (_joint_score, motion_frame, row) in enumerate(selected_with_motion, start=1):
        raw_audio_frames.append(int(float(row["audio_frame"])))
        frame = motion_frame
        event_by_frame[frame] = row
        matched_motion_frames.append(motion_frame)
        contact_part = "paddle_face" if index % 2 else "practice_wall"
        for active_frame in range(frame - activation_radius_frames, frame + activation_radius_frames + 1):
            if active_frame in coordinates:
                event_windows[active_frame] = (row, contact_part)
        classified.append({
            **{field: row.get(field, "") for field in EVENT_FIELDS},
            "event": f"impact_{index:04d}",
            "event_type": "impact",
            "audio_frame": str(frame),
            "start_frame": str(frame),
            "end_frame": str(frame),
            "detector": "joint_audio_motion_impact_gate",
            "source": str(sample_dir / "audio.wav"),
        })
    write(result_dir / "contact_candidates_internal/audio_events.csv", EVENT_FIELDS, classified)

    states: list[dict[str, object]] = []
    constraints: list[dict[str, object]] = []
    for row in identity:
        frame = int(float(row["frame"]))
        window_event = event_windows.get(frame)
        event = window_event[0] if window_event else None
        contact_part = window_event[1] if window_event else "none"
        active = window_event is not None
        score = float(event.get("audio_score", 0.0)) if event else 0.0
        states.append({
            "frame": frame,
            "time": row.get("time", ""),
            "contact_active": "1" if active else "0",
            "human_contact_state": "1" if active else "0",
            "contact_label": "audio_timed_impact" if active else "none",
            "contact_part": contact_part,
            "contact_conf": f"{score:.6f}",
            "anchor_score": f"{score:.6f}",
            "anchor_update": "1" if active else "0",
            "visibility": "visible",
            "source": "joint_audio_motion_impact_state_no_metric_contact",
        })
        constraints.append({
            "frame": frame, "time": row.get("time", ""), "contact_active": "0",
            "human_part": "none", "human_side": "", "object_part": "none",
            "object_local_id": "", "contact_u": "", "contact_v": "",
            "contact_depth_offset_m": "", "contact_conf": "0.000000",
            "anchor_score": "0.000000", "source": "audio_has_no_metric_xyz",
        })
    write(result_dir / "impact_interaction_states.csv", STATE_FIELDS, states)
    write(result_dir / "sphere_no_metric_contact_constraints.csv", CONSTRAINT_FIELDS, constraints)
    manifest = {
        "schema_version": 1,
        "impact_count": len(classified),
        "impact_frames": sorted(event_by_frame),
        "raw_audio_frames": raw_audio_frames,
        "matched_motion_frames": matched_motion_frames,
        "active_window_frames": sorted(event_windows),
        "raw_audio_candidate_count": len(audio_candidates),
        "expected_impact_count": expected_impact_count,
        "evidence_window_frames": evidence_window_frames,
        "activation_radius_frames": activation_radius_frames,
        "minimum_audio_score": minimum_score,
        "audio_enabled": not disable_audio,
        "metric_contact_constraints": 0,
        "policy": "audio peak and visual direction-change jointly gate impact timing; audio never supplies 3D position",
    }
    (result_dir / "impact_sequence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--minimum-score", type=float, default=0.35)
    parser.add_argument("--disable-audio", action="store_true")
    parser.add_argument("--expected-impact-count", type=int)
    parser.add_argument("--evidence-window-frames", type=int, default=3)
    parser.add_argument("--activation-radius-frames", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(
        args.sample_dir,
        args.result_dir,
        args.minimum_score,
        args.disable_audio,
        args.expected_impact_count,
        args.evidence_window_frames,
        args.activation_radius_frames,
    ), indent=2))


if __name__ == "__main__":
    main()
