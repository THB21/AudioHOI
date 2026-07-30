from __future__ import annotations

import csv
import json
from pathlib import Path

from .types import FrameInteractionState, InteractionTimeline, frame_record


def _interval_key(state: FrameInteractionState) -> tuple[str, str, str, str]:
    return (
        state.visibility_state.value,
        state.contact_state.value,
        state.contact_mode.value,
        state.motion_mode.value,
    )


def interaction_intervals(timeline: InteractionTimeline) -> list[dict[str, object]]:
    if not timeline.frames:
        return []
    intervals: list[dict[str, object]] = []
    start = timeline.frames[0]
    previous = start
    current_key = _interval_key(start)
    for state in timeline.frames[1:]:
        key = _interval_key(state)
        if key != current_key:
            intervals.append(
                {
                    "start_frame": start.frame,
                    "end_frame": previous.frame,
                    "start_time": f"{start.time:.6f}",
                    "end_time": f"{previous.time:.6f}",
                    "visibility_state": current_key[0],
                    "contact_state": current_key[1],
                    "contact_mode": current_key[2],
                    "motion_mode": current_key[3],
                }
            )
            start = state
            current_key = key
        previous = state
    intervals.append(
        {
            "start_frame": start.frame,
            "end_frame": previous.frame,
            "start_time": f"{start.time:.6f}",
            "end_time": f"{previous.time:.6f}",
            "visibility_state": current_key[0],
            "contact_state": current_key[1],
            "contact_mode": current_key[2],
            "motion_mode": current_key[3],
        }
    )
    return intervals


def write_interaction_timeline(timeline: InteractionTimeline, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "interaction_timeline.jsonl"
    intervals_path = out_dir / "interaction_intervals.csv"
    metrics_path = out_dir / "interaction_state_metrics.json"
    with jsonl_path.open("w") as handle:
        for state in timeline.frames:
            handle.write(json.dumps(frame_record(state), ensure_ascii=False, sort_keys=True) + "\n")
    intervals = interaction_intervals(timeline)
    fields = [
        "start_frame",
        "end_frame",
        "start_time",
        "end_time",
        "visibility_state",
        "contact_state",
        "contact_mode",
        "motion_mode",
    ]
    with intervals_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(intervals)
    metrics_path.write_text(json.dumps(timeline.metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return {
        "timeline": jsonl_path,
        "intervals": intervals_path,
        "metrics": metrics_path,
    }
