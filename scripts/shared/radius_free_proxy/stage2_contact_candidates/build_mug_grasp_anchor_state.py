#!/usr/bin/env python3
"""Build a stable object-local grasp-anchor state for the mug sequence.

This script turns the per-frame Articraft contact diagnostics into a state
track that separates:
  - confirmed hand-handle grasp anchors,
  - hidden/occluded frames that should reuse the previous anchor,
  - rim drinking contacts that must not overwrite the hand anchor,
  - debug/misaligned/no-contact frames.

The output is intentionally conservative: only frames explicitly marked by the
M13 contact export as `use_this_point_for_hand_attachment == 1` are allowed to
update the object-local anchor.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "frame",
    "time",
    "object_contact_event",
    "hand_mug_contact_state",
    "nearest_articraft_part",
    "semantic_contact_part",
    "active_label",
    "vlm_visibility",
    "vlm_hand_contact_part",
    "vlm_handle_contact",
    "use_this_point_for_hand_attachment",
    "use_previous_grasp_for_hand_attachment",
    "rim_drinking_contact",
    "frame_mode",
    "stable_grasp_local_x",
    "stable_grasp_local_y",
    "stable_grasp_local_z",
    "stable_grasp_conf",
    "stable_grasp_source_frame",
    "stable_grasp_object_part",
    "stable_grasp_object_region",
    "stable_grasp_hand_side",
    "stable_grasp_primary_finger",
    "stable_grasp_fingers",
    "stable_grasp_type",
    "vlm_keyframe_source_frame",
    "vlm_keyframe_object_part",
    "vlm_keyframe_object_region",
    "vlm_keyframe_primary_finger",
    "vlm_keyframe_contact_fingers",
    "vlm_keyframe_grasp_type",
    "current_candidate_local_x",
    "current_candidate_local_y",
    "current_candidate_local_z",
    "hand_to_object_contact_px",
    "hand_contact_3d_gap_m",
    "hand_object_z_gap_m",
    "hand_depth_alignment_needed",
    "reason",
]


def truthy(value: str | int | float | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default




def read_vlm_keyframes(path: Path | None) -> dict[int, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open(newline="") as fh:
        return {int(float(row["frame"])): row for row in csv.DictReader(fh)}


def is_good_vlm_grasp(row: dict[str, str] | None) -> bool:
    if not row:
        return False
    if not truthy(row.get("contact_visible")):
        return False
    if not truthy(row.get("use_as_stable_grasp_keyframe")):
        return False
    return row.get("object_part") == "handle"


def nearest_vlm_keyframe(frame: int, vlm_rows: dict[int, dict[str, str]], max_gap: int = 3) -> tuple[int | None, dict[str, str] | None]:
    best_frame = None
    best_gap = max_gap + 1
    for fr, row in vlm_rows.items():
        gap = abs(fr - frame)
        if gap < best_gap and is_good_vlm_grasp(row):
            best_frame = fr
            best_gap = gap
    if best_frame is None:
        return None, None
    return best_frame, vlm_rows[best_frame]

def is_confirmed_anchor(row: dict[str, str]) -> bool:
    """Only confirmed hand-handle contact may update the stable grasp."""
    if not truthy(row.get("use_this_point_for_hand_attachment")):
        return False
    if truthy(row.get("rim_drinking_contact")):
        return False
    if row.get("object_contact_event") != "hand_handle_grasp":
        return False
    if row.get("hand_mug_contact_state") != "direct_hand_grasp_point":
        return False
    if row.get("semantic_contact_part") != "handle_loop":
        return False
    if row.get("vlm_hand_contact_part") not in {"handle", "handle_region", ""}:
        return False
    return True


def wants_previous_anchor(row: dict[str, str]) -> bool:
    if truthy(row.get("use_previous_grasp_for_hand_attachment")):
        return True
    event = row.get("object_contact_event", "")
    state = row.get("hand_mug_contact_state", "")
    if event in {
        "occluded_hand_mug_grasp",
        "handle_contact_point_misaligned",
        "unconfirmed_handle_candidate",
        "hand_handle_grasp_depth_misaligned",
    }:
        return True
    if state in {
        "held_but_handle_not_observed",
        "visual_contact_needs_depth_alignment",
    }:
        return True
    return False


def classify_row(row: dict[str, str], has_stable: bool) -> tuple[str, str]:
    if is_confirmed_anchor(row):
        return "direct_grasp_anchor", "confirmed hand_handle_grasp updates stable object-local anchor"
    if truthy(row.get("rim_drinking_contact")):
        if has_stable or wants_previous_anchor(row):
            return "rim_contact_keep_previous_grasp_anchor", "rim/mouth contact is not a hand anchor; keep previous grasp"
        return "rim_contact_no_hand_anchor", "rim/mouth contact is not a hand anchor and no previous grasp exists"
    if wants_previous_anchor(row):
        if has_stable:
            return "keep_previous_grasp_anchor", "hidden/occluded/misaligned frame keeps previous stable grasp"
        return "needs_previous_grasp_but_none", "frame requests previous grasp but no stable anchor exists yet"
    return "no_attachment", "no confirmed hand-handle anchor for this frame"


def build_state(input_csv: Path, vlm_keyframe_csv: Path | None = None) -> tuple[list[dict[str, str]], dict[str, object]]:
    rows_out: list[dict[str, str]] = []
    stable = None
    stable_conf = 0.0
    stable_source = ""
    stable_meta = {
        "object_part": "",
        "object_region": "",
        "hand_side": "",
        "primary_finger": "",
        "fingers": "",
        "grasp_type": "",
        "vlm_source_frame": "",
    }
    vlm_rows = read_vlm_keyframes(vlm_keyframe_csv)
    counts: dict[str, int] = {}
    update_frames: list[int] = []
    keep_frames: list[int] = []
    rim_frames: list[int] = []

    with input_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            frame = int(float(row["frame"]))
            mode, reason = classify_row(row, stable is not None)
            if mode == "direct_grasp_anchor":
                stable = (
                    f(row, "mug_local_x"),
                    f(row, "mug_local_y"),
                    f(row, "mug_local_z"),
                )
                stable_conf = f(row, "contact_conf", 1.0)
                stable_source = str(frame)
                vlm_fr, vlm_row = nearest_vlm_keyframe(frame, vlm_rows)
                if vlm_row is not None:
                    stable_meta = {
                        "object_part": vlm_row.get("object_part", ""),
                        "object_region": vlm_row.get("object_region", ""),
                        "hand_side": vlm_row.get("hand_side", ""),
                        "primary_finger": vlm_row.get("primary_contact_finger", ""),
                        "fingers": vlm_row.get("contact_fingers", ""),
                        "grasp_type": vlm_row.get("handle_grasp_type", ""),
                        "vlm_source_frame": str(vlm_fr),
                    }
                update_frames.append(frame)
            elif "keep_previous" in mode:
                keep_frames.append(frame)
            if truthy(row.get("rim_drinking_contact")):
                rim_frames.append(frame)

            counts[mode] = counts.get(mode, 0) + 1
            stable_vals = stable if stable is not None else ("", "", "")
            vlm_fr, vlm_row = nearest_vlm_keyframe(frame, vlm_rows)
            rows_out.append(
                {
                    "frame": str(frame),
                    "time": row.get("time", ""),
                    "object_contact_event": row.get("object_contact_event", ""),
                    "hand_mug_contact_state": row.get("hand_mug_contact_state", ""),
                    "nearest_articraft_part": row.get("nearest_articraft_part", ""),
                    "semantic_contact_part": row.get("semantic_contact_part", ""),
                    "active_label": row.get("active_label", ""),
                    "vlm_visibility": row.get("vlm_visibility", ""),
                    "vlm_hand_contact_part": row.get("vlm_hand_contact_part", ""),
                    "vlm_handle_contact": row.get("vlm_handle_contact", ""),
                    "use_this_point_for_hand_attachment": row.get("use_this_point_for_hand_attachment", ""),
                    "use_previous_grasp_for_hand_attachment": row.get("use_previous_grasp_for_hand_attachment", ""),
                    "rim_drinking_contact": row.get("rim_drinking_contact", ""),
                    "frame_mode": mode,
                    "stable_grasp_local_x": "" if stable is None else f"{stable_vals[0]:.6f}",
                    "stable_grasp_local_y": "" if stable is None else f"{stable_vals[1]:.6f}",
                    "stable_grasp_local_z": "" if stable is None else f"{stable_vals[2]:.6f}",
                    "stable_grasp_conf": "" if stable is None else f"{stable_conf:.6f}",
                    "stable_grasp_source_frame": stable_source,
                    "stable_grasp_object_part": "" if stable is None else stable_meta.get("object_part", ""),
                    "stable_grasp_object_region": "" if stable is None else stable_meta.get("object_region", ""),
                    "stable_grasp_hand_side": "" if stable is None else stable_meta.get("hand_side", ""),
                    "stable_grasp_primary_finger": "" if stable is None else stable_meta.get("primary_finger", ""),
                    "stable_grasp_fingers": "" if stable is None else stable_meta.get("fingers", ""),
                    "stable_grasp_type": "" if stable is None else stable_meta.get("grasp_type", ""),
                    "vlm_keyframe_source_frame": "" if vlm_row is None else str(vlm_fr),
                    "vlm_keyframe_object_part": "" if vlm_row is None else vlm_row.get("object_part", ""),
                    "vlm_keyframe_object_region": "" if vlm_row is None else vlm_row.get("object_region", ""),
                    "vlm_keyframe_primary_finger": "" if vlm_row is None else vlm_row.get("primary_contact_finger", ""),
                    "vlm_keyframe_contact_fingers": "" if vlm_row is None else vlm_row.get("contact_fingers", ""),
                    "vlm_keyframe_grasp_type": "" if vlm_row is None else vlm_row.get("handle_grasp_type", ""),
                    "current_candidate_local_x": row.get("mug_local_x", ""),
                    "current_candidate_local_y": row.get("mug_local_y", ""),
                    "current_candidate_local_z": row.get("mug_local_z", ""),
                    "hand_to_object_contact_px": row.get("hand_to_object_contact_px", ""),
                    "hand_contact_3d_gap_m": row.get("hand_contact_3d_gap_m", ""),
                    "hand_object_z_gap_m": row.get("hand_object_z_gap_m", ""),
                    "hand_depth_alignment_needed": row.get("hand_depth_alignment_needed", ""),
                    "reason": reason,
                }
            )

    summary = {
        "input_csv": str(input_csv),
        "vlm_keyframe_csv": "" if vlm_keyframe_csv is None else str(vlm_keyframe_csv),
        "n_frames": len(rows_out),
        "mode_counts": counts,
        "n_anchor_updates": len(update_frames),
        "anchor_update_frames": update_frames,
        "n_keep_previous_frames": len(keep_frames),
        "keep_previous_frames_sample": keep_frames[:30],
        "n_rim_drinking_frames": len(rim_frames),
        "rim_drinking_frames_sample": rim_frames[:30],
        "final_stable_grasp_local": None if stable is None else list(stable),
        "final_stable_grasp_source_frame": stable_source,
        "final_stable_grasp_meta": stable_meta,
    }
    return rows_out, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("samples_known_object/02_mug/results/mug_articraft_contact_points/mug_articraft_contact_points.csv"),
    )
    parser.add_argument(
        "--vlm-keyframe-csv",
        type=Path,
        default=Path("samples_known_object/02_mug/annotations/vlm_mug_contact_keyframes/mug_contact_keyframe_annotations.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("samples_known_object/02_mug/results/mug_grasp_anchor_state"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = build_state(args.input_csv, args.vlm_keyframe_csv)

    out_csv = args.output_dir / "mug_grasp_anchor_state.csv"
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    out_json = args.output_dir / "mug_grasp_anchor_state_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"wrote: {out_csv}")
    print(f"wrote: {out_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
