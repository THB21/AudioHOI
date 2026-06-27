#!/usr/bin/env python3
"""Evaluate chair pose quality against semantic 2D anchors and palm contacts."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[6]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.components.pose.solvers import fit_chair_metric_6d_pose as fit  # noqa: E402
from scripts.shared.generic_contact_pipeline.components.render.scenes.render_chair_articraft_3d_scene import (  # noqa: E402
    palm_centers,
    read_human_result,
)
from scripts.shared.generic_contact_pipeline.components.render.scenes.render_chair_articraft_visual_model import build_body_joints  # noqa: E402


POSE_KEYS = ["rx_delta", "ry_delta", "rz_delta", "tx", "ty", "tz", "rear_joint_angle", "seat_joint_angle"]
SEGMENT_OBS = {
    "backrest_top_edge": ("top_rail_left_u", "top_rail_left_v", "top_rail_right_u", "top_rail_right_v"),
    "seat_front_edge": ("seat_front_left_u", "seat_front_left_v", "seat_front_right_u", "seat_front_right_v"),
    "front_leg_left": ("front_leg_left_top_u", "front_leg_left_top_v", "front_leg_left_bottom_u", "front_leg_left_bottom_v"),
    "front_leg_right": ("front_leg_right_top_u", "front_leg_right_top_v", "front_leg_right_bottom_u", "front_leg_right_bottom_v"),
    "rear_leg_left": ("rear_leg_left_top_u", "rear_leg_left_top_v", "rear_leg_left_bottom_u", "rear_leg_left_bottom_v"),
    "rear_leg_right": ("rear_leg_right_top_u", "rear_leg_right_top_v", "rear_leg_right_bottom_u", "rear_leg_right_bottom_v"),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows


def ff(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def pose_params(row: dict[str, str]) -> np.ndarray:
    return np.asarray([ff(row, key, 0.0) for key in POSE_KEYS], dtype=float)


def summarize(values: list[float]) -> dict[str, float | int]:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return {"n": 0, "median": math.nan, "p90": math.nan, "mean": math.nan, "max": math.nan}
    return {
        "n": len(vals),
        "median": float(np.median(vals)),
        "p90": float(vals[int(0.9 * (len(vals) - 1))]),
        "mean": float(np.mean(vals)),
        "max": float(max(vals)),
    }


def observed_segment(row: dict[str, str], keys: tuple[str, str, str, str]) -> np.ndarray | None:
    vals = [ff(row, key) for key in keys]
    if not all(math.isfinite(v) for v in vals):
        return None
    return np.asarray([[vals[0], vals[1]], [vals[2], vals[3]]], dtype=float)


def semantic_2d_metrics(pose_csv: Path, observations_csv: Path, segments_csv: Path) -> dict[str, object]:
    pose_rows = {int(row["frame"]): row for row in read_rows(pose_csv)}
    obs_rows = {int(row["frame"]): row for row in read_rows(observations_csv)}
    segs = fit.load_segments(segments_csv)
    all_errors: list[float] = []
    by_segment: dict[str, dict[str, float | int]] = {}
    for sid, keys in SEGMENT_OBS.items():
        errors: list[float] = []
        if sid not in segs:
            by_segment[sid] = summarize(errors)
            continue
        for frame, pose_row in pose_rows.items():
            obs_row = obs_rows.get(frame)
            if obs_row is None:
                continue
            obs = observed_segment(obs_row, keys)
            if obs is None:
                continue
            params = pose_params(pose_row)
            k = (ff(pose_row, "fx"), ff(pose_row, "fy"), ff(pose_row, "cx"), ff(pose_row, "cy"))
            uv, _cam, _local = fit.project_segment(params, sid, segs, k)
            err = np.linalg.norm(uv - obs, axis=1).tolist()
            errors.extend(float(v) for v in err)
            all_errors.extend(float(v) for v in err)
        by_segment[sid] = summarize(errors)
    return {"all": summarize(all_errors), "by_segment": by_segment}


def contact_gap_metrics(
    sample_dir: Path,
    pose_csv: Path,
    contacts_csv: Path,
    body_model_root: Path,
) -> dict[str, object]:
    pose_rows = {int(row["frame"]): row for row in read_rows(pose_csv)}
    contacts_by_frame: dict[int, list[dict[str, str]]] = {}
    for row in read_rows(contacts_csv):
        if row.get("contact_active") == "1":
            contacts_by_frame.setdefault(int(row["frame"]), []).append(row)
    human = read_human_result(sample_dir / "results/gvhmr/result.pkl")
    joints = build_body_joints(body_model_root, human)
    gaps: list[float] = []
    by_frame: list[dict[str, float | int]] = []
    for frame in sorted(contacts_by_frame):
        pose_row = pose_rows.get(frame)
        if pose_row is None:
            continue
        left_palm, right_palm = palm_centers(joints[frame - 1])
        palms = {"left_hand": left_palm, "right_hand": right_palm}
        params = pose_params(pose_row)
        frame_gaps: list[float] = []
        for contact in contacts_by_frame[frame]:
            hand = palms.get(contact.get("hand_side", ""))
            if hand is None:
                continue
            local = np.asarray([ff(contact, "chair_local_x"), ff(contact, "chair_local_y"), ff(contact, "chair_local_z")], dtype=float)
            if not np.all(np.isfinite(local)):
                continue
            gap = float(np.linalg.norm(fit.transform(params, local[None, :])[0] - hand))
            gaps.append(gap)
            frame_gaps.append(gap)
        if frame_gaps:
            by_frame.append({"frame": frame, "median": float(np.median(frame_gaps)), "max": float(max(frame_gaps))})
    return {"all": summarize(gaps), "by_frame": by_frame}


def freeze_metrics(pose_csv: Path, first_active: int, last_active: int) -> dict[str, object]:
    rows = read_rows(pose_csv)
    by_frame = {int(row["frame"]): row for row in rows}
    keys = POSE_KEYS + ["tx", "ty", "tz"]

    def max_delta(frame_range: range, anchor: int) -> float:
        anchor_row = by_frame.get(anchor)
        if anchor_row is None:
            return math.inf
        out = 0.0
        for frame in frame_range:
            row = by_frame.get(frame)
            if row is None:
                continue
            for key in keys:
                out = max(out, abs(ff(row, key, 0.0) - ff(anchor_row, key, 0.0)))
        return out

    prefix = max_delta(range(1, first_active), first_active) if first_active > 1 else 0.0
    suffix = max_delta(range(last_active + 1, max(by_frame) + 1), last_active) if last_active < max(by_frame) else 0.0
    return {
        "first_active": first_active,
        "last_active": last_active,
        "prefix_max_abs_delta": float(prefix),
        "suffix_max_abs_delta": float(suffix),
        "pass": prefix <= 1e-6 and suffix <= 1e-6,
    }


def quality_gate(candidate: dict[str, object], baseline: dict[str, object]) -> dict[str, object]:
    cand_2d = candidate["semantic_2d"]["all"]  # type: ignore[index]
    base_2d = baseline["semantic_2d"]["all"]  # type: ignore[index]
    cand_contact = candidate["contact_gap"]["all"]  # type: ignore[index]
    base_contact = baseline["contact_gap"]["all"]  # type: ignore[index]
    checks = {
        "semantic_2d_median_within_1p10": float(cand_2d["median"]) <= float(base_2d["median"]) * 1.10,  # type: ignore[index]
        "semantic_2d_p90_within_1p10": float(cand_2d["p90"]) <= float(base_2d["p90"]) * 1.10,  # type: ignore[index]
        "contact_median_within_1p20": float(cand_contact["median"]) <= float(base_contact["median"]) * 1.20,  # type: ignore[index]
        "contact_p90_within_1p35": float(cand_contact["p90"]) <= float(base_contact["p90"]) * 1.35,  # type: ignore[index]
        "freeze_pass": bool(candidate["freeze"]["pass"]),  # type: ignore[index]
    }
    return {"pass": all(checks.values()), "checks": checks}


def evaluate_pose(args: argparse.Namespace, pose_csv: Path) -> dict[str, object]:
    return {
        "pose_csv": str(pose_csv),
        "semantic_2d": semantic_2d_metrics(pose_csv, args.observations_csv, args.segments_csv),
        "contact_gap": contact_gap_metrics(args.sample_dir, pose_csv, args.contacts_csv, args.body_model_root),
        "freeze": freeze_metrics(pose_csv, args.first_active, args.last_active),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--candidate-pose-csv", type=Path, required=True)
    ap.add_argument("--baseline-pose-csv", type=Path, required=True)
    ap.add_argument("--observations-csv", type=Path, required=True)
    ap.add_argument("--segments-csv", type=Path, required=True)
    ap.add_argument("--contacts-csv", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--body-model-root", type=Path, default=Path("third-party/GVHMR/inputs/checkpoints/body_models"))
    ap.add_argument("--first-active", type=int, default=20)
    ap.add_argument("--last-active", type=int, default=144)
    args = ap.parse_args()

    candidate = evaluate_pose(args, args.candidate_pose_csv)
    baseline = evaluate_pose(args, args.baseline_pose_csv)
    report = {
        "candidate": candidate,
        "baseline": baseline,
        "quality_gate": quality_gate(candidate, baseline),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["quality_gate"], indent=2))


if __name__ == "__main__":
    main()
