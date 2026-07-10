from __future__ import annotations

import math
from typing import Any

from .schemas import EvaluationPaths, MetricBlock
from .utils import f, load_json, mean, read_rows, write_json, write_rows


PAIR_FIELDS = [
    "frame",
    "time",
    "human_part",
    "human_side",
    "object_part",
    "object_local_id",
    "expected",
    "observed",
    "persistent",
    "rel_static",
    "min_distance_m",
    "surface_gap_m",
    "penetration_depth_m",
    "contact_confidence",
    "contact_state",
    "anchor_update_allowed",
    "pose_anchor_allowed",
    "stable_local_x",
    "stable_local_y",
    "stable_local_z",
    "stable_local_s",
    "observed_local_x",
    "observed_local_y",
    "observed_local_z",
    "observed_local_s",
    "contact_u",
    "contact_v",
    "contact_depth_offset_m",
    "local_s_drift",
    "source",
]

INTERVAL_FIELDS = [
    "human_part",
    "human_side",
    "object_part",
    "start_frame",
    "end_frame",
    "n_frames",
    "mean_contact_confidence",
    "mean_surface_gap_m",
    "mean_anchor_drift",
    "source",
]


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y", "active", "observed"}


def _frame_int(row: dict[str, str]) -> int:
    value = f(row.get("frame"), 0.0) or 0.0
    return int(value)


def _anchor_key(row: dict[str, str]) -> tuple[int, str, str, str]:
    return (
        _frame_int(row),
        str(row.get("human_side", "")),
        str(row.get("human_part", "")),
        str(row.get("object_part", "")),
    )


def _anchor_index(paths: EvaluationPaths) -> dict[tuple[int, str, str, str], dict[str, str]]:
    anchors = {}
    for row in read_rows(paths.anchor_state_csv):
        anchors[_anchor_key(row)] = row
    return anchors


def _candidate_rows(paths: EvaluationPaths) -> list[dict[str, str]]:
    rows = read_rows(paths.contact_points_csv)
    if rows:
        return rows
    return read_rows(paths.result_dir / "contact_candidates.csv")


def contact_proxy_from_gap(contact_gap_mm: float | None, *, sigma_gap_mm: float = 50.0) -> float | None:
    if contact_gap_mm is None:
        return None
    return math.exp(-max(0.0, contact_gap_mm) / sigma_gap_mm)


def _anchor_for(row: dict[str, str], anchors: dict[tuple[int, str, str, str], dict[str, str]]) -> dict[str, str]:
    exact = anchors.get(_anchor_key(row))
    if exact is not None:
        return exact
    frame = _frame_int(row)
    side = str(row.get("human_side", ""))
    part = str(row.get("human_part", ""))
    for key, value in anchors.items():
        if key[0] == frame and key[1] == side and key[2] == part:
            return value
    return {}


def _candidate_to_pair(row: dict[str, str], anchor: dict[str, str]) -> dict[str, Any]:
    observed = _truth(row.get("contact_active")) or _truth(row.get("contact")) or _truth(anchor.get("contact_observed"))
    persistent = _truth(anchor.get("contact_persistent"))
    update_allowed = _truth(anchor.get("anchor_update_allowed"))
    pose_allowed = _truth(anchor.get("pose_anchor_allowed"))
    human_part = row.get("human_part") or row.get("contact_region") or row.get("contact_part") or anchor.get("human_part") or ""
    human_side = row.get("human_side") or anchor.get("human_side") or ""
    object_part = row.get("object_part") or anchor.get("object_part") or ("surface" if row.get("contact_region") else "")
    local_s_drift = f(row.get("local_s_drift"))
    if local_s_drift is None:
        stable_s = f(row.get("stable_object_local_s"), f(anchor.get("stable_local_s")))
        observed_s = f(row.get("object_local_s"), f(anchor.get("observed_local_s")))
        if stable_s is not None and observed_s is not None:
            local_s_drift = abs(observed_s - stable_s)
    contact_conf = f(row.get("contact_conf"), f(row.get("anchor_score"), f(anchor.get("contact_confidence"))))
    surface_gap = f(row.get("contact_depth_offset_m"))
    if surface_gap is None:
        px_gap = f(row.get("palm_to_line_px"))
        surface_gap = px_gap / 1000.0 if px_gap is not None else None
    contact_state = (
        "persistent_anchor"
        if observed and persistent and pose_allowed
        else "observed_anchor"
        if observed and pose_allowed
        else "observed_no_pose_anchor"
        if observed
        else "inactive"
    )
    return {
        "frame": str(_frame_int(row)),
        "time": row.get("time", anchor.get("time", "")),
        "human_part": human_part,
        "human_side": human_side,
        "object_part": object_part,
        "object_local_id": row.get("object_local_id", ""),
        "expected": int(observed or persistent),
        "observed": int(observed),
        "persistent": int(persistent),
        "rel_static": int(local_s_drift is not None and local_s_drift <= 0.05 and persistent),
        "min_distance_m": surface_gap,
        "surface_gap_m": surface_gap,
        "penetration_depth_m": "",
        "contact_confidence": contact_conf,
        "contact_state": contact_state,
        "anchor_update_allowed": int(update_allowed),
        "pose_anchor_allowed": int(pose_allowed),
        "stable_local_x": row.get("stable_local_x", anchor.get("stable_local_x", "")),
        "stable_local_y": row.get("stable_local_y", anchor.get("stable_local_y", "")),
        "stable_local_z": row.get("stable_local_z", anchor.get("stable_local_z", "")),
        "stable_local_s": row.get("stable_object_local_s", anchor.get("stable_local_s", "")),
        "observed_local_x": row.get("object_local_x", anchor.get("observed_local_x", "")),
        "observed_local_y": row.get("object_local_y", anchor.get("observed_local_y", "")),
        "observed_local_z": row.get("object_local_z", anchor.get("observed_local_z", "")),
        "observed_local_s": row.get("object_local_s", anchor.get("observed_local_s", "")),
        "contact_u": row.get("contact_u", ""),
        "contact_v": row.get("contact_v", ""),
        "contact_depth_offset_m": row.get("contact_depth_offset_m", ""),
        "local_s_drift": local_s_drift,
        "source": row.get("source", anchor.get("source", "tom_body_surface_contacts" if row.get("contact_region") else "")),
    }


def _build_pair_rows(paths: EvaluationPaths) -> list[dict[str, Any]]:
    anchors = _anchor_index(paths)
    pairs = []
    for row in _candidate_rows(paths):
        human_part = str(row.get("human_part") or row.get("contact_region") or row.get("contact_part") or "")
        object_part = str(row.get("object_part", ""))
        if (
            human_part in {"", "none"}
            and object_part in {"", "none"}
            and not _truth(row.get("contact_active"))
            and not _truth(row.get("contact"))
        ):
            continue
        pairs.append(_candidate_to_pair(row, _anchor_for(row, anchors)))
    return pairs


def _build_intervals(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active = [row for row in pairs if _truth(row.get("observed"))]
    active.sort(key=lambda row: (row.get("human_part", ""), row.get("human_side", ""), row.get("object_part", ""), int(row.get("frame", 0))))
    intervals = []
    current: list[dict[str, Any]] = []
    current_key: tuple[str, str, str] | None = None
    last_frame: int | None = None
    for row in active:
        key = (str(row.get("human_part", "")), str(row.get("human_side", "")), str(row.get("object_part", "")))
        frame = int(row.get("frame", 0))
        if current and (key != current_key or last_frame is None or frame > last_frame + 1):
            intervals.append(_interval_from_rows(current))
            current = []
        current.append(row)
        current_key = key
        last_frame = frame
    if current:
        intervals.append(_interval_from_rows(current))
    return intervals


def _interval_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    return {
        "human_part": first.get("human_part", ""),
        "human_side": first.get("human_side", ""),
        "object_part": first.get("object_part", ""),
        "start_frame": rows[0].get("frame", ""),
        "end_frame": rows[-1].get("frame", ""),
        "n_frames": len(rows),
        "mean_contact_confidence": mean(f(row.get("contact_confidence")) for row in rows),
        "mean_surface_gap_m": mean(f(row.get("surface_gap_m")) for row in rows),
        "mean_anchor_drift": mean(f(row.get("local_s_drift")) for row in rows),
        "source": first.get("source", ""),
    }


def compute_hoi_contact_metrics(paths: EvaluationPaths) -> MetricBlock:
    hoi = load_json(paths.hoi_eval_json)
    pair_rows = _build_pair_rows(paths)
    interval_rows = _build_intervals(pair_rows)
    anchor_drifts = [f(row.get("local_s_drift")) for row in pair_rows if _truth(row.get("observed"))]
    observed_rows = [row for row in pair_rows if _truth(row.get("observed"))]
    persistent_rows = [row for row in pair_rows if _truth(row.get("persistent"))]
    contact_gap = f(hoi.get("contact_gap_mm"))
    metrics: dict[str, Any] = {
        "contact_frame_ratio": f(hoi.get("contact_frame_ratio")),
        "contact_gap_mm": contact_gap,
        "contact_proxy": contact_proxy_from_gap(contact_gap),
        "contact_proxy_source": "contact_gap_mm_exp_decay_sigma_50" if contact_gap is not None else "missing_contact_gap_mm",
        "part_correct_ratio": f(hoi.get("part_correct_ratio")),
        "contact_ratio_audio_windows": f(hoi.get("contact_ratio_audio_windows")),
        "grasp_stability_mm": f(hoi.get("grasp_stability_mm")),
        "mdev_star_mm": f(hoi.get("mdev_star_mm")),
        "hoi_contact_pair_rows": len(pair_rows),
        "hoi_contact_interval_count": len(interval_rows),
        "hoi_observed_contact_rows": len(observed_rows),
        "hoi_persistent_contact_rows": len(persistent_rows),
        "contact_anchor_drift_mean": mean(anchor_drifts),
        "contact_anchor_drift_max": max((v for v in anchor_drifts if v is not None), default=None),
        "source": str(paths.hoi_eval_json) if paths.hoi_eval_json.exists() else "missing_hoi_eval",
    }
    pairs_csv = write_rows(paths.evaluation_dir / "hoi_contact_pairs.csv", pair_rows, PAIR_FIELDS)
    intervals_csv = write_rows(paths.evaluation_dir / "hoi_contact_intervals.csv", interval_rows, INTERVAL_FIELDS)
    out_json = write_json(paths.evaluation_dir / "hoi_contact_metrics.json", metrics)
    out_csv = write_rows(paths.evaluation_dir / "hoi_contact_metrics.csv", [metrics])
    return MetricBlock(
        "hoi_contact",
        metrics,
        {"json": str(out_json), "csv": str(out_csv), "pairs_csv": str(pairs_csv), "intervals_csv": str(intervals_csv)},
    )
