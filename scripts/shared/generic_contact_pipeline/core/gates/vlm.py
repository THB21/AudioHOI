from __future__ import annotations

import csv
import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

from ..base.config import CaseProfile
from ..semantics.hoi_profile import profile_choices_for_vlm
from ..base.io import read_csv, write_csv, write_json
from ..base.schema import stage_paths
from .amodal_mask_completion import (
    CHOICES as AMODAL_MASK_CHOICES,
    QUERY_TYPE as AMODAL_MASK_QUERY_TYPE,
    completion_intervals,
    materialize_candidate_evidence,
)
from .interval_candidate_selection import QUERY_TYPE as INTERVAL_SELECTION_QUERY_TYPE
from .semantic_relations import (
    PREDICATE_BY_QUERY_TYPE,
    SEMANTIC_QUERY_TYPES,
    profile_uncertain_windows,
    question_for_query_type,
)


QUERY_FIELDS = [
    "query_id",
    "case_name",
    "stage",
    "frame",
    "start_frame",
    "end_frame",
    "time",
    "query_type",
    "input_image_path",
    "input_render_path",
    "highlight_layer",
    "question",
    "choices",
    "hoi_profile_source",
    "object_part_choices",
    "contact_relation_choices",
    "gate_policy",
    "evidence_sha256",
    "visual_factor_roles",
    "contact_factor_roles",
    "candidate_a_role",
    "candidate_b_role",
    "semantic_predicate",
    "subject_entity",
    "object_entity",
]

RESULT_FIELDS = [
    "case_name",
    "stage",
    "frame",
    "query_type",
    "normalized_label",
    "pass_gate",
    "repair_action",
    "used_for_anchor_update",
    "used_for_contact_residual",
    "used_for_report",
]

GATE_PASS = "pass"
GATE_REJECT = "reject"
GATE_UNCLEAR = "unclear"
GATE_NOT_EVALUATED = "not_evaluated"

STAGE_QUERY_TYPES = {
    "stage0": ["target_mask_check"],
    "stage1": ["keypart_identity_check", "track_stability_check"],
    "stage2": ["contact_relation_check", "keypart_visibility_check"],
    "stage3": ["overlay_alignment_check"],
    "stage4": ["anchor_update_check", "contact_relation_check", "overlay_alignment_check", "temporal_motion_check", "constraint_reliability_check", AMODAL_MASK_QUERY_TYPE, INTERVAL_SELECTION_QUERY_TYPE, *SEMANTIC_QUERY_TYPES],
    "stage5": ["post_render_sanity_check", "temporal_motion_check"],
    "stage6": ["baseline_regression_check"],
    "stage7": ["loss_diagnostic_check"],
}

COMMON_REPAIR_ACTIONS = [
    "accept_candidate",
    "reject_candidate",
    "disable_contact_frame",
    "keep_previous_anchor",
    "downweight_correspondence",
    "use_mask_fallback",
    "retry_with_alternative_keypart",
    "allow_small_se3_refine",
    "freeze_before_after_contact",
    "mark_for_report_only",
    "unclear_no_update",
    "keep_stable_candidate",
    "select_occlusion_challenger",
    "reject_candidate_pair",
]


def fuse_stage_decision(case_name: str, stage: str, results: list[dict[str, object]], *, mode: str) -> dict[str, object]:
    gate_counts: dict[str, int] = {}
    query_counts: dict[str, int] = {}
    accepted_frames: list[int] = []
    blocked_frames: list[int] = []
    unclear_frames: list[int] = []
    report_only_reject_frames: list[int] = []
    repair_actions_by_frame: dict[str, list[str]] = {}
    result_rows: list[dict[str, object]] = []
    for row in results:
        gate = str(row.get("pass_gate", ""))
        qtype = str(row.get("query_type", ""))
        action = str(row.get("repair_action", ""))
        try:
            frame = int(float(str(row.get("frame", ""))))
        except Exception:
            frame = -1
        gate_counts[gate] = gate_counts.get(gate, 0) + 1
        query_counts[qtype] = query_counts.get(qtype, 0) + 1
        is_report_only = action == "mark_for_report_only"
        if gate == GATE_PASS and frame >= 0 and frame not in accepted_frames:
            accepted_frames.append(frame)
        elif gate == GATE_REJECT and is_report_only and frame >= 0 and frame not in report_only_reject_frames:
            report_only_reject_frames.append(frame)
        elif gate == GATE_REJECT and frame >= 0 and frame not in blocked_frames:
            blocked_frames.append(frame)
        elif gate in {GATE_UNCLEAR, GATE_NOT_EVALUATED} and frame >= 0 and frame not in unclear_frames:
            unclear_frames.append(frame)
        if frame >= 0 and action and action != "none":
            repair_actions_by_frame.setdefault(str(frame), [])
            if action not in repair_actions_by_frame[str(frame)]:
                repair_actions_by_frame[str(frame)].append(action)
        if gate == GATE_REJECT:
            result_rows.append(row)

    blocking = mode == "qwen_vl" and len(blocked_frames) > 0
    if mode == "dry_run":
        decision = "not_blocking_dry_run"
    elif blocking:
        decision = "blocked_by_vlm_reject"
    elif report_only_reject_frames:
        decision = "pass_with_report_only_failures"
    elif gate_counts.get(GATE_UNCLEAR, 0) > 0:
        decision = "pass_with_unclear_frames"
    else:
        decision = "pass"

    return {
        "case_name": case_name,
        "stage": stage,
        "mode": mode,
        "decision": decision,
        "blocking": blocking,
        "query_count": len(results),
        "gate_counts": gate_counts,
        "query_counts": query_counts,
        "accepted_frames": sorted(accepted_frames),
        "blocked_frames": sorted(blocked_frames),
        "report_only_reject_frames": sorted(report_only_reject_frames),
        "unclear_frames": sorted(unclear_frames),
        "repair_actions_by_frame": repair_actions_by_frame,
        "rejected_results": result_rows[:20],
        "allowed_repair_actions": COMMON_REPAIR_ACTIONS,
        "note": (
            "Dry-run outputs generate schema/evidence only and never block."
            if mode == "dry_run"
            else "Qwen-VL forced-choice labels gate only predefined update/report actions; they do not directly optimize pose."
        ),
    }


def object_vlm_profile(profile: CaseProfile) -> dict[str, object]:
    resolved = profile_choices_for_vlm(profile)
    if resolved is not None:
        return resolved
    configured = profile.data.get("vlm")
    if isinstance(configured, dict):
        target = configured.get("target") or profile.data.get("object_family") or profile.case_name
        parts = configured.get("parts") or ["main_body", "support_region", "contact_region", "background", "unclear"]
        contacts = configured.get("contacts") or ["hand", "foot", "floor", "table", "nothing_nearby", "unclear"]
        stable_refs = configured.get("stable_refs") or ["center", "visible_semantic_part", "support_point", "unclear"]
        return {
            "target": str(target),
            "parts": list(parts),
            "contacts": list(contacts),
            "stable_refs": list(stable_refs),
        }
    case = profile.case_name
    if case == "basketball":
        return {
            "target": "basketball",
            "parts": ["ball_center", "ball_boundary", "ball_bottom", "background", "unclear"],
            "contacts": ["hand", "floor", "body", "nothing_nearby", "unclear"],
            "stable_refs": ["center", "bottom_support", "boundary_point", "unclear"],
        }
    if case == "football":
        return {
            "target": "football",
            "parts": ["ball_center", "ball_boundary", "ball_bottom", "background", "unclear"],
            "contacts": ["foot", "floor", "hand_body", "nothing_nearby", "unclear"],
            "stable_refs": ["center", "bottom_support", "boundary_point", "unclear"],
        }
    if case == "mug":
        return {
            "target": "mug",
            "parts": ["handle", "cup_body", "rim", "bottom", "background", "unclear"],
            "contacts": ["palm", "fingertip", "rim_mouth", "table", "nothing_nearby", "unclear"],
            "stable_refs": ["mug_body", "handle_loop", "rim", "bottom", "unclear"],
        }
    if case == "chair":
        return {
            "target": "chair",
            "parts": ["front_leg", "rear_leg", "seat", "backrest", "top_rail", "stretcher", "hole", "background", "unclear"],
            "contacts": ["left_palm", "right_palm", "both_hands", "body", "floor", "nothing_nearby", "unclear"],
            "stable_refs": ["front_leg", "top_rail_endpoint", "seat_edge", "backrest_hole", "stretcher", "unclear"],
        }
    return {
        "target": str(profile.data.get("object_family", "object")),
        "parts": ["main_body", "support_region", "contact_region", "background", "unclear"],
        "contacts": ["hand", "foot", "floor", "table", "nothing_nearby", "unclear"],
        "stable_refs": ["center", "visible_semantic_part", "support_point", "unclear"],
    }


def _frame_path(sample_dir: Path, frame: int) -> str:
    for name in (f"{frame:05d}.png", f"{frame:06d}.png", f"{frame:04d}.png", f"{frame:03d}.png"):
        path = sample_dir / "frames" / name
        if path.exists():
            return str(path)
    return ""


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        return read_csv(path)
    except Exception:
        return []


def _row_frame(row: dict[str, str], default: int) -> int:
    try:
        return int(float(row.get("frame", default)))
    except Exception:
        return default


def _row_time(row: dict[str, str]) -> str:
    return row.get("time", "")


def _float(row: dict[str, str], key: str) -> float | None:
    try:
        value = row.get(key)
        if value in {"", None}:
            return None
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return None


def _rows_by_frame(path: Path) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for row in _read_rows(path):
        out[_row_frame(row, 1)] = row
    return out


def _unique_limited(values: Iterable[int], limit: int = 5) -> list[int]:
    out: list[int] = []
    for value in values:
        if value not in out:
            out.append(value)
        if len(out) >= limit:
            break
    return out


def stage_keyframes(profile: CaseProfile, stage: str) -> list[tuple[int, str]]:
    paths = stage_paths(profile)
    frame_policy = profile.data.get("vlm", {}).get("frame_policy", {}) if isinstance(profile.data.get("vlm", {}), dict) else {}
    if isinstance(frame_policy, dict) and str(frame_policy.get(stage, "")).lower() == "all":
        source_by_stage_all = {
            "stage0": paths["object_observations"],
            "stage1": paths["object_observations"],
            "stage2": paths["contact_candidates"],
            "stage3": paths["object_pose_init"],
            "stage4": paths["object_contact_points"],
            "stage5": paths["object_pose"],
            "stage6": paths["object_pose"],
            "stage7": paths["loss_residuals"],
        }
        source = source_by_stage_all.get(stage, paths["object_pose"])
        rows_all = _read_rows(source)
        if stage == "stage4" and not rows_all:
            for fallback in (paths["contact_state"], paths["contact_candidates"], paths["object_pose"]):
                rows_all = _read_rows(fallback)
                if rows_all:
                    break
        out: list[tuple[int, str]] = []
        seen: set[int] = set()
        for row in rows_all:
            fr = _row_frame(row, 1)
            if fr in seen:
                continue
            seen.add(fr)
            out.append((fr, _row_time(row)))
        if out:
            return out
    if stage == "stage7":
        rows = _read_rows(paths["loss_residuals"])
        if not rows:
            return [(1, "")]
        scored: list[tuple[float, dict[str, str]]] = []
        for row in rows:
            try:
                scored.append((float(row.get("E_total", "0") or 0), row))
            except Exception:
                scored.append((0.0, row))
        top = [row for _score, row in sorted(scored, key=lambda item: item[0], reverse=True)[:3]]
        idxs = [0, len(rows) // 2, len(rows) - 1]
        chosen_rows = top + [rows[i] for i in idxs]
        frames = _unique_limited((_row_frame(row, 1) for row in chosen_rows), 5)
        by_frame = {_row_frame(row, 1): _row_time(row) for row in rows}
        return [(fr, by_frame.get(fr, "")) for fr in frames]
    source_by_stage = {
        "stage0": paths["object_observations"],
        "stage1": paths["object_observations"],
        "stage2": paths["contact_candidates"],
        "stage3": paths["object_pose_init"],
        "stage4": paths["object_contact_points"],
        "stage5": paths["object_pose"],
        "stage6": paths["object_pose"],
    }
    rows = _read_rows(source_by_stage.get(stage, paths["object_pose"]))
    if not rows:
        return [(1, "")]
    active = [
        row
        for row in rows
        if row.get("contact_active") == "1"
        or row.get("phase", "").lower() not in {"", "static_no_contact", "no_contact"}
        or row.get("active_label", "").lower() not in {"", "none"}
    ]
    chosen_rows: list[dict[str, str]] = []
    if active:
        idxs = [0, len(active) // 2, len(active) - 1]
        chosen_rows.extend(active[i] for i in _unique_limited(idxs, 3))
    idxs = [0, len(rows) // 2, len(rows) - 1]
    chosen_rows.extend(rows[i] for i in idxs)
    frames = _unique_limited((_row_frame(row, 1) for row in chosen_rows), 5)
    by_frame = {_row_frame(row, 1): _row_time(row) for row in rows}
    return [(fr, by_frame.get(fr, "")) for fr in frames]


def temporal_risk_keyframes(profile: CaseProfile, stage: str, *, limit: int = 80) -> list[tuple[int, str]]:
    paths = stage_paths(profile)
    risk_frames: list[int] = []
    for row in _read_rows(paths["pose_jump_audit"]):
        if row.get("visual_spike") == "1" or row.get("contact_spike") == "1" or row.get("smoothness_spike") == "1":
            risk_frames.append(_row_frame(row, 1))
    if not risk_frames:
        return stage_keyframes(profile, stage)[: min(5, limit)]
    out: list[int] = []
    for fr in sorted(set(risk_frames)):
        for near in (max(1, fr - 1), fr, fr + 1):
            if near not in out:
                out.append(near)
            if len(out) >= limit:
                break
        if len(out) >= limit:
            break
    pose_by_frame = {_row_frame(row, 1): _row_time(row) for row in _read_rows(paths["object_pose"])}
    return [(fr, pose_by_frame.get(fr, "")) for fr in out]


def constraint_risk_intervals(profile: CaseProfile, *, maximum_window_frames: int = 24) -> list[tuple[int, int, int, str]]:
    """Find generic visual/contact conflict windows for VLM arbitration."""

    if maximum_window_frames < 1:
        raise ValueError("constraint reliability window must contain at least one frame")
    line_frames = {
        _row_frame(row, 1)
        for row in _read_rows(profile.result_dir / "line_observations.csv")
        if str(row.get("line_observation_trusted", "1")).strip().lower() not in {"0", "false"}
    }
    paths = stage_paths(profile)
    contact_rows: list[dict[str, str]] = []
    for path in (paths["object_contact_points"], paths["contact_state"], paths["contact_candidates"]):
        contact_rows = _read_rows(path)
        if contact_rows:
            break
    contact_frames = {
        _row_frame(row, 1)
        for row in contact_rows
        if any(
            str(row.get(field, "")).strip().lower() in {"1", "1.0", "true", "active"}
            for field in ("contact_active", "human_contact_state", "anchor_contact_state", "floor_contact_state")
        )
    }
    frames = sorted((line_frames & contact_frames) if line_frames else contact_frames)
    if not frames:
        return [(frame, frame, frame, time) for frame, time in stage_keyframes(profile, "stage4")]
    times = {_row_frame(row, 1): _row_time(row) for row in contact_rows}
    contiguous: list[tuple[int, int]] = []
    start = frames[0]
    previous = frames[0]
    for frame in frames[1:]:
        if frame != previous + 1:
            contiguous.append((start, previous))
            start = frame
        previous = frame
    contiguous.append((start, previous))
    windows: list[tuple[int, int, int, str]] = []
    for interval_start, interval_end in contiguous:
        window_start = interval_start
        while window_start <= interval_end:
            window_end = min(interval_end, window_start + maximum_window_frames - 1)
            representative = (window_start + window_end) // 2
            windows.append((representative, window_start, window_end, times.get(representative, "")))
            window_start = window_end + 1
    return windows


def _render_video_for_stage(profile: CaseProfile, stage: str) -> Path | None:
    if stage not in {"stage3", "stage4", "stage5", "stage6"}:
        return None
    if stage == "stage4":
        candidate = (
            profile.result_dir
            / "generic_stage4_candidate"
            / "vlm_evidence"
            / "object_only"
            / "overlay.mp4"
        )
        if candidate.exists():
            return candidate
    names = [
        "with_human/overlay.mp4",
        "with_human/contact_overlay_solid.mp4",
        "object_only/overlay.mp4",
        "object_only/overlay_solid.mp4",
    ]
    for name in names:
        path = profile.render_dir / name
        if path.exists():
            return path
    return None


def _pil_modules():
    try:
        from PIL import Image, ImageDraw

        return Image, ImageDraw
    except Exception:
        return None, None


def _extract_video_frame(video_path: Path, frame: int) -> Any | None:
    Image, _ImageDraw = _pil_modules()
    if Image is None:
        return None
    try:
        import cv2
    except Exception:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame - 1))
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def _read_frame(sample_dir: Path, frame: int) -> Any | None:
    Image, _ImageDraw = _pil_modules()
    if Image is None:
        return None
    path = _frame_path(sample_dir, frame)
    if not path:
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def _mask_path(sample_dir: Path, frame: int) -> Path | None:
    for name in (f"{frame:05d}_mask.png", f"{frame:06d}_mask.png", f"{frame:04d}_mask.png"):
        path = sample_dir / "results" / "segmentation" / "masks" / name
        if path.exists():
            return path
    return None


def _draw_label(img: Any, text: str, xy: tuple[int, int], color: tuple[int, int, int]) -> None:
    _Image, ImageDraw = _pil_modules()
    if ImageDraw is None:
        return
    draw = ImageDraw.Draw(img)
    x, y = xy
    draw.text((x - 1, y - 1), text, fill=(20, 20, 20))
    draw.text((x + 1, y - 1), text, fill=(20, 20, 20))
    draw.text((x - 1, y + 1), text, fill=(20, 20, 20))
    draw.text((x + 1, y + 1), text, fill=(20, 20, 20))
    draw.text((x, y), text, fill=color)


def _draw_point(img: Any, u: float | None, v: float | None, label: str, color: tuple[int, int, int]) -> None:
    if u is None or v is None:
        return
    _Image, ImageDraw = _pil_modules()
    if ImageDraw is None:
        return
    draw = ImageDraw.Draw(img)
    x, y = int(round(u)), int(round(v))
    draw.ellipse((x - 13, y - 13, x + 13, y + 13), fill=(20, 20, 20))
    draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color)
    _draw_label(img, label, (x + 12, max(18, y - 18)), color)


def _draw_circle(
    img: Any,
    u: float | None,
    v: float | None,
    radius: float | None,
    label: str,
    color: tuple[int, int, int],
) -> None:
    if u is None or v is None:
        return
    if radius is None or radius <= 1.0:
        _draw_point(img, u, v, label, color)
        return
    _Image, ImageDraw = _pil_modules()
    if ImageDraw is None:
        return
    draw = ImageDraw.Draw(img)
    x, y, r = int(round(u)), int(round(v)), int(round(radius))
    draw.ellipse((x - r, y - r, x + r, y + r), outline=(20, 20, 20), width=9)
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=5)
    _draw_label(img, label, (x + r + 10, max(18, y - r)), color)


def _draw_line(img: Any, a: tuple[float, float] | None, b: tuple[float, float] | None, label: str, color: tuple[int, int, int]) -> None:
    if a is None or b is None:
        return
    _Image, ImageDraw = _pil_modules()
    if ImageDraw is None:
        return
    draw = ImageDraw.Draw(img)
    p1 = (int(round(a[0])), int(round(a[1])))
    p2 = (int(round(b[0])), int(round(b[1])))
    draw.line((p1, p2), fill=(20, 20, 20), width=9)
    draw.line((p1, p2), fill=color, width=5)
    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    _draw_label(img, label, (mid[0] + 8, max(18, mid[1] - 24)), color)


def _tracking_line(profile: CaseProfile, frame: int) -> tuple[tuple[float, float], tuple[float, float]] | None:
    tracks = profile.sample_dir / "results" / "tracking" / "object_mesh_tracks_test.csv"
    pts: dict[str, tuple[float, float]] = {}
    for row in _read_rows(tracks):
        if _row_frame(row, 1) != frame:
            continue
        pid = row.get("point_id", "")
        if pid in {"tip_a", "tip_b"}:
            u = _float(row, "x")
            v = _float(row, "y")
            if u is not None and v is not None:
                pts[pid] = (u, v)
    if "tip_a" in pts and "tip_b" in pts:
        return pts["tip_a"], pts["tip_b"]
    return None


def _visual_line_measurement(profile: CaseProfile, frame: int) -> tuple[tuple[float, float], tuple[float, float]] | None:
    row = _rows_by_frame(profile.result_dir / "line_observations.csv").get(frame, {})
    for prefix in ("physical", "visible"):
        values = tuple(_float(row, f"{prefix}_{key}") for key in ("x1", "y1", "x2", "y2"))
        if all(value is not None for value in values):
            x1, y1, x2, y2 = values
            return (float(x1), float(y1)), (float(x2), float(y2))  # type: ignore[arg-type]
    return None


def _draw_stage4_context(profile: CaseProfile, img: Any, frame: int) -> None:
    line_rows = [
        row for row in _read_rows(profile.result_dir / "line_observations.csv")
        if _row_frame(row, 1) == frame
    ]
    lines: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for row in line_rows:
        values = tuple(_float(row, f"physical_{key}") for key in ("x1", "y1", "x2", "y2"))
        if all(value is not None for value in values):
            x1, y1, x2, y2 = values
            lines.append(((float(x1), float(y1)), (float(x2), float(y2))))  # type: ignore[arg-type]
    for index, line in enumerate(lines):
        _draw_line(img, line[0], line[1], f"visual rigid line {index + 1}", (40, 255, 80))
    paths = stage_paths(profile)
    observation = _rows_by_frame(paths["object_observations"]).get(frame, {})
    if not lines:
        _draw_circle(
            img,
            _float(observation, "ref_u"),
            _float(observation, "ref_v"),
            _float(observation, "radius_px"),
            "visual object measurement",
            (40, 255, 80),
        )
    rows = [row for row in _read_rows(paths["object_contact_points"]) if _row_frame(row, 1) == frame]
    if not rows:
        rows = [row for row in _read_rows(paths["contact_state"]) if _row_frame(row, 1) == frame]
    for row in rows:
        u = _float(row, "palm_u")
        v = _float(row, "palm_v")
        if u is None:
            u = _float(row, "state_active_part_u")
        if v is None:
            v = _float(row, "state_active_part_v")
        side = row.get("human_side") or row.get("contact_side") or row.get("contact_label") or "palm"
        human_active = any(
            str(row.get(field, "")).strip().lower() in {"1", "1.0", "true", "active"}
            for field in ("contact_active", "human_contact_state", "anchor_contact_state")
        )
        floor_active = str(row.get("floor_contact_state", "")).strip().lower() in {"1", "1.0", "true", "active"}
        if floor_active and not human_active:
            u = _float(observation, "support_u") or _float(observation, "ref_u")
            v = _float(observation, "support_v") or _float(observation, "ref_v")
            side = "floor"
        active = human_active or floor_active
        color = (255, 230, 0) if active else (160, 160, 160)
        label = f"{side} active" if active else f"{side} inactive"
        _draw_point(img, u, v, label, color)
    _draw_label(img, "dark render = predicted object; green = visual measurement", (24, 42), (40, 255, 80))
    _draw_label(img, "yellow = active read-only human contact site; gray = inactive", (24, 76), (255, 230, 0))


def _crop_stage4_evidence(profile: CaseProfile, img: Any, frame: int) -> Any:
    """Crop a generic point/line interaction region without object identity rules."""

    row = _rows_by_frame(stage_paths(profile)["object_observations"]).get(frame, {})
    u = _float(row, "ref_u")
    v = _float(row, "ref_v")
    if u is None or v is None:
        return img
    width, height = img.size
    crop_w = min(width, 720)
    crop_h = min(height, 405)
    left = int(round(u - crop_w / 2))
    top = int(round(v - crop_h / 2))
    left = max(0, min(left, width - crop_w))
    top = max(0, min(top, height - crop_h))
    return img.crop((left, top, left + crop_w, top + crop_h))


def _draw_mask(img: Any, mask_path: Path | None) -> None:
    if mask_path is None:
        return
    Image, ImageDraw = _pil_modules()
    if Image is None or ImageDraw is None:
        return
    try:
        mask = Image.open(mask_path).convert("L")
    except Exception:
        return
    if mask.size != img.size:
        mask = mask.resize(img.size)
    color = Image.new("RGB", img.size, (0, 200, 255))
    alpha = mask.point(lambda p: 82 if p > 0 else 0)
    img.paste(Image.blend(img, color, 0.45), (0, 0), alpha)
    bbox = mask.getbbox()
    if bbox:
        draw = ImageDraw.Draw(img)
        draw.rectangle(bbox, outline=(0, 120, 255), width=4)


def _draw_observation_points(profile: CaseProfile, img: Any, frame: int) -> None:
    row = _rows_by_frame(stage_paths(profile)["object_observations"]).get(frame, {})
    if profile.case_name in {"basketball", "football"}:
        _draw_point(img, _float(row, "ref_u"), _float(row, "ref_v"), "object center", (0, 0, 255))
        _draw_point(img, _float(row, "support_u"), _float(row, "support_v"), "support", (0, 255, 255))
    elif profile.case_name == "mug":
        _draw_label(img, "mug body pose / handle phase observation", (24, 42), (0, 0, 255))
    elif profile.case_name == "chair":
        pairs = [
            ("top_rail_left_u", "top_rail_left_v", "top rail L", (0, 0, 255)),
            ("top_rail_right_u", "top_rail_right_v", "top rail R", (0, 0, 255)),
            ("seat_front_left_u", "seat_front_left_v", "seat L", (255, 0, 255)),
            ("seat_front_right_u", "seat_front_right_v", "seat R", (255, 0, 255)),
            ("front_leg_left_top_u", "front_leg_left_top_v", "front leg top", (0, 170, 255)),
            ("front_leg_right_top_u", "front_leg_right_top_v", "front leg top", (0, 170, 255)),
        ]
        for ukey, vkey, label, color in pairs:
            _draw_point(img, _float(row, ukey), _float(row, vkey), label, color)


def _draw_contact_points(profile: CaseProfile, img: Any, frame: int, stage: str) -> None:
    paths = stage_paths(profile)
    source = paths["object_contact_points"] if stage == "stage4" else paths["contact_candidates"]
    rows = [row for row in _read_rows(source) if _row_frame(row, 1) == frame]
    for row in rows:
        u = _float(row, "contact_u")
        v = _float(row, "contact_v")
        if u is None:
            u = _float(row, "contact_point_2d_u")
        if v is None:
            v = _float(row, "contact_point_2d_v")
        label = row.get("object_part") or row.get("contact_object_part") or row.get("chair_local_point_id") or "contact"
        hand = row.get("human_part") or row.get("hand_side") or ""
        if hand and hand != "none":
            label = f"{hand}->{label}"
        _draw_point(img, u, v, label, (0, 255, 255))


def _evidence_points(profile: CaseProfile, frame: int, query_type: str, stage: str) -> list[tuple[float, float, str]]:
    points: list[tuple[float, float, str]] = []
    if query_type in {"contact_relation_check", "keypart_visibility_check"}:
        paths = stage_paths(profile)
        source = paths["object_contact_points"] if stage == "stage4" else paths["contact_candidates"]
        for row in _read_rows(source):
            if _row_frame(row, 1) != frame:
                continue
            u = _float(row, "contact_u")
            v = _float(row, "contact_v")
            if u is None:
                u = _float(row, "contact_point_2d_u")
            if v is None:
                v = _float(row, "contact_point_2d_v")
            if u is not None and v is not None:
                points.append((u, v, row.get("human_side") or row.get("hand_side") or "contact"))
    elif query_type in {"keypart_identity_check", "track_stability_check"}:
        row = _rows_by_frame(stage_paths(profile)["object_observations"]).get(frame, {})
        if profile.case_name in {"basketball", "football"}:
            for ukey, vkey, label in [("ref_u", "ref_v", "center"), ("support_u", "support_v", "support")]:
                u, v = _float(row, ukey), _float(row, vkey)
                if u is not None and v is not None:
                    points.append((u, v, label))
        elif profile.case_name == "chair":
            for ukey, vkey, label in [
                ("top_rail_left_u", "top_rail_left_v", "topL"),
                ("top_rail_right_u", "top_rail_right_v", "topR"),
                ("seat_front_left_u", "seat_front_left_v", "seatL"),
                ("seat_front_right_u", "seat_front_right_v", "seatR"),
            ]:
                u, v = _float(row, ukey), _float(row, vkey)
                if u is not None and v is not None:
                    points.append((u, v, label))
    return points


def _escape_drawtext(text: str) -> str:
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _ffmpeg_point_evidence(profile: CaseProfile, stage: str, frame: int, query_type: str, out_path: Path) -> str:
    ffmpeg = shutil.which("ffmpeg")
    frame_path = _frame_path(profile.sample_dir, frame)
    if not ffmpeg or not frame_path:
        return ""
    points = _evidence_points(profile, frame, query_type, stage)
    if not points:
        return ""
    filters = []
    for idx, (u, v, label) in enumerate(points[:8]):
        x = max(0, int(round(u)) - 12)
        y = max(0, int(round(v)) - 12)
        color = "cyan" if idx % 2 == 0 else "yellow"
        filters.append(f"drawbox=x={x}:y={y}:w=24:h=24:color={color}@0.85:t=fill")
        filters.append(
            "drawtext="
            f"text='{_escape_drawtext(label[:18])}':x={x + 28}:y={max(0, y - 4)}:"
            f"fontsize=22:fontcolor={color}:box=1:boxcolor=black@0.45"
        )
    filters.append(
        "drawtext="
        f"text='{_escape_drawtext(query_type)}':x=24:y=32:"
        "fontsize=26:fontcolor=lime:box=1:boxcolor=black@0.45"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-v", "error", "-i", frame_path, "-vf", ",".join(filters), "-frames:v", "1", str(out_path)]
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        return ""
    return str(out_path) if out_path.exists() else ""


def _loss_plot_evidence(profile: CaseProfile, frame: int, query_type: str) -> Any | None:
    Image, ImageDraw = _pil_modules()
    if Image is None or ImageDraw is None:
        return None
    paths = stage_paths(profile)
    plot_names = ["E_total.png", "E_visual.png", "E_contact.png", "E_support.png", "E_smooth.png", "E_reg.png"]
    plots = []
    for name in plot_names:
        path = paths["loss_dir"] / name
        if path.exists():
            try:
                plots.append((name[:-4], Image.open(path).convert("RGB")))
            except Exception:
                pass
    if not plots:
        return None
    tile_w, tile_h = 360, 120
    cols = 2
    rows = (len(plots) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h + 54), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), f"{profile.case_name} {query_type} frame {frame:05d}: loss diagnostic plots", fill=(20, 20, 20))
    for idx, (label, plot) in enumerate(plots):
        x = (idx % cols) * tile_w
        y = 54 + (idx // cols) * tile_h
        plot = plot.resize((tile_w, tile_h - 20))
        canvas.paste(plot, (x, y + 18))
        draw.text((x + 12, y + 2), label, fill=(35, 95, 180))
    rows_by_fr = _rows_by_frame(paths["loss_residuals"])
    row = rows_by_fr.get(frame, {})
    if row:
        text = " ".join(f"{key}={row.get(key, '')}" for key in ["E_total", "E_visual", "E_contact", "E_support", "E_smooth"] if row.get(key, ""))
        draw.text((16, canvas.size[1] - 18), text[:150], fill=(90, 90, 90))
    return canvas


def materialize_evidence(profile: CaseProfile, stage: str, frame: int, query_type: str) -> str:
    paths = stage_paths(profile)
    out_dir = paths["vlm_dir"] / stage / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"frame{frame:05d}_{query_type}.png"
    img: Any | None
    if stage == "stage7" or query_type == "loss_diagnostic_check":
        img = _loss_plot_evidence(profile, frame, query_type)
        if img is None:
            return ""
        img.save(out_path)
        return str(out_path)
    if stage == "stage4" and query_type in SEMANTIC_QUERY_TYPES:
        Image, ImageDraw = _pil_modules()
        if Image is not None and ImageDraw is not None:
            semantic_config = dict(
                dict(profile.data.get("vlm", {})).get("semantic_orientation", {})
            )
            semantic_radius = int(semantic_config.get("temporal_radius_frames", 1))
            configured_offsets = semantic_config.get("evidence_offsets_frames")
            if isinstance(configured_offsets, (list, tuple)) and configured_offsets:
                offsets = tuple(sorted({int(value) for value in configured_offsets}))
            else:
                offsets = (-2 * semantic_radius, -semantic_radius, 0, semantic_radius, 2 * semantic_radius)
            selected_frames = list(dict.fromkeys(max(1, frame + offset) for offset in offsets))
            original_panels = []
            evidence_panels = []
            for selected_frame in selected_frames:
                original = _read_frame(profile.sample_dir, selected_frame)
                if original is None:
                    continue
                evidence = original.copy()
                _draw_mask(evidence, _mask_path(profile.sample_dir, selected_frame))
                _draw_stage4_context(profile, evidence, selected_frame)
                for panel in (original, evidence):
                    _draw_label(panel, f"frame {selected_frame:03d}", (18, 24), (255, 255, 255))
                original_panels.append(_crop_stage4_evidence(profile, original, selected_frame).resize((320, 180)))
                evidence_panels.append(_crop_stage4_evidence(profile, evidence, selected_frame).resize((320, 180)))
            if original_panels and len(original_panels) == len(evidence_panels):
                label_height = 32
                panel_width, panel_height = 320, 180
                reference_path = semantic_config.get("asset_reference_image")
                reference = None
                if reference_path:
                    candidate = Path(str(reference_path))
                    if not candidate.is_absolute():
                        candidate = profile.sample_dir / candidate
                    if candidate.is_file():
                        try:
                            reference = Image.open(candidate).convert("RGB")
                        except Exception:
                            reference = None
                reference_height = 300 if reference is not None else 0
                reference_label_height = label_height if reference is not None else 0
                canvas = Image.new(
                    "RGB",
                    (
                        panel_width * len(original_panels),
                        2 * (panel_height + label_height) + reference_height + reference_label_height,
                    ),
                    (245, 245, 245),
                )
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, 0, canvas.width, label_height), fill=(20, 20, 20))
                draw.text(
                    (12, 8),
                    "ORIGINAL RGB TEMPORAL STRIP: EARLIER  ->  LATER",
                    fill=(255, 255, 255),
                )
                second_label_y = panel_height + label_height
                draw.rectangle(
                    (0, second_label_y, canvas.width, second_label_y + label_height),
                    fill=(20, 20, 20),
                )
                draw.text(
                    (12, second_label_y + 8),
                    "TYPED EVIDENCE: MASK + RAILS + READ-ONLY CONTACT",
                    fill=(255, 255, 255),
                )
                for index, panel in enumerate(original_panels):
                    canvas.paste(panel, (index * panel_width, label_height))
                for index, panel in enumerate(evidence_panels):
                    canvas.paste(panel, (index * panel_width, second_label_y + label_height))
                if reference is not None:
                    reference_label_y = second_label_y + label_height + panel_height
                    draw.rectangle(
                        (0, reference_label_y, canvas.width, reference_label_y + label_height),
                        fill=(20, 20, 20),
                    )
                    draw.text(
                        (12, reference_label_y + 8),
                        "ASSET IDENTITY REFERENCE: BROAD / NARROW FACES, TWO RAILS, HANDLE, WHEELS",
                        fill=(255, 255, 255),
                    )
                    reference.thumbnail((canvas.width, reference_height))
                    ref_x = (canvas.width - reference.width) // 2
                    ref_y = reference_label_y + label_height + (reference_height - reference.height) // 2
                    canvas.paste(reference, (ref_x, ref_y))
                canvas.save(out_path)
                return str(out_path)

    video = _render_video_for_stage(profile, stage)
    if stage in {"stage4", "stage5"} and query_type in {"temporal_motion_check", "constraint_reliability_check", *SEMANTIC_QUERY_TYPES} and video is not None:
        Image, ImageDraw = _pil_modules()
        if Image is not None:
            frames = []
            semantic_radius = int(
                dict(dict(profile.data.get("vlm", {})).get("semantic_orientation", {})).get(
                    "temporal_radius_frames", 1
                )
            )
            offsets = [-semantic_radius, 0, semantic_radius] if query_type in SEMANTIC_QUERY_TYPES else [-1, 0, 1]
            for offset in offsets:
                fr = max(1, frame + offset)
                panel = _extract_video_frame(video, fr)
                if panel is None:
                    panel = _read_frame(profile.sample_dir, fr)
                if panel is not None:
                    if query_type in SEMANTIC_QUERY_TYPES:
                        _draw_mask(panel, _mask_path(profile.sample_dir, fr))
                    _draw_stage4_context(profile, panel, fr)
                    _draw_label(panel, f"frame {fr:03d}", (18, 24), (255, 255, 255))
                    panel = _crop_stage4_evidence(profile, panel, fr)
                    panel = panel.resize((426, 240))
                    frames.append(panel)
            if frames:
                canvas = Image.new("RGB", (426 * len(frames), 240), (245, 245, 245))
                for idx, panel in enumerate(frames):
                    canvas.paste(panel, (idx * 426, 0))
                canvas.save(out_path)
                return str(out_path)
    if video is not None:
        img = _extract_video_frame(video, frame)
    else:
        img = _read_frame(profile.sample_dir, frame)
    if img is None:
        return _ffmpeg_point_evidence(profile, stage, frame, query_type, out_path)

    if query_type == "target_mask_check":
        _draw_mask(img, _mask_path(profile.sample_dir, frame))
        _draw_label(img, "highlighted target mask", (24, 42), (0, 120, 255))
    elif query_type in {"keypart_identity_check", "track_stability_check"}:
        _draw_observation_points(profile, img, frame)
        _draw_label(img, "highlighted observation / tracked semantic part", (24, 42), (0, 0, 255))
    elif query_type in {"contact_relation_check", "keypart_visibility_check"}:
        _draw_contact_points(profile, img, frame, stage)
        _draw_label(img, "highlighted contact candidate", (24, 42), (0, 255, 255))
    elif query_type == "overlay_alignment_check":
        _draw_label(img, "rendered overlay alignment check", (24, 42), (0, 255, 0))
    elif stage == "stage4" and query_type in {"anchor_update_check", "constraint_reliability_check"}:
        _draw_stage4_context(profile, img, frame)
    elif query_type == "temporal_motion_check":
        _draw_label(img, "temporal motion check: inspect neighboring context from stage rows", (24, 42), (255, 200, 0))
    elif query_type == "post_render_sanity_check":
        _draw_label(img, "final render sanity check", (24, 42), (0, 255, 0))
    elif query_type == "baseline_regression_check":
        _draw_label(img, "baseline regression check", (24, 42), (0, 255, 0))
    elif query_type == "loss_diagnostic_check":
        _draw_label(img, "loss diagnostic check", (24, 42), (255, 160, 0))

    img.save(out_path)
    return str(out_path)


def _materialize_interval_candidate_evidence(
    profile: CaseProfile,
    *,
    frame: int,
    start: int,
    end: int,
    stable_first: bool = True,
) -> str:
    Image, ImageDraw = _pil_modules()
    if Image is None:
        return ""
    candidate_root = profile.result_dir / "generic_stage4_candidate"
    stable_video = candidate_root / "stable" / "object_only" / "overlay.mp4"
    challenger_video = candidate_root / "occlusion_challenger" / "object_only" / "overlay.mp4"
    videos = (
        ("CANDIDATE A", candidate_root / "stable" / "object_only" / "overlay.mp4"),
        (
            "CANDIDATE B",
            candidate_root / "occlusion_challenger" / "object_only" / "overlay.mp4",
        ),
    )
    if not stable_first:
        videos = (("CANDIDATE A", challenger_video), ("CANDIDATE B", stable_video))
    if any(not path.is_file() for _label, path in videos):
        return ""
    out_dir = stage_paths(profile)["vlm_dir"] / "stage4" / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    order = "stable_first" if stable_first else "challenger_first"
    out_path = out_dir / f"frame{frame:05d}_{INTERVAL_SELECTION_QUERY_TYPE}_{order}.png"
    columns = [max(start, frame - 1), frame, min(end, frame + 1)]
    tile_width, tile_height, label_height = 426, 240, 32
    canvas = Image.new(
        "RGB",
        (tile_width * len(columns), (tile_height + label_height) * len(videos)),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, (label, video) in enumerate(videos):
        y = row_index * (tile_height + label_height)
        draw.rectangle((0, y, canvas.width, y + label_height), fill=(20, 20, 20))
        draw.text((12, y + 8), f"{label}  interval {start}-{end}", fill=(255, 255, 255))
        for column_index, selected_frame in enumerate(columns):
            panel = _extract_video_frame(video, selected_frame)
            if panel is None:
                return ""
            _draw_mask(panel, _mask_path(profile.sample_dir, selected_frame))
            _draw_stage4_context(profile, panel, selected_frame)
            panel = _crop_stage4_evidence(profile, panel, selected_frame).resize(
                (tile_width, tile_height)
            )
            _draw_label(panel, f"frame {selected_frame:03d}", (16, 20), (255, 255, 255))
            canvas.paste(panel, (column_index * tile_width, y + label_height))
    canvas.save(out_path)
    return str(out_path)


def render_reference(profile: CaseProfile, stage: str, frame: int, query_type: str) -> str:
    evidence = materialize_evidence(profile, stage, frame, query_type)
    if evidence:
        return evidence
    return ""


def question_for(profile: CaseProfile, query_type: str) -> tuple[str, list[str], str]:
    obj = object_vlm_profile(profile)
    target = str(obj["target"])
    parts = list(obj["parts"])  # type: ignore[arg-type]
    contacts = list(obj["contacts"])  # type: ignore[arg-type]
    stable_refs = list(obj["stable_refs"])  # type: ignore[arg-type]
    if query_type in PREDICATE_BY_QUERY_TYPE:
        spec = question_for_query_type(target, query_type)
        return (
            str(spec["question"]),
            str(spec["choices"]).split("|"),
            "temporal_rigid_face_rails_wheels_handle_support_overlay",
        )
    if query_type == "target_mask_check":
        return (
            f"Does the highlighted mask correspond to the target {target}?",
            ["yes", "partially", "wrong_object", "missing_object", "unclear"],
            "mask_or_bbox_overlay",
        )
    if query_type == "keypart_identity_check":
        return (f"Which {target} part is highlighted?", parts, "semantic_part_overlay")
    if query_type == "keypart_visibility_check":
        return (
            f"Is the highlighted key {target} part visible in this frame?",
            ["visible", "partially_visible", "hidden", "occluded_by_human", "unclear"],
            "visibility_crop_or_overlay",
        )
    if query_type == "track_stability_check":
        return (
            "Is the highlighted reference point or part stable and visually trackable across neighboring frames?",
            ["stable", "unstable_due_to_occlusion", "unstable_not_real_part", "wrong_part", "unclear"],
            "track_triplet_overlay:" + ",".join(stable_refs),
        )
    if query_type == "contact_relation_check":
        return ("What is the highlighted object contact region closest to?", contacts, "contact_candidate_overlay")
    if query_type == "overlay_alignment_check":
        return (
            "Does the rendered object overlay sit on top of the same visible physical object in the video? Choose aligned only if the rendered object follows the real visible object; choose lateral_shift, wrong_scale, or rotation_error if it is offset, too long/short, or at a different angle.",
            ["aligned", "lateral_shift", "depth_shift", "wrong_scale", "rotation_error", "wrong_object", "unclear"],
            "render_overlay",
        )
    if query_type == "anchor_update_check":
        return (
            "Using the visual guide in the image: the dark render is the predicted object geometry, the green overlay is the visual object measurement, yellow marks are active read-only human contact sites, and gray marks are inactive sites. Does the predicted object preserve both visible alignment and active contact without creating a wrong contact?",
            ["improvement", "no_change", "worse_overlay", "wrong_contact", "unclear"],
            "before_after_anchor_update_overlay",
        )
    if query_type == "temporal_motion_check":
        return (
            "In the three-panel neighboring-frame view, do the predicted object geometry and visual object measurement move consistently without sudden jumps, wrong-object tracking, or physically implausible motion?",
            ["consistent", "jumps_unnaturally", "wrong_object_tracking", "physically_implausible", "unclear"],
            "temporal_triplet_overlay",
        )
    if query_type == "constraint_reliability_check":
        return (
            "Judge each overlay against the physical object pixels, not against the dark render. The green overlay is a visual object measurement. Yellow marks are active human contact sites from a read-only body estimate. Use this rubric: choose both_consistent when green follows the visible physical object and every yellow active site lies on that same object at a plausible hand contact; choose visual_observation_reliable when green follows the physical object but a yellow site is visibly off it; choose contact_relation_reliable when yellow sites lie on the physical object but green tracks the wrong location, extent, or angle; otherwise choose unclear. Never choose visual_observation_reliable merely because green agrees with the dark render.",
            ["both_consistent", "visual_observation_reliable", "contact_relation_reliable", "unclear"],
            "visual_contact_constraint_temporal_overlay",
        )
    if query_type == INTERVAL_SELECTION_QUERY_TYPE:
        return (
            "Compare Candidate A and Candidate B against the same real object pixels and neighboring frames. Judge rigid body and rail geometry, continuous hand-handle relation, floor support, and temporal motion. Choose candidate_a or candidate_b only when that row is visibly more reliable inside the bounded interval without introducing a jump. Choose reject_both for a clear physical or alignment failure in both. Choose unclear when the occlusion prevents a reliable comparison. The row names carry no quality meaning. Do not estimate pose coordinates or loss weights.",
            ["candidate_a", "candidate_b", "reject_both", "unclear"],
            "stable_vs_occlusion_challenger_temporal_overlay",
        )
    if query_type == AMODAL_MASK_QUERY_TYPE:
        return (
            "The left CURRENT panel shows the original incomplete object mask in white/cyan over the RGB frame. The middle and right panels show the last and next frames where both rigid parallel rails are visible. Decide whether the missing region inside the CURRENT suitcase body outline is caused by human occlusion and should be completed as one rigid body. The completion will preserve every original mask pixel and fill only the body-region gap; it will not invent handle or person pixels.",
            list(AMODAL_MASK_CHOICES),
            "amodal_rigid_mask_candidates",
        )
    if query_type == "post_render_sanity_check":
        return (
            "Inspect the dark rendered object over the video. Does it accurately cover the real visible object without floating away, using the wrong angle/scale, or incorrectly covering the human? Choose pass only if alignment is visually good.",
            ["pass", "overlay_shift", "wrong_contact", "object_jitter", "depth_mismatch", "render_missing", "unclear"],
            "final_render_preview",
        )
    if query_type == "baseline_regression_check":
        return (
            "Compared with the solved baseline, does the current render show a regression?",
            ["no_regression", "worse_overlay", "worse_contact", "worse_jitter", "missing_output", "unclear"],
            "baseline_comparison_preview",
        )
    if query_type == "loss_diagnostic_check":
        return (
            "Which label best describes the highlighted loss diagnostic plots?",
            ["normal", "visual_spike", "contact_spike", "smoothness_spike", "support_spike", "missing_loss", "unclear"],
            "loss_diagnostic_plots",
        )
    raise ValueError(f"Unknown VLM query type: {query_type}")


def query_types_for_stage(profile: CaseProfile, stage: str) -> list[str]:
    if "disable_vlm_semantic_evidence" in set(profile.data.get("ablation_flags", ())):
        return []
    obj = object_vlm_profile(profile)
    policy = obj.get("vlm_query_policy")
    selected: list[str] = []
    if isinstance(policy, dict):
        values = policy.get(stage)
        if isinstance(values, list):
            out = [str(v) for v in values if str(v) in STAGE_QUERY_TYPES.get(stage, [])]
            if out:
                selected = out
    if not selected:
        selected = list(STAGE_QUERY_TYPES.get(stage, []))
    if stage == "stage4":
        semantic = dict(dict(profile.data.get("vlm", {})).get("semantic_orientation", {}))
        disabled = "disable_vlm_semantic_evidence" in set(profile.data.get("ablation_flags", ()))
        configured = {str(value) for value in semantic.get("predicates", ())}
        selected = [
            query_type for query_type in selected
            if (
                not disabled
                or query_type not in {AMODAL_MASK_QUERY_TYPE, INTERVAL_SELECTION_QUERY_TYPE}
            )
            and (
                query_type not in PREDICATE_BY_QUERY_TYPE
                or (
                    semantic.get("enabled", False)
                    and not disabled
                    and PREDICATE_BY_QUERY_TYPE[query_type] in configured
                )
            )
        ]
    runtime = profile.data.get("factor_runtime")
    if stage == "stage4" and isinstance(runtime, dict):
        kinds = {str(value) for value in runtime}
        visual = {"point_reprojection", "line_reprojection", "mask_silhouette", "metric_depth", "depth_order"}
        contact = {"contact_distance", "contact_relative_velocity", "local_anchor_constancy"}
        if kinds & visual and kinds & contact and "constraint_reliability_check" not in selected:
            selected.append("constraint_reliability_check")
    return selected


def build_queries(
    profile: CaseProfile,
    stage: str,
    query_type_filter: str | None = None,
) -> list[dict[str, object]]:
    queries: list[dict[str, object]] = []
    obj = object_vlm_profile(profile)
    for query_type in query_types_for_stage(profile, stage):
        if query_type_filter and query_type != query_type_filter:
            continue
        if query_type in PREDICATE_BY_QUERY_TYPE:
            windows = profile_uncertain_windows(profile)
        elif query_type == "constraint_reliability_check":
            windows = constraint_risk_intervals(profile)
        elif query_type in {AMODAL_MASK_QUERY_TYPE, INTERVAL_SELECTION_QUERY_TYPE}:
            windows = completion_intervals(profile)
        else:
            frames = temporal_risk_keyframes(profile, stage) if stage == "stage5" and query_type == "temporal_motion_check" else stage_keyframes(profile, stage)
            windows = [(frame, frame, frame, time) for frame, time in frames]
        for frame, start_frame, end_frame, time in windows:
            question, choices, highlight = question_for(profile, query_type)
            if query_type == AMODAL_MASK_QUERY_TYPE:
                evidence_path = materialize_candidate_evidence(
                    profile,
                    frame=frame,
                    start=start_frame,
                    end=end_frame,
                    out_path=stage_paths(profile)["vlm_dir"] / stage / "evidence" / f"frame{frame:05d}_{query_type}.png",
                )
            elif query_type == INTERVAL_SELECTION_QUERY_TYPE:
                evidence_path = _materialize_interval_candidate_evidence(
                    profile,
                    frame=frame,
                    start=start_frame,
                    end=end_frame,
                )
            else:
                evidence_path = render_reference(profile, stage, frame, query_type)
            if query_type == INTERVAL_SELECTION_QUERY_TYPE and not evidence_path:
                # Pairwise arbitration must never degrade to a raw single frame;
                # both current-attempt renders are required evidence.
                continue
            input_path = evidence_path or _frame_path(profile.sample_dir, frame)
            evidence_sha256 = hashlib.sha256(Path(input_path).read_bytes()).hexdigest() if input_path and Path(input_path).is_file() else ""
            query_id = (
                f"{stage}:{query_type}:{frame:05d}:{start_frame:05d}-{end_frame:05d}"
                if query_type in {AMODAL_MASK_QUERY_TYPE, INTERVAL_SELECTION_QUERY_TYPE}
                else f"{stage}:{query_type}:{start_frame:05d}-{end_frame:05d}"
            )
            queries.append(
                {
                    "query_id": query_id,
                    "case_name": profile.case_name,
                    "stage": stage,
                    "frame": frame,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "time": time,
                    "query_type": query_type,
                    "input_image_path": input_path,
                    "input_render_path": evidence_path,
                    "highlight_layer": highlight,
                    "question": question,
                    "choices": "|".join(choices),
                    "hoi_profile_source": str(obj.get("source", "case_config")),
                    "object_part_choices": "|".join(str(v) for v in obj.get("parts", [])),
                    "contact_relation_choices": "|".join(str(v) for v in obj.get("contacts", [])),
                    "gate_policy": "forced_choice_gate_only_no_continuous_pose_or_loss",
                    "evidence_sha256": evidence_sha256,
                    "visual_factor_roles": "point_reprojection|line_reprojection|mask_silhouette|metric_depth|depth_order",
                    "contact_factor_roles": "contact_distance|contact_relative_velocity|local_anchor_constancy",
                    "candidate_a_role": (
                        "stable" if query_type == INTERVAL_SELECTION_QUERY_TYPE else ""
                    ),
                    "candidate_b_role": (
                        "occlusion_challenger" if query_type == INTERVAL_SELECTION_QUERY_TYPE else ""
                    ),
                    "semantic_predicate": PREDICATE_BY_QUERY_TYPE.get(query_type, ""),
                    "subject_entity": "target_object" if query_type in PREDICATE_BY_QUERY_TYPE else "",
                    "object_entity": "human" if query_type in PREDICATE_BY_QUERY_TYPE else "",
                }
            )
    mirrored: list[dict[str, object]] = []
    for query in queries:
        if query.get("query_type") != INTERVAL_SELECTION_QUERY_TYPE:
            continue
        evidence_path = _materialize_interval_candidate_evidence(
            profile,
            frame=int(query["frame"]),
            start=int(query["start_frame"]),
            end=int(query["end_frame"]),
            stable_first=False,
        )
        if not evidence_path:
            continue
        mirrored_query = dict(query)
        mirrored_query.update(
            {
                "query_id": str(query["query_id"]) + ":mirrored",
                "input_image_path": evidence_path,
                "input_render_path": evidence_path,
                "evidence_sha256": hashlib.sha256(Path(evidence_path).read_bytes()).hexdigest(),
                "candidate_a_role": "occlusion_challenger",
                "candidate_b_role": "stable",
            }
        )
        mirrored.append(mirrored_query)
    queries.extend(mirrored)
    return queries


def dry_run_results(queries: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for query in queries:
        rows.append(
            {
                "case_name": query["case_name"],
                "stage": query["stage"],
                "frame": query["frame"],
                "query_type": query["query_type"],
                "normalized_label": "dry_run_not_evaluated",
                "pass_gate": "not_evaluated",
                "repair_action": "none",
                "used_for_anchor_update": "0",
                "used_for_contact_residual": "0",
                "used_for_report": "1",
            }
        )
    return rows


def write_stage_verification(
    profile: CaseProfile,
    stage: str,
    query_type_filter: str | None = None,
) -> dict[str, object]:
    paths = stage_paths(profile)
    out_dir = paths["vlm_dir"] / stage
    selected_queries = build_queries(profile, stage, query_type_filter)
    selected_results = dry_run_results(selected_queries)
    if query_type_filter:
        existing_queries = [
            row
            for row in _read_rows(out_dir / "vlm_queries.csv")
            if row.get("query_type") != query_type_filter
        ]
        existing_results = [
            row
            for row in _read_rows(out_dir / "vlm_results.csv")
            if row.get("query_type") != query_type_filter
        ]
        queries = [*existing_queries, *selected_queries]
        results = [*existing_results, *selected_results]
    else:
        queries = selected_queries
        results = selected_results
    write_csv(out_dir / "vlm_queries.csv", queries, QUERY_FIELDS)
    write_csv(out_dir / "vlm_results.csv", results, RESULT_FIELDS)
    decision = fuse_stage_decision(profile.case_name, stage, results, mode="dry_run")
    write_json(out_dir / "stage_decision.json", decision)
    write_aggregate(profile)
    return decision


def write_aggregate(profile: CaseProfile) -> None:
    paths = stage_paths(profile)
    root = paths["vlm_dir"]
    all_queries: list[dict[str, object]] = []
    all_results: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    for stage in STAGE_QUERY_TYPES:
        qpath = root / stage / "vlm_queries.csv"
        rpath = root / stage / "vlm_results.csv"
        dpath = root / stage / "stage_decision.json"
        if qpath.exists():
            with qpath.open(newline="") as f:
                all_queries.extend(dict(row) for row in csv.DictReader(f))
        if rpath.exists():
            with rpath.open(newline="") as f:
                all_results.extend(dict(row) for row in csv.DictReader(f))
        if dpath.exists():
            try:
                import json

                payload = json.loads(dpath.read_text())
            except Exception:
                payload = {}
            decisions.append({"stage": stage, "decision_path": str(dpath), **payload})
    write_csv(paths["vlm_queries"], all_queries, QUERY_FIELDS)
    write_csv(paths["vlm_results"], all_results, RESULT_FIELDS)
    modes = sorted({str(item.get("mode", "unknown")) for item in decisions if item.get("mode")})
    if not modes:
        mode_line = "Mode: no stage decisions found."
    elif modes == ["dry_run"]:
        mode_line = "Mode: dry-run query generation only. No VLM model was called."
    elif modes == ["qwen_vl"]:
        mode_line = "Mode: local Qwen-VL forced-choice verification."
    else:
        mode_line = f"Mode: mixed stage decisions ({', '.join(modes)})."
    lines = [
        f"# VLM Verification Summary: {profile.case_name}",
        "",
        mode_line,
        "",
        f"- Total queries: {len(all_queries)}",
        f"- Total result rows: {len(all_results)}",
        "",
        "## Stage Decisions",
        "",
    ]
    for item in decisions:
        lines.append(
            f"- {item['stage']}: {item.get('decision', 'unknown')} "
            f"blocking={item.get('blocking', '')} gates={item.get('gate_counts', {})} "
            f"`{item['decision_path']}`"
        )
    paths["vlm_summary"].write_text("\n".join(lines) + "\n")
