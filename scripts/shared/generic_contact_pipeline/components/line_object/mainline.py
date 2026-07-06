from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from ...core.base.config import CaseProfile
from ...core.base.io import read_csv, write_csv, write_json
from ...core.base.schema import stage_paths
from ...core.gates.vlm_gates import disabled_anchor_frames, disabled_contact_frames


def _f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        return default if value in {"", None} else float(value)
    except Exception:
        return default


def _frame(row: dict[str, str]) -> int:
    return int(float(row.get("frame", "0") or 0))


def _angle(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.atan2(y2 - y1, x2 - x1)


def _angle_gap(a: float, b: float) -> float:
    d = abs(math.atan2(math.sin(a - b), math.cos(a - b)))
    return min(d, abs(math.pi - d))


def _line_from_tracks(rows: list[dict[str, str]]) -> dict[int, dict[str, float | str]]:
    grouped: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in rows:
        pid = row.get("point_id", "")
        if pid not in {"tip_a", "tip_b"}:
            continue
        try:
            grouped[_frame(row)][pid] = {
                "x": float(row["x"]),
                "y": float(row["y"]),
                "visible": _f(row, "visible", 1.0),
            }
        except Exception:
            continue
    out: dict[int, dict[str, float | str]] = {}
    for fr, pts in grouped.items():
        if "tip_a" in pts and "tip_b" in pts:
            a = pts["tip_a"]
            b = pts["tip_b"]
            visibility = min(float(a.get("visible", 1.0)), float(b.get("visible", 1.0)))
            out[fr] = {
                "x1": float(a["x"]),
                "y1": float(a["y"]),
                "x2": float(b["x"]),
                "y2": float(b["y"]),
                "source": "cotracker_sam2_mask_points",
                "cotracker_visibility": visibility,
            }
    return out


def _line_from_sam2_rows(rows: list[dict[str, str]]) -> dict[int, dict[str, float | str]]:
    out: dict[int, dict[str, float | str]] = {}
    for row in rows:
        try:
            fr = _frame(row)
            out[fr] = {
                "x1": float(row["x1"]),
                "y1": float(row["y1"]),
                "x2": float(row["x2"]),
                "y2": float(row["y2"]),
                "source": row.get("source", "sam2_mask_major_axis") or "sam2_mask_major_axis",
                "mask_area": _f(row, "mask_area", 0.0),
                "mask_major_px": _f(row, "mask_major_px", math.nan),
            }
        except Exception:
            continue
    return out


def _sam2_line_rows_from_masks(masks_dir: Path) -> list[dict[str, str]]:
    if not masks_dir.exists():
        return []
    import cv2
    import numpy as np

    rows: list[dict[str, str]] = []
    for mask_path in sorted(masks_dir.glob("*_mask.png")):
        try:
            fr = int(mask_path.stem.split("_")[0])
        except Exception:
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        ys, xs = np.where(mask > 0)
        if len(xs) < 8:
            continue
        pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        center = pts.mean(axis=0)
        centered = pts - center
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        if axis[0] < 0:
            axis = -axis
        proj = centered @ axis
        lo = float(np.percentile(proj, 1.0))
        hi = float(np.percentile(proj, 99.0))
        p1 = center + axis * lo
        p2 = center + axis * hi
        rows.append(
            {
                "frame": str(fr),
                "x1": f"{float(p1[0]):.3f}",
                "y1": f"{float(p1[1]):.3f}",
                "x2": f"{float(p2[0]):.3f}",
                "y2": f"{float(p2[1]):.3f}",
                "mask_area": str(int(len(xs))),
                "mask_major_px": f"{float(hi - lo):.6f}",
                "source": "sam2_mask_major_axis",
            }
        )
    return rows


def build_line_correspondence_rows(
    track_rows: list[dict[str, str]],
    *,
    line_length_m: float,
    sam2_line_rows: list[dict[str, str]] | None = None,
) -> list[dict[str, object]]:
    """Convert per-frame visible line tracks into physical endpoint state.

    SAM2 mask geometry is the primary visible line evidence. CoTracker is a
    low-trust correspondence hint and is never allowed to make a low-visibility
    raw track trusted by itself.
    """
    cotracker_tracks = _line_from_tracks(track_rows)
    sam2_tracks = _line_from_sam2_rows(sam2_line_rows or [])
    tracks = {
        fr: (sam2_tracks.get(fr) or cotracker_tracks[fr])
        for fr in sorted(set(cotracker_tracks) | set(sam2_tracks))
        if fr in sam2_tracks or fr in cotracker_tracks
    }
    rows: list[dict[str, object]] = []
    ref_len: float | None = None
    prev_phys: tuple[float, float, float, float] | None = None
    prev_angle: float | None = None
    prev_center: tuple[float, float] | None = None
    for fr in sorted(tracks):
        obs = tracks[fr]
        x1 = float(obs["x1"])
        y1 = float(obs["y1"])
        x2 = float(obs["x2"])
        y2 = float(obs["y2"])
        primary_source = str(obs.get("source", "unknown"))
        cotracker_visibility = float(obs.get("cotracker_visibility", math.nan))
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        length = math.hypot(x2 - x1, y2 - y1)
        angle = _angle(x1, y1, x2, y2)
        if ref_len is None:
            ref_len = max(length, 1.0)
        visible_fraction = max(0.0, min(1.0, length / max(ref_len, 1.0)))
        center_jump = math.hypot(cx - prev_center[0], cy - prev_center[1]) if prev_center is not None else 0.0
        angle_jump = _angle_gap(angle, prev_angle) if prev_angle is not None else 0.0
        too_short = visible_fraction < 0.68
        too_long = length > max(ref_len * 1.55, ref_len + 220.0)
        jumpy = center_jump > 100.0 or angle_jump > math.radians(42.0)
        primary_is_sam2 = primary_source.startswith("sam2")
        low_cotracker_visibility = primary_source.startswith("cotracker") and (
            not math.isfinite(cotracker_visibility) or cotracker_visibility < 0.5
        )
        temporal_untrusted = jumpy and not primary_is_sam2
        trusted = not (too_short or too_long or temporal_untrusted or low_cotracker_visibility)
        if trusted:
            ref_len = 0.93 * ref_len + 0.07 * length
            phys = (x1, y1, x2, y2)
            source = primary_source
            state = "visible"
            conf = 1.0 if primary_source.startswith("sam2") else 0.55
        elif prev_phys is not None:
            phys = prev_phys
            source = "propagated_physical_endpoints"
            state = "partial_visible" if too_short else "occluded" if (jumpy or low_cotracker_visibility) else "motion_blur_or_spurious"
            conf = 0.35
        else:
            phys = (x1, y1, x2, y2)
            source = "untrusted_visible_segment"
            state = "partial_visible" if too_short else "occluded"
            conf = 0.25
        prev_phys = phys
        prev_angle = angle if trusted else prev_angle
        prev_center = (cx, cy) if trusted else prev_center
        rows.append(
            {
                "frame": fr,
                "visible_x1": f"{x1:.3f}",
                "visible_y1": f"{y1:.3f}",
                "visible_x2": f"{x2:.3f}",
                "visible_y2": f"{y2:.3f}",
                "visible_len_px": f"{length:.6f}",
                "visible_angle_rad": f"{angle:.9f}",
                "physical_x1": f"{phys[0]:.3f}",
                "physical_y1": f"{phys[1]:.3f}",
                "physical_x2": f"{phys[2]:.3f}",
                "physical_y2": f"{phys[3]:.3f}",
                "physical_length_m": f"{line_length_m:.6f}",
                "visible_fraction": f"{visible_fraction:.6f}",
                "occlusion_state": state,
                "endpoint_correspondence_source": source,
                "endpoint_track_conf": f"{conf:.6f}",
                "line_observation_trusted": "1" if trusted else "0",
                "center_jump_px": f"{center_jump:.6f}",
                "angle_jump_deg": f"{math.degrees(angle_jump):.6f}",
                "primary_observation_source": primary_source,
                "cotracker_visibility": f"{cotracker_visibility:.6f}" if math.isfinite(cotracker_visibility) else "",
            }
        )
    return rows


def write_line_correspondence(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    track_path = profile.sample_dir / "results" / "tracking" / "object_mesh_tracks_test.csv"
    if not track_path.exists():
        return {"component": "generic_line_correspondence", "enabled": False, "reason": f"missing {track_path}"}
    sam2_rows = _sam2_line_rows_from_masks(profile.sample_dir / "results" / "segmentation" / "masks")
    rows = build_line_correspondence_rows(
        read_csv(track_path),
        line_length_m=float(profile.data.get("line_object", {}).get("length_m", 1.0)),
        sam2_line_rows=sam2_rows,
    )
    corr = write_csv(paths["line_correspondence"], rows)
    obs = write_csv(paths["line_observations"], rows)
    return {
        "component": "generic_line_correspondence",
        "enabled": True,
        "line_correspondence": str(corr),
        "line_observations": str(obs),
        "rows": len(rows),
        "trusted_rows": sum(1 for r in rows if r.get("line_observation_trusted") == "1"),
        "untrusted_rows": sum(1 for r in rows if r.get("line_observation_trusted") == "0"),
        "sam2_rows": len(sam2_rows),
        "policy": "SAM2 mask major-axis is primary line observation; CoTracker is a low-trust correspondence hint unless visible and temporally consistent",
    }


def build_anchor_state_rows(
    contact_rows: list[dict[str, str]],
    correspondence_by_frame: dict[int, dict[str, str]],
    *,
    vlm_disabled_contact_frames: set[int] | None = None,
    vlm_disabled_anchor_frames: set[int] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    previous_stable: dict[str, float] = {}
    relock_history: dict[str, list[float]] = defaultdict(list)
    disabled_contact = vlm_disabled_contact_frames or set()
    disabled_anchor = vlm_disabled_anchor_frames or set()
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in contact_rows:
        grouped[_frame(row)].append(row)
    suppressed: set[tuple[int, str]] = set()
    for fr, frame_rows in grouped.items():
        active = [
            row
            for row in frame_rows
            if str(row.get("contact_active", "0")).strip().lower() in {"1", "true", "yes"}
            and math.isfinite(_f(row, "hand_joint_median_line_px"))
        ]
        if len(active) < 2:
            continue
        ranked = sorted(active, key=lambda row: (_f(row, "hand_joint_median_line_px", 999.0), _f(row, "fingertip_median_line_px", 999.0)))
        best = ranked[0]
        best_score = _f(best, "hand_joint_median_line_px", 999.0)
        for row in ranked[1:]:
            score = _f(row, "hand_joint_median_line_px", 999.0)
            tip_score = _f(row, "fingertip_median_line_px", score)
            if score >= max(16.0, best_score * 1.45) and tip_score >= max(16.0, _f(best, "fingertip_median_line_px", best_score) * 1.35):
                suppressed.add((fr, row.get("human_side", row.get("contact_side", ""))))
    for row in sorted(contact_rows, key=lambda r: (_frame(r), r.get("human_side", ""))):
        fr = _frame(row)
        side = row.get("human_side", row.get("contact_side", ""))
        corr = correspondence_by_frame.get(fr, {})
        trusted_line = str(corr.get("line_observation_trusted", "1")).strip() == "1"
        occlusion_state = corr.get("occlusion_state", "visible")
        raw_contact_observed = str(row.get("contact_active", "0")).strip().lower() in {"1", "true", "yes"}
        contact_disambiguation = "suppressed_weaker_hand_joint_contact" if (fr, side) in suppressed else "kept"
        contact_observed = raw_contact_observed and fr not in disabled_contact and (fr, side) not in suppressed
        dist = _f(row, "palm_to_line_px", 999.0)
        observed_s = _f(row, "object_local_s")
        prior_stable_s = previous_stable.get(side, _f(row, "stable_object_local_s", observed_s))
        stable_s = prior_stable_s if side in previous_stable else _f(row, "stable_object_local_s", prior_stable_s)
        local_s_drift = abs(observed_s - prior_stable_s) if math.isfinite(observed_s) and math.isfinite(prior_stable_s) else math.inf
        strong_contact = contact_observed and math.isfinite(dist) and dist <= 55.0
        precise_contact = contact_observed and math.isfinite(dist) and dist <= 18.0
        stable_update_consistent = not (side in previous_stable and local_s_drift > 0.18)
        relock_candidate = precise_contact and trusted_line and occlusion_state == "visible" and math.isfinite(observed_s) and local_s_drift > 0.10
        if relock_candidate:
            relock_history[side].append(observed_s)
            relock_history[side] = relock_history[side][-3:]
        else:
            relock_history[side] = []
        relock_allowed = False
        if len(relock_history[side]) >= 3:
            vals = np.asarray(relock_history[side], dtype=float)
            relock_allowed = float(np.max(vals) - np.min(vals)) <= 0.035
        update_allowed = (
            strong_contact
            and trusted_line
            and occlusion_state == "visible"
            and ((stable_update_consistent and fr not in disabled_anchor) or relock_allowed)
        )
        persistent_contact = side in previous_stable or math.isfinite(stable_s)
        pose_allowed = contact_observed and math.isfinite(dist) and dist <= 75.0 and (trusted_line or persistent_contact)
        if update_allowed and math.isfinite(observed_s):
            stable_s = observed_s
            previous_stable[side] = stable_s
            action = "trusted_contact_relock" if relock_allowed and fr in disabled_anchor else "update"
        else:
            if math.isfinite(stable_s):
                previous_stable[side] = stable_s
            action = "propagate" if contact_observed else "inactive"
        rows.append(
            {
                "frame": fr,
                "time": row.get("time", ""),
                "human_side": side,
                "raw_contact_observed": "1" if raw_contact_observed else "0",
                "contact_observed": "1" if contact_observed else "0",
                "contact_persistent": "1" if side in previous_stable else "0",
                "anchor_update_allowed": "1" if update_allowed else "0",
                "pose_anchor_allowed": "1" if pose_allowed else "0",
                "anchor_action": action,
                "vlm_contact_disabled": "1" if fr in disabled_contact else "0",
                "vlm_anchor_disabled": "1" if fr in disabled_anchor else "0",
                "contact_disambiguation": contact_disambiguation,
                "hand_joint_median_line_px": row.get("hand_joint_median_line_px", ""),
                "fingertip_median_line_px": row.get("fingertip_median_line_px", ""),
                "observed_object_local_s": f"{observed_s:.6f}" if math.isfinite(observed_s) else "",
                "stable_object_local_s": f"{stable_s:.6f}" if math.isfinite(stable_s) else "",
                "anchor_u": row.get("palm_u", row.get("state_active_part_u", "")),
                "anchor_v": row.get("palm_v", row.get("state_active_part_v", "")),
                "anchor_z_m": row.get("anchor_z_m", row.get("palm_z", "")),
                "palm_to_line_px": row.get("palm_to_line_px", ""),
                "line_observation_trusted": "1" if trusted_line else "0",
                "occlusion_state": occlusion_state,
            }
        )
    return rows


def write_anchor_state(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    if not paths["contact_candidates"].exists() or not paths["line_correspondence"].exists():
        return {"component": "generic_anchor_state", "enabled": False, "reason": "missing contact candidates or line correspondence"}
    corr_by_frame = {_frame(r): r for r in read_csv(paths["line_correspondence"])}
    disabled_contact = disabled_contact_frames(profile, stage="stage2")
    disabled_anchor = disabled_anchor_frames(profile, stage="stage1") | disabled_anchor_frames(profile, stage="stage4")
    rows = build_anchor_state_rows(
        read_csv(paths["contact_candidates"]),
        corr_by_frame,
        vlm_disabled_contact_frames=disabled_contact,
        vlm_disabled_anchor_frames=disabled_anchor,
    )
    out = write_csv(paths["anchor_state"], rows)
    metrics = {
        "component": "generic_anchor_state",
        "enabled": True,
        "anchor_state": str(out),
        "rows": len(rows),
        "anchor_updates": sum(1 for r in rows if r.get("anchor_update_allowed") == "1"),
        "pose_anchor_rows": sum(1 for r in rows if r.get("pose_anchor_allowed") == "1"),
        "vlm_disabled_contact_frames": sorted(disabled_contact),
        "vlm_disabled_anchor_frames": sorted(disabled_anchor),
        "policy": "mainline stable contact anchor propagation; untrusted/occluded frames propagate but do not update",
    }
    write_json(profile.result_dir / "anchor_state_metrics.json", metrics)
    return metrics
