#!/usr/bin/env python3
"""Build chair semantic 2D observations from SAM2 masks and CoTracker tracks.

This stage does not use whole-chair silhouette as the pose anchor. The mask is
used to find semantic part hypotheses, and CoTracker points provide stable
tracked samples for later propagation / robust fitting.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter


FIELDS = [
    "frame",
    "time",
    "mask_bbox_x1",
    "mask_bbox_y1",
    "mask_bbox_x2",
    "mask_bbox_y2",
    "top_rail_left_u",
    "top_rail_left_v",
    "top_rail_right_u",
    "top_rail_right_v",
    "top_rail_center_u",
    "top_rail_center_v",
    "backrest_hole_u",
    "backrest_hole_v",
    "backrest_hole_track_id",
    "seat_front_left_u",
    "seat_front_left_v",
    "seat_front_right_u",
    "seat_front_right_v",
    "seat_center_u",
    "seat_center_v",
    "front_left_foot_u",
    "front_left_foot_v",
    "front_right_foot_u",
    "front_right_foot_v",
    "support_center_u",
    "support_center_v",
    "floor_support_proxy_left_u",
    "floor_support_proxy_left_v",
    "floor_support_proxy_right_u",
    "floor_support_proxy_right_v",
    "floor_support_proxy_center_u",
    "floor_support_proxy_center_v",
    "top_rail_left_track_id",
    "top_rail_right_track_id",
    "seat_front_left_track_id",
    "seat_front_right_track_id",
    "front_left_foot_track_id",
    "front_right_foot_track_id",
    "front_leg_left_top_u",
    "front_leg_left_top_v",
    "front_leg_left_bottom_u",
    "front_leg_left_bottom_v",
    "front_leg_right_top_u",
    "front_leg_right_top_v",
    "front_leg_right_bottom_u",
    "front_leg_right_bottom_v",
    "rear_leg_left_top_u",
    "rear_leg_left_top_v",
    "rear_leg_left_bottom_u",
    "rear_leg_left_bottom_v",
    "rear_leg_right_top_u",
    "rear_leg_right_top_v",
    "rear_leg_right_bottom_u",
    "rear_leg_right_bottom_v",
    "front_leg_parallel_error_deg",
    "rear_leg_parallel_error_deg",
    "front_leg_pair_valid",
    "rear_leg_pair_valid",
    "leg_parallel_family_a_left_top_u",
    "leg_parallel_family_a_left_top_v",
    "leg_parallel_family_a_left_bottom_u",
    "leg_parallel_family_a_left_bottom_v",
    "leg_parallel_family_a_right_top_u",
    "leg_parallel_family_a_right_top_v",
    "leg_parallel_family_a_right_bottom_u",
    "leg_parallel_family_a_right_bottom_v",
    "leg_parallel_family_b_left_top_u",
    "leg_parallel_family_b_left_top_v",
    "leg_parallel_family_b_left_bottom_u",
    "leg_parallel_family_b_left_bottom_v",
    "leg_parallel_family_b_right_top_u",
    "leg_parallel_family_b_right_top_v",
    "leg_parallel_family_b_right_bottom_u",
    "leg_parallel_family_b_right_bottom_v",
    "leg_parallel_family_a_error_deg",
    "leg_parallel_family_b_error_deg",
    "leg_parallel_family_a_valid",
    "leg_parallel_family_b_valid",
    "semantic_conf",
    "source",
]


def read_tracks(path: Path) -> dict[int, list[dict[str, str]]]:
    if not path.exists():
        return {}
    out: dict[int, list[dict[str, str]]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out.setdefault(int(float(row["frame"])), []).append(row)
    return out


def finite_pt(u: float, v: float) -> bool:
    return math.isfinite(u) and math.isfinite(v)


def nearest_track(rows: list[dict[str, str]], u: float, v: float, max_dist: float = 32.0) -> tuple[float, float, str]:
    if not rows or not finite_pt(u, v):
        return u, v, ""
    best = None
    best_d = float("inf")
    for row in rows:
        try:
            vis = float(row.get("visible", "0") or 0)
            if vis < 0.35:
                continue
            x = float(row["x"])
            y = float(row["y"])
        except Exception:
            continue
        d = math.hypot(x - u, y - v)
        if d < best_d:
            best = row
            best_d = d
    if best is None or best_d > max_dist:
        return u, v, ""
    return float(best["x"]), float(best["y"]), best["point_id"]


def smooth_values(vals: np.ndarray, window: int = 7) -> np.ndarray:
    vals = vals.astype(float)
    ok = np.isfinite(vals)
    if not np.any(ok):
        return vals
    idx = np.arange(len(vals))
    filled = vals.copy()
    filled[~ok] = np.interp(idx[~ok], idx[ok], vals[ok])
    win = min(window, len(vals) if len(vals) % 2 else len(vals) - 1)
    if win >= 5:
        return savgol_filter(filled, win, 2, mode="interp")
    return filled


def span_in_band(xs: np.ndarray, ys: np.ndarray, y0: float, y1: float, min_pts: int = 12) -> tuple[np.ndarray, np.ndarray, float]:
    sel = (ys >= y0) & (ys <= y1)
    if int(sel.sum()) < min_pts:
        return np.array([math.nan, math.nan]), np.array([math.nan, math.nan]), 0.0
    px = xs[sel].astype(float)
    py = ys[sel].astype(float)
    # Reject tiny outliers by percentiles.
    lx = float(np.percentile(px, 4))
    rx = float(np.percentile(px, 96))
    # Use the median y in the band. For a thin rail/seat this is much less
    # jittery than min/max contour y.
    yy = float(np.median(py))
    return np.array([lx, yy]), np.array([rx, yy]), float(sel.sum())


def derive_semantics_from_mask(mask: np.ndarray) -> dict[str, np.ndarray | float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        nan = np.array([math.nan, math.nan])
        return {
            "bbox": np.array([math.nan, math.nan, math.nan, math.nan]),
            "top_l": nan,
            "top_r": nan,
            "seat_l": nan,
            "seat_r": nan,
            "foot_l": nan,
            "foot_r": nan,
            "conf": 0.0,
        }
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    h = max(1.0, y2 - y1)

    # Primary: upper backrest/top rail. Restrict to upper mask band; this keeps
    # the seat and lower stretchers from stealing the body anchor.
    top_l, top_r, top_n = span_in_band(xs, ys, y1, y1 + 0.20 * h)
    if not np.all(np.isfinite(top_l)):
        top_l, top_r, top_n = span_in_band(xs, ys, y1, y1 + 0.30 * h)

    # Secondary: seat front edge. It is usually the widest middle horizontal
    # part. Search a middle-lower band, not the bottom feet.
    seat_l, seat_r, seat_n = span_in_band(xs, ys, y1 + 0.34 * h, y1 + 0.62 * h)
    if not np.all(np.isfinite(seat_l)):
        seat_l, seat_r, seat_n = span_in_band(xs, ys, y1 + 0.28 * h, y1 + 0.72 * h)

    # Support: feet at lower mask extrema. These are weak anchors for lift/place,
    # not the main body anchor during carry.
    bottom_sel = ys >= y1 + 0.78 * h
    if int(bottom_sel.sum()) >= 8:
        bx = xs[bottom_sel].astype(float)
        by = ys[bottom_sel].astype(float)
        left_gate = np.percentile(bx, 28)
        right_gate = np.percentile(bx, 72)
        lsel = bx <= left_gate
        rsel = bx >= right_gate
        foot_l = np.array([float(np.median(bx[lsel])), float(np.percentile(by[lsel], 92))]) if np.any(lsel) else np.array([math.nan, math.nan])
        foot_r = np.array([float(np.median(bx[rsel])), float(np.percentile(by[rsel], 92))]) if np.any(rsel) else np.array([math.nan, math.nan])
    else:
        foot_l = np.array([math.nan, math.nan])
        foot_r = np.array([math.nan, math.nan])

    area = float(len(xs))
    conf = min(1.0, area / 12000.0)
    return {
        "bbox": np.array([x1, y1, x2, y2]),
        "top_l": top_l,
        "top_r": top_r,
        "seat_l": seat_l,
        "seat_r": seat_r,
        "foot_l": foot_l,
        "foot_r": foot_r,
        "conf": conf,
        "top_n": top_n,
        "seat_n": seat_n,
    }


def line_angle_deg(seg: np.ndarray) -> float:
    a, b = seg
    ang = math.degrees(math.atan2(float(b[1] - a[1]), float(b[0] - a[0])))
    while ang <= -90:
        ang += 180
    while ang > 90:
        ang -= 180
    return ang


def ordered_top_bottom(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    a = np.array([x1, y1], dtype=float)
    b = np.array([x2, y2], dtype=float)
    return np.stack([a, b]) if a[1] <= b[1] else np.stack([b, a])


def detect_leg_segments(
    mask: np.ndarray,
    image_bgr: np.ndarray | None,
    top_l: np.ndarray,
    top_r: np.ndarray,
    seat_l: np.ndarray,
    seat_r: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Find four long slanted leg/support segments inside the SAM2 mask.

    The chair legs are thin internal wooden bars, so mask-boundary Hough lines
    are not enough: they confuse seat edges and silhouette edges with legs. Use
    RGB edges inside the SAM2 mask first, then fall back to mask edges.
    """
    binary = (mask.astype(np.uint8) * 255)
    ys, xs = np.where(binary > 0)
    nan_seg = np.full((2, 2), math.nan, dtype=float)
    if len(xs) == 0:
        return {
            "front_left": nan_seg,
            "front_right": nan_seg,
            "rear_left": nan_seg,
            "rear_right": nan_seg,
            "front_parallel_error_deg": math.nan,
            "rear_parallel_error_deg": math.nan,
        }
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    h = max(1.0, y2 - y1)
    dilated_mask = cv2.dilate(binary, np.ones((11, 11), np.uint8), iterations=1)
    if image_bgr is not None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 40, 120)
        edges = cv2.bitwise_and(edges, edges, mask=dilated_mask)
    else:
        edges = cv2.Canny(binary, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=14, minLineLength=34, maxLineGap=10)
    cands: list[dict[str, object]] = []
    if lines is not None:
        for raw in lines[:, 0, :]:
            lx1, ly1, lx2, ly2 = [float(v) for v in raw]
            length = math.hypot(lx2 - lx1, ly2 - ly1)
            if length < 36:
                continue
            seg = ordered_top_bottom(lx1, ly1, lx2, ly2)
            angle = line_angle_deg(seg)
            if abs(angle) < 20 or abs(angle) > 89:
                continue
            mid = seg.mean(axis=0)
            if dilated_mask[int(np.clip(mid[1], 0, dilated_mask.shape[0] - 1)), int(np.clip(mid[0], 0, dilated_mask.shape[1] - 1))] == 0:
                continue
            if mid[1] < y1 + 0.15 * h or mid[1] > y2 + 0.98 * h:
                continue
            cands.append({"seg": seg, "angle": angle, "mid": mid, "length": length})

    y_ref = y1 + 0.55 * h

    def x_at_y(seg: np.ndarray, y: float) -> float:
        a, b = seg
        dy = float(b[1] - a[1])
        if abs(dy) < 1e-4:
            return float((a[0] + b[0]) * 0.5)
        t = (y - float(a[1])) / dy
        return float(a[0] + t * (float(b[0] - a[0])))

    def merge_cluster(cluster: list[dict[str, object]]) -> tuple[np.ndarray, float, float, float]:
        points = np.vstack([np.asarray(c["seg"], dtype=float) for c in cluster])
        center = points.mean(axis=0)
        centered = points - center
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        direction = vt[0]
        if direction[1] < 0:
            direction = -direction
        proj = centered @ direction
        top = center + direction * float(proj.min())
        bottom = center + direction * float(proj.max())
        seg = ordered_top_bottom(float(top[0]), float(top[1]), float(bottom[0]), float(bottom[1]))
        total_len = float(sum(float(c["length"]) for c in cluster))
        angle = line_angle_deg(seg)
        xref = x_at_y(seg, y_ref)
        score = total_len + 0.8 * seg_length(seg)
        return seg, angle, xref, score

    def cluster_candidates(items: list[dict[str, object]], min_len: float = 45.0) -> list[tuple[np.ndarray, float, float, float]]:
        fam = list(items)
        fam = sorted(fam, key=lambda c: float(c["length"]), reverse=True)
        clusters: list[list[dict[str, object]]] = []
        refs: list[float] = []
        angles: list[float] = []
        for c in fam:
            seg = np.asarray(c["seg"], dtype=float)
            ref = x_at_y(seg, y_ref)
            angle = float(c["angle"])
            hit = -1
            for i, (old_ref, old_angle) in enumerate(zip(refs, angles)):
                if abs(ref - old_ref) <= 30.0 and abs(angle - old_angle) <= 14.0:
                    hit = i
                    break
            if hit < 0:
                clusters.append([c])
                refs.append(ref)
                angles.append(angle)
            else:
                clusters[hit].append(c)
                refs[hit] = float(np.mean([x_at_y(np.asarray(cc["seg"], dtype=float), y_ref) for cc in clusters[hit]]))
                angles[hit] = float(np.mean([float(cc["angle"]) for cc in clusters[hit]]))
        merged = [merge_cluster(cluster) for cluster in clusters]
        merged = [m for m in merged if seg_length(m[0]) >= min_len]
        merged.sort(key=lambda x: x[3], reverse=True)
        picked: list[tuple[np.ndarray, float, float, float]] = []
        for item in merged:
            if all(abs(item[2] - old[2]) > 36.0 for old in picked):
                picked.append(item)
            if len(picked) >= 2:
                break
        picked.sort(key=lambda x: x[2])
        return picked

    def cluster_family(sign: int) -> list[tuple[np.ndarray, float, float, float]]:
        fam = [c for c in cands if (float(c["angle"]) < 0) == (sign < 0)]
        return cluster_candidates(fam, min_len=45.0)

    def role_family(role: str) -> list[tuple[np.ndarray, float, float, float]]:
        seat_y = float((seat_l[1] + seat_r[1]) * 0.5)
        if role == "front":
            min_len = max(85.0, 0.42 * h)

            def keep(c: dict[str, object]) -> bool:
                seg = np.asarray(c["seg"], dtype=float)
                angle = float(c["angle"])
                dy = float(seg[1, 1] - seg[0, 1])
                return (
                    float(c["length"]) >= min_len
                    and angle < -35.0
                    and abs(angle) <= 78.0
                    and dy >= 0.62 * float(c["length"])
                    and float(seg[0, 1]) <= seat_y - 12.0
                    and float(seg[1, 1]) >= seat_y + 15.0
                )

        elif role == "rear":
            min_len = max(70.0, 0.28 * h)

            def keep(c: dict[str, object]) -> bool:
                seg = np.asarray(c["seg"], dtype=float)
                angle = float(c["angle"])
                dy = float(seg[1, 1] - seg[0, 1])
                return (
                    float(c["length"]) >= min_len
                    and abs(angle) >= 72.0
                    and dy >= 0.82 * float(c["length"])
                    and float(seg[1, 1]) >= y1 + 0.33 * h
                )

        else:
            raise ValueError(role)
        return cluster_candidates([c for c in cands if keep(c)], min_len=min_len)

    def pick_family(sign: int) -> tuple[np.ndarray, np.ndarray, float]:
        picked = cluster_family(sign)
        if len(picked) < 2:
            return nan_seg, nan_seg, math.nan
        a = np.asarray(picked[0][0], dtype=float)
        b = np.asarray(picked[1][0], dtype=float)
        err = abs(float(picked[0][1]) - float(picked[1][1]))
        err = min(err, 180.0 - err)
        return a, b, err

    def pick_role(role: str) -> tuple[np.ndarray, np.ndarray, float]:
        picked = role_family(role)
        if len(picked) < 2:
            return nan_seg, nan_seg, math.nan
        a = np.asarray(picked[0][0], dtype=float)
        b = np.asarray(picked[1][0], dtype=float)
        err = abs(float(picked[0][1]) - float(picked[1][1]))
        err = min(err, 180.0 - err)
        return a, b, err

    # Semantic roles are not the same as raw slope sign in this video. The
    # rear support legs can appear with either sign and are often near-vertical;
    # using only the positive-slope family lets short stretcher bars become
    # "rear_leg". Keep raw slope-family outputs for diagnostics, but choose the
    # primary semantic legs by role-specific geometry.
    front_left, front_right, front_err = pick_role("front")
    rear_left, rear_right, rear_err = pick_role("rear")

    def parallel_err(a: np.ndarray, b: np.ndarray) -> float:
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            return math.nan
        err = abs(line_angle_deg(a) - line_angle_deg(b))
        return min(err, 180.0 - err)

    fam_a_left, fam_a_right, fam_a_err = pick_family(-1)
    fam_b_left, fam_b_right, fam_b_err = pick_family(1)
    return {
        "front_left": front_left,
        "front_right": front_right,
        "rear_left": rear_left,
        "rear_right": rear_right,
        "front_parallel_error_deg": front_err,
        "rear_parallel_error_deg": rear_err,
        "family_a_left": fam_a_left,
        "family_a_right": fam_a_right,
        "family_b_left": fam_b_left,
        "family_b_right": fam_b_right,
        "family_a_error_deg": fam_a_err,
        "family_b_error_deg": fam_b_err,
        "candidate_segments": cands,
    }


def snap_segment_to_tracks(seg: np.ndarray, tracks: list[dict[str, str]], max_dist: float = 34.0) -> np.ndarray:
    if not np.all(np.isfinite(seg)):
        return seg
    out = []
    for p in seg:
        u, v, _ = nearest_track(tracks, float(p[0]), float(p[1]), max_dist=max_dist)
        out.append([u, v])
    return np.asarray(out, dtype=float)


def seg_row(prefix: str, seg: np.ndarray) -> dict[str, str]:
    return {
        f"{prefix}_top_u": f"{float(seg[0, 0]):.3f}",
        f"{prefix}_top_v": f"{float(seg[0, 1]):.3f}",
        f"{prefix}_bottom_u": f"{float(seg[1, 0]):.3f}",
        f"{prefix}_bottom_v": f"{float(seg[1, 1]):.3f}",
    }


def seg_length(seg: np.ndarray) -> float:
    if not np.all(np.isfinite(seg)):
        return math.nan
    return float(np.linalg.norm(seg[1] - seg[0]))


def pair_valid(a: np.ndarray, b: np.ndarray, err_deg: float) -> bool:
    la = seg_length(a)
    lb = seg_length(b)
    dya = float(a[1, 1] - a[0, 1]) if np.all(np.isfinite(a)) else math.nan
    dyb = float(b[1, 1] - b[0, 1]) if np.all(np.isfinite(b)) else math.nan
    return (
        np.all(np.isfinite(a))
        and np.all(np.isfinite(b))
        and math.isfinite(err_deg)
        and la >= 70.0
        and lb >= 70.0
        and dya >= 0.58 * la
        and dyb >= 0.58 * lb
        and err_deg <= 18.0
    )


def x_at_y(seg: np.ndarray, y: float) -> float:
    a, b = seg
    dy = float(b[1] - a[1])
    if abs(dy) < 1e-4:
        return float((a[0] + b[0]) * 0.5)
    t = (y - float(a[1])) / dy
    return float(a[0] + t * (float(b[0] - a[0])))


def line_direction(seg: np.ndarray) -> np.ndarray:
    d = np.asarray(seg[1] - seg[0], dtype=float)
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        return np.array([math.nan, math.nan], dtype=float)
    if d[1] < 0:
        d = -d
    return d / n


def extend_candidate_from_top(top: np.ndarray, cand_seg: np.ndarray, min_bottom_y: float) -> np.ndarray:
    d = line_direction(cand_seg)
    if not np.all(np.isfinite(d)):
        return np.full((2, 2), math.nan, dtype=float)
    bottom = np.asarray(cand_seg[1], dtype=float)
    if float(bottom[1]) < min_bottom_y:
        dy = max(70.0, min_bottom_y - float(top[1]))
        bottom = top + d * (dy / max(float(d[1]), 1e-4))
    return np.stack([top, bottom])


def top_anchored_front_pair(
    candidates: list[dict[str, float]],
    top_l: np.ndarray,
    top_r: np.ndarray,
    seat_l: np.ndarray,
    seat_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    if not all(np.all(np.isfinite(p)) for p in (top_l, top_r, seat_l, seat_r)):
        return None
    seat_y = float((seat_l[1] + seat_r[1]) * 0.5)
    usable = []
    for cand in candidates:
        if cand.get("family") != "front_leg_candidate":
            continue
        seg = np.array([[cand["top_u"], cand["top_v"]], [cand["bottom_u"], cand["bottom_v"]]], dtype=float)
        angle = float(cand["angle_deg"])
        length = float(cand["length_px"])
        dy = float(seg[1, 1] - seg[0, 1])
        if length < 60.0 or angle >= -30.0 or abs(angle) > 82.0 or dy < 0.50 * length:
            continue
        if float(seg[1, 1]) < seat_y + 20.0:
            continue
        usable.append(seg)
    if len(usable) < 2:
        return None

    def score(seg: np.ndarray, top: np.ndarray, side: str) -> float:
        xt = x_at_y(seg, float(top[1]))
        d = line_direction(seg)
        if not np.all(np.isfinite(d)):
            return float("inf")
        projected_bottom = extend_candidate_from_top(top, seg, seat_y + 35.0)[1]
        # The right front leg often has its observed Hough segment starting
        # below the top rail. Extrapolating to the top rail is intentional.
        lateral_penalty = abs(float(xt - top[0]))
        vertical_penalty = max(0.0, float(top[1] - seg[0, 1]) - 90.0)
        lower_penalty = max(0.0, seat_y + 25.0 - float(projected_bottom[1]))
        side_penalty = 0.0
        if side == "left" and projected_bottom[0] > top[0] + 12.0:
            side_penalty += 40.0
        if side == "right" and projected_bottom[0] > top[0] + 5.0:
            side_penalty += 30.0
        return lateral_penalty + 0.35 * vertical_penalty + 0.8 * lower_penalty + side_penalty

    best = None
    best_score = float("inf")
    for i, left_seg in enumerate(usable):
        for j, right_seg in enumerate(usable):
            if i == j:
                continue
            left = extend_candidate_from_top(top_l, left_seg, seat_y + 35.0)
            right = extend_candidate_from_top(top_r, right_seg, seat_y + 35.0)
            if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
                continue
            if left[1, 0] >= right[1, 0] - 18.0:
                continue
            err = abs(line_angle_deg(left) - line_angle_deg(right))
            err = min(err, 180.0 - err)
            if err > 22.0:
                continue
            s = score(left_seg, top_l, "left") + score(right_seg, top_r, "right") + 1.2 * err
            if s < best_score:
                best = (left, right, err)
                best_score = s
    return best


def build(
    sample: Path,
    fps: float,
    tracks_csv: Path | None,
    smooth: bool,
    out_dir: Path | None = None,
    top_anchored_front: bool = False,
    write_previews: bool = False,
) -> tuple[Path, Path]:
    masks_dir = sample / "results" / "segmentation" / "masks"
    track_rows = read_tracks(tracks_csv or (sample / "results" / "tracking" / "object_mesh_tracks_test.csv"))
    rows: list[dict[str, str]] = []
    candidate_rows: list[dict[str, str]] = []
    candidates_by_frame: dict[int, list[dict[str, float]]] = {}
    nan_seg = np.full((2, 2), math.nan, dtype=float)
    for mask_path in sorted(masks_dir.glob("*_mask.png")):
        fr = int(mask_path.name.split("_")[0])
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        image_bgr = cv2.imread(str(sample / "frames" / f"{fr:05d}.png"))
        sem = derive_semantics_from_mask(mask > 0)
        bbox = sem["bbox"]
        top_l = np.asarray(sem["top_l"], dtype=float)
        top_r = np.asarray(sem["top_r"], dtype=float)
        seat_l = np.asarray(sem["seat_l"], dtype=float)
        seat_r = np.asarray(sem["seat_r"], dtype=float)
        foot_l = np.asarray(sem["foot_l"], dtype=float)
        foot_r = np.asarray(sem["foot_r"], dtype=float)
        legs = detect_leg_segments(mask > 0, image_bgr, top_l, top_r, seat_l, seat_r)
        raw_candidates = sorted(
            list(legs.get("candidate_segments", [])),
            key=lambda c: float(c.get("length", 0.0)),
            reverse=True,
        )
        candidates_by_frame[fr] = []
        for cid, cand in enumerate(raw_candidates):
            seg = np.asarray(cand["seg"], dtype=float)
            angle = float(cand["angle"])
            length = float(cand["length"])
            dy = float(seg[1, 1] - seg[0, 1])
            if angle < -35.0 and abs(angle) <= 78.0 and dy >= 0.62 * length:
                family = "front_leg_candidate"
            elif abs(angle) >= 72.0 and dy >= 0.82 * length:
                family = "rear_leg_candidate"
            else:
                family = "support_or_stretcher_candidate"
            candidate = {
                "frame": fr,
                "candidate_id": cid,
                "family": family,
                "angle_deg": angle,
                "length_px": length,
                "top_u": float(seg[0, 0]),
                "top_v": float(seg[0, 1]),
                "bottom_u": float(seg[1, 0]),
                "bottom_v": float(seg[1, 1]),
            }
            candidates_by_frame[fr].append(candidate)
            candidate_rows.append({
                "frame": str(fr),
                "time": f"{(fr - 1) / fps:.6f}",
                "candidate_id": str(cid),
                "family": family,
                "angle_deg": f"{angle:.3f}",
                "length_px": f"{length:.3f}",
                "top_u": f"{float(seg[0, 0]):.3f}",
                "top_v": f"{float(seg[0, 1]):.3f}",
                "bottom_u": f"{float(seg[1, 0]):.3f}",
                "bottom_v": f"{float(seg[1, 1]):.3f}",
                "source": "rgb_edges_inside_sam2_mask",
            })

        tracks = track_rows.get(fr, [])
        top_l_u, top_l_v, top_l_id = nearest_track(tracks, float(top_l[0]), float(top_l[1]))
        top_r_u, top_r_v, top_r_id = nearest_track(tracks, float(top_r[0]), float(top_r[1]))
        seat_l_u, seat_l_v, seat_l_id = nearest_track(tracks, float(seat_l[0]), float(seat_l[1]))
        seat_r_u, seat_r_v, seat_r_id = nearest_track(tracks, float(seat_r[0]), float(seat_r[1]))
        foot_l_u, foot_l_v, foot_l_id = nearest_track(tracks, float(foot_l[0]), float(foot_l[1]), max_dist=45.0)
        foot_r_u, foot_r_v, foot_r_id = nearest_track(tracks, float(foot_r[0]), float(foot_r[1]), max_dist=45.0)
        front_leg_left = snap_segment_to_tracks(np.asarray(legs["front_left"], dtype=float), tracks)
        front_leg_right = snap_segment_to_tracks(np.asarray(legs["front_right"], dtype=float), tracks)
        front_err_deg = float(legs["front_parallel_error_deg"])
        anchored_front = None
        if top_anchored_front:
            anchored_front = top_anchored_front_pair(
                candidates_by_frame[fr],
                np.array([top_l_u, top_l_v], dtype=float),
                np.array([top_r_u, top_r_v], dtype=float),
                np.array([seat_l_u, seat_l_v], dtype=float),
                np.array([seat_r_u, seat_r_v], dtype=float),
            )
        if anchored_front is not None:
            front_leg_left = snap_segment_to_tracks(anchored_front[0], tracks)
            front_leg_right = snap_segment_to_tracks(anchored_front[1], tracks)
            front_err_deg = float(anchored_front[2])
        rear_leg_left = snap_segment_to_tracks(np.asarray(legs["rear_left"], dtype=float), tracks)
        rear_leg_right = snap_segment_to_tracks(np.asarray(legs["rear_right"], dtype=float), tracks)
        family_a_left = snap_segment_to_tracks(np.asarray(legs["family_a_left"], dtype=float), tracks)
        family_a_right = snap_segment_to_tracks(np.asarray(legs["family_a_right"], dtype=float), tracks)
        family_b_left = snap_segment_to_tracks(np.asarray(legs["family_b_left"], dtype=float), tracks)
        family_b_right = snap_segment_to_tracks(np.asarray(legs["family_b_right"], dtype=float), tracks)
        front_leg_valid = pair_valid(front_leg_left, front_leg_right, front_err_deg)
        rear_leg_valid = pair_valid(rear_leg_left, rear_leg_right, float(legs["rear_parallel_error_deg"]))
        family_a_valid = pair_valid(family_a_left, family_a_right, float(legs["family_a_error_deg"]))
        family_b_valid = pair_valid(family_b_left, family_b_right, float(legs["family_b_error_deg"]))
        if not front_leg_valid:
            front_leg_left = nan_seg.copy()
            front_leg_right = nan_seg.copy()
        if not rear_leg_valid:
            rear_leg_left = nan_seg.copy()
            rear_leg_right = nan_seg.copy()

        top_c = np.array([(top_l_u + top_r_u) * 0.5, (top_l_v + top_r_v) * 0.5])
        seat_c = np.array([(seat_l_u + seat_r_u) * 0.5, (seat_l_v + seat_r_v) * 0.5])
        support_c = np.array([(foot_l_u + foot_r_u) * 0.5, (foot_l_v + foot_r_v) * 0.5])
        row = {
            "frame": str(fr),
            "time": f"{(fr - 1) / fps:.6f}",
            "mask_bbox_x1": f"{bbox[0]:.3f}",
            "mask_bbox_y1": f"{bbox[1]:.3f}",
            "mask_bbox_x2": f"{bbox[2]:.3f}",
            "mask_bbox_y2": f"{bbox[3]:.3f}",
            "top_rail_left_u": f"{top_l_u:.3f}",
            "top_rail_left_v": f"{top_l_v:.3f}",
            "top_rail_right_u": f"{top_r_u:.3f}",
            "top_rail_right_v": f"{top_r_v:.3f}",
            "top_rail_center_u": f"{top_c[0]:.3f}",
            "top_rail_center_v": f"{top_c[1]:.3f}",
            "seat_front_left_u": f"{seat_l_u:.3f}",
            "seat_front_left_v": f"{seat_l_v:.3f}",
            "seat_front_right_u": f"{seat_r_u:.3f}",
            "seat_front_right_v": f"{seat_r_v:.3f}",
            "seat_center_u": f"{seat_c[0]:.3f}",
            "seat_center_v": f"{seat_c[1]:.3f}",
            "front_left_foot_u": f"{foot_l_u:.3f}",
            "front_left_foot_v": f"{foot_l_v:.3f}",
            "front_right_foot_u": f"{foot_r_u:.3f}",
            "front_right_foot_v": f"{foot_r_v:.3f}",
            "support_center_u": f"{support_c[0]:.3f}",
            "support_center_v": f"{support_c[1]:.3f}",
            "floor_support_proxy_left_u": f"{foot_l_u:.3f}",
            "floor_support_proxy_left_v": f"{foot_l_v:.3f}",
            "floor_support_proxy_right_u": f"{foot_r_u:.3f}",
            "floor_support_proxy_right_v": f"{foot_r_v:.3f}",
            "floor_support_proxy_center_u": f"{support_c[0]:.3f}",
            "floor_support_proxy_center_v": f"{support_c[1]:.3f}",
            "top_rail_left_track_id": top_l_id,
            "top_rail_right_track_id": top_r_id,
            "seat_front_left_track_id": seat_l_id,
            "seat_front_right_track_id": seat_r_id,
            "front_left_foot_track_id": foot_l_id,
            "front_right_foot_track_id": foot_r_id,
            **seg_row("front_leg_left", front_leg_left),
            **seg_row("front_leg_right", front_leg_right),
            **seg_row("rear_leg_left", rear_leg_left),
            **seg_row("rear_leg_right", rear_leg_right),
            "front_leg_parallel_error_deg": f"{front_err_deg:.3f}",
            "rear_leg_parallel_error_deg": f"{float(legs['rear_parallel_error_deg']):.3f}",
            "front_leg_pair_valid": "1" if front_leg_valid else "0",
            "rear_leg_pair_valid": "1" if rear_leg_valid else "0",
            **seg_row("leg_parallel_family_a_left", family_a_left),
            **seg_row("leg_parallel_family_a_right", family_a_right),
            **seg_row("leg_parallel_family_b_left", family_b_left),
            **seg_row("leg_parallel_family_b_right", family_b_right),
            "leg_parallel_family_a_error_deg": f"{float(legs['family_a_error_deg']):.3f}",
            "leg_parallel_family_b_error_deg": f"{float(legs['family_b_error_deg']):.3f}",
            "leg_parallel_family_a_valid": "1" if family_a_valid else "0",
            "leg_parallel_family_b_valid": "1" if family_b_valid else "0",
            "semantic_conf": f"{float(sem['conf']):.6f}",
            "source": "sam2_mask_semantic_bands_cotracker_snap_leg_parallelograms_topanchored_front" if top_anchored_front else "sam2_mask_semantic_bands_cotracker_snap_leg_parallelograms",
        }
        rows.append(row)

    if smooth and rows:
        for key in [
            "top_rail_left_u", "top_rail_left_v", "top_rail_right_u", "top_rail_right_v",
            "top_rail_center_u", "top_rail_center_v", "seat_front_left_u", "seat_front_left_v",
            "seat_front_right_u", "seat_front_right_v", "seat_center_u", "seat_center_v",
        ]:
            vals = np.array([float(r[key]) for r in rows])
            sm = smooth_values(vals, window=5)
            for i, v in enumerate(sm):
                rows[i][key] = f"{float(v):.3f}"

    out_dir = out_dir or (sample / "results" / "chair_semantic_observations")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "chair_semantic_observations.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    candidate_csv_path = out_dir / "chair_leg_candidates.csv"
    with candidate_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "time",
                "candidate_id",
                "family",
                "angle_deg",
                "length_px",
                "top_u",
                "top_v",
                "bottom_u",
                "bottom_v",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(candidate_rows)

    preview_dir = out_dir / "previews"
    parallel_preview_dir = out_dir / "previews_parallel"
    candidate_preview_dir = out_dir / "previews_leg_candidates"
    if write_previews:
        preview_dir.mkdir(parents=True, exist_ok=True)
        parallel_preview_dir.mkdir(parents=True, exist_ok=True)
        candidate_preview_dir.mkdir(parents=True, exist_ok=True)
        by_frame = {int(r["frame"]): r for r in rows}
        for fr in [1, 50, 100, 140, 180, 192]:
            r = by_frame.get(fr)
            img = cv2.imread(str(sample / "frames" / f"{fr:05d}.png"))
            if r is None or img is None:
                continue

            def draw_items(canvas: np.ndarray, items: list[tuple[str, str, str, str, tuple[int, int, int], str, str]]) -> None:
                cv2.rectangle(canvas, (int(float(r["mask_bbox_x1"])), int(float(r["mask_bbox_y1"]))), (int(float(r["mask_bbox_x2"])), int(float(r["mask_bbox_y2"]))), (0, 200, 0), 2)
                for au, av, bu, bv, color, name, gate in items:
                    if gate and r.get(gate, "0") != "1":
                        continue
                    vals = [float(r[au]), float(r[av]), float(r[bu]), float(r[bv])]
                    if all(math.isfinite(x) for x in vals):
                        a = (int(vals[0]), int(vals[1]))
                        b = (int(vals[2]), int(vals[3]))
                        cv2.line(canvas, a, b, color, 3, cv2.LINE_AA)
                        cv2.circle(canvas, a, 5, color, -1, cv2.LINE_AA)
                        cv2.circle(canvas, b, 5, color, -1, cv2.LINE_AA)
                        cv2.putText(canvas, name, (a[0] + 4, a[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            semantic_items = [
                ("top_rail_left_u", "top_rail_left_v", "top_rail_right_u", "top_rail_right_v", (0, 0, 255), "top_rail", ""),
                ("seat_front_left_u", "seat_front_left_v", "seat_front_right_u", "seat_front_right_v", (255, 0, 255), "seat_front", ""),
                ("front_leg_left_top_u", "front_leg_left_top_v", "front_leg_left_bottom_u", "front_leg_left_bottom_v", (0, 170, 255), "front_leg", "front_leg_pair_valid"),
                ("front_leg_right_top_u", "front_leg_right_top_v", "front_leg_right_bottom_u", "front_leg_right_bottom_v", (0, 170, 255), "front_leg", "front_leg_pair_valid"),
                ("rear_leg_left_top_u", "rear_leg_left_top_v", "rear_leg_left_bottom_u", "rear_leg_left_bottom_v", (255, 180, 0), "rear_leg", "rear_leg_pair_valid"),
                ("rear_leg_right_top_u", "rear_leg_right_top_v", "rear_leg_right_bottom_u", "rear_leg_right_bottom_v", (255, 180, 0), "rear_leg", "rear_leg_pair_valid"),
            ]
            parallel_items = [
                ("top_rail_left_u", "top_rail_left_v", "top_rail_right_u", "top_rail_right_v", (0, 0, 255), "top_rail", ""),
                ("seat_front_left_u", "seat_front_left_v", "seat_front_right_u", "seat_front_right_v", (255, 0, 255), "seat_front", ""),
                ("leg_parallel_family_a_left_top_u", "leg_parallel_family_a_left_top_v", "leg_parallel_family_a_left_bottom_u", "leg_parallel_family_a_left_bottom_v", (0, 255, 180), "para_A", "leg_parallel_family_a_valid"),
                ("leg_parallel_family_a_right_top_u", "leg_parallel_family_a_right_top_v", "leg_parallel_family_a_right_bottom_u", "leg_parallel_family_a_right_bottom_v", (0, 255, 180), "para_A", "leg_parallel_family_a_valid"),
                ("leg_parallel_family_b_left_top_u", "leg_parallel_family_b_left_top_v", "leg_parallel_family_b_left_bottom_u", "leg_parallel_family_b_left_bottom_v", (180, 255, 0), "para_B", "leg_parallel_family_b_valid"),
                ("leg_parallel_family_b_right_top_u", "leg_parallel_family_b_right_top_v", "leg_parallel_family_b_right_bottom_u", "leg_parallel_family_b_right_bottom_v", (180, 255, 0), "para_B", "leg_parallel_family_b_valid"),
            ]
            semantic_canvas = img.copy()
            parallel_canvas = img.copy()
            candidate_canvas = img.copy()
            draw_items(semantic_canvas, semantic_items)
            draw_items(parallel_canvas, parallel_items)
            cv2.rectangle(candidate_canvas, (int(float(r["mask_bbox_x1"])), int(float(r["mask_bbox_y1"]))), (int(float(r["mask_bbox_x2"])), int(float(r["mask_bbox_y2"]))), (0, 200, 0), 2)
            for cand in candidates_by_frame.get(fr, [])[:28]:
                color = (0, 165, 255) if float(cand["angle_deg"]) < 0 else (255, 120, 0)
                a = (int(float(cand["top_u"])), int(float(cand["top_v"])))
                b = (int(float(cand["bottom_u"])), int(float(cand["bottom_v"])))
                cv2.line(candidate_canvas, a, b, color, 3, cv2.LINE_AA)
                cv2.putText(
                    candidate_canvas,
                    f"{int(cand['candidate_id'])}:{float(cand['angle_deg']):.0f}",
                    (a[0] + 3, a[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            cv2.imwrite(str(preview_dir / f"chair_semantic_{fr:03d}.png"), semantic_canvas)
            cv2.imwrite(str(parallel_preview_dir / f"chair_parallel_{fr:03d}.png"), parallel_canvas)
            cv2.imwrite(str(candidate_preview_dir / f"chair_leg_candidates_{fr:03d}.png"), candidate_canvas)

    valid_stats = {}
    if rows:
        for key in ["front_leg_pair_valid", "rear_leg_pair_valid", "leg_parallel_family_a_valid", "leg_parallel_family_b_valid"]:
            valid_stats[key] = sum(1 for row in rows if row.get(key) == "1")
    summary = {
        "csv": str(csv_path),
        "candidate_csv": str(candidate_csv_path),
        "rows": len(rows),
        "candidate_rows": len(candidate_rows),
        "preview_dir": str(preview_dir) if write_previews else "",
        "parallel_preview_dir": str(parallel_preview_dir) if write_previews else "",
        "candidate_preview_dir": str(candidate_preview_dir) if write_previews else "",
        "write_previews": write_previews,
        "valid_stats": valid_stats,
        "note": "support_center/floor_support_proxy are lift-place/static floor proxies, not chair body anchors. 6D pose should use top_rail, seat_front, RGB leg candidates, and valid leg_parallel_family_a/b constraints with robust loss.",
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return csv_path, preview_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-dir", type=Path, default=Path("samples_known_object/05_chair"))
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--tracks-csv", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--top-anchored-front", action="store_true", help="Experimental: force front-leg top endpoints to top rail endpoints. Off by default.")
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--write-previews", action="store_true")
    args = ap.parse_args()
    build(args.sample_dir, args.fps, args.tracks_csv, smooth=not args.no_smooth, out_dir=args.out_dir, top_anchored_front=args.top_anchored_front, write_previews=args.write_previews)


if __name__ == "__main__":
    main()
