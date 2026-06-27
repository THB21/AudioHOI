#!/usr/bin/env python3
"""Build chunk-local CoTracker front-frame quadrilateral observations."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


FIELDS = [
    "frame",
    "time",
    "source_chunk",
    "query_frame",
    "seed_frame",
    "front_frame_track_conf",
    "front_frame_lt_track_id",
    "front_frame_lb_track_id",
    "front_frame_rt_track_id",
    "front_frame_rb_track_id",
    "front_frame_lt_u",
    "front_frame_lt_v",
    "front_frame_lb_u",
    "front_frame_lb_v",
    "front_frame_rt_u",
    "front_frame_rt_v",
    "front_frame_rb_u",
    "front_frame_rb_v",
    "source",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def finite_pair(row: dict[str, str], prefix: str) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        a = np.array([float(row[f"{prefix}_top_u"]), float(row[f"{prefix}_top_v"])], dtype=float)
        b = np.array([float(row[f"{prefix}_bottom_u"]), float(row[f"{prefix}_bottom_v"])], dtype=float)
    except Exception:
        return None
    if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
        return a, b
    return None


def finite_point(row: dict[str, str], prefix: str) -> np.ndarray | None:
    try:
        p = np.array([float(row[f"{prefix}_u"]), float(row[f"{prefix}_v"])], dtype=float)
    except Exception:
        return None
    if np.all(np.isfinite(p)):
        return p
    return None


def seed_front_frame_points(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Use semantic top rail as the front-frame top edge and tracked feet/leg bottoms as bottom edge."""
    lt = finite_point(row, "top_rail_left")
    rt = finite_point(row, "top_rail_right")
    if lt is None or rt is None:
        return None
    left_leg = finite_pair(row, "front_leg_left")
    right_leg = finite_pair(row, "front_leg_right")
    if left_leg is not None and right_leg is not None:
        lb = left_leg[1]
        rb = right_leg[1]
    else:
        lb = finite_point(row, "front_left_foot")
        rb = finite_point(row, "front_right_foot")
    if lb is None or rb is None:
        return None
    side_len = 0.5 * (np.linalg.norm(lb - lt) + np.linalg.norm(rb - rt))
    top_len = np.linalg.norm(rt - lt)
    bottom_len = np.linalg.norm(rb - lb)
    if side_len < 60.0 or top_len < 25.0 or bottom_len < 25.0:
        return None
    return lt, lb, rt, rb


def nearest_track(rows: list[dict[str, str]], pt: np.ndarray, max_dist: float = 42.0) -> tuple[str, float] | None:
    best_id = ""
    best_d = float("inf")
    for r in rows:
        try:
            if float(r.get("visible", "0") or 0) < 0.35:
                continue
            xy = np.array([float(r["x"]), float(r["y"])], dtype=float)
        except Exception:
            continue
        d = float(np.linalg.norm(xy - pt))
        if d < best_d:
            best_id = r["point_id"]
            best_d = d
    if best_id and best_d <= max_dist:
        return best_id, best_d
    return None


def chunk_key(row: dict[str, str]) -> tuple[str, str]:
    return row.get("source_chunk", ""), row.get("query_frame", "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-dir", type=Path, default=Path("samples_known_object/05_chair"))
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--tracks-csv", type=Path, default=None)
    ap.add_argument("--semantic-csv", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--write-previews", action="store_true")
    args = ap.parse_args()

    sample = args.sample_dir
    tracks_path = args.tracks_csv or (sample / "results/tracking/object_mesh_tracks_test.csv")
    semantic_path = args.semantic_csv or (sample / "results/chair_semantic_observations/chair_semantic_observations.csv")
    out_dir = args.out_dir or (sample / "results/chair_semantic_observations")
    out_dir.mkdir(parents=True, exist_ok=True)

    track_rows = read_rows(tracks_path)
    sem_rows = {int(float(r["frame"])): r for r in read_rows(semantic_path)}
    tracks_by_frame: dict[int, list[dict[str, str]]] = {}
    chunk_frames: dict[tuple[str, str], list[int]] = {}
    for r in track_rows:
        fr = int(float(r["frame"]))
        tracks_by_frame.setdefault(fr, []).append(r)
        chunk_frames.setdefault(chunk_key(r), []).append(fr)
    chunk_ranges = {k: (min(v), max(v)) for k, v in chunk_frames.items()}

    rows_out: list[dict[str, str]] = []
    seed_info: dict[str, dict[str, object]] = {}
    for key, (start, end) in sorted(chunk_ranges.items(), key=lambda kv: kv[1][0]):
        valid = []
        for fr in range(start, end + 1):
            row = sem_rows.get(fr)
            if not row:
                continue
            seed_pts = seed_front_frame_points(row)
            if seed_pts is None:
                continue
            lt, lb, rt, rb = seed_pts
            side_len = 0.5 * (np.linalg.norm(lb - lt) + np.linalg.norm(rb - rt))
            top_len = np.linalg.norm(rt - lt)
            front_bonus = 1.0 if row.get("front_leg_pair_valid") == "1" else 0.0
            valid.append((fr, front_bonus, side_len, top_len, lt, lb, rt, rb))
        if not valid:
            continue
        center = (start + end) * 0.5
        seed = min(valid, key=lambda item: (abs(item[0] - center), -item[1], -item[2]))
        seed_frame, _, _, _, lt, lb, rt, rb = seed
        seed_tracks = tracks_by_frame.get(seed_frame, [])
        bound = {
            "lt": nearest_track(seed_tracks, lt),
            "lb": nearest_track(seed_tracks, lb),
            "rt": nearest_track(seed_tracks, rt),
            "rb": nearest_track(seed_tracks, rb),
        }
        if any(v is None for v in bound.values()):
            continue
        ids = {name: val[0] for name, val in bound.items() if val is not None}
        max_bind_dist = max(float(val[1]) for val in bound.values() if val is not None)
        seed_info[f"{key[0]}:{key[1]}"] = {
            "frame_range": [start, end],
            "seed_frame": seed_frame,
            "track_ids": ids,
            "max_bind_dist": max_bind_dist,
        }
        for fr in range(start, end + 1):
            by_id = {r["point_id"]: r for r in tracks_by_frame.get(fr, [])}
            pts: dict[str, np.ndarray] = {}
            vis_vals = []
            ok = True
            for name, pid in ids.items():
                tr = by_id.get(pid)
                if tr is None:
                    ok = False
                    break
                vis = float(tr.get("visible", "0") or 0)
                if vis < 0.35:
                    ok = False
                    break
                pts[name] = np.array([float(tr["x"]), float(tr["y"])], dtype=float)
                vis_vals.append(vis)
            if not ok:
                continue
            side_len = 0.5 * (np.linalg.norm(pts["lb"] - pts["lt"]) + np.linalg.norm(pts["rb"] - pts["rt"]))
            top_len = np.linalg.norm(pts["rt"] - pts["lt"])
            bottom_len = np.linalg.norm(pts["rb"] - pts["lb"])
            if side_len < 45.0 or top_len < 18.0 or bottom_len < 18.0:
                continue
            conf = min(vis_vals) * float(np.clip(side_len / 130.0, 0.35, 1.0))
            rows_out.append(
                {
                    "frame": str(fr),
                    "time": f"{(fr - 1) / args.fps:.6f}",
                    "source_chunk": key[0],
                    "query_frame": key[1],
                    "seed_frame": str(seed_frame),
                    "front_frame_track_conf": f"{conf:.6f}",
                    "front_frame_lt_track_id": ids["lt"],
                    "front_frame_lb_track_id": ids["lb"],
                    "front_frame_rt_track_id": ids["rt"],
                    "front_frame_rb_track_id": ids["rb"],
                    "front_frame_lt_u": f"{pts['lt'][0]:.3f}",
                    "front_frame_lt_v": f"{pts['lt'][1]:.3f}",
                    "front_frame_lb_u": f"{pts['lb'][0]:.3f}",
                    "front_frame_lb_v": f"{pts['lb'][1]:.3f}",
                    "front_frame_rt_u": f"{pts['rt'][0]:.3f}",
                    "front_frame_rt_v": f"{pts['rt'][1]:.3f}",
                    "front_frame_rb_u": f"{pts['rb'][0]:.3f}",
                    "front_frame_rb_v": f"{pts['rb'][1]:.3f}",
                    "source": "chunk_local_cotracker_front_frame_seed",
                }
            )

    csv_path = out_dir / "chair_front_frame_cotracker_observations.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)

    preview_dir = out_dir / "previews_front_frame_cotracker"
    if args.write_previews:
        preview_dir.mkdir(parents=True, exist_ok=True)
        by_frame = {int(r["frame"]): r for r in rows_out}
        for fr in [50, 86, 100, 140, 145, 180]:
            img = cv2.imread(str(sample / "frames" / f"{fr:05d}.png"))
            r = by_frame.get(fr)
            if img is None or r is None:
                continue
            pts = []
            for name in ["lt", "lb", "rb", "rt"]:
                pts.append([float(r[f"front_frame_{name}_u"]), float(r[f"front_frame_{name}_v"])])
            poly = np.round(np.asarray(pts)).astype(np.int32)
            cv2.polylines(img, [poly], True, (0, 220, 255), 4, cv2.LINE_AA)
            for p, label in zip(poly, ["lt", "lb", "rb", "rt"]):
                cv2.circle(img, tuple(p), 6, (0, 220, 255), -1, cv2.LINE_AA)
                cv2.putText(img, label, tuple(p + np.array([4, -4])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, label, tuple(p + np.array([4, -4])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"front frame cotracker conf={float(r['front_frame_track_conf']):.2f} seed={r['seed_frame']}", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, f"front frame cotracker conf={float(r['front_frame_track_conf']):.2f} seed={r['seed_frame']}", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(preview_dir / f"front_frame_cotracker_{fr:03d}.png"), img)

    summary = {
        "csv": str(csv_path),
        "rows": len(rows_out),
        "seed_info": seed_info,
        "preview_dir": str(preview_dir) if args.write_previews else "",
        "write_previews": args.write_previews,
            "note": "Chunk-local CoTracker front-frame quadrilateral observations. Each chunk binds top_rail_left/right plus front bottom support points to four track IDs and propagates that quadrilateral inside the chunk.",
    }
    (out_dir / "front_frame_cotracker_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
