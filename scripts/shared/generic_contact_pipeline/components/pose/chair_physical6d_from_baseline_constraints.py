#!/usr/bin/env python3
"""Fit a physical camera-projected chair pose from stable baseline 2D cues.

This is the no-cheating version for the chair wire: no free projective q and no
editing of Articraft local geometry. The target Articraft-ratio segments are
projected with the same rigid/articulated camera model used by the stage3 6D
pipeline. Baseline 2D overlay is used only as the observation source because
its leg lines/top rail are visually stable.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[5]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.components.render import fit_chair_metric_6d_pose as fit


COLORS = {
    "orange_front_leg": (0, 165, 255),
    "blue_rear_leg": (255, 160, 0),
    "top_rail_observation": (0, 0, 255),
    "board_bottom_edge": (0, 90, 255),
    "seat_front_observation": (255, 0, 255),
    "seat_hinge_observation": (255, 0, 255),
    "seat_depth_line": (180, 120, 255),
    "front_floor_support": (0, 220, 220),
    "rear_floor_support": (0, 190, 170),
    "front_lower_stretcher": (0, 180, 220),
    "rear_lower_stretcher": (0, 150, 190),
    "side_lower_stretcher": (0, 210, 220),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows


LEG_SIDS = ("front_leg_left", "front_leg_right", "rear_leg_left", "rear_leg_right")
SIDE_SIDS = (
    ("left_side_lower_stretcher", "front_leg_left", "rear_leg_left"),
    ("right_side_lower_stretcher", "front_leg_right", "rear_leg_right"),
)
RAIL_SIDS = ("backrest_top_edge", "backrest_bottom_edge", "seat_front_edge")
OUT_FIELDS = [
    "frame",
    "time",
    "tx",
    "ty",
    "tz",
    "qx",
    "qy",
    "qz",
    "qw",
    "rx_delta",
    "ry_delta",
    "rz_delta",
    "rear_joint_angle",
    "seat_joint_angle",
    "fx",
    "fy",
    "cx",
    "cy",
    "line_rmse_px",
    "endpoint_rmse_px",
    "side_ratio_rmse_px",
    "rail_rmse_px",
    "source",
]


def ff(row: dict[str, str], key: str, default: float = math.nan) -> float:
    val = row.get(key, "")
    if val in ("", None):
        return default
    return float(val)


def pose_params(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [
            ff(row, "rx_delta", 0.0),
            ff(row, "ry_delta", 0.0),
            ff(row, "rz_delta", 0.0),
            ff(row, "tx", 0.0),
            ff(row, "ty", 0.0),
            ff(row, "tz", 3.5),
            ff(row, "rear_joint_angle", 0.0),
            ff(row, "seat_joint_angle", 0.0),
        ],
        dtype=float,
    )


def intrinsics(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (ff(row, "fx", 1085.976736362), ff(row, "fy", 1085.976736362), ff(row, "cx", 640.0), ff(row, "cy", 360.0))


def local_rear_origin(segs: dict[str, tuple[str, str, np.ndarray, np.ndarray]]) -> np.ndarray:
    tops = []
    for sid in ("rear_leg_left", "rear_leg_right"):
        if sid in segs:
            tops.append(segs[sid][2])
    if tops:
        return np.mean(np.vstack(tops), axis=0)
    return np.array([0.0, 0.040, 0.548], dtype=float)


def local_seat_origin(segs: dict[str, tuple[str, str, np.ndarray, np.ndarray]]) -> np.ndarray:
    if "seat_hinge_edge" in segs:
        _, _, a, b = segs["seat_hinge_edge"]
        return (a + b) * 0.5
    return np.array([0.0, 0.090, 0.440], dtype=float)


def articulate_segment_points(
    params: np.ndarray,
    sid: str,
    part: str,
    pts: np.ndarray,
    segs: dict[str, tuple[str, str, np.ndarray, np.ndarray]],
) -> np.ndarray:
    rear_angle = float(params[6]) if len(params) > 6 else 0.0
    seat_angle = float(params[7]) if len(params) > 7 else 0.0
    rear_origin = local_rear_origin(segs)
    seat_origin = local_seat_origin(segs)
    if part in {"rear_leg", "rear_stretcher"} or sid == "rear_feet_line":
        return fit.rotate_x(pts, rear_origin, rear_angle)
    if part == "side_stretcher" and pts.shape[0] == 2:
        out = pts.copy()
        out[1:] = fit.rotate_x(out[1:], rear_origin, rear_angle)
        return out
    if part in {"seat", "seat_slat"} or sid == "seat_hinge_edge":
        return fit.rotate_x(pts, seat_origin, -seat_angle)
    return pts


def project_seg(params: np.ndarray, sid: str, segs: dict, K: tuple[float, float, float, float]) -> np.ndarray:
    part, _, a, b = segs[sid]
    pts = articulate_segment_points(params, sid, part, np.vstack([a, b]), segs)
    uv, _ = fit.project(params, pts, K)
    return uv


def point_line_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-9:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def point_fraction(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-9:
        return 0.5
    return float(np.clip(((point - a) @ ab) / denom, 0.0, 1.0))


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a * (1.0 - t) + b * t


def baseline_observations(
    pose_row: dict[str, str],
    baseline_segs: dict[str, tuple[str, str, np.ndarray, np.ndarray]],
    target_segs: dict[str, tuple[str, str, np.ndarray, np.ndarray]],
    semantic_row: dict[str, str] | None = None,
) -> dict[str, object]:
    base_params = pose_params(pose_row)
    K = intrinsics(pose_row)
    leg_lines: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sid in LEG_SIDS:
        leg_lines[sid] = tuple(project_seg(base_params, sid, baseline_segs, K))  # type: ignore[arg-type]

    rails: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sid in RAIL_SIDS:
        if sid in baseline_segs:
            rails[sid] = tuple(project_seg(base_params, sid, baseline_segs, K))  # type: ignore[arg-type]
    if semantic_row is not None:
        top_l = np.asarray([ff(semantic_row, "top_rail_left_u"), ff(semantic_row, "top_rail_left_v")], dtype=float)
        top_r = np.asarray([ff(semantic_row, "top_rail_right_u"), ff(semantic_row, "top_rail_right_v")], dtype=float)
        if np.all(np.isfinite(top_l)) and np.all(np.isfinite(top_r)):
            rails["backrest_top_edge"] = (top_l, top_r)

    side_targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for side_sid, front_sid, rear_sid in SIDE_SIDS:
        if side_sid not in target_segs or front_sid not in target_segs or rear_sid not in target_segs:
            continue
        _, _, side_a, side_b = target_segs[side_sid]
        _, _, front_a, front_b = target_segs[front_sid]
        _, _, rear_a, rear_b = target_segs[rear_sid]
        tf = point_fraction(side_a, front_a, front_b)
        tr = point_fraction(side_b, rear_a, rear_b)
        front_top, front_bottom = leg_lines[front_sid]
        rear_top, rear_bottom = leg_lines[rear_sid]
        side_targets[side_sid] = (lerp(front_top, front_bottom, tf), lerp(rear_top, rear_bottom, tr))

    return {"K": K, "leg_lines": leg_lines, "rails": rails, "side_targets": side_targets}


def endpoint_res(pred: np.ndarray, obs: tuple[np.ndarray, np.ndarray], sigma: float, weight: float) -> list[float]:
    return (math.sqrt(weight) * np.concatenate([pred[0] - obs[0], pred[1] - obs[1]]) / sigma).tolist()


def smooth_params(params: np.ndarray) -> np.ndarray:
    out = params.copy()
    if len(out) >= 7:
        win = min(9, len(out) if len(out) % 2 else len(out) - 1)
        for j in range(out.shape[1]):
            out[:, j] = savgol_filter(out[:, j], win, 2, mode="interp")
    return out


def fit_frame(
    start: np.ndarray,
    baseline_start: np.ndarray,
    prev: np.ndarray | None,
    obs: dict[str, object],
    target_segs: dict[str, tuple[str, str, np.ndarray, np.ndarray]],
    weights: dict[str, float],
) -> tuple[np.ndarray, dict[str, float]]:
    K = obs.get("fit_K", obs["K"])  # type: ignore[assignment]
    leg_lines = obs["leg_lines"]  # type: ignore[assignment]
    rails = obs["rails"]  # type: ignore[assignment]
    side_targets = obs["side_targets"]  # type: ignore[assignment]

    def residual(x: np.ndarray) -> np.ndarray:
        vals: list[float] = []
        for sid in LEG_SIDS:
            uv = project_seg(x, sid, target_segs, K)
            la, lb = leg_lines[sid]
            # Strong line residual keeps the old good baseline leg strokes.
            vals.extend([
                math.sqrt(weights["leg_line"]) * point_line_distance(uv[0], la, lb) / 5.0,
                math.sqrt(weights["leg_line"]) * point_line_distance(uv[1], la, lb) / 5.0,
            ])
            # Weak endpoint residual prevents sliding too far along those lines.
            vals.extend(endpoint_res(uv, (la, lb), 18.0, weights["leg_endpoint"]))

        for side_sid, _, _ in SIDE_SIDS:
            if side_sid not in target_segs or side_sid not in side_targets:
                continue
            uv = project_seg(x, side_sid, target_segs, K)
            vals.extend(endpoint_res(uv, side_targets[side_sid], 10.0, weights["side"]))

        for sid, weight in [
            ("backrest_top_edge", weights["top_rail"]),
            ("backrest_bottom_edge", weights["bottom_rail"]),
            ("seat_front_edge", weights["seat_front"]),
        ]:
            if sid in target_segs and sid in rails:
                uv = project_seg(x, sid, target_segs, K)
                vals.extend(endpoint_res(uv, rails[sid], 9.0, weight))

        # Baseline pose is a stabilizer, not the target geometry. Keep depth and
        # articulation sane so the physical projection does not exploit joints.
        vals.extend((weights["pose_prior"] * (x[:3] - baseline_start[:3]) / np.array([0.55, 0.55, 0.55])).tolist())
        vals.extend((weights["pose_prior"] * (x[3:6] - baseline_start[3:6]) / np.array([0.65, 0.65, 0.85])).tolist())
        vals.extend((weights["joint_prior"] * (x[6:8] - baseline_start[6:8]) / np.array([0.32, 0.32])).tolist())
        if prev is not None:
            vals.extend((0.45 * (x - prev) / np.array([0.12, 0.12, 0.12, 0.18, 0.18, 0.22, 0.10, 0.10])).tolist())
        return np.asarray(vals, dtype=float)

    bounds = (
        baseline_start + np.array([-1.2, -1.2, -1.2, -1.0, -1.0, -1.4, -0.55, -0.55], dtype=float),
        baseline_start + np.array([1.2, 1.2, 1.2, 1.0, 1.0, 1.4, 0.55, 0.55], dtype=float),
    )
    res = least_squares(residual, np.clip(start, bounds[0], bounds[1]), bounds=bounds, loss="soft_l1", f_scale=1.0, max_nfev=140)
    x = res.x

    line_vals: list[float] = []
    endpoint_vals: list[float] = []
    for sid in LEG_SIDS:
        uv = project_seg(x, sid, target_segs, K)
        la, lb = leg_lines[sid]
        line_vals.extend([point_line_distance(uv[0], la, lb), point_line_distance(uv[1], la, lb)])
        endpoint_vals.extend(np.linalg.norm(uv - np.vstack([la, lb]), axis=1).tolist())
    side_vals: list[float] = []
    for side_sid, _, _ in SIDE_SIDS:
        if side_sid in side_targets:
            side_vals.extend(np.linalg.norm(project_seg(x, side_sid, target_segs, K) - np.vstack(side_targets[side_sid]), axis=1).tolist())
    rail_vals: list[float] = []
    for sid in RAIL_SIDS:
        if sid in target_segs and sid in rails:
            rail_vals.extend(np.linalg.norm(project_seg(x, sid, target_segs, K) - np.vstack(rails[sid]), axis=1).tolist())

    metrics = {
        "line_rmse_px": float(np.sqrt(np.mean(np.square(line_vals)))) if line_vals else math.nan,
        "endpoint_rmse_px": float(np.sqrt(np.mean(np.square(endpoint_vals)))) if endpoint_vals else math.nan,
        "side_ratio_rmse_px": float(np.sqrt(np.mean(np.square(side_vals)))) if side_vals else math.nan,
        "rail_rmse_px": float(np.sqrt(np.mean(np.square(rail_vals)))) if rail_vals else math.nan,
    }
    return x, metrics


def draw_wire(img: np.ndarray, params: np.ndarray, segs: dict, K: tuple[float, float, float, float]) -> np.ndarray:
    canvas = img.copy()
    for sid, (_, role, _, _) in segs.items():
        color = COLORS.get(role)
        if color is None:
            continue
        uv = project_seg(params, sid, segs, K)
        aa = tuple(np.round(uv[0]).astype(int))
        bb = tuple(np.round(uv[1]).astype(int))
        cv2.line(canvas, aa, bb, (20, 20, 20), 6, cv2.LINE_AA)
        cv2.line(canvas, aa, bb, color, 3, cv2.LINE_AA)
        cv2.circle(canvas, aa, 4, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, bb, 4, color, -1, cv2.LINE_AA)
    return canvas


def finalize_video(tmp: Path, out: Path) -> None:
    ffmpeg = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    try:
        subprocess.run([ffmpeg, "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        subprocess.run([ffmpeg, "-y", "-i", str(tmp), "-c:v", "libopenh264", "-pix_fmt", "yuv420p", str(out)], check=True, capture_output=True, text=True)
    tmp.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-dir", type=Path, default=Path("samples_known_object/05_chair"))
    ap.add_argument("--baseline-pose-csv", type=Path, default=None)
    ap.add_argument("--baseline-segments-csv", type=Path, default=None)
    ap.add_argument("--target-segments-csv", type=Path, required=True)
    ap.add_argument("--semantic-observations-csv", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--fit-fx", type=float, default=None, help="Override fitted/projected camera fx, e.g. GVHMR K_fullimg.")
    ap.add_argument("--fit-fy", type=float, default=None, help="Override fitted/projected camera fy, e.g. GVHMR K_fullimg.")
    ap.add_argument("--fit-cx", type=float, default=None, help="Override fitted/projected camera cx.")
    ap.add_argument("--fit-cy", type=float, default=None, help="Override fitted/projected camera cy.")
    ap.add_argument("--leg-line-weight", type=float, default=3.4)
    ap.add_argument("--leg-endpoint-weight", type=float, default=0.35)
    ap.add_argument("--side-weight", type=float, default=2.4)
    ap.add_argument("--top-rail-weight", type=float, default=2.4)
    ap.add_argument("--bottom-rail-weight", type=float, default=1.8)
    ap.add_argument("--seat-front-weight", type=float, default=0.75)
    ap.add_argument("--pose-prior-weight", type=float, default=1.0)
    ap.add_argument("--joint-prior-weight", type=float, default=1.0)
    ap.add_argument("--no-render", action="store_true", help="Only write physical6d_pose.csv and summary.json; skip diagnostic MP4.")
    args = ap.parse_args()

    sample = args.sample_dir
    baseline_pose_csv = args.baseline_pose_csv or sample / "results/final_result/object_pose.csv"
    baseline_segments_csv = args.baseline_segments_csv or sample / "results/chair_semantic_local_points/chair_semantic_local_segments.csv"
    pose_rows = read_rows(baseline_pose_csv)
    semantic_by_frame: dict[int, dict[str, str]] = {}
    if args.semantic_observations_csv is not None:
        semantic_by_frame = {int(r["frame"]): r for r in read_rows(args.semantic_observations_csv)}
    baseline_segs = fit.load_segments(baseline_segments_csv)
    target_segs = fit.load_segments(args.target_segments_csv)
    weights = {
        "leg_line": args.leg_line_weight,
        "leg_endpoint": args.leg_endpoint_weight,
        "side": args.side_weight,
        "top_rail": args.top_rail_weight,
        "bottom_rail": args.bottom_rail_weight,
        "seat_front": args.seat_front_weight,
        "pose_prior": args.pose_prior_weight,
        "joint_prior": args.joint_prior_weight,
    }

    fitted: list[np.ndarray] = []
    metrics: list[dict[str, float]] = []
    prev: np.ndarray | None = None
    for row in pose_rows:
        base = pose_params(row)
        obs = baseline_observations(row, baseline_segs, target_segs, semantic_by_frame.get(int(row["frame"])))
        obs_k = obs["K"]
        obs["fit_K"] = (
            args.fit_fx if args.fit_fx is not None else obs_k[0],
            args.fit_fy if args.fit_fy is not None else obs_k[1],
            args.fit_cx if args.fit_cx is not None else obs_k[2],
            args.fit_cy if args.fit_cy is not None else obs_k[3],
        )
        start = prev.copy() if prev is not None else base.copy()
        x, m = fit_frame(start, base, prev, obs, target_segs, weights)
        fitted.append(x)
        metrics.append(m)
        prev = x

    params = smooth_params(np.vstack(fitted))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "physical6d_pose.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        for row, x, m in zip(pose_rows, params, metrics):
            r_total = Rotation.from_matrix(Rotation.from_rotvec(x[:3]).as_matrix() @ fit.BASE_LOCAL_TO_CAM)
            qx, qy, qz, qw = r_total.as_quat()
            out = {
                "frame": int(row["frame"]),
                "time": row.get("time", "0"),
                "tx": x[3],
                "ty": x[4],
                "tz": x[5],
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "qw": qw,
                "rx_delta": x[0],
                "ry_delta": x[1],
                "rz_delta": x[2],
                "rear_joint_angle": x[6],
                "seat_joint_angle": x[7],
                "fx": args.fit_fx if args.fit_fx is not None else ff(row, "fx", 1085.976736362),
                "fy": args.fit_fy if args.fit_fy is not None else ff(row, "fy", 1085.976736362),
                "cx": args.fit_cx if args.fit_cx is not None else ff(row, "cx", 640.0),
                "cy": args.fit_cy if args.fit_cy is not None else ff(row, "cy", 360.0),
                **m,
                "source": "physical6d_baseline_lines_top_side_ratio_no_projective_q",
            }
            writer.writerow({k: (f"{v:.9f}" if isinstance(v, float) else v) for k, v in out.items()})

    out_video = args.out_dir / "physical6d_wire.mp4"
    if not args.no_render:
        first = cv2.imread(str(sample / "frames/00001.png"))
        if first is None:
            raise RuntimeError(f"Missing first frame under {sample / 'frames'}")
        h, w = first.shape[:2]
        tmp = args.out_dir / "physical6d_wire.tmp.mp4"
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open {tmp}")
        for row, x in zip(pose_rows, params):
            frame = int(row["frame"])
            img = cv2.imread(str(sample / "frames" / f"{frame:05d}.png"))
            if img is None:
                continue
            row_k = intrinsics(row)
            K = (
                args.fit_fx if args.fit_fx is not None else row_k[0],
                args.fit_fy if args.fit_fy is not None else row_k[1],
                args.fit_cx if args.fit_cx is not None else row_k[2],
                args.fit_cy if args.fit_cy is not None else row_k[3],
            )
            over = draw_wire(img, x, target_segs, K)
            label = f"frame {frame:03d} physical 6D Articraft ratio"
            cv2.putText(over, label, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(over, label, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (40, 240, 40), 1, cv2.LINE_AA)
            writer.write(over)
        writer.release()
        finalize_video(tmp, out_video)

    summary = {
        "overlay_video": str(out_video) if not args.no_render else "",
        "pose_csv": str(csv_path),
        "baseline_pose_csv": str(baseline_pose_csv),
        "baseline_segments_csv": str(baseline_segments_csv),
        "target_segments_csv": str(args.target_segments_csv),
        "frames": len(params),
        "median_line_rmse_px": float(np.nanmedian([m["line_rmse_px"] for m in metrics])),
        "median_endpoint_rmse_px": float(np.nanmedian([m["endpoint_rmse_px"] for m in metrics])),
        "median_side_ratio_rmse_px": float(np.nanmedian([m["side_ratio_rmse_px"] for m in metrics])),
        "median_rail_rmse_px": float(np.nanmedian([m["rail_rmse_px"] for m in metrics])),
        "weights": weights,
        "fit_K_override": [args.fit_fx, args.fit_fy, args.fit_cx, args.fit_cy],
        "render_skipped": bool(args.no_render),
        "note": "Physical camera projection only: optimizes R/t and chair joints; no projective q and no Articraft local-geometry edits.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
