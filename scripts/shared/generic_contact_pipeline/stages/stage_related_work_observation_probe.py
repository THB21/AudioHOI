#!/usr/bin/env python3
from __future__ import annotations

import argparse
import binascii
import functools
import json
import math
import statistics
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[4]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
else:
    REPO = Path(__file__).resolve().parents[4]

from scripts.shared.generic_contact_pipeline.core.config import available_cases, load_case_profile  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.io import read_csv, repo_path, write_csv, write_json  # noqa: E402


OUT_DIR = REPO / "docs" / "related_work_audit"


def finite_float(value: object) -> float | None:
    try:
        if value in ("", None):
            return None
        out = float(value)  # type: ignore[arg-type]
        return out if math.isfinite(out) else None
    except Exception:
        return None


def mean(vals: list[float]) -> float | None:
    vals = [v for v in vals if math.isfinite(v)]
    return statistics.fmean(vals) if vals else None


def median(vals: list[float]) -> float | None:
    vals = [v for v in vals if math.isfinite(v)]
    return statistics.median(vals) if vals else None


def p95(vals: list[float]) -> float | None:
    vals = sorted(v for v in vals if math.isfinite(v))
    if not vals:
        return None
    return vals[min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))]


def cv(vals: list[float]) -> float | None:
    vals = [v for v in vals if math.isfinite(v)]
    if len(vals) < 3:
        return None
    m = statistics.fmean(vals)
    if abs(m) < 1e-12:
        return None
    return statistics.pstdev(vals) / abs(m)


def first_float(row: dict[str, str], keys: list[str]) -> float | None:
    for key in keys:
        val = finite_float(row.get(key))
        if val is not None:
            return val
    return None


def read_mask_stats(mask_dir: Path, frame: int) -> dict[str, float] | None:
    candidates = [
        mask_dir / f"{frame:05d}_mask.png",
        mask_dir / f"{frame - 1:05d}_mask.png",
        mask_dir / f"{frame:06d}_mask.png",
        mask_dir / f"{frame - 1:06d}_mask.png",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None
    return read_mask_stats_path(str(path))


@functools.lru_cache(maxsize=4096)
def read_mask_stats_path(path_str: str) -> dict[str, float] | None:
    path = Path(path_str)
    points = read_png_nonzero_points_fast(path)
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = float(min(xs)), float(max(xs))
    y1, y2 = float(min(ys)), float(max(ys))
    area = float(len(xs))
    return {
        "mask_center_x": float(sum(xs) / area),
        "mask_center_y": float(sum(ys) / area),
        "mask_area_px": area,
        "mask_bbox_x1": x1,
        "mask_bbox_y1": y1,
        "mask_bbox_x2": x2,
        "mask_bbox_y2": y2,
        "mask_bbox_w_px": float(x2 - x1 + 1.0),
        "mask_bbox_h_px": float(y2 - y1 + 1.0),
        "mask_bottom_v": y2,
    }


def read_png_nonzero_points_fast(path: Path) -> list[tuple[int, int]]:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            if img.ndim == 2:
                active = img > 0
            elif img.shape[2] == 4:
                active = (img[:, :, 3] > 0) & (img[:, :, :3].sum(axis=2) > 0)
            else:
                active = img[:, :, :3].sum(axis=2) > 0
            ys, xs = np.nonzero(active)
            return list(zip(xs.astype(int).tolist(), ys.astype(int).tolist()))
    except Exception:
        pass
    return read_png_nonzero_points(path)


def read_png_nonzero_points(path: Path) -> list[tuple[int, int]]:
    """Tiny dependency-free PNG reader for binary/grayscale/RGB SAM masks."""
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"Not a PNG file: {path}")
    pos = 8
    width = height = bit_depth = color_type = None
    idat = bytearray()
    palette: list[tuple[int, int, int]] | None = None
    while pos < len(data):
        if pos + 8 > len(data):
            break
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        crc_read = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])[0]
        crc_calc = binascii.crc32(ctype + chunk) & 0xFFFFFFFF
        if crc_read != crc_calc:
            raise ValueError(f"PNG CRC mismatch in {path} chunk {ctype!r}")
        pos += 12 + length
        if ctype == b"IHDR":
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(">IIBBBBB", chunk)
            if compression != 0 or filter_method != 0 or interlace != 0:
                raise ValueError(f"Unsupported PNG encoding in {path}")
            if bit_depth != 8:
                raise ValueError(f"Unsupported PNG bit depth {bit_depth} in {path}; expected 8")
        elif ctype == b"PLTE":
            palette = [(chunk[i], chunk[i + 1], chunk[i + 2]) for i in range(0, len(chunk), 3)]
        elif ctype == b"IDAT":
            idat.extend(chunk)
        elif ctype == b"IEND":
            break
    if None in (width, height, bit_depth, color_type):
        raise ValueError(f"Missing IHDR in {path}")
    channels_by_type = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    channels = channels_by_type.get(int(color_type))  # type: ignore[arg-type]
    if channels is None:
        raise ValueError(f"Unsupported PNG color type {color_type} in {path}")
    raw = zlib.decompress(bytes(idat))
    stride = int(width) * channels  # type: ignore[arg-type]
    rows: list[bytearray] = []
    rp = 0
    prev = bytearray(stride)
    bpp = channels
    for _ in range(int(height)):  # type: ignore[arg-type]
        filter_type = raw[rp]
        rp += 1
        scan = bytearray(raw[rp:rp + stride])
        rp += stride
        recon = bytearray(stride)
        for i, x in enumerate(scan):
            left = recon[i - bpp] if i >= bpp else 0
            up = prev[i]
            up_left = prev[i - bpp] if i >= bpp else 0
            if filter_type == 0:
                val = x
            elif filter_type == 1:
                val = x + left
            elif filter_type == 2:
                val = x + up
            elif filter_type == 3:
                val = x + ((left + up) // 2)
            elif filter_type == 4:
                val = x + paeth(left, up, up_left)
            else:
                raise ValueError(f"Unsupported PNG filter {filter_type} in {path}")
            recon[i] = val & 0xFF
        rows.append(recon)
        prev = recon

    pts: list[tuple[int, int]] = []
    for y, row in enumerate(rows):
        for x in range(int(width)):  # type: ignore[arg-type]
            off = x * channels
            if color_type == 0:
                active = row[off] > 0
            elif color_type == 2:
                active = (row[off] | row[off + 1] | row[off + 2]) > 0
            elif color_type == 3:
                idx = row[off]
                if palette:
                    rgb = palette[idx] if idx < len(palette) else (0, 0, 0)
                    active = (rgb[0] | rgb[1] | rgb[2]) > 0
                else:
                    active = idx > 0
            elif color_type == 4:
                active = row[off] > 0 or row[off + 1] > 0
            elif color_type == 6:
                active = row[off + 3] > 0 and (row[off] | row[off + 1] | row[off + 2]) > 0
            else:
                active = False
            if active:
                pts.append((x, y))
    return pts


def paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def observation_center(row: dict[str, str]) -> tuple[float, float] | None:
    u = first_float(row, ["u_proj", "u_obs", "ref_u_fit", "ref_u_smooth", "ref_u", "center_x", "mask_center_x", "top_rail_center_u", "seat_center_u"])
    v = first_float(row, ["v_proj", "v_obs", "ref_v_fit", "ref_v_smooth", "ref_v", "center_y", "mask_center_y", "top_rail_center_v", "seat_center_v"])
    if u is None or v is None:
        return None
    return u, v


def observation_bbox(row: dict[str, str]) -> tuple[float, float, float, float] | None:
    x1 = first_float(row, ["bbox_x1", "body_bbox_x1", "mask_bbox_x1"])
    y1 = first_float(row, ["bbox_y1", "body_bbox_y1", "mask_bbox_y1"])
    x2 = first_float(row, ["bbox_x2", "body_bbox_x2", "mask_bbox_x2"])
    y2 = first_float(row, ["bbox_y2", "body_bbox_y2", "mask_bbox_y2"])
    if None in (x1, y1, x2, y2):
        r = first_float(row, ["radius_proj_px", "radius_obs_px", "radius_px"])
        center = observation_center(row)
        if r is not None and center is not None:
            u, v = center
            return u - r, v - r, u + r, v + r
        return None
    return x1, y1, x2, y2  # type: ignore[return-value]


def point_outside_bbox(u: float, v: float, bbox: tuple[float, float, float, float], margin: float = 4.0) -> float:
    x1, y1, x2, y2 = bbox
    dx = max(x1 - margin - u, 0.0, u - (x2 + margin))
    dy = max(y1 - margin - v, 0.0, v - (y2 + margin))
    return math.hypot(dx, dy)


def semantic_points(row: dict[str, str]) -> list[tuple[str, float, float]]:
    keys = [
        "top_rail_left",
        "top_rail_right",
        "seat_front_left",
        "seat_front_right",
        "front_left_foot",
        "front_right_foot",
        "floor_support_proxy_left",
        "floor_support_proxy_right",
        "front_leg_left_top",
        "front_leg_right_top",
        "rear_leg_left_top",
        "rear_leg_right_top",
        "rear_leg_left_bottom",
        "rear_leg_right_bottom",
    ]
    pts: list[tuple[str, float, float]] = []
    for prefix in keys:
        u = finite_float(row.get(f"{prefix}_u"))
        v = finite_float(row.get(f"{prefix}_v"))
        if u is not None and v is not None:
            pts.append((prefix, u, v))
    return pts


def row_metrics(case: str, row: dict[str, str], mask_stats: dict[str, float] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    if mask_stats:
        if case != "chair":
            center = observation_center(row)
            if center:
                out["opengaus_mask_center_loss_px"] = math.hypot(center[0] - mask_stats["mask_center_x"], center[1] - mask_stats["mask_center_y"])
            bbox = observation_bbox(row)
            if bbox:
                x1, y1, x2, y2 = bbox
                pred_area = max(0.0, x2 - x1 + 1.0) * max(0.0, y2 - y1 + 1.0)
                mask_area = max(mask_stats["mask_area_px"], 1.0)
                out["opengaus_mask_size_rel"] = abs(pred_area - mask_area) / mask_area
                pred_center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
                mask_bbox_center = (
                    (mask_stats["mask_bbox_x1"] + mask_stats["mask_bbox_x2"]) * 0.5,
                    (mask_stats["mask_bbox_y1"] + mask_stats["mask_bbox_y2"]) * 0.5,
                )
                out["opengaus_bbox_center_loss_px"] = math.hypot(pred_center[0] - mask_bbox_center[0], pred_center[1] - mask_bbox_center[1])
        if case in {"basketball", "football"}:
            support_v = first_float(row, ["support_v", "bottom_proj_v"])
            if support_v is not None:
                out["mover_support_bottom_gap_px"] = abs(support_v - mask_stats["mask_bottom_v"])
        if case == "chair":
            bbox_tuple = (
                mask_stats["mask_bbox_x1"],
                mask_stats["mask_bbox_y1"],
                mask_stats["mask_bbox_x2"],
                mask_stats["mask_bbox_y2"],
            )
            outside = [point_outside_bbox(u, v, bbox_tuple) for _, u, v in semantic_points(row)]
            if outside:
                out["mover_semantic_outside_mask_p95_px"] = p95(outside) or 0.0
                out["mover_semantic_outside_mask_mean_px"] = mean(outside) or 0.0

    line = first_float(row, ["line_rmse_px", "endpoint_rmse_px", "rail_rmse_px", "residual_px"])
    if line is not None:
        out["opengaus_semantic_or_reprojection_residual_px"] = abs(line)
    depth_gap = first_float(row, ["contact_depth_gap", "hand_contact_3d_gap_m", "contact_depth_offset_m", "hand_object_z_gap_m"])
    if depth_gap is not None:
        out["mover_depth_contact_gap_m"] = abs(depth_gap)
    return out


def choose_obs_path(case: str, variant: str) -> Path | None:
    profile = load_case_profile(case)
    if variant == "generic":
        path = profile.result_dir / "object_observations.csv"
        return path if path.exists() else None
    for key in ["object_observations_csv", "semantic_observations_csv", "object_proxy_observations_csv"]:
        rel = profile.baseline.get(key)
        if rel:
            path = repo_path(rel)
            if path.exists():
                return path
    return None


def choose_pose_path(case: str, variant: str) -> Path | None:
    profile = load_case_profile(case)
    if variant == "generic":
        path = profile.result_dir / "object_pose.csv"
        return path if path.exists() else None
    rel = profile.baseline.get("final_pose_csv")
    if not rel:
        return None
    path = repo_path(rel)
    return path if path.exists() else None


def merge_by_frame(*tables: list[dict[str, str]]) -> list[dict[str, str]]:
    by_frame: dict[int, dict[str, str]] = {}
    for table in tables:
        for row in table:
            frame = int(finite_float(row.get("frame")) or len(by_frame) + 1)
            dst = by_frame.setdefault(frame, {"frame": str(frame)})
            for k, v in row.items():
                if v not in ("", None):
                    dst.setdefault(k, v)
    return [by_frame[k] for k in sorted(by_frame)]


def analyze_variant(case: str, variant: str) -> dict[str, Any]:
    profile = load_case_profile(case)
    obs_path = choose_obs_path(case, variant)
    pose_path = choose_pose_path(case, variant)
    obs_rows = read_csv(obs_path) if obs_path else []
    pose_rows = read_csv(pose_path) if pose_path else []
    rows = merge_by_frame(obs_rows, pose_rows)
    mask_dir = profile.sample_dir / "results" / "segmentation" / "masks"

    metric_lists: dict[str, list[float]] = {}
    per_frame: list[dict[str, Any]] = []
    for row in rows:
        frame = int(finite_float(row.get("frame")) or len(per_frame) + 1)
        masks = read_mask_stats(mask_dir, frame) if mask_dir.exists() else None
        metrics = row_metrics(case, row, masks)
        out = {"case": case, "variant": variant, "frame": frame}
        out.update(metrics)
        per_frame.append(out)
        for key, val in metrics.items():
            metric_lists.setdefault(key, []).append(val)

    summary: dict[str, Any] = {
        "case": case,
        "variant": variant,
        "observation_csv": str(obs_path) if obs_path else "",
        "pose_csv": str(pose_path) if pose_path else "",
        "frames": len(rows),
        "mask_frames_used": sum(1 for r in per_frame if any(k.startswith("opengaus_mask") or k.startswith("mover_semantic") or k.startswith("mover_support") for k in r)),
    }
    for key, vals in metric_lists.items():
        summary[f"{key}_median"] = median(vals)
        summary[f"{key}_p95"] = p95(vals)
        summary[f"{key}_cv"] = cv(vals)
        summary[f"{key}_samples"] = len(vals)
    return {"summary": summary, "per_frame": per_frame}


def ratio(a: Any, b: Any) -> float | None:
    aa = finite_float(a)
    bb = finite_float(b)
    if aa is None or bb is None or abs(bb) < 1e-12:
        return None
    return aa / bb


def fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}" if math.isfinite(v) else ""
    return str(v)


def write_markdown(summaries: list[dict[str, Any]], ratios: list[dict[str, Any]]) -> Path:
    path = OUT_DIR / "related_work_observation_probe_summary.md"
    lines = [
        "# Related Work Observation-Level Probe",
        "",
        "Generated by `scripts/shared/generic_contact_pipeline/stages/stage_related_work_observation_probe.py`.",
        "",
        "This probe measures related-work-inspired terms directly from AudioHOI observations and SAM2 masks:",
        "",
        "- Open3DHOI-style mask center and mask size alignment.",
        "- Open3DHOI-style semantic/reprojection residual where projected/semantic residual columns exist.",
        "- MOVER-style support-bottom gap and semantic graph outside-mask sanity.",
        "- MOVER-style contact/depth gap when present in pose/contact CSV columns.",
        "",
        "It is still not a full differentiable Gaussian/MOVER reproduction because RGB render losses, SDF collision, and rendered ordinal-depth buffers are not available for every case.",
        "",
        "## Variant Metrics",
        "",
        "| case | variant | frames | mask frames | mask center med px | mask size med rel | semantic/residual med px | support gap med px | outside-mask p95 med px | depth/contact med m |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            "| {case} | {variant} | {frames} | {mask_frames} | {center} | {size} | {sem} | {support} | {outside} | {depth} |".format(
                case=s["case"],
                variant=s["variant"],
                frames=s["frames"],
                mask_frames=s["mask_frames_used"],
                center=fmt(s.get("opengaus_mask_center_loss_px_median")),
                size=fmt(s.get("opengaus_mask_size_rel_median")),
                sem=fmt(s.get("opengaus_semantic_or_reprojection_residual_px_median")),
                support=fmt(s.get("mover_support_bottom_gap_px_median")),
                outside=fmt(s.get("mover_semantic_outside_mask_p95_px_median")),
                depth=fmt(s.get("mover_depth_contact_gap_m_median")),
            )
        )
    lines.extend(["", "## Generic / Baseline Ratios", ""])
    lines.append("| case | mask center | mask size | semantic/residual | support gap | outside-mask | depth/contact |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in ratios:
        lines.append(
            "| {case} | {center} | {size} | {sem} | {support} | {outside} | {depth} |".format(
                case=r["case"],
                center=fmt(r.get("opengaus_mask_center_loss_px_median")),
                size=fmt(r.get("opengaus_mask_size_rel_median")),
                sem=fmt(r.get("opengaus_semantic_or_reprojection_residual_px_median")),
                support=fmt(r.get("mover_support_bottom_gap_px_median")),
                outside=fmt(r.get("mover_semantic_outside_mask_p95_px_median")),
                depth=fmt(r.get("mover_depth_contact_gap_m_median")),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Lower is better for all listed losses/ratios.",
            "- Empty cells mean the corresponding observation is not present in that case/baseline schema.",
            "- This table complements `method_loss_probe_summary.md`: that one works from final CSV trajectories; this one works from SAM2/semantic observations.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")
    return path


def run(cases: list[str]) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    per_frame_rows: list[dict[str, Any]] = []
    for case in cases:
        for variant in ["generic", "baseline"]:
            result = analyze_variant(case, variant)
            summaries.append(result["summary"])
            per_frame_rows.extend(result["per_frame"])

    by_case = {}
    for row in summaries:
        by_case.setdefault(row["case"], {})[row["variant"]] = row
    ratios: list[dict[str, Any]] = []
    metric_keys = sorted({k for row in summaries for k in row if k.endswith("_median")})
    for case, variants in by_case.items():
        g = variants.get("generic")
        b = variants.get("baseline")
        if not g or not b:
            continue
        out: dict[str, Any] = {"case": case, "variant": "generic_over_baseline_ratio"}
        for key in metric_keys:
            out[key] = ratio(g.get(key), b.get(key))
        ratios.append(out)

    write_csv(OUT_DIR / "related_work_observation_probe_metrics.csv", summaries)
    write_csv(OUT_DIR / "related_work_observation_probe_ratios.csv", ratios)
    write_csv(OUT_DIR / "related_work_observation_probe_per_frame.csv", per_frame_rows)
    write_json(OUT_DIR / "related_work_observation_probe_metrics.json", {"summaries": summaries, "ratios": ratios})
    md = write_markdown(summaries, ratios)
    return {
        "cases": cases,
        "summary_rows": len(summaries),
        "per_frame_rows": len(per_frame_rows),
        "metrics_csv": str(OUT_DIR / "related_work_observation_probe_metrics.csv"),
        "ratios_csv": str(OUT_DIR / "related_work_observation_probe_ratios.csv"),
        "summary_md": str(md),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run observation-level related-work proxy tests from SAM2 masks and object observations.")
    ap.add_argument("--case", default="all", help="case name or all")
    args = ap.parse_args()
    cases = available_cases() if args.case == "all" else [args.case]
    print(json.dumps(run(cases), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
