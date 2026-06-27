from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path
from typing import Iterable

from ..base.config import CaseProfile
from ..base.io import read_csv, write_csv, write_json
from ..base.schema import stage_paths


LOSS_FIELDS = [
    "frame",
    "time",
    "E_total",
    "E_2d",
    "E_depth",
    "E_visual",
    "E_mask",
    "E_contact",
    "E_audio",
    "E_support",
    "E_smooth",
    "E_temporal",
    "E_static",
    "E_penetration",
    "E_prior",
    "E_reg",
    "vlm_contact_gate",
    "vlm_anchor_gate",
    "contact_active",
    "static_active",
    "failure_label",
    "contact_state",
    "visibility_state",
    "source",
]

TRACE_FIELDS = [
    "iteration",
    "E_total",
    "E_visual",
    "E_mask",
    "E_contact",
    "E_audio",
    "E_support",
    "E_smooth",
    "E_reg",
    "active_frames",
    "active_contact_frames",
    "active_audio_frames",
]

POSE_KEYS = {
    "basketball": ["tx", "ty", "tz"],
    "football": ["tx", "ty", "tz"],
    "mug": ["x", "y", "z", "yaw", "pitch", "roll"],
    "chair": ["tx", "ty", "tz", "qx", "qy", "qz", "qw"],
}


def f(row: dict[str, str], key: str) -> float | None:
    try:
        value = row.get(key)
        if value in {"", None}:
            return None
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return None


def frame_id(row: dict[str, str]) -> int:
    try:
        return int(float(row.get("frame", "0") or 0))
    except Exception:
        return 0


def by_frame(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    return {frame_id(row): row for row in rows}


def gates_by_frame(path: Path, query_types: set[str]) -> dict[int, str]:
    if not path.exists():
        return {}
    out: dict[int, str] = {}
    try:
        rows = read_csv(path)
    except Exception:
        return out
    for row in rows:
        if row.get("query_type") not in query_types:
            continue
        fr = frame_id(row)
        gate = row.get("pass_gate", "")
        if gate:
            out[fr] = gate
    return out


def sq(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value * value


def diff_norm(row: dict[str, str], prev: dict[str, str] | None, keys: Iterable[str]) -> float | None:
    if prev is None:
        return None
    vals = []
    for key in keys:
        a = f(row, key)
        b = f(prev, key)
        if a is not None and b is not None:
            vals.append((a - b) ** 2)
    if not vals:
        return None
    return math.sqrt(sum(vals))


def reg_norm(row: dict[str, str], init: dict[str, str] | None, keys: Iterable[str]) -> float | None:
    if init is None:
        return None
    vals = []
    for key in keys:
        a = f(row, key)
        b = f(init, key)
        if a is not None and b is not None:
            vals.append((a - b) ** 2)
    if not vals:
        return None
    return math.sqrt(sum(vals))


def visual_residual(profile: CaseProfile, pose: dict[str, str]) -> float | None:
    if profile.case_name in {"basketball", "football"}:
        direct = f(pose, "residual_px")
        if direct is not None:
            return abs(direct)
        up = f(pose, "u_proj")
        uo = f(pose, "u_obs")
        vp = f(pose, "v_proj")
        vo = f(pose, "v_obs")
        if None not in {up, uo, vp, vo}:
            return math.sqrt((up - uo) ** 2 + (vp - vo) ** 2)  # type: ignore[operator]
    if profile.case_name == "chair":
        vals = [f(pose, key) for key in ["line_rmse_px", "endpoint_rmse_px", "rail_rmse_px"]]
        vals = [v for v in vals if v is not None]
        if vals:
            return math.sqrt(sum(v * v for v in vals) / len(vals))
    return None


def support_residual(profile: CaseProfile, pose: dict[str, str], obs: dict[str, str] | None) -> float | None:
    if profile.case_name in {"basketball", "football"}:
        bottom = f(pose, "bottom_proj_v")
        floor = f(pose, "floor_v")
        if bottom is not None and floor is not None:
            return abs(bottom - floor)
    if profile.case_name == "chair":
        vals = [f(pose, "side_ratio_rmse_px"), f(pose, "endpoint_rmse_px")]
        vals = [v for v in vals if v is not None]
        if vals:
            return math.sqrt(sum(v * v for v in vals) / len(vals))
    if obs:
        gap = f(obs, "support_gap_px")
        if gap is not None:
            return abs(gap)
    return None


def contact_residual(profile: CaseProfile, pose: dict[str, str], contact: dict[str, str] | None) -> float | None:
    if profile.case_name in {"basketball", "football"}:
        gap = f(pose, "contact_depth_gap")
        if gap is not None:
            return abs(gap)
        if contact:
            off = f(contact, "contact_depth_offset_m")
            if off is not None:
                return abs(off)
    if profile.case_name == "mug" and contact:
        gap = f(contact, "hand_contact_3d_gap_m")
        if gap is not None:
            return abs(gap)
        z = f(contact, "hand_object_z_gap_m")
        if z is not None:
            return abs(z)
    if profile.case_name == "chair" and contact:
        u = f(contact, "contact_point_2d_u") or f(contact, "contact_u")
        v = f(contact, "contact_point_2d_v") or f(contact, "contact_v")
        if u is not None and v is not None and contact.get("contact_active") == "1":
            return 0.0
    return None


def mask_residual(_profile: CaseProfile, obs: dict[str, str] | None) -> float | None:
    if not obs:
        return None
    jitter = f(obs, "proxy_jitter_px")
    if jitter is not None:
        return abs(jitter)
    conf = f(obs, "semantic_conf")
    if conf is not None:
        return max(0.0, 1.0 - conf)
    return None


def audio_residual(pose: dict[str, str], phase: dict[str, str] | None) -> float | None:
    if pose.get("audio_contact_frame") in {"1", "1.0"}:
        return 0.0
    if phase and phase.get("audio_event"):
        return 0.0
    return None


def sum_terms(row: dict[str, object]) -> float:
    total = 0.0
    for key in ["E_visual", "E_mask", "E_contact", "E_audio", "E_support", "E_smooth", "E_reg"]:
        value = row.get(key)
        try:
            if value not in {"", None}:
                total += float(value)  # type: ignore[arg-type]
        except Exception:
            pass
    return total


def build_loss_rows(profile: CaseProfile) -> list[dict[str, object]]:
    paths = stage_paths(profile)
    pose_rows = read_csv(paths["object_pose"])
    init_by_frame = by_frame(read_csv(paths["object_pose_init"])) if paths["object_pose_init"].exists() else {}
    obs_by_frame = by_frame(read_csv(paths["object_observations"])) if paths["object_observations"].exists() else {}
    contact_by_frame = by_frame(read_csv(paths["object_contact_points"])) if paths["object_contact_points"].exists() else {}
    phase_by_frame = by_frame(read_csv(paths["object_phase"])) if paths["object_phase"].exists() else {}
    contact_gate_by_frame = gates_by_frame(paths["vlm_dir"] / "stage2" / "vlm_gates.csv", {"contact_relation_check"})
    anchor_gate_by_frame: dict[int, str] = {}
    anchor_gate_by_frame.update(gates_by_frame(paths["vlm_dir"] / "stage1" / "vlm_gates.csv", {"keypart_identity_check", "track_stability_check", "keypart_visibility_check"}))
    anchor_gate_by_frame.update(gates_by_frame(paths["vlm_dir"] / "stage4" / "vlm_gates.csv", {"anchor_update_check"}))
    keys = POSE_KEYS.get(profile.case_name, ["tx", "ty", "tz"])
    out: list[dict[str, object]] = []
    prev: dict[str, str] | None = None
    for pose in pose_rows:
        fr = frame_id(pose)
        obs = obs_by_frame.get(fr)
        contact = contact_by_frame.get(fr)
        phase = phase_by_frame.get(fr)
        values = {
            "E_visual": sq(visual_residual(profile, pose)),
            "E_mask": sq(mask_residual(profile, obs)),
            "E_contact": sq(contact_residual(profile, pose, contact)),
            "E_audio": sq(audio_residual(pose, phase)),
            "E_support": sq(support_residual(profile, pose, obs)),
            "E_smooth": sq(diff_norm(pose, prev, keys)),
            "E_reg": sq(reg_norm(pose, init_by_frame.get(fr), keys)),
        }
        contact_state = (contact or {}).get("contact_active", pose.get("contact_frame", ""))
        static_active = str(phase.get("static_active", "") if phase else pose.get("static_active", ""))
        if not static_active and str((phase or {}).get("phase", "")).lower().startswith("static"):
            static_active = "1"
        row: dict[str, object] = {
            "frame": fr,
            "time": pose.get("time", ""),
            "contact_state": contact_state,
            "visibility_state": (contact or {}).get("vlm_visibility", pose.get("vlm_visibility", "")),
            "vlm_contact_gate": contact_gate_by_frame.get(fr, ""),
            "vlm_anchor_gate": anchor_gate_by_frame.get(fr, ""),
            "contact_active": contact_state,
            "static_active": static_active,
            "failure_label": pose.get("failure_label", ""),
            "source": "non_invasive_loss_audit_from_generic_outputs",
        }
        row.update({key: "" if val is None else f"{val:.9f}" for key, val in values.items()})
        row["E_2d"] = row.get("E_visual", "")
        row["E_depth"] = row.get("E_support", "")
        row["E_temporal"] = row.get("E_smooth", "")
        row["E_static"] = "0.000000000" if static_active in {"1", "1.0", "true", "True"} else ""
        row["E_penetration"] = ""
        row["E_prior"] = row.get("E_reg", "")
        row["E_total"] = f"{sum_terms(row):.9f}"
        out.append(row)
        prev = pose
    return out


def aggregate_trace(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    trace: dict[str, object] = {"iteration": 0}
    for key in ["E_total", "E_visual", "E_mask", "E_contact", "E_audio", "E_support", "E_smooth", "E_reg"]:
        vals = []
        for row in rows:
            try:
                value = row.get(key)
                if value not in {"", None}:
                    vals.append(float(value))  # type: ignore[arg-type]
            except Exception:
                pass
        trace[key] = f"{sum(vals):.9f}" if vals else ""
    trace["active_frames"] = len(rows)
    trace["active_contact_frames"] = sum(1 for r in rows if str(r.get("contact_state", "")) in {"1", "1.0", "true", "True"})
    trace["active_audio_frames"] = sum(1 for r in rows if r.get("E_audio") not in {"", None})
    return [trace]


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _draw_line(pixels: list[list[tuple[int, int, int]]], x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    h = len(pixels)
    w = len(pixels[0]) if h else 0
    while True:
        if 0 <= x < w and 0 <= y < h:
            pixels[y][x] = color
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def write_simple_png(path: Path, series: list[float], width: int = 720, height: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = [[(255, 255, 255) for _ in range(width)] for _ in range(height)]
    for x in range(45, width - 15):
        pixels[height - 30][x] = (190, 190, 190)
    for y in range(15, height - 30):
        pixels[y][45] = (190, 190, 190)
    finite = [v for v in series if math.isfinite(v)]
    if len(finite) >= 2:
        lo, hi = min(finite), max(finite)
        if abs(hi - lo) < 1e-12:
            hi = lo + 1.0
        pts = []
        for i, val in enumerate(series):
            x = 45 + int(round((width - 65) * i / max(1, len(series) - 1)))
            y = height - 30 - int(round((height - 50) * (val - lo) / (hi - lo)))
            pts.append((x, y))
        for a, b in zip(pts, pts[1:]):
            _draw_line(pixels, a[0], a[1], b[0], b[1], (35, 95, 180))
    raw = b"".join(b"\x00" + bytes(channel for pixel in row for channel in pixel) for row in pixels)
    data = b"\x89PNG\r\n\x1a\n"
    data += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    data += _png_chunk(b"IDAT", zlib.compress(raw, 9))
    data += _png_chunk(b"IEND", b"")
    path.write_bytes(data)


def write_plots(out_dir: Path, rows: list[dict[str, object]]) -> list[str]:
    written: list[str] = []
    for key in ["E_total", "E_visual", "E_contact", "E_support", "E_smooth", "E_reg"]:
        series = []
        for row in rows:
            try:
                value = row.get(key)
                if value in {"", None}:
                    series.append(math.nan)
                else:
                    series.append(float(value))  # type: ignore[arg-type]
            except Exception:
                series.append(math.nan)
        finite = [v for v in series if math.isfinite(v)]
        if len(finite) >= 2:
            path = out_dir / f"{key}.png"
            write_simple_png(path, [0.0 if not math.isfinite(v) else v for v in series])
            written.append(str(path))
    return written


def run_loss_analysis(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    out_dir = profile.result_dir / "loss_analysis"
    rows = build_loss_rows(profile)
    trace = aggregate_trace(rows)
    per_frame = write_csv(out_dir / "per_frame_residuals.csv", rows, LOSS_FIELDS)
    stage7_alias = write_csv(profile.result_dir / "stage7_loss_residuals.csv", rows, LOSS_FIELDS)
    loss_trace = write_csv(out_dir / "loss_trace.csv", trace, TRACE_FIELDS)
    plots = write_plots(out_dir, rows)
    summary = {
        "case_name": profile.case_name,
        "mode": "non_invasive_loss_audit",
        "per_frame_residuals": str(per_frame),
        "stage7_loss_residuals": str(stage7_alias),
        "loss_trace": str(loss_trace),
        "plots": plots,
        "active_terms": [key for key in TRACE_FIELDS if key.startswith("E_") and trace and trace[0].get(key) not in {"", None}],
        "note": "This audit computes comparable residual terms from existing generic outputs. It does not replace the optimizer internals yet.",
    }
    write_json(out_dir / "loss_summary.json", summary)
    return summary
