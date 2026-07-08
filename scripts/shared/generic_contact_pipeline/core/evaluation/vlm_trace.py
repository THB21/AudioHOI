from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import zlib
from array import array
from ast import literal_eval
from pathlib import Path
from typing import Any

from ..base.config import CaseProfile
from ..base.io import read_csv, write_csv, write_json
from ..base.schema import stage_paths


TRACE_DIRS = [
    "00_input",
    "01_observation",
    "02_contact_candidates",
    "03_vlm_reasoning",
    "04_gating",
    "05_optimization",
    "06_evaluation",
]

TRACE_REQUIRED_ARTIFACTS = [
    "input_video",
    "sampled_frames",
    "object_observations",
    "object_correspondence",
    "object_surface_points",
    "object_semantic_points",
    "contact_candidates",
    "object_mask_overlay",
    "human_pose_overlay",
    "depth_overlay",
    "contact_candidate_overlay",
    "before_after_overlay",
    "gate_timeline_png",
]


def _safe_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        return read_csv(path)
    except Exception:
        return []


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _artifact_record(path: Path, *, stage: str, target: Path | None = None) -> dict[str, object]:
    suffix = path.suffix.lower()
    rows = len(_safe_rows(path)) if suffix == ".csv" and path.exists() else None
    return {
        "stage": stage,
        "source_path": str(path),
        "target_path": str(target) if target else "",
        "exists": path.exists(),
        "rows": rows if rows is not None else "",
    }


def _materialize(src: Path, dst: Path) -> dict[str, object]:
    record = _artifact_record(src, stage="", target=dst)
    if not src.exists():
        record["storage_mode"] = "missing"
        return record
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        rel = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel)
        record["storage_mode"] = "relative_symlink"
    except Exception:
        shutil.copy2(src, dst)
        record["storage_mode"] = "copy"
    return record


def _first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def _materialize_optional(src: Path | None, dst: Path, *, stage: str) -> dict[str, object]:
    if src is None:
        return {
            "stage": stage,
            "source_path": "",
            "target_path": str(dst),
            "exists": False,
            "rows": "",
            "storage_mode": "missing",
        }
    if src.resolve() == dst.resolve():
        record = _artifact_record(src, stage=stage, target=dst)
        record["storage_mode"] = "generated"
        return record
    record = _materialize(src, dst)
    record["stage"] = stage
    return record


def _write_png(path: Path, pixels: list[list[tuple[int, int, int]]]) -> Path:
    height = len(pixels)
    width = len(pixels[0]) if height else 1
    raw = b"".join(b"\x00" + bytes(channel for pixel in row for channel in pixel) for row in pixels)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    return path


def _write_gate_timeline_png(path: Path, rows: list[dict[str, object]]) -> Path:
    width = max(64, min(1024, len(rows) * 8 if rows else 64))
    height = 48
    pixels = [[(248, 248, 248) for _ in range(width)] for _ in range(height)]
    colors = {
        "pass": (39, 174, 96),
        "unclear": (241, 196, 15),
        "reject": (192, 57, 43),
        "not_evaluated": (149, 165, 166),
        "": (189, 195, 199),
    }
    if rows:
        bar_w = max(1, width // len(rows))
        for idx, row in enumerate(rows):
            color = colors.get(str(row.get("source_gate", "")), (127, 140, 141))
            x0 = idx * bar_w
            x1 = width if idx == len(rows) - 1 else min(width, (idx + 1) * bar_w)
            for y in range(8, height - 8):
                for x in range(x0, x1):
                    pixels[y][x] = color
    return _write_png(path, pixels)


def _read_npy_2d(path: Path) -> tuple[int, int, list[float]] | None:
    with path.open("rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            return None
        major = f.read(1)[0]
        f.read(1)
        header_len_size = 2 if major == 1 else 4
        header_len = int.from_bytes(f.read(header_len_size), "little")
        header = literal_eval(f.read(header_len).decode("latin1").strip())
        shape = tuple(header.get("shape", ()))
        if len(shape) != 2:
            return None
        height, width = int(shape[0]), int(shape[1])
        descr = str(header.get("descr", ""))
        payload = f.read()
    code = {"<f4": "f", "|f4": "f", "<f8": "d", "|f8": "d", "<u2": "H", "|u2": "H", "<u1": "B", "|u1": "B"}.get(descr)
    if code is None:
        return None
    arr = array(code)
    arr.frombytes(payload)
    if descr.startswith(">"):
        arr.byteswap()
    values = [float(v) for v in arr[: height * width]]
    return width, height, values


def _write_depth_pgm(src: Path, dst: Path) -> bool:
    parsed = _read_npy_2d(src)
    if parsed is None:
        return False
    width, height, values = parsed
    finite = [v for v in values if v == v and abs(v) != float("inf")]
    if not finite:
        return False
    lo = min(finite)
    hi = max(finite)
    if hi <= lo:
        hi = lo + 1.0
    scale = 255.0 / (hi - lo)
    pixels = bytearray()
    for value in values:
        if value != value:
            pixels.append(0)
        else:
            pixels.append(max(0, min(255, int(round((value - lo) * scale)))))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(f"P5\n{width} {height}\n255\n".encode("ascii") + bytes(pixels))
    return True


def _generate_depth_overlay_from_npy(profile: CaseProfile, dst: Path) -> Path | None:
    scene_dir = profile.sample_dir / "results" / "da3" / "scene_depth"
    npy_files = sorted(path for path in scene_dir.glob("*.npy") if path.stem.isdigit())
    if not npy_files:
        return None
    frames_dir = dst.parent / "depth_overlay_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for idx, src in enumerate(npy_files, start=1):
        if _write_depth_pgm(src, frames_dir / f"{idx:05d}.pgm"):
            written += 1
    if written == 0:
        return None
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        "24",
        "-i",
        str(frames_dir / "%05d.pgm"),
        "-pix_fmt",
        "yuv420p",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        return None
    return dst if dst.exists() else None


def _find_input_video(profile: CaseProfile) -> Path | None:
    for name in ("video.mp4", f"{profile.case_name}.mp4"):
        candidate = profile.sample_dir / name
        if candidate.exists():
            return candidate
    metadata = profile.sample_dir / "metadata.json"
    data = _safe_json(metadata)
    for key in ("video", "video_path", "source_video"):
        value = data.get(key)
        if value:
            candidate = Path(str(value))
            if not candidate.is_absolute():
                candidate = profile.sample_dir / candidate
            if candidate.exists():
                return candidate
    return None


def _summarize_csv(path: Path) -> dict[str, object]:
    rows = _safe_rows(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "rows": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
        "sample": rows[:3],
    }


def _contact_intervals(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    intervals: list[dict[str, object]] = []
    start: int | None = None
    prev: int | None = None
    part = ""
    confs: list[float] = []
    for row in rows:
        try:
            frame = int(float(row.get("frame", "0") or 0))
        except Exception:
            continue
        active = row.get("contact_observed", row.get("contact_active", "0")) in {"1", "1.0", "true", "True"}
        if active and start is None:
            start = frame
            part = row.get("contact_part", row.get("human_part", ""))
            confs = []
        if active:
            prev = frame
            try:
                confs.append(float(row.get("contact_confidence", "0") or 0))
            except Exception:
                pass
        if not active and start is not None:
            intervals.append(
                {
                    "start_frame": start,
                    "end_frame": prev if prev is not None else start,
                    "candidate_type": part,
                    "confidence": sum(confs) / len(confs) if confs else "",
                    "source": "contact_candidates",
                }
            )
            start = None
            prev = None
            confs = []
    if start is not None:
        intervals.append(
            {
                "start_frame": start,
                "end_frame": prev if prev is not None else start,
                "candidate_type": part,
                "confidence": sum(confs) / len(confs) if confs else "",
                "source": "contact_candidates",
            }
        )
    return intervals


def _aggregate_vlm_rows(profile: CaseProfile, filename: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    vlm_dir = stage_paths(profile)["vlm_dir"]
    for stage_dir in sorted(vlm_dir.glob("stage*")):
        path = stage_dir / filename
        for row in _safe_rows(path):
            if "stage" not in row:
                row = {"stage": stage_dir.name, **row}
            out.append(row)
    path = vlm_dir / filename
    if path.exists() and not out:
        out.extend(_safe_rows(path))
    return out


def _gate_timeline(gates: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in gates:
        rows.append(
            {
                "frame": row.get("frame", ""),
                "stage": row.get("stage", ""),
                "constraint": row.get("query_type", row.get("check_type", "")),
                "active": row.get("is_effective", row.get("allow_pose_refine", "")),
                "weight": row.get("residual_weight", ""),
                "source_gate": row.get("pass_gate", ""),
                "reason": row.get("reason", row.get("normalized_label", "")),
            }
        )
    return rows


def export_vlm_trace(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    trace_root = profile.result_dir / "vlm_trace"
    dirs = {name: trace_root / name for name in TRACE_DIRS}
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, dict[str, object]] = {}

    input_video = _find_input_video(profile)
    if input_video:
        artifacts["input_video"] = _materialize(input_video, dirs["00_input"] / "video.mp4")
    else:
        artifacts["input_video"] = {"stage": "00_input", "source_path": "", "target_path": str(dirs["00_input"] / "video.mp4"), "exists": False, "storage_mode": "missing"}
    write_json(
        dirs["00_input"] / "target_object.json",
        {
            "case_name": profile.case_name,
            "object_family": profile.data.get("object_family", ""),
            "result_name": profile.result_name,
        },
    )
    sampled_dir = dirs["00_input"] / "sampled_frames"
    sampled_dir.mkdir(parents=True, exist_ok=True)
    sampled_sources = sorted((profile.sample_dir / "results" / "segmentation" / "sam2_jpg_frames").glob("*.jpg"))[:8]
    if not sampled_sources:
        sampled_sources = sorted(paths["vlm_dir"].glob("stage*/evidence/*.*"))[:8]
    sampled_records = []
    for src in sampled_sources:
        sampled_records.append(_materialize(src, sampled_dir / src.name))
    artifacts["sampled_frames"] = {
        "stage": "00_input",
        "source_path": str(sampled_sources[0].parent) if sampled_sources else "",
        "target_path": str(sampled_dir),
        "exists": bool(sampled_sources),
        "rows": len(sampled_sources),
        "storage_mode": "relative_symlink_or_copy",
    }
    write_json(dirs["00_input"] / "sampled_frames_manifest.json", {"frames": sampled_records})

    depth_overlay_target = dirs["01_observation"] / "depth_overlay.mp4"
    depth_overlay_source = _first_existing([profile.sample_dir / "results" / "da3" / "depth_overlay.mp4", profile.render_dir / "depth_overlay.mp4"])
    if depth_overlay_source is None:
        depth_overlay_source = _generate_depth_overlay_from_npy(profile, depth_overlay_target)
    visual_slots = {
        "object_mask_overlay": (
            dirs["01_observation"] / "object_mask_overlay.mp4",
            _first_existing(
                [
                    profile.render_dir / "with_human" / "overlay_vlm_masked.mp4",
                    profile.render_dir / "object_only" / "overlay.mp4",
                    profile.render_dir / "with_human" / "overlay.mp4",
                ]
            ),
            "01_observation",
        ),
        "human_pose_overlay": (
            dirs["01_observation"] / "human_pose_overlay.mp4",
            _first_existing([profile.render_dir / "with_human" / "overlay.mp4"]),
            "01_observation",
        ),
        "depth_overlay": (
            depth_overlay_target,
            depth_overlay_source,
            "01_observation",
        ),
        "contact_candidate_overlay": (
            dirs["02_contact_candidates"] / "contact_candidate_overlay.mp4",
            _first_existing([profile.render_dir / "with_human" / "contact_overlay.mp4", profile.render_dir / "with_human" / "overlay.mp4"]),
            "02_contact_candidates",
        ),
        "before_after_overlay": (
            dirs["05_optimization"] / "before_after_overlay.mp4",
            _first_existing([profile.render_dir / "with_human" / "overlay.mp4", profile.render_dir / "object_only" / "overlay.mp4"]),
            "05_optimization",
        ),
    }
    for name, (target, source, stage) in visual_slots.items():
        artifacts[name] = _materialize_optional(source, target, stage=stage)

    observation_sources = {
        "object_observations": paths["object_observations"],
        "object_correspondence": paths["object_correspondence"],
        "object_surface_points": paths["object_surface_points"],
        "object_semantic_points": paths["object_semantic_points"],
        "line_correspondence": paths["line_correspondence"],
        "line_observations": paths["line_observations"],
    }
    for name, source in observation_sources.items():
        target = dirs["01_observation"] / source.name
        artifacts[name] = _materialize(source, target)
        artifacts[name]["stage"] = "01_observation"
    write_json(dirs["01_observation"] / "observation_summary.json", {name: _summarize_csv(path) for name, path in observation_sources.items()})

    contacts = _safe_rows(paths["contact_candidates"])
    artifacts["contact_candidates"] = _materialize(paths["contact_candidates"], dirs["02_contact_candidates"] / "contact_candidates.csv")
    artifacts["contact_candidates"]["stage"] = "02_contact_candidates"
    intervals = _contact_intervals(contacts)
    write_csv(dirs["02_contact_candidates"] / "contact_intervals.csv", intervals, ["start_frame", "end_frame", "candidate_type", "confidence", "source"])

    queries = _aggregate_vlm_rows(profile, "vlm_queries.csv")
    results = _aggregate_vlm_rows(profile, "vlm_results.csv")
    gates = _aggregate_vlm_rows(profile, "vlm_gates.csv")
    write_csv(dirs["03_vlm_reasoning"] / "framewise_reasoning.csv", results)
    prompt_text = "\n".join(str(row.get("question", row.get("query_type", ""))) for row in queries)
    (dirs["03_vlm_reasoning"] / "prompt.txt").write_text(prompt_text + ("\n" if prompt_text else ""))
    raw_text = "\n".join(str(row.get("raw_response", row.get("normalized_label", ""))) for row in results)
    (dirs["03_vlm_reasoning"] / "raw_response.txt").write_text(raw_text + ("\n" if raw_text else ""))
    write_json(dirs["03_vlm_reasoning"] / "parsed_response.json", {"rows": len(results), "labels": sorted({row.get("normalized_label", "") for row in results if row.get("normalized_label")})})

    gate_rows = _gate_timeline(gates)
    write_csv(dirs["04_gating"] / "gate_timeline.csv", gate_rows, ["frame", "stage", "constraint", "active", "weight", "source_gate", "reason"])
    gate_png = _write_gate_timeline_png(dirs["04_gating"] / "gate_timeline.png", gate_rows)
    artifacts["gate_timeline_png"] = _artifact_record(gate_png, stage="04_gating", target=gate_png)
    active: dict[str, dict[str, object]] = {}
    for row in gate_rows:
        key = str(row.get("constraint", ""))
        if key:
            active.setdefault(key, {"frames": [], "pass_gates": set(), "active_count": 0})
            if row.get("frame") not in {"", None}:
                active[key]["frames"].append(row.get("frame"))  # type: ignore[index,union-attr]
            if row.get("source_gate"):
                active[key]["pass_gates"].add(row.get("source_gate"))  # type: ignore[union-attr]
            if str(row.get("active")) in {"1", "1.0", "true", "True"}:
                active[key]["active_count"] = int(active[key]["active_count"]) + 1
    write_json(
        dirs["04_gating"] / "activated_constraints.json",
        {key: {**value, "pass_gates": sorted(value["pass_gates"])} for key, value in active.items()},
    )

    opt_inputs = {
        "object_pose_init": _summarize_csv(paths["object_pose_init"]),
        "anchor_state": _summarize_csv(paths["anchor_state"]),
        "optimizer_decisions": _summarize_csv(paths["optimizer_decisions"]),
    }
    opt_outputs = {
        "object_pose_pre_smooth": _summarize_csv(paths["object_pose_pre_smooth"]),
        "object_pose": _summarize_csv(paths["object_pose"]),
        "physical_smooth_residuals": _summarize_csv(paths["physical_smooth_residuals"]),
        "pose_jump_audit": _summarize_csv(paths["pose_jump_audit"]),
    }
    write_json(dirs["05_optimization"] / "optimization_input.json", opt_inputs)
    write_json(dirs["05_optimization"] / "optimization_output.json", opt_outputs)
    (dirs["05_optimization"] / "optimization_log.txt").write_text(
        "Generated from existing generic pipeline artifacts; VLM/LLM gates are audit constraints, not direct pose outputs.\n"
    )
    write_json(dirs["05_optimization"] / "failure_flags.json", {"status": "pending_final_evaluator"})

    write_json(dirs["06_evaluation"] / "evaluation_summary.json", {"status": "pending_final_evaluator"})

    manifest = {
        "case_name": profile.case_name,
        "result_name": profile.result_name,
        "trace_root": str(trace_root),
        "directories": {name: str(path) for name, path in dirs.items()},
        "artifacts": artifacts,
    }
    completeness = {
        name: {
            "required": True,
            "exists": bool(artifacts.get(name, {}).get("exists")),
            "path": artifacts.get(name, {}).get("target_path", ""),
            "stage": artifacts.get(name, {}).get("stage", ""),
        }
        for name in TRACE_REQUIRED_ARTIFACTS
    }
    missing_required = [name for name, item in completeness.items() if not item["exists"]]
    completeness_summary = {
        "required_count": len(TRACE_REQUIRED_ARTIFACTS),
        "present_count": len(TRACE_REQUIRED_ARTIFACTS) - len(missing_required),
        "missing_required": missing_required,
        "items": completeness,
    }
    write_json(trace_root / "trace_completeness.json", completeness_summary)
    manifest["trace_completeness"] = completeness_summary
    write_json(trace_root / "trace_manifest.json", manifest)
    return manifest
