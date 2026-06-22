#!/usr/bin/env python3
"""Prepare an Articraft-compatible mug proxy handoff.

This script does not run Articraft itself. It creates the stable files that
connect an Articraft-generated canonical mug to the radius-free pipeline:

  keyframes/              selected reference frames for clean-image/contact work
  articraft/prompt_*.txt  prompts to run Articraft externally
  proxy/mug_proxy.json    pipeline-owned canonical mug proxy

Stage1/Stage2 should consume mug_proxy.json, not Articraft's internal asset
format.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import struct
import subprocess
from collections import Counter
from pathlib import Path


TEXT_ONLY_PROMPT = """Create a minimal mug geometry proxy. The object must consist of exactly four semantic parts: body, handle, rim, and bottom. The body is a vertical cylinder. The handle is a simple C-shaped or torus-like part attached to one side of the body. The bottom is a flat disk. The rim is a thin ring. This object is for contact reasoning, not photorealistic rendering. Expose clearly named semantic parts: body, handle, rim, bottom."""


IMAGE_CONDITIONED_PROMPT = """Create a simple 3D mug based on this reference image. The asset will be used as a contact-reasoning proxy, not as a photorealistic reconstruction. The mug must expose semantic parts named body, handle, rim, and bottom. The body should be a cylindrical cup body. The handle should be a separate C-shaped side part attached to the body. Prefer robust simple geometry and clearly named parts over visual detail."""


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def parse_float(value: str | None, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_int(value: str | None, default: int = 0) -> int:
    v = parse_float(value, math.nan)
    return default if not math.isfinite(v) else int(round(v))


def choose_frame(rows: list[dict[str, str]], score_key: str, fallback_index: int) -> int:
    best_frame = None
    best_score = -math.inf
    for row in rows:
        score = parse_float(row.get(score_key), -math.inf)
        if score > best_score:
            best_score = score
            best_frame = parse_int(row.get("frame"))
    if best_frame is not None and best_frame > 0:
        return best_frame
    if rows:
        idx = min(max(fallback_index, 0), len(rows) - 1)
        return parse_int(rows[idx].get("frame"), 1)
    return 1


def copy_frame(sample_dir: Path, frame: int, out_path: Path) -> None:
    candidates = [
        sample_dir / "frames" / f"{frame:05d}.png",
        sample_dir / "frames" / f"{frame:05d}.jpg",
        sample_dir / "frames" / f"{frame:05d}.jpeg",
    ]
    for src in candidates:
        if src.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out_path)
            return
    raise FileNotFoundError(f"Could not find frame {frame:05d} under {sample_dir / 'frames'}")


def infer_side(rows: list[dict[str, str]], keys: list[str], default: str = "unknown") -> str:
    votes: list[str] = []
    for row in rows:
        for key in keys:
            value = (row.get(key) or "").strip().lower()
            if value in {"left", "right"}:
                votes.append(value)
    if not votes:
        return default
    return Counter(votes).most_common(1)[0][0]


def robust_body_dimensions(rows: list[dict[str, str]]) -> tuple[float, float]:
    widths = []
    heights = []
    for row in rows:
        w = parse_float(row.get("body_bbox_w_px"))
        h = parse_float(row.get("body_bbox_h_px"))
        if math.isfinite(w) and w > 1.0:
            widths.append(w)
        if math.isfinite(h) and h > 1.0:
            heights.append(h)
    if not widths or not heights:
        return 0.35, 1.0
    median_w = sorted(widths)[len(widths) // 2]
    median_h = sorted(heights)[len(heights) // 2]
    radius = max(0.1, min(0.5, 0.5 * median_w / max(median_h, 1.0)))
    return float(radius), 1.0


def parse_articraft_constants(model_py: Path) -> dict[str, float]:
    if not model_py or not model_py.exists():
        return {}
    constants: dict[str, float] = {}
    pattern = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([-+]?\d+(?:\.\d+)?)\s*$")
    keyword_pattern = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([-+]?\d+(?:\.\d+)?)\s*,?")
    for line in model_py.read_text().splitlines():
        stripped = line.strip()
        match = pattern.match(stripped)
        if match:
            constants[match.group(1)] = float(match.group(2))
            continue
        for match in keyword_pattern.finditer(stripped):
            constants.setdefault(match.group(1), float(match.group(2)))
    return constants


def apply_articraft_geometry(proxy: dict[str, object], model_py: Path | None, record_id: str, observed_side: str) -> dict[str, object]:
    constants = parse_articraft_constants(model_py) if model_py else {}
    if not constants:
        return proxy

    outer_radius = constants.get("OUTER_RADIUS", constants.get("outer_radius", 0.04))
    inner_radius = constants.get("INNER_RADIUS", constants.get("inner_radius", 0.0355))
    bottom_height = constants.get("BOTTOM_HEIGHT", constants.get("bottom_height", 0.006))
    body_height = constants.get("BODY_HEIGHT", constants.get("body_height", 0.095))
    rim_tube = constants.get("rim_tube", constants.get("tube", 0.003))
    rim_height = constants.get("RIM_HEIGHT", 2.0 * rim_tube)
    rim_outer_radius = constants.get("RIM_OUTER_RADIUS", constants.get("radius", outer_radius))
    rim_inner_radius = constants.get("RIM_INNER_RADIUS", inner_radius)
    handle_embed = constants.get("HANDLE_EMBED", 0.006)

    handle_center_z = constants.get("HANDLE_CENTER_Z", constants.get("center_z", 0.055))
    radius_x = constants.get("HANDLE_RADIUS_X", constants.get("radius_x", 0.030))
    radius_z = constants.get("HANDLE_RADIUS_Z", constants.get("radius_z", 0.031))
    tube_radius = constants.get("HANDLE_TUBE_RADIUS", constants.get("tube_radius", 0.0048))
    # Refined Articraft mug uses a swept C-shaped tube. Preserve this canonical
    # semantic handle for contact instead of a frame-varying hand-side guess.
    endpoint_x = outer_radius + max(handle_embed, 0.006)
    grip_x = endpoint_x + radius_x
    upper_z = handle_center_z + radius_z
    lower_z = handle_center_z - radius_z
    grip_height = 2.0 * radius_z + 2.0 * tube_radius
    total_height = max(1e-6, body_height + rim_height)
    body_center_z = 0.5 * total_height
    sign = -1.0 if observed_side == "left" else 1.0

    def nz(z: float) -> float:
        return round((z - body_center_z) / total_height, 6)

    def nx(x: float) -> float:
        return round(sign * x / total_height, 6)

    proxy["source"] = "articraft_canonical_geometry"
    proxy["articraft_role"] = "canonical_geometry_source"
    proxy["articraft_record"] = {
        "record_id": record_id,
        "model_py": str(model_py) if model_py else "",
        "semantic_parts": ["body", "handle", "rim", "bottom"],
        "canonical_handle_side": "right",
        "observed_contact_side": observed_side,
        "mirror_to_observed_side": observed_side == "left",
    }
    proxy["units"] = {
        "articraft_metric": "meters",
        "pipeline_scale": "normalized_by_articraft_total_height",
        "total_height_m": round(total_height, 6),
    }
    proxy["parts"] = {
        "body": {
            "type": "hollow_cylinder",
            "radius": round(outer_radius / total_height, 6),
            "inner_radius": round(inner_radius / total_height, 6),
            "height": round(body_height / total_height, 6),
            "metric_radius_m": outer_radius,
            "metric_inner_radius_m": inner_radius,
            "metric_height_m": body_height,
            "local_center": [0.0, 0.0, nz(0.5 * body_height)],
            "source_part": "body",
            "source_mesh": "body_shell.obj",
        },
        "handle": {
            "type": "smooth_c_shaped_tube_handle",
            "side": observed_side,
            "canonical_side": "right",
            "local_center": [nx(grip_x), 0.0, nz(handle_center_z)],
            "metric_local_center_m": [sign * grip_x, 0.0, round(handle_center_z - body_center_z, 6)],
            "visual_part": True,
            "source_part": "handle",
            "elements": {
                "swept_loop": {
                    "center_m": [round(sign * grip_x, 6), 0.0, round(handle_center_z - body_center_z, 6)],
                    "radius_x_m": round(radius_x, 6),
                    "radius_z_m": round(radius_z, 6),
                    "tube_radius_m": round(tube_radius, 6),
                    "vertical_span_m": round(grip_height, 6),
                },
                "attachment_zone": {
                    "body_side_x_m": round(sign * outer_radius, 6),
                    "endpoint_x_m": round(sign * endpoint_x, 6),
                    "upper_z_m": round(upper_z - body_center_z, 6),
                    "lower_z_m": round(lower_z - body_center_z, 6),
                },
            },
        },
        "rim": {
            "type": "ring",
            "radius": round(rim_outer_radius / total_height, 6),
            "inner_radius": round(rim_inner_radius / total_height, 6),
            "height": round(rim_height / total_height, 6),
            "local_center": [0.0, 0.0, nz(body_height + 0.5 * rim_height)],
            "source_part": "rim",
            "source_mesh": "rim_lip.obj",
        },
        "bottom": {
            "type": "disk",
            "radius": round(outer_radius / total_height, 6),
            "height": round(bottom_height / total_height, 6),
            "local_center": [0.0, 0.0, nz(0.5 * bottom_height)],
            "support_region": True,
            "source_part": "bottom",
            "source_mesh": "bottom_disk.obj",
        },
    }
    proxy["contact_region"] = {
        "type": "handle",
        "part_name": "handle",
        "side": observed_side,
        "source": "articraft_handle_part_mirrored_to_observed_contact_side",
        "confidence": 1.0,
    }
    return proxy


def build_mug_proxy(rows: list[dict[str, str]]) -> dict[str, object]:
    handle_side = infer_side(rows, ["contact_region_side", "hand_contact_side", "latent_handle_side", "visual_handle_side", "handle_side"], "unknown")
    body_radius, body_height = robust_body_dimensions(rows)
    side_sign = -1.0 if handle_side == "left" else 1.0
    if handle_side == "unknown":
        side_sign = 1.0

    return {
        "object_type": "mug",
        "geometry_type": "cylinder_with_handle",
        "ref_proxy": "body_center",
        "scale_mode": "normalized",
        "source": "articraft_or_manual_canonical_proxy",
        "articraft_role": "optional_canonical_geometry_generator",
        "parts": {
            "body": {
                "type": "cylinder",
                "radius": round(body_radius, 6),
                "height": body_height,
                "local_center": [0.0, 0.0, 0.0],
            },
            "handle": {
                "type": "side_handle",
                "side": handle_side,
                "local_center": [round(side_sign * (body_radius + 0.12), 6), 0.0, 0.05],
                "visual_part": True,
            },
            "rim": {
                "type": "ring",
                "local_center": [0.0, 0.0, 0.5 * body_height],
            },
            "bottom": {
                "type": "disk",
                "local_center": [0.0, 0.0, -0.5 * body_height],
                "support_region": True,
            },
        },
        "visual_handle": {
            "type": "rigid_handle_geometry",
            "part_name": "handle",
            "side": handle_side,
            "source": "articraft_handle_geometry",
            "confidence": 1.0 if handle_side in {"left", "right"} else 0.0,
        },
        "contact_region": {
            "type": "object_surface_contact_region",
            "part_name": "",
            "allowed_parts": ["handle", "body", "rim"],
            "side": handle_side,
            "source": "hand_object_contact_region_not_visual_handle",
            "confidence": 1.0 if handle_side in {"left", "right"} else 0.0,
            "note": "Object-side surface touched by the hand; not necessarily the visual handle.",
        },
        "stage_policy": {
            "stage1_ref": "track body_center only",
            "stage2_contact": "anchor hand to contact_region, not mug center",
        },
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


def get_row_for_frame(rows: list[dict[str, str]], frame: int) -> dict[str, str]:
    if not rows:
        return {}
    return min(rows, key=lambda row: abs(parse_int(row.get("frame"), frame) - frame))


def row_bbox(row: dict[str, str]) -> tuple[int, int, int, int] | None:
    x1 = parse_float(row.get("body_bbox_x1"))
    y1 = parse_float(row.get("body_bbox_y1"))
    x2 = parse_float(row.get("body_bbox_x2"))
    y2 = parse_float(row.get("body_bbox_y2"))
    if not all(math.isfinite(v) for v in [x1, y1, x2, y2]) or x2 <= x1 or y2 <= y1:
        x1 = parse_float(row.get("bbox_x1"))
        y1 = parse_float(row.get("bbox_y1"))
        x2 = parse_float(row.get("bbox_x2"))
        y2 = parse_float(row.get("bbox_y2"))
    if not all(math.isfinite(v) for v in [x1, y1, x2, y2]) or x2 <= x1 or y2 <= y1:
        cx = parse_float(row.get("body_center_x"), parse_float(row.get("center_x")))
        cy = parse_float(row.get("body_center_y"), parse_float(row.get("center_y")))
        w = parse_float(row.get("body_bbox_w_px"), parse_float(row.get("bbox_w_px"), 40.0))
        h = parse_float(row.get("body_bbox_h_px"), parse_float(row.get("bbox_h_px"), 50.0))
        if not all(math.isfinite(v) for v in [cx, cy, w, h]):
            return None
        x1, x2 = cx - 0.5 * w, cx + 0.5 * w
        y1, y2 = cy - 0.5 * h, cy + 0.5 * h
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        header = f.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")
    return struct.unpack(">II", header[16:24])


def write_contact_region_mask(sample_dir: Path, rows: list[dict[str, str]], frame: int, out_dir: Path) -> None:
    frame_path = sample_dir / "frames" / f"{frame:05d}.png"
    if not frame_path.exists():
        write_text(out_dir / "001_contact_region_mask_unavailable.txt", f"Missing frame image: {frame_path}")
        return

    row = get_row_for_frame(rows, frame)
    bbox = row_bbox(row)
    if bbox is None:
        write_text(out_dir / "001_contact_region_mask_unavailable.txt", f"Missing body/object bbox for frame {frame}.")
        return

    w, h = png_size(frame_path)
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    side = infer_side([row], ["contact_region_side", "hand_contact_side", "latent_handle_side", "visual_handle_side", "handle_side"], "left")

    obj_w = max(1, x2 - x1)
    obj_h = max(1, y2 - y1)
    band_w = max(6, int(round(0.30 * obj_w)))
    pad = max(4, int(round(0.10 * obj_w)))
    cy = int(round((y1 + y2) * 0.5))
    handle_h = max(8, int(round(0.55 * obj_h)))
    hy1 = max(0, cy - handle_h // 2)
    hy2 = min(h - 1, cy + handle_h // 2)
    if side == "right":
        hx1 = min(w - 1, x2 - pad)
        hx2 = min(w - 1, x2 + band_w)
    else:
        hx1 = max(0, x1 - band_w)
        hx2 = max(0, x1 + pad)
    box_w = max(1, hx2 - hx1)
    box_h = max(1, hy2 - hy1)

    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / "001_contact_region_mask.png"
    preview_path = out_dir / "001_contact_region_preview.png"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"color=black:s={w}x{h}",
            "-vf", f"drawbox=x={hx1}:y={hy1}:w={box_w}:h={box_h}:color=white:t=fill",
            "-frames:v", "1", str(mask_path),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(frame_path),
            "-vf", f"drawbox=x={hx1}:y={hy1}:w={box_w}:h={box_h}:color=red@0.45:t=fill,drawbox=x={hx1}:y={hy1}:w={box_w}:h={box_h}:color=red:t=2",
            "-frames:v", "1", str(preview_path),
        ],
        check=True,
    )
    unavailable = out_dir / "001_contact_region_mask_unavailable.txt"
    if unavailable.exists():
        unavailable.unlink()
    write_text(
        out_dir / "001_contact_region_mask.json",
        json.dumps(
            {
                "frame": frame,
                "source": "articraft_handle_part_mirrored_to_observed_contact_side",
                "side": side,
                "bbox_xyxy": [hx1, hy1, hx2, hy2],
                "note": "2D contact-region preview derived from observed hand-contact side. This region is not necessarily the visual handle; drinking frames may contact the body/rim while the handle is occluded.",
            },
            indent=2,
        ),
    )


def export_stage_aliases(sample_dir: Path, rows: list[dict[str, str]]) -> None:
    results_dir = sample_dir / "results"
    stage1_path = results_dir / "stage1_mug_body_trajectory.csv"
    stage1_path.parent.mkdir(parents=True, exist_ok=True)
    with stage1_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "time", "body_center_u", "body_center_v", "contact_region_side", "source"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "frame": row.get("frame", ""),
                    "time": row.get("time", ""),
                    "body_center_u": row.get("body_center_x", row.get("center_x", "")),
                    "body_center_v": row.get("body_center_y", row.get("center_y", "")),
                    "contact_region_side": row.get("contact_region_side", ""),
                    "source": "object_observations_body_center",
                }
            )

    src = results_dir / "contact_candidates_object_proxy" / "contact_candidates_labeled.csv"
    dst = results_dir / "stage2_mug_contact_test.csv"
    if src.exists():
        shutil.copy2(src, dst)


def write_articraft_record(sample_dir: Path, proxy: dict[str, object], frame: int) -> None:
    record_dir = sample_dir / "articraft" / "generated_record"
    record_dir.mkdir(parents=True, exist_ok=True)
    record_id = str(proxy.get("articraft_record", {}).get("record_id", "") if isinstance(proxy.get("articraft_record"), dict) else "")
    if record_id:
        readme = (
            "This folder records the Articraft mug proxy handoff. Articraft was run with "
            "the Codex CLI provider, and the generated record is copied under this directory. "
            "The downstream radius-free pipeline still reads ../../proxy/mug_proxy.json as the stable interface.\n\n"
            f"record_id: {record_id}"
        )
    else:
        readme = (
            "This folder records the Articraft handoff. Articraft has not been run yet; "
            "the current canonical proxy is stored in ../../proxy/mug_proxy.json."
        )
    write_text(record_dir / "README.md", readme)
    write_text(record_dir / "mug_proxy_record.json", json.dumps({"keyframe": frame, "proxy": proxy}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Articraft mug proxy handoff files.")
    parser.add_argument("--sample-dir", type=Path, default=Path("samples_known_object/02_mug"))
    parser.add_argument("--object-observations", type=Path, default=None)
    parser.add_argument("--copy-keyframes", action="store_true")
    parser.add_argument("--articraft-model", type=Path, default=None)
    parser.add_argument("--articraft-record-id", default="")
    args = parser.parse_args()

    sample_dir = args.sample_dir
    obs_csv = args.object_observations or (sample_dir / "results" / "object_observations" / "object_observations.csv")
    rows = read_rows(obs_csv)

    keyframe_dir = sample_dir / "keyframes"
    articraft_dir = sample_dir / "articraft"
    proxy_dir = sample_dir / "proxy"
    annotations_dir = sample_dir / "annotations"
    for directory in [keyframe_dir, articraft_dir, proxy_dir, annotations_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    if args.copy_keyframes:
        hand_frame = choose_frame(rows, "hand_contact_conf", 0)
        clean_frame = choose_frame(rows, "handle_conf", 0)
        mid_frame = parse_int(rows[len(rows) // 2].get("frame"), clean_frame) if rows else clean_frame
        putdown_frame = parse_int(rows[-1].get("frame"), clean_frame) if rows else clean_frame
        copy_frame(sample_dir, hand_frame, keyframe_dir / "001_hand_holding.png")
        copy_frame(sample_dir, clean_frame, keyframe_dir / "001_mug_clean_reference.png")
        copy_frame(sample_dir, mid_frame, keyframe_dir / "002_contact_mid.png")
        copy_frame(sample_dir, putdown_frame, keyframe_dir / "003_putdown.png")

    write_text(articraft_dir / "prompt_mug_proxy_text_only.txt", TEXT_ONLY_PROMPT)
    write_text(articraft_dir / "prompt_mug_proxy_image_conditioned.txt", IMAGE_CONDITIONED_PROMPT)
    write_text(
        articraft_dir / "README.md",
        "Run Articraft externally and convert/abstract its result into ../proxy/mug_proxy.json. "
        "The radius-free pipeline reads mug_proxy.json. Visual handle geometry and hand contact region are separate semantics.",
    )

    proxy = build_mug_proxy(rows)
    observed_side = infer_side(rows, ["contact_region_side", "hand_contact_side", "latent_handle_side", "visual_handle_side", "handle_side"], "unknown")
    if args.articraft_model:
        proxy = apply_articraft_geometry(proxy, args.articraft_model, args.articraft_record_id, observed_side)
    proxy_path = proxy_dir / "mug_proxy.json"
    proxy_path.write_text(json.dumps(proxy, indent=2) + "\n")
    hand_frame = choose_frame(rows, "hand_contact_conf", 0)
    write_contact_region_mask(sample_dir, rows, hand_frame, annotations_dir)
    write_articraft_record(sample_dir, proxy, hand_frame)
    export_stage_aliases(sample_dir, rows)

    print(f"mug_proxy_json: {proxy_path}")
    print(f"articraft_prompt_text_only: {articraft_dir / 'prompt_mug_proxy_text_only.txt'}")
    print(f"articraft_prompt_image_conditioned: {articraft_dir / 'prompt_mug_proxy_image_conditioned.txt'}")
    print(f"annotations_dir: {annotations_dir}")
    print(f"stage1_alias: {sample_dir / 'results' / 'stage1_mug_body_trajectory.csv'}")
    print(f"stage2_alias: {sample_dir / 'results' / 'stage2_mug_contact_test.csv'}")


if __name__ == "__main__":
    main()
