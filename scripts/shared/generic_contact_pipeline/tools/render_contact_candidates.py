#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[4]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.base.config import load_case_profile, with_runtime_overrides  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.base.io import read_csv, write_csv, write_json  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.base.schema import stage_paths  # noqa: E402


COLORS = {
    "left": (75, 235, 90),
    "right": (95, 220, 255),
    "suppressed": (80, 80, 230),
    "visible": (0, 165, 255),
    "physical": (255, 170, 40),
    "text": (245, 245, 245),
    "shadow": (15, 15, 15),
}


def _frame(row: dict[str, str]) -> int:
    return int(float(row.get("frame", "0") or 0))


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value in {"", None} else float(value)
    except Exception:
        return default


def frame_path(sample_dir: Path, frame: int) -> Path:
    for name in (f"{frame:05d}.png", f"{frame:06d}.png", f"{frame:04d}.png", f"{frame:03d}.png"):
        path = sample_dir / "frames" / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No frame image for frame {frame} under {sample_dir / 'frames'}")


def put_label(img, text: str, xy: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = xy
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, COLORS["shadow"], 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 1, cv2.LINE_AA)


def line(img, row: dict[str, str], prefix: str, color: tuple[int, int, int], label: str, thickness: int = 3) -> None:
    x1 = _f(row, f"{prefix}_x1")
    y1 = _f(row, f"{prefix}_y1")
    x2 = _f(row, f"{prefix}_x2")
    y2 = _f(row, f"{prefix}_y2")
    cv2.line(img, (round(x1), round(y1)), (round(x2), round(y2)), color, thickness, cv2.LINE_AA)
    put_label(img, label, (round(min(x1, x2)), max(24, round(min(y1, y2)) - 8)), color)


def write_video(frames: list[Path], out_path: Path, fps: float) -> None:
    if not frames:
        return
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    tmp = out_path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {tmp}")
    for path in frames:
        writer.write(cv2.imread(str(path)))
    writer.release()
    ffmpeg = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    subprocess.run([ffmpeg, "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)], check=True, capture_output=True, text=True)
    tmp.unlink(missing_ok=True)


def build_labeled_rows(contact_rows: list[dict[str, str]], anchor_rows: list[dict[str, str]], line_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    anchor_by_key = {(_frame(row), row.get("human_side", "")): row for row in anchor_rows}
    line_by_frame = {_frame(row): row for row in line_rows}
    out: list[dict[str, object]] = []
    for row in contact_rows:
        fr = _frame(row)
        side = row.get("human_side", "")
        anchor = anchor_by_key.get((fr, side), {})
        line_state = line_by_frame.get(fr, {})
        contact_observed = anchor.get("contact_observed", row.get("contact_active", ""))
        pose_anchor_allowed = anchor.get("pose_anchor_allowed", "")
        update_allowed = anchor.get("anchor_update_allowed", "")
        disamb = anchor.get("contact_disambiguation", "")
        label = (
            "pose_anchor"
            if pose_anchor_allowed == "1"
            else "update_only"
            if update_allowed == "1"
            else "suppressed"
            if contact_observed == "0" and row.get("contact_active") == "1"
            else "candidate_only"
            if row.get("contact_active") == "1"
            else "inactive"
        )
        out.append(
            {
                "frame": fr,
                "time": row.get("time", ""),
                "human_side": side,
                "contact_label": label,
                "raw_contact_active": row.get("contact_active", ""),
                "contact_observed": contact_observed,
                "contact_persistent": anchor.get("contact_persistent", ""),
                "pose_anchor_allowed": pose_anchor_allowed,
                "anchor_update_allowed": update_allowed,
                "anchor_action": anchor.get("anchor_action", ""),
                "contact_disambiguation": disamb,
                "line_observation_trusted": line_state.get("line_observation_trusted", ""),
                "occlusion_state": line_state.get("occlusion_state", ""),
                "palm_u": row.get("palm_u", ""),
                "palm_v": row.get("palm_v", ""),
                "contact_u": row.get("contact_u", ""),
                "contact_v": row.get("contact_v", ""),
                "object_local_s": row.get("object_local_s", ""),
                "stable_object_local_s": row.get("stable_object_local_s", ""),
                "local_s_drift": row.get("local_s_drift", ""),
                "palm_to_line_px": row.get("palm_to_line_px", ""),
                "hand_joint_median_line_px": row.get("hand_joint_median_line_px", ""),
                "fingertip_median_line_px": row.get("fingertip_median_line_px", ""),
                "hand_joint_near_count_25px": row.get("hand_joint_near_count_25px", ""),
                "contact_conf": row.get("contact_conf", ""),
                "anchor_z_m": row.get("anchor_z_m", anchor.get("anchor_z_m", "")),
            }
        )
    return out


def render(profile, out_dir: Path, *, fps: float = 24.0) -> dict[str, object]:
    paths = stage_paths(profile)
    contact_rows = read_csv(paths["contact_candidates"])
    anchor_rows = read_csv(paths["anchor_state"]) if paths["anchor_state"].exists() else []
    line_rows = read_csv(paths["line_correspondence"]) if paths["line_correspondence"].exists() else []
    labeled_rows = build_labeled_rows(contact_rows, anchor_rows, line_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled_csv = write_csv(out_dir / "contact_candidates_labeled.csv", labeled_rows)

    by_frame: dict[int, list[dict[str, object]]] = {}
    for row in labeled_rows:
        by_frame.setdefault(int(row["frame"]), []).append(row)
    line_by_frame = {_frame(row): row for row in line_rows}
    frame_paths: list[Path] = []
    preview_frames = {1, 21, 22, 55, 56, 62, 80, 180}
    for fr in sorted(by_frame):
        img = cv2.imread(str(frame_path(profile.sample_dir, fr)), cv2.IMREAD_COLOR)
        if img is None:
            continue
        line_row = line_by_frame.get(fr)
        if line_row:
            line(img, line_row, "visible", COLORS["visible"], f"visible trusted={line_row.get('line_observation_trusted')} {line_row.get('occlusion_state')}", 2)
            line(img, line_row, "physical", COLORS["physical"], "physical endpoints", 1)
        put_label(img, f"frame {fr:03d}", (18, 32), COLORS["text"])
        y = 58
        for item in sorted(by_frame[fr], key=lambda r: str(r.get("human_side"))):
            side = str(item.get("human_side", ""))
            label = str(item.get("contact_label", ""))
            kept = item.get("contact_observed") == "1"
            color = COLORS.get(side, (255, 255, 255)) if kept else COLORS["suppressed"]
            pu = round(float(item.get("palm_u") or 0.0))
            pv = round(float(item.get("palm_v") or 0.0))
            cu = round(float(item.get("contact_u") or 0.0))
            cv = round(float(item.get("contact_v") or 0.0))
            cv2.circle(img, (cu, cv), 7, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(img, (cu, cv), 5, color, -1, cv2.LINE_AA)
            cv2.circle(img, (pu, pv), 12, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(img, (pu, pv), 9, color, -1, cv2.LINE_AA)
            cv2.line(img, (pu, pv), (cu, cv), color, 2, cv2.LINE_AA)
            text = (
                f"{side}: {label} obs={item.get('contact_observed')} pose={item.get('pose_anchor_allowed')} "
                f"upd={item.get('anchor_update_allowed')} d={item.get('palm_to_line_px')} "
                f"hj={item.get('hand_joint_median_line_px')} tip={item.get('fingertip_median_line_px')} "
                f"s={item.get('object_local_s')} stable={item.get('stable_object_local_s')} {item.get('contact_disambiguation')}"
            )
            put_label(img, text, (18, y), color)
            y += 22
        png = out_dir / "frames" / f"frame_{fr:03d}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(png), img)
        frame_paths.append(png)
        if fr in preview_frames:
            cv2.imwrite(str(out_dir / f"preview_frame_{fr:03d}.png"), img)
    video = out_dir / "contact_points_overlay.mp4"
    write_video(frame_paths, video, fps)
    metrics = {
        "contact_candidates_labeled": str(labeled_csv),
        "contact_points_overlay": str(video),
        "frames_dir": str(out_dir / "frames"),
        "rows": len(labeled_rows),
        "frames": len(frame_paths),
        "pose_anchor_rows": sum(1 for r in labeled_rows if r.get("pose_anchor_allowed") == "1"),
        "suppressed_rows": sum(1 for r in labeled_rows if r.get("contact_label") == "suppressed"),
    }
    write_json(out_dir / "contact_candidate_render_manifest.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Render per-frame contact candidate labels over video frames.")
    parser.add_argument("--case", required=True)
    parser.add_argument("--result-name", required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=24.0)
    args = parser.parse_args()
    profile = with_runtime_overrides(load_case_profile(args.case), result_name=args.result_name)
    out_dir = args.out_dir or profile.result_dir / "contact_candidates_object_proxy"
    print(json.dumps(render(profile, out_dir, fps=args.fps), indent=2))


if __name__ == "__main__":
    main()
