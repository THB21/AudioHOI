from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from ..base.config import CaseProfile
from ..base.io import count_rows, float_or_none, read_csv, repo_path
from ..base.schema import REQUIRED_RENDER_FILES, stage_paths


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, int(round((pct / 100.0) * (len(vals) - 1)))))
    return vals[idx]


def _numeric_rows(rows: list[dict[str, str]], candidates: list[str]) -> list[list[float]]:
    values: list[list[float]] = []
    for row in rows:
        current: list[float] = []
        for key in candidates:
            v = float_or_none(row.get(key))
            if v is not None:
                current.append(v)
        if current:
            values.append(current)
    return values


def pose_delta_summary(new_csv: Path, baseline_csv: Path, keys: list[str]) -> dict[str, object]:
    new_rows = read_csv(new_csv)
    base_rows = read_csv(baseline_csv)
    n = min(len(new_rows), len(base_rows))
    new_header = set(new_rows[0].keys()) if new_rows else set()
    base_header = set(base_rows[0].keys()) if base_rows else set()
    common_keys = [key for key in keys if key in new_header and key in base_header]
    if not common_keys:
        return {"comparable": False, "reason": "no_common_numeric_columns", "requested_keys": keys}
    new_vals = _numeric_rows(new_rows[:n], common_keys)
    base_vals = _numeric_rows(base_rows[:n], common_keys)
    if not new_vals or not base_vals or len(new_vals) != len(base_vals):
        return {"comparable": False, "reason": "missing_numeric_columns"}
    diff: list[float] = []
    for a, b in zip(new_vals, base_vals):
        if len(a) != len(b):
            continue
        diff.extend(abs(x - y) for x, y in zip(a, b))
    if not diff:
        return {"comparable": False, "reason": "no_matching_numeric_values"}
    return {
        "comparable": True,
        "keys": common_keys,
        "median_abs_delta": float(_median(diff)),
        "p95_abs_delta": float(_percentile(diff, 95)),
        "max_abs_delta": float(max(diff)),
    }


def render_video_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"exists": False, "bytes": 0, "codec": "", "width": 0, "height": 0, "frames": 0, "pass": False}
    summary: dict[str, object] = {"exists": True, "bytes": path.stat().st_size, "codec": "", "width": 0, "height": 0, "frames": 0}
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,nb_frames",
                "-of",
                "csv=p=0",
                str(path),
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        row = next(csv.reader([proc.stdout.strip()]))
        if len(row) >= 4:
            summary["codec"] = row[0]
            summary["width"] = int(float(row[1])) if row[1] else 0
            summary["height"] = int(float(row[2])) if row[2] else 0
            summary["frames"] = int(float(row[3])) if row[3] and row[3] != "N/A" else 0
    except Exception as exc:
        summary["ffprobe_error"] = str(exc)
    summary["pass"] = (
        bool(summary["exists"])
        and int(summary["bytes"]) > 0
        and summary.get("codec") == "h264"
        and int(summary.get("width", 0)) > 0
        and int(summary.get("height", 0)) > 0
        and int(summary.get("frames", 0)) > 0
    )
    return summary


def _stage4_component_items(stage4: dict[str, object]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for key in ("components", "compatibility_adapters"):
        value = stage4.get(key, [])
        if isinstance(value, list):
            items.extend(item for item in value if isinstance(item, dict))
    if stage4.get("component"):
        items.append(stage4)
    return items


def _chair_quality_gate_from_stage4(stage4: dict[str, object]) -> tuple[bool, dict[str, object]]:
    for item in _stage4_component_items(stage4):
        if item.get("component") != "small_se3" and item.get("adapter_component") != "small_se3":
            continue
        summary = item.get("generic_pairprop_quality_summary")
        if not isinstance(summary, dict):
            quality_path = item.get("generic_pairprop_quality")
            if quality_path:
                try:
                    with Path(str(quality_path)).open() as f:
                        summary = json.load(f)
                except Exception:
                    summary = {}
        gate = summary.get("quality_gate", {}) if isinstance(summary, dict) else {}
        accepted = bool(item.get("generic_pairprop_pose_accepted"))
        return bool(accepted and isinstance(gate, dict) and gate.get("pass")), gate if isinstance(gate, dict) else {}
    return False, {}


def compare_case(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    baseline = profile.baseline
    final_pose = paths["object_pose"]
    baseline_pose = repo_path(baseline["final_pose_csv"]) if baseline.get("final_pose_csv") else None
    report: dict[str, object] = {
        "case_name": profile.case_name,
        "result_dir": str(profile.result_dir),
        "render_dir": str(profile.render_dir),
        "checks": {},
    }
    checks = report["checks"]  # type: ignore[assignment]
    required_csvs = {
        "object_observations": paths["object_observations"],
        "contact_candidates": paths["contact_candidates"],
        "object_pose_init": paths["object_pose_init"],
        "object_pose": paths["object_pose"],
    }
    if profile.case_name in {"mug", "chair"}:
        required_csvs["object_phase"] = paths["object_phase"]
    required_csvs["object_contact_points"] = paths["object_contact_points"]
    csv_checks = {}
    for name, path in required_csvs.items():
        exists = path.exists()
        rows = count_rows(path) if exists else 0
        csv_checks[name] = {"exists": exists, "rows": rows, "pass": exists and rows >= 0}
    checks["required_csvs"] = csv_checks
    checks["required_csvs_pass"] = all(item["pass"] for item in csv_checks.values())
    checks["frame_count"] = {
        "new": count_rows(final_pose) if final_pose.exists() else 0,
        "baseline": count_rows(baseline_pose) if baseline_pose and baseline_pose.exists() else 0,
        "pass": (
            final_pose.exists()
            and (
                (baseline_pose.exists() and count_rows(final_pose) == count_rows(baseline_pose))
                if baseline_pose
                else count_rows(final_pose) > 0
            )
        ),
    }
    if profile.case_name in {"basketball", "football"}:
        keys = ["tx", "ty", "tz", "u_obs", "v_obs", "z_ref_anchor_segment"]
    elif profile.case_name == "mug":
        keys = ["x", "y", "z", "yaw", "pitch", "roll", "scale"]
    else:
        keys = ["tx", "ty", "tz", "qx", "qy", "qz", "qw", "line_rmse_px", "endpoint_rmse_px"]
    if baseline_pose and final_pose.exists():
        checks["pose_delta"] = pose_delta_summary(final_pose, baseline_pose, keys)
    else:
        checks["pose_delta"] = {"comparable": False, "reason": "no_baseline_for_new_case"}
    pose_delta = checks["pose_delta"]
    if isinstance(pose_delta, dict) and pose_delta.get("comparable"):
        # tight enough to catch coordinate/camera regressions, loose enough for CSV rounding
        checks["pose_delta_pass"] = float(pose_delta.get("max_abs_delta", 999.0)) <= 0.01
    elif not baseline_pose:
        checks["pose_delta_pass"] = final_pose.exists() and count_rows(final_pose) > 0
    else:
        checks["pose_delta_pass"] = False
    if profile.case_name == "chair" and paths["stage4_metrics"].exists():
        try:
            with paths["stage4_metrics"].open() as f:
                stage4 = json.load(f)
            chair_quality_pass, quality_gate = _chair_quality_gate_from_stage4(stage4)
            checks["chair_quality_pose_gate"] = {
                "pass": chair_quality_pass,
                "note": "semantic 2D/contact/freeze gate for generic chair pairprop pose",
                "quality_gate": quality_gate,
            }
            checks["pose_delta_pass"] = bool(checks["pose_delta_pass"] or chair_quality_pass)
        except Exception as exc:
            checks["chair_quality_pose_gate"] = {"pass": False, "error": str(exc)}
    phase_csv = paths["object_phase"]
    baseline_phase = profile.baseline.get("final_phase_csv")
    if baseline_phase and phase_csv.exists():
        if profile.case_name == "mug":
            phase_keys = ["m17_phase_rad", "m43_phase_rad", "m17_phase_deg", "m43_phase_deg"]
        else:
            phase_keys = ["contact_start_frame", "lift_frame", "place_frame", "static_start_frame"]
        checks["phase_delta"] = pose_delta_summary(phase_csv, repo_path(baseline_phase), phase_keys)
        phase_delta = checks["phase_delta"]
        checks["phase_delta_pass"] = bool(
            isinstance(phase_delta, dict)
            and phase_delta.get("comparable")
            and float(phase_delta.get("max_abs_delta", 999.0)) <= 0.01
        )
    else:
        checks["phase_delta"] = {"comparable": True, "reason": "no_phase_baseline"}
        checks["phase_delta_pass"] = True
    render_checks = {}
    for rel in REQUIRED_RENDER_FILES:
        path = profile.render_dir / rel
        render_checks[rel] = render_video_summary(path)
    checks["renders"] = render_checks
    checks["required_renders_pass"] = all(bool(item["pass"]) for item in render_checks.values())
    checks["overall_pass"] = bool(
        checks["frame_count"]["pass"] and checks["required_renders_pass"] and checks["required_csvs_pass"]  # type: ignore[index]
        and checks["pose_delta_pass"]
        and checks["phase_delta_pass"]
    )
    return report
