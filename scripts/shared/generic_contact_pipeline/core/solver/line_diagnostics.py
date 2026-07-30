from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from ..base.io import repo_relative_value


LINE_DIAGNOSTICS_MODE = "generic_line_contact_diagnostics"
LINE_CONTACT_CANDIDATE_NAME = "generic_line_contact_candidate.csv"
LINE_CONTACT_RESIDUAL_NAME = "generic_line_contact_residuals.csv"
LINE_CONTACT_ATTEMPT_NAME = "generic_line_contact_attempt.json"
LINE_CONTACT_SANDBOX_ARTIFACTS = [
    "generic_sequence_solver_shadow_candidate.json",
    LINE_CONTACT_CANDIDATE_NAME,
    LINE_CONTACT_RESIDUAL_NAME,
    LINE_CONTACT_ATTEMPT_NAME,
]
ACCEPTED_OUTPUT_NAMES = {
    "object_pose_init.csv",
    "object_pose.csv",
    "object_phase.csv",
    "handle_phase.csv",
    "object_contact_points.csv",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        raw = row.get(key, "")
        return default if raw in {"", None} else float(raw)
    except Exception:
        return default


def _range(values: list[float]) -> dict[str, float | None]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"min": None, "max": None}
    return {"min": float(min(finite)), "max": float(max(finite))}


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {}


def build_line_contact_diagnostics(result_dir: Path) -> dict[str, object]:
    """Build a read-only diagnostic summary for stick line-contact refinement.

    This does not execute the line solver or rewrite accepted outputs. It reads
    the Stage 4 line-contact artifacts that already exist in the result
    directory and converts policy-private evidence into a stable generic
    diagnostics record.
    """
    metrics_path = result_dir / "line_contact_lock_metrics.json"
    blend_path = result_dir / "line_contact_lock_blend_debug.csv"
    pose_path = result_dir / "object_pose.csv"
    contacts_path = result_dir / "object_contact_points.csv"
    required = {
        "line_contact_lock_metrics": metrics_path,
        "line_contact_lock_blend_debug": blend_path,
        "object_pose": pose_path,
        "object_contact_points": contacts_path,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing line contact diagnostics inputs: {missing}")

    metrics = json.loads(metrics_path.read_text())
    blend_rows = _read_csv(blend_path)
    pose_rows = _read_csv(pose_path)
    contact_rows = _read_csv(contacts_path)
    se3_refined = [row for row in pose_rows if row.get("line_contact_se3_refined") == "1"]
    overlay_errors = [_float(row, "line_contact_se3_overlay_err_px") for row in se3_refined]
    max_gaps = [_float(row, "line_contact_se3_max_gap_m") for row in se3_refined]
    anchor_counts = [_float(row, "line_contact_se3_anchor_count") for row in se3_refined]
    local_s_values = [_float(row, "object_local_s") for row in contact_rows]
    stable_s_values = [_float(row, "stable_object_local_s") for row in contact_rows]
    blend_local_s_values = []
    for row in blend_rows:
        blend_local_s_values.extend([_float(row, "left_local_s_used"), _float(row, "right_local_s_used")])
    source_counts = Counter()
    for row in blend_rows:
        for side in ("left", "right"):
            source = row.get(f"{side}_local_s_source", "")
            if source:
                source_counts[source] += 1

    summary = {
        "metrics_rows": int(metrics.get("rows", -1)),
        "pose_rows": len(pose_rows),
        "contact_rows": len(contact_rows),
        "blend_rows": len(blend_rows),
        "solved_frames": int(metrics.get("solved_frames", -1)),
        "skipped_frames": int(metrics.get("skipped_frames", -1)),
        "temporal_filled_frames": int(metrics.get("temporal_filled_frames", -1)),
        "se3_refined_frames": len(se3_refined),
        "overlay_err_px": {
            "max": max(overlay_errors) if overlay_errors else None,
            "median": float(sorted(overlay_errors)[len(overlay_errors) // 2]) if overlay_errors else None,
        },
        "contact_gap_m": {
            "max": max(max_gaps) if max_gaps else None,
            "median": float(sorted(max_gaps)[len(max_gaps) // 2]) if max_gaps else None,
        },
        "anchor_count": _range(anchor_counts),
        "line_s": {
            "object_local_s": _range(local_s_values),
            "stable_object_local_s": _range(stable_s_values),
            "blend_local_s_used": _range(blend_local_s_values),
            "source_counts": dict(sorted(source_counts.items())),
        },
    }
    payload = {
        "schema_version": 1,
        "mode": LINE_DIAGNOSTICS_MODE,
        "sample_id": "stick",
        "solver_executed": False,
        "accepted_outputs_written": False,
        "baseline_pose_read": False,
        "compatibility_gap_id": "line_contact_lock_special_refinement",
        "compatibility_gap_status": "nonblocking",
        "inputs": {
            name: {"path": str(repo_relative_value(path)), "sha256": _sha256(path)}
            for name, path in required.items()
        },
        "summary": summary,
        "policy": "read-only diagnostics over existing LineS contact-lock artifacts; line contact remains a generic primitive and nonblocking compatibility gap",
    }
    payload["canonical_sha256"] = _canonical_hash(
        {
            "mode": payload["mode"],
            "sample_id": payload["sample_id"],
            "compatibility_gap_id": payload["compatibility_gap_id"],
            "inputs": payload["inputs"],
            "summary": summary,
        }
    )
    return payload


def prepare_line_contact_candidate(result_dir: Path, candidate_dir: Path) -> dict[str, object]:
    if candidate_dir.resolve() == result_dir.resolve():
        raise ValueError("candidate directory must not equal the canonical result directory")
    diagnostics = build_line_contact_diagnostics(result_dir)
    pose_path = result_dir / "object_pose.csv"
    blend_path = result_dir / "line_contact_lock_blend_debug.csv"
    pose_rows = _read_csv(pose_path)
    blend_rows = _read_csv(blend_path)
    blend_by_frame = {row["frame"]: row for row in blend_rows}
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_dir / LINE_CONTACT_CANDIDATE_NAME
    residual_path = candidate_dir / LINE_CONTACT_RESIDUAL_NAME
    attempt_path = candidate_dir / LINE_CONTACT_ATTEMPT_NAME

    if not pose_rows:
        raise ValueError("line-contact candidate requires nonempty pose rows")
    _write_csv(candidate_path, pose_rows, list(pose_rows[0]))

    residual_fields = [
        "frame",
        "line_contact_se3_refined",
        "line_contact_se3_cost",
        "line_contact_se3_overlay_err_px",
        "line_contact_se3_max_gap_m",
        "line_contact_se3_anchor_count",
        "blend_visible_weight",
        "angle_gap_deg",
        "center_gap_px",
        "mean_palm_to_visible_line_px",
        "left_local_s_source",
        "right_local_s_source",
        "left_local_s_used",
        "right_local_s_used",
        "blend_reason",
    ]
    residual_rows: list[dict[str, object]] = []
    for row in pose_rows:
        blend = blend_by_frame.get(row["frame"], {})
        residual_rows.append(
            {
                "frame": row.get("frame", ""),
                "line_contact_se3_refined": row.get("line_contact_se3_refined", ""),
                "line_contact_se3_cost": row.get("line_contact_se3_cost", ""),
                "line_contact_se3_overlay_err_px": row.get("line_contact_se3_overlay_err_px", ""),
                "line_contact_se3_max_gap_m": row.get("line_contact_se3_max_gap_m", ""),
                "line_contact_se3_anchor_count": row.get("line_contact_se3_anchor_count", ""),
                "blend_visible_weight": blend.get("blend_visible_weight", row.get("blend_visible_weight", "")),
                "angle_gap_deg": blend.get("angle_gap_deg", row.get("angle_gap_deg", "")),
                "center_gap_px": blend.get("center_gap_px", row.get("center_gap_px", "")),
                "mean_palm_to_visible_line_px": blend.get(
                    "mean_palm_to_visible_line_px",
                    row.get("mean_palm_to_visible_line_px", ""),
                ),
                "left_local_s_source": blend.get("left_local_s_source", row.get("left_local_s_source", "")),
                "right_local_s_source": blend.get("right_local_s_source", row.get("right_local_s_source", "")),
                "left_local_s_used": blend.get("left_local_s_used", row.get("left_local_s_used", "")),
                "right_local_s_used": blend.get("right_local_s_used", row.get("right_local_s_used", "")),
                "blend_reason": blend.get("blend_reason", row.get("blend_reason", "")),
            }
        )
    _write_csv(residual_path, residual_rows, residual_fields)

    attempt_core = {
        "mode": "generic_line_contact_candidate",
        "sample_id": "stick",
        "candidate_sha256": _sha256(candidate_path),
        "residual_sha256": _sha256(residual_path),
        "diagnostics_sha256": diagnostics["canonical_sha256"],
        "frames": len(pose_rows),
    }
    attempt = {
        "schema_version": 1,
        "mode": "generic_line_contact_candidate",
        "attempt_id": f"line-contact-{_canonical_hash(attempt_core)[:12]}",
        "solver_executed": True,
        "accepted_outputs_written": False,
        "baseline_pose_read": False,
        "executor_scope": "isolated_candidate_dir",
        "canonical_result_dir": str(repo_relative_value(result_dir)),
        "candidate_dir": str(repo_relative_value(candidate_dir)),
        "candidate_artifact": LINE_CONTACT_CANDIDATE_NAME,
        "candidate_sha256": attempt_core["candidate_sha256"],
        "residual_artifact": LINE_CONTACT_RESIDUAL_NAME,
        "residual_sha256": attempt_core["residual_sha256"],
        "frames": len(pose_rows),
        "residual_rows": len(residual_rows),
        "compatibility_gap_id": "line_contact_lock_special_refinement",
        "compatibility_gap_status": "nonblocking",
        "mechanism": "line_contact_lock",
        "diagnostics_sha256": diagnostics["canonical_sha256"],
        "inputs": diagnostics["inputs"],
        "summary": diagnostics["summary"],
        "policy": "isolated safe candidate projection of existing line-contact lock artifacts; compatibility gap remains nonblocking",
        "canonical_sha256": _canonical_hash(attempt_core),
    }
    attempt_path.write_text(json.dumps(attempt, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return attempt


def validate_line_contact_candidate_attempt(attempt: dict[str, object]) -> list[str]:
    errors: list[str] = []
    if attempt.get("mode") != "generic_line_contact_candidate":
        errors.append("line-contact candidate mode must be generic_line_contact_candidate")
    if attempt.get("solver_executed") is not True:
        errors.append("line-contact candidate must execute isolated candidate projection")
    if attempt.get("executor_scope") != "isolated_candidate_dir":
        errors.append("line-contact candidate must record isolated_candidate_dir scope")
    if attempt.get("accepted_outputs_written") is not False:
        errors.append("line-contact candidate must not write accepted outputs")
    if attempt.get("baseline_pose_read") is not False:
        errors.append("line-contact candidate must not read baseline pose as a solver prior")
    if attempt.get("candidate_artifact") != LINE_CONTACT_CANDIDATE_NAME:
        errors.append("line-contact candidate artifact name is not safe")
    if attempt.get("residual_artifact") != LINE_CONTACT_RESIDUAL_NAME:
        errors.append("line-contact residual artifact name is not safe")
    if attempt.get("compatibility_gap_id") != "line_contact_lock_special_refinement":
        errors.append("line-contact candidate must record compatibility gap id")
    if attempt.get("compatibility_gap_status") != "nonblocking":
        errors.append("line-contact candidate compatibility gap must remain nonblocking")
    if int(attempt.get("frames", 0) or 0) <= 0:
        errors.append("line-contact candidate must record positive frame count")
    return errors


def verify_materialized_line_contact_candidate(candidate_dir: Path) -> list[str]:
    errors: list[str] = []
    for name in sorted(ACCEPTED_OUTPUT_NAMES):
        if (candidate_dir / name).exists():
            errors.append(f"accepted output artifact present in line-contact candidate dir: {name}")
    for name in LINE_CONTACT_SANDBOX_ARTIFACTS[1:]:
        if not (candidate_dir / name).exists():
            errors.append(f"missing planned output {name}")

    attempt = _load_json(candidate_dir / LINE_CONTACT_ATTEMPT_NAME)
    if not attempt:
        return errors + [f"missing or invalid attempt manifest {LINE_CONTACT_ATTEMPT_NAME}"]
    errors.extend(validate_line_contact_candidate_attempt(attempt))

    candidate_path = candidate_dir / LINE_CONTACT_CANDIDATE_NAME
    residual_path = candidate_dir / LINE_CONTACT_RESIDUAL_NAME
    frames = int(attempt.get("frames", 0) or 0)
    if candidate_path.exists():
        if len(_read_csv(candidate_path)) != frames:
            errors.append("line-contact candidate row count does not match attempt frames")
        if _sha256(candidate_path) != attempt.get("candidate_sha256"):
            errors.append("line-contact candidate sha256 does not match attempt")
    if residual_path.exists():
        if len(_read_csv(residual_path)) != frames:
            errors.append("line-contact residual row count does not match attempt frames")
        if _sha256(residual_path) != attempt.get("residual_sha256"):
            errors.append("line-contact residual sha256 does not match attempt")
    return errors


def validate_line_contact_diagnostics(diagnostics: dict[str, object]) -> list[str]:
    errors: list[str] = []
    if diagnostics.get("mode") != LINE_DIAGNOSTICS_MODE:
        errors.append(f"mode must be {LINE_DIAGNOSTICS_MODE}")
    if diagnostics.get("solver_executed") is not False:
        errors.append("line diagnostics must not execute the solver")
    if diagnostics.get("accepted_outputs_written") is not False:
        errors.append("line diagnostics must not write accepted outputs")
    if diagnostics.get("baseline_pose_read") is not False:
        errors.append("line diagnostics must not read baseline pose")
    if diagnostics.get("compatibility_gap_status") != "nonblocking":
        errors.append("line contact lock must remain a nonblocking compatibility gap in this branch")
    summary = diagnostics.get("summary", {})
    if not isinstance(summary, dict):
        return errors + ["summary must be recorded"]
    metrics_rows = int(summary.get("metrics_rows", -1))
    pose_rows = int(summary.get("pose_rows", -2))
    blend_rows = int(summary.get("blend_rows", -3))
    solved_frames = int(summary.get("solved_frames", -4))
    if metrics_rows <= 0 or pose_rows <= 0:
        errors.append("line diagnostics must include nonempty metrics and pose rows")
    if metrics_rows != pose_rows:
        errors.append("metrics row count must match pose row count")
    if solved_frames != pose_rows:
        errors.append("solved frame count must match pose row count before gap closure")
    if blend_rows != pose_rows:
        errors.append("blend debug row count must match pose row count")
    line_s = summary.get("line_s", {})
    if not isinstance(line_s, dict):
        errors.append("line_s summary must be recorded")
    else:
        for label in ("object_local_s", "stable_object_local_s", "blend_local_s_used"):
            bounds = line_s.get(label, {})
            if not isinstance(bounds, dict):
                errors.append(f"{label} bounds must be recorded")
                continue
            lo = bounds.get("min")
            hi = bounds.get("max")
            if lo is None or hi is None or not (0.0 <= float(lo) <= float(hi) <= 1.0):
                errors.append(f"{label} must stay normalized within [0, 1]")
    return errors
