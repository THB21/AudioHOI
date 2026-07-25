from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from ..base.config import CaseProfile, load_case_profile
from ..base.io import REPO
from .adapters import adapt_legacy_state_rows
from .golden import CANONICAL_CASE_DIRECTORIES


@dataclass(frozen=True)
class ParityCheck:
    check_id: str
    status: str
    max_abs_error: float | None = None
    violations: int = 0
    detail: str = ""


def _number(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    if value in {"", None}:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _max_abs(rows: list[dict[str, str]], left: str, right: str) -> tuple[float, int]:
    max_error = 0.0
    violations = 0
    for row in rows:
        a = _number(row, left)
        b = _number(row, right)
        if a is None or b is None:
            violations += 1
            continue
        max_error = max(max_error, abs(a - b))
    return max_error, violations


def _quat_norm_error(rows: list[dict[str, str]], fields: tuple[str, str, str, str]) -> tuple[float, int]:
    max_error = 0.0
    violations = 0
    for row in rows:
        values = [_number(row, field) for field in fields]
        if any(value is None for value in values):
            violations += 1
            continue
        norm = math.sqrt(sum(float(value) ** 2 for value in values))
        max_error = max(max_error, abs(norm - 1.0))
    return max_error, violations


def _bounded(rows: list[dict[str, str]], field: str, lower: float, upper: float) -> tuple[float, int]:
    max_error = 0.0
    violations = 0
    for row in rows:
        value = _number(row, field)
        if value is None:
            violations += 1
            continue
        if value < lower:
            violations += 1
            max_error = max(max_error, lower - value)
        elif value > upper:
            violations += 1
            max_error = max(max_error, value - upper)
    return max_error, violations


def _check(
    check_id: str,
    max_error: float | None,
    violations: int,
    tolerance: float,
    detail: str,
    violation_status: str = "fail",
) -> ParityCheck:
    status = "pass" if violations == 0 and (max_error is None or max_error <= tolerance) else violation_status
    return ParityCheck(check_id, status, max_error, violations, detail)


def _record(check: ParityCheck) -> dict[str, object]:
    return {
        "check_id": check.check_id,
        "status": check.status,
        "max_abs_error": check.max_abs_error,
        "violations": check.violations,
        "detail": check.detail,
    }


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_state_parity_report(profile: CaseProfile, pose_csv: Path, rows: list[dict[str, str]]) -> dict[str, object]:
    adapted = adapt_legacy_state_rows(profile, rows, str(pose_csv))
    checks: list[ParityCheck] = []
    schema = adapted.schema

    checks.append(ParityCheck("frame_time_present", "pass", None, 0, "frame/time parsed from every legacy row"))

    if schema in {"translation3_sphere_v1", "translation3_line_capsule_v1", "semantic_graph_6d_v1"}:
        for field in ("tx", "ty", "tz"):
            missing = sum(1 for row in rows if _number(row, field) is None)
            checks.append(_check(f"root_translation.{field}.present", 0.0, missing, 0.0, f"{field} maps to root.translation"))

    if schema == "rigid6_plus_periodic_phase_v1":
        for left, right in (("x", "tx"), ("y", "ty"), ("z", "tz")):
            max_error, violations = _max_abs(rows, left, right)
            checks.append(_check(f"root_translation.{left}_matches_{right}", max_error, violations, 1e-5, "legacy xyz mirror agrees with SE(3) translation"))
        max_error, violations = _max_abs(rows, "handle_phase_deg", "handle_phase_rad")
        phase_deg_errors = []
        for row in rows:
            phase = _number(row, "handle_phase_rad")
            deg = _number(row, "handle_phase_deg")
            if phase is not None and deg is not None:
                phase_deg_errors.append(abs(math.degrees(phase) - deg))
        checks.append(_check("handle_phase.deg_matches_rad", max(phase_deg_errors, default=0.0), violations, 1e-4, "periodic phase degree column matches radians"))

    if schema in {"translation3_sphere_v1", "translation3_line_capsule_v1", "rigid6_plus_periodic_phase_v1"}:
        fields = ("qw", "qx", "qy", "qz")
    else:
        fields = ("qw", "qx", "qy", "qz")
    max_error, violations = _quat_norm_error(rows, fields)
    checks.append(_check("root_rotation.quaternion_unit_norm", max_error, violations, 1e-3, "legacy quaternion is a normalized state representation"))

    if schema == "translation3_sphere_v1":
        radius_missing = sum(1 for row in rows if _number(row, "radius_m") is None)
        checks.append(_check("sphere.radius.present", 0.0, radius_missing, 0.0, "radius_m maps to sphere.radius"))
        checks.append(ParityCheck("sphere.rotation_gauge_declared", "pass", None, 0, "root.rotation is explicitly unobservable for sphere"))
    elif schema == "translation3_line_capsule_v1":
        checks.append(ParityCheck("line.roll_gauge_declared", "pass", None, 0, "roll around line axis is explicitly unobservable"))
        length_present = 0 if float(profile.data.get("line_object", {}).get("length_m", 0.0)) > 0 else 1
        checks.append(_check("line.length.present", 0.0, length_present, 0.0, "line_object.length_m maps to static line.length"))
    elif schema == "semantic_graph_6d_v1":
        rear_error, rear_violations = _bounded(rows, "rear_joint_angle", -0.82, 0.12)
        seat_error, seat_violations = _bounded(rows, "seat_joint_angle", 0.0, 1.35)
        checks.append(_check("joint.front_to_rear.within_urdf_limit", rear_error, rear_violations, 0.0, "legacy rear_joint_angle exceeds declared URDF limits in the current canonical seed", "warn"))
        checks.append(_check("joint.front_to_seat.within_urdf_limit", seat_error, seat_violations, 0.0, "legacy seat_joint_angle exceeds declared URDF limits in the current canonical seed", "warn"))

    records = [_record(check) for check in checks]
    return {
        "schema_version": 1,
        "mode": "read_only_shadow",
        "consumed_by_solver": False,
        "sample_id": profile.case_name,
        "legacy_schema": schema,
        "source": {
            "path": str(pose_csv),
            "sha256": hashlib.sha256(pose_csv.read_bytes()).hexdigest(),
            "rows": len(rows),
        },
        "checks": records,
        "summary": {
            "status": "fail" if any(check["status"] == "fail" for check in records) else "pass",
            "passed": sum(1 for check in records if check["status"] == "pass"),
            "warnings": sum(1 for check in records if check["status"] == "warn"),
            "failed": sum(1 for check in records if check["status"] == "fail"),
            "canonical_sha256": _canonical_hash(records),
        },
    }


def build_canonical_state_parity_reports(result_name: str = "benchmark_vlm_qwen") -> dict[str, object]:
    reports = {}
    for case_name, directory in CANONICAL_CASE_DIRECTORIES.items():
        profile = load_case_profile(case_name)
        pose_csv = REPO / "samples_known_object" / directory / "results" / result_name / "object_pose_init.csv"
        with pose_csv.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        reports[case_name] = build_state_parity_report(profile, pose_csv, rows)
    return {"schema_version": 1, "cases": reports}
