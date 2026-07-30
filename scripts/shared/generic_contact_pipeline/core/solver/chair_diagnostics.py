from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from ..base.io import repo_relative_value


CHAIR_DIAGNOSTICS_MODE = "generic_chair_contact_diagnostics"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _small_se3_report(stage4_metrics: dict[str, object]) -> dict[str, object]:
    if stage4_metrics.get("component") == "small_se3":
        return stage4_metrics
    for adapter in stage4_metrics.get("compatibility_adapters", []):
        if isinstance(adapter, dict):
            report = adapter.get("report", {})
            if isinstance(report, dict) and report.get("component") == "small_se3":
                return report
    raise ValueError("stage4_metrics does not contain a small_se3 report")


def _finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) else None


def build_chair_contact_diagnostics(result_dir: Path) -> dict[str, object]:
    """Build read-only diagnostics for chair pairprop/contact-chord migration.

    This records the current pairprop seed policy, contact quality and gate
    status without running the chair solver or writing accepted outputs.
    """
    stage4_metrics_path = result_dir / "stage4_metrics.json"
    if not stage4_metrics_path.exists():
        raise FileNotFoundError(f"missing chair stage4 metrics: {stage4_metrics_path}")
    stage4_metrics = json.loads(stage4_metrics_path.read_text())
    report = _small_se3_report(stage4_metrics)
    pairprop_summary = report.get("generic_pairprop_summary", {})
    if not isinstance(pairprop_summary, dict):
        pairprop_summary = {}
    quality_summary = report.get("generic_pairprop_quality_summary", {})
    if not isinstance(quality_summary, dict):
        quality_summary = {}
    seed_info = report.get("generic_pairprop_seed_info", {})
    if not isinstance(seed_info, dict):
        seed_info = {}
    seed_policy = str(seed_info.get("policy", "unknown"))
    seed_source = str(seed_info.get("seed_source", ""))
    seed_is_current_run = seed_policy == "current_stage3_observation_fit"
    historical_references = sorted(
        key
        for key in ("mainline_pose_csv", "physical6d_seed_csv", "snapshot_pose_csv")
        if seed_info.get(key)
    )
    contact_chord_gate = pairprop_summary.get("contact_chord_constraint_gate", {})
    if not isinstance(contact_chord_gate, dict):
        contact_chord_gate = {}
    metrics = pairprop_summary.get("metrics", [])
    metrics_rows = metrics if isinstance(metrics, list) else []
    contact_gaps = [
        value
        for value in (_finite(row.get("median_contact_gap_m")) for row in metrics_rows if isinstance(row, dict))
        if value is not None
    ]
    summary = {
        "pairprop_pose_accepted": bool(report.get("generic_pairprop_pose_accepted")),
        "standard_quality_pass": bool(report.get("generic_pairprop_standard_quality_pass")),
        "constraint_quality_pass": bool(report.get("generic_pairprop_constraint_quality_pass")),
        "seed_policy": seed_policy,
        "seed_is_current_run": seed_is_current_run,
        "historical_seed_reference_fields": historical_references,
        "snapshot_fallback_used": bool(seed_info.get("snapshot_fallback_used")),
        "active_frames": int(pairprop_summary.get("active_frames", 0) or 0),
        "first_active": int(pairprop_summary.get("first_active", 0) or 0),
        "last_active": int(pairprop_summary.get("last_active", 0) or 0),
        "metrics_rows": len(metrics_rows),
        "median_optimized_contact_gap_m": _finite(pairprop_summary.get("median_optimized_contact_gap_m")),
        "p90_optimized_contact_gap_m": _finite(pairprop_summary.get("p90_optimized_contact_gap_m")),
        "per_frame_contact_gap_max_m": max(contact_gaps) if contact_gaps else None,
        "contact_chord_constraint_gate": {
            "present": bool(contact_chord_gate),
            "pass": bool(contact_chord_gate.get("pass")) if contact_chord_gate else False,
        },
        "quality_gate_present": bool(quality_summary),
    }
    gap_status = "nonblocking" if seed_is_current_run and summary["constraint_quality_pass"] else "blocking"
    payload = {
        "schema_version": 1,
        "mode": CHAIR_DIAGNOSTICS_MODE,
        "sample_id": "chair",
        "solver_executed": False,
        "accepted_outputs_written": False,
        "baseline_pose_read": False,
        "compatibility_gap_id": "semantic_graph_solver_private",
        "compatibility_gap_status": gap_status,
        "inputs": {
            "stage4_metrics": {
                "path": str(repo_relative_value(stage4_metrics_path)),
                "sha256": _sha256(stage4_metrics_path),
            }
        },
        "summary": summary,
        "policy": "read-only diagnostics over chair pairprop/contact-chord artifacts; does not close the semantic graph solver gap",
    }
    payload["canonical_sha256"] = _canonical_hash(
        {
            "mode": payload["mode"],
            "sample_id": payload["sample_id"],
            "compatibility_gap_id": payload["compatibility_gap_id"],
            "compatibility_gap_status": payload["compatibility_gap_status"],
            "inputs": payload["inputs"],
            "summary": summary,
        }
    )
    return payload


def validate_chair_contact_diagnostics(diagnostics: dict[str, object]) -> list[str]:
    errors: list[str] = []
    if diagnostics.get("mode") != CHAIR_DIAGNOSTICS_MODE:
        errors.append(f"mode must be {CHAIR_DIAGNOSTICS_MODE}")
    if diagnostics.get("solver_executed") is not False:
        errors.append("chair diagnostics must not execute the solver")
    if diagnostics.get("accepted_outputs_written") is not False:
        errors.append("chair diagnostics must not write accepted outputs")
    if diagnostics.get("baseline_pose_read") is not False:
        errors.append("chair diagnostics must not read baseline pose")
    if diagnostics.get("compatibility_gap_id") != "semantic_graph_solver_private":
        errors.append("chair diagnostics must track semantic_graph_solver_private")
    summary = diagnostics.get("summary", {})
    if not isinstance(summary, dict):
        return errors + ["summary must be recorded"]
    if int(summary.get("active_frames", 0) or 0) <= 0:
        errors.append("chair diagnostics must include active contact frames")
    if int(summary.get("metrics_rows", 0) or 0) <= 0:
        errors.append("chair diagnostics must include pairprop per-frame metrics")
    seed_is_current_run = bool(summary.get("seed_is_current_run"))
    gap_status = diagnostics.get("compatibility_gap_status")
    if not seed_is_current_run and gap_status != "blocking":
        errors.append("historical/rebuilt chair seed must keep the compatibility gap blocking")
    if seed_is_current_run and gap_status == "blocking" and summary.get("constraint_quality_pass") is True:
        errors.append("current-run passing chair seed should not remain a blocking diagnostics state")
    if summary.get("historical_seed_reference_fields") and seed_is_current_run:
        errors.append("current-run chair seed must not report historical seed reference fields")
    median_gap = summary.get("median_optimized_contact_gap_m")
    p90_gap = summary.get("p90_optimized_contact_gap_m")
    if median_gap is None or p90_gap is None or float(median_gap) < 0.0 or float(p90_gap) < 0.0:
        errors.append("chair diagnostics must record nonnegative contact gap summaries")
    return errors
