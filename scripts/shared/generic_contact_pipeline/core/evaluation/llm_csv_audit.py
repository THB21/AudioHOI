from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import requests

from ..base.config import CaseProfile
from ..base.io import float_or_none, read_csv, read_csv_header, write_csv, write_json
from ..base.schema import stage_paths
from ..gates.llm_provider import load_llm_provider


PASS = "pass"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        return read_csv(path)
    except Exception:
        return []


def _header(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        return read_csv_header(path)
    except Exception:
        return []


def _metric_json_summary(path: Path) -> dict[str, object]:
    data = _read_json(path)
    return {
        "exists": path.exists(),
        "keys": sorted(data.keys())[:40],
        "empty": not bool(data),
    }


def _csv_summary(path: Path, *, sample_limit: int = 3) -> dict[str, object]:
    rows = _rows(path)
    header = _header(path)
    return {
        "exists": path.exists(),
        "row_count": len(rows),
        "columns": header,
        "sample_rows": rows[:sample_limit],
    }


def _has_any_col(columns: list[str], choices: set[str]) -> bool:
    return any(col in choices for col in columns)


def _numeric_series(rows: list[dict[str, str]], *keys: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = None
        for key in keys:
            value = float_or_none(row.get(key))
            if value is not None:
                break
        if value is not None and math.isfinite(value):
            out.append(value)
    return out


def _max_step(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return max(abs(b - a) for a, b in zip(values[:-1], values[1:]))


def _nonempty(rows: list[dict[str, str]], *keys: str) -> bool:
    for row in rows:
        for key in keys:
            value = row.get(key)
            if value not in ("", None):
                return True
    return False


def _query(query_id: str, label_space: list[str], question: str, evidence: dict[str, object]) -> dict[str, object]:
    return {
        "query_id": query_id,
        "question": question,
        "allowed_labels": label_space,
        "evidence": evidence,
    }


def _safe_frame(value: object, fallback: int) -> int:
    parsed = float_or_none(value)
    if parsed is None:
        return fallback
    return int(parsed)


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def build_queries(profile: CaseProfile) -> list[dict[str, object]]:
    paths = stage_paths(profile)
    common_labels = [
        PASS,
        "schema_missing",
        "stage_inconsistent",
        "contact_empty",
        "contact_side_swapped",
        "static_drift",
        "rotation_jump",
        "depth_outlier",
        "vlm_gate_ignored",
        "unclear",
    ]
    evidence = {
        "case_name": profile.case_name,
        "object_family": profile.data.get("object_family", ""),
        "fixed_mainline": "preprocess -> generic observation -> generic contact/anchor -> generic SE3 pose init -> generic sequence_se3_optimizer -> render/eval",
        "legacy_yaml_selectors_role": "compatibility adapters only",
        "legacy_observation_model": profile.data.get("observation_model", ""),
        "legacy_contact_policy": profile.data.get("contact_policy", ""),
        "legacy_pose_model": profile.data.get("pose_model", ""),
        "legacy_refinement_policy": profile.data.get("refinement_policy", []),
        "render_backend": profile.component("render_backend"),
    }
    return [
        _query(
            "schema_completeness",
            common_labels,
            "Do the required CSV outputs contain the expected schema for this case?",
            evidence,
        ),
        _query(
            "stage_consistency",
            common_labels,
            "Are Stage2 contact candidates, Stage4 contact outputs, VLM gates, and metrics mutually consistent?",
            evidence,
        ),
        _query(
            "object_specific_rules",
            common_labels,
            "Do the case-specific semantic rules hold for contact, phase, static/freeze, and side assignments?",
            evidence,
        ),
        _query(
            "failure_range_summary",
            common_labels,
            "Which frame ranges should be inspected for depth outliers, rotation jumps, static drift, or missing contact?",
            evidence,
        ),
    ]


def _schema_audit(profile: CaseProfile, summaries: dict[str, dict[str, object]]) -> dict[str, object]:
    missing: list[str] = []
    for name in (
        "object_observations",
        "object_correspondence",
        "contact_candidates",
        "anchor_state",
        "object_pose_init",
        "object_pose",
        "physical_smooth_residuals",
        "motion_regime",
        "pose_jump_audit",
        "optimizer_decisions",
    ):
        if not summaries[name]["exists"] or int(summaries[name]["row_count"]) == 0:
            missing.append(name)
    pose_cols = list(summaries["object_pose"]["columns"])
    required_any = [{"frame"}, {"tx"}, {"ty"}, {"tz"}, {"qw"}, {"qx"}, {"qy"}, {"qz"}]
    for choices in required_any:
        if not _has_any_col(pose_cols, choices):
            missing.append("object_pose:" + "|".join(sorted(choices)))
    anchor_cols = list(summaries["anchor_state"]["columns"])
    for col in ("contact_observed", "contact_persistent", "anchor_update_allowed", "pose_anchor_allowed"):
        if col not in anchor_cols:
            missing.append(f"anchor_state:{col}")
    decision_cols = list(summaries["optimizer_decisions"]["columns"])
    for col in (
        "motion_regime",
        "sequence_se3_optimizer_enabled",
        "velocity_residual_enabled",
        "acceleration_residual_enabled",
        "smooth_weight",
        "accel_weight",
        "feedback_reoptimized",
    ):
        if col not in decision_cols:
            missing.append(f"optimizer_decisions:{col}")

    contact_cols = list(summaries["object_contact_points"]["columns"])
    phase_cols = list(summaries["object_phase"]["columns"])
    if profile.case_name in {"mug", "chair"}:
        if int(summaries["object_contact_points"]["row_count"]) == 0:
            missing.append("object_contact_points")
    if profile.case_name == "mug" and int(summaries["object_phase"]["row_count"]) == 0 and not any("phase" in c for c in phase_cols):
        missing.append("object_phase_or_handle_phase")
    if profile.case_name == "chair":
        if not _has_any_col(contact_cols, {"hand_side", "contact_side", "human_part"}):
            missing.append("chair_contact_side_column")
        if not _has_any_col(contact_cols, {"object_part", "contact_part", "target_object_part"}):
            missing.append("chair_object_part_column")

    label = PASS if not missing else "schema_missing"
    return {
        "query_id": "schema_completeness",
        "label": label,
        "blocking": label != PASS,
        "evidence": {"missing": missing},
        "short_reason": "required schemas are present" if label == PASS else "missing required schema fields/files",
    }


def _stage_consistency_audit(profile: CaseProfile, paths: dict[str, Path], summaries: dict[str, dict[str, object]]) -> dict[str, object]:
    contact_candidates = _rows(paths["contact_candidates"])
    contact_points = _rows(paths["object_contact_points"])
    gates = _rows(paths["vlm_gates"])
    decisions = _rows(paths["optimizer_decisions"])
    regimes = _rows(paths["motion_regime"])
    failures: list[str] = []
    if contact_candidates and profile.case_name in {"mug", "chair"} and not contact_points:
        failures.append("stage2_has_contact_but_stage4_contact_empty")
    rejected_frames = {
        row.get("frame", "")
        for row in gates
        if str(row.get("gate", row.get("decision", ""))).lower() in {"reject", "rejected", "disable", "disabled"}
    }
    contact_frames = {row.get("frame", "") for row in contact_points if row.get("frame", "")}
    ignored = sorted(frame for frame in rejected_frames if frame and frame in contact_frames)
    if ignored:
        failures.append("vlm_rejected_frames_present_in_contact_points")
    if decisions and not any(row.get("sequence_se3_optimizer_enabled") == "1" for row in decisions):
        failures.append("sequence_se3_optimizer_not_enabled")
    if not decisions:
        failures.append("missing_optimizer_decisions")
    label = PASS if not failures else "stage_inconsistent"
    if ignored:
        label = "vlm_gate_ignored"
    return {
        "query_id": "stage_consistency",
        "label": label,
        "blocking": label != PASS,
        "evidence": {
            "contact_candidates_rows": len(contact_candidates),
            "object_contact_points_rows": len(contact_points),
            "vlm_gates_rows": len(gates),
            "optimizer_decision_rows": len(decisions),
            "motion_regime_rows": len(regimes),
            "ignored_rejected_frames": ignored[:20],
            "failures": failures,
        },
        "short_reason": "stage outputs are mutually consistent" if label == PASS else "; ".join(failures),
    }


def _object_specific_audit(profile: CaseProfile, paths: dict[str, Path]) -> dict[str, object]:
    pose_rows = _rows(paths["object_pose"])
    contact_rows = _rows(paths["object_contact_points"])
    phase_rows = _rows(paths["object_phase"])
    failures: list[str] = []
    label = PASS

    z_values = _numeric_series(pose_rows, "tz", "z", "Z")
    z_step = _max_step(z_values)
    if z_step > 0.35:
        failures.append(f"large_depth_step:{z_step:.3f}m")
        label = "depth_outlier"

    if profile.case_name == "mug":
        phase_values = _numeric_series(phase_rows, "handle_phase", "phase", "m43_phase_rad", "m17_phase_rad")
        phase_step = _max_step(phase_values)
        if phase_values and phase_step > 0.75:
            failures.append(f"large_handle_phase_step:{phase_step:.3f}rad")
            label = "rotation_jump"
        if contact_rows and not _nonempty(contact_rows, "object_part", "contact_part", "nearest_articraft_part"):
            failures.append("mug_contact_part_empty")
            label = "contact_empty"

    if profile.case_name == "chair":
        active_contacts = [row for row in contact_rows if str(row.get("contact_active", "0")).strip() in {"1", "true", "True"}]
        left_ok = any("left" in (row.get("hand_side", "") + row.get("human_side", "") + row.get("contact_side", "") + row.get("human_part", "")).lower() for row in active_contacts)
        right_ok = any("right" in (row.get("hand_side", "") + row.get("human_side", "") + row.get("contact_side", "") + row.get("human_part", "")).lower() for row in active_contacts)
        if active_contacts and not (left_ok and right_ok):
            failures.append("chair_missing_left_or_right_hand_contact")
            label = "contact_side_swapped"
        if active_contacts and not _nonempty(active_contacts, "object_part", "contact_part", "target_object_part"):
            failures.append("chair_contact_object_part_empty")
            label = "contact_empty"

    if profile.case_name in {"basketball", "football"}:
        active_contacts = [row for row in contact_rows if str(row.get("contact_active", "0")).strip() in {"1", "true", "True"}]
        if active_contacts and not _nonempty(active_contacts, "contact_part", "contact_label", "anchor_type", "human_part", "object_part"):
            failures.append("ball_contact_label_empty")
            label = "contact_empty"

    return {
        "query_id": "object_specific_rules",
        "label": label,
        "blocking": label not in {PASS, "static_drift"},
        "evidence": {"failures": failures, "max_depth_step_m": round(z_step, 6)},
        "short_reason": "case-specific rules pass" if label == PASS else "; ".join(failures),
    }


def _failure_range_audit(profile: CaseProfile, paths: dict[str, Path]) -> dict[str, object]:
    pose_rows = _rows(paths["object_pose"])
    contact_rows = _rows(paths["object_contact_points"])
    pose_jump_rows = _rows(paths["pose_jump_audit"])
    decision_rows = _rows(paths["optimizer_decisions"])
    trusted_pose_lock = bool(decision_rows) and all(row.get("pose_lock_reason") for row in decision_rows)
    frames = [_safe_frame(row.get("frame"), i + 1) for i, row in enumerate(pose_rows)]
    z_values = _numeric_series(pose_rows, "tz", "z", "Z")
    jump_frames: list[int] = []
    if len(z_values) == len(frames):
        for frame, a, b in zip(frames[1:], z_values[:-1], z_values[1:]):
            if abs(b - a) > 0.25:
                jump_frames.append(frame)
    contact_frames = sorted(
        int(float(row.get("frame", 0) or 0))
        for row in contact_rows
        if float_or_none(row.get("frame")) is not None
    )
    label = "depth_outlier" if jump_frames else PASS
    audited_spike_frames = [
        _safe_frame(row.get("frame"), -1)
        for row in pose_jump_rows
        if row.get("visual_spike") == "1" or row.get("smoothness_spike") == "1" or row.get("contact_spike") == "1"
    ]
    low_trust_frames = [
        _safe_frame(row.get("frame"), -1)
        for row in decision_rows
        if (float_or_none(row.get("trust_weight")) or 1.0) < 0.35
    ]
    if audited_spike_frames and label == PASS and not trusted_pose_lock:
        label = "rotation_jump"
    return {
        "query_id": "failure_range_summary",
        "label": label,
        "blocking": False,
        "evidence": {
            "depth_jump_frames": jump_frames[:30],
            "contact_frame_min": min(contact_frames) if contact_frames else "",
            "contact_frame_max": max(contact_frames) if contact_frames else "",
            "contact_frame_count": len(contact_frames),
            "pose_jump_audit_frames": [fr for fr in audited_spike_frames[:30] if fr >= 0],
            "trusted_solved_pose_lock": trusted_pose_lock,
            "report_only_pose_jump_frames": [fr for fr in audited_spike_frames[:30] if fr >= 0] if trusted_pose_lock else [],
            "low_trust_optimizer_frames": [fr for fr in low_trust_frames[:30] if fr >= 0],
        },
        "short_reason": (
            "pose jump frames are report-only under trusted solved pose lock"
            if trusted_pose_lock and audited_spike_frames
            else "inspect listed frames"
            if (jump_frames or audited_spike_frames)
            else "no obvious numeric jump range found"
        ),
    }


MISTRAL_SYSTEM_PROMPT = """You are a CSV/data audit reviewer for AudioHOI.

Return JSON only. You may only choose labels from each query's allowed_labels.
Do not output pose, coordinates, loss weights, SE(3), or continuous corrections.
Your role is to inspect structured evidence and label schema/semantic consistency.
"""


MISTRAL_USER_PROMPT = """Review the AudioHOI pipeline CSV audit evidence.

For every query, return:
{{
  "results": [
    {{
      "query_id": "...",
      "label": "one allowed label",
      "blocking": true or false,
      "short_reason": "short reason",
      "evidence": {{"key": "small evidence summary"}}
    }}
  ]
}}

Allowed labels are provided per query. Use "unclear" if evidence is insufficient.
Never invent continuous values or pose updates.
If evidence contains trusted_solved_pose_lock=true, pose_jump_audit frames are report-only solved-pose diagnostics; do not label them rotation_jump unless another non-locked failure source is present.

Case:
{case_name}

Queries:
{queries}

CSV/JSON summaries:
{summaries}

Deterministic baseline audit:
{deterministic_results}
"""


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = text.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _ask_mistral(profile: CaseProfile, queries: list[dict[str, object]], summaries: dict[str, dict[str, object]], deterministic_results: list[dict[str, object]]) -> tuple[list[dict[str, object]] | None, dict[str, object]]:
    provider = load_llm_provider()
    trace: dict[str, object] = {"provider": provider.doctor(), "used": False, "error": ""}
    prompt = MISTRAL_USER_PROMPT.format(
        case_name=profile.case_name,
        queries=json.dumps(queries, ensure_ascii=False, indent=2),
        summaries=json.dumps(summaries, ensure_ascii=False, indent=2),
        deterministic_results=json.dumps(deterministic_results, ensure_ascii=False, indent=2),
    )
    text, err = provider.complete(MISTRAL_SYSTEM_PROMPT, prompt)
    if text is None:
        trace["error"] = err
        return None, trace
    parsed = _extract_json_object(text)
    trace.update({"used": True, "raw_text": text[:4000]})
    if not parsed or not isinstance(parsed.get("results"), list):
        trace["error"] = "response did not contain results list"
        return None, trace

    allowed = {str(q["query_id"]): set(str(label) for label in q["allowed_labels"]) for q in queries}
    deterministic_by_id = {str(row["query_id"]): row for row in deterministic_results}
    reviewed: list[dict[str, object]] = []
    for item in parsed["results"]:
        if not isinstance(item, dict):
            continue
        query_id = str(item.get("query_id", ""))
        if query_id not in allowed:
            continue
        label = str(item.get("label", "unclear"))
        if label not in allowed[query_id]:
            label = "unclear"
        base = dict(deterministic_by_id.get(query_id, {"query_id": query_id}))
        base.update(
            {
                "query_id": query_id,
                "label": label,
                "blocking": bool(item.get("blocking", label not in {PASS, "static_drift"})),
                "short_reason": str(item.get("short_reason", ""))[:240] or str(base.get("short_reason", "")),
                "evidence": item.get("evidence", base.get("evidence", {})),
                "llm_reviewed": True,
            }
        )
        reviewed.append(base)
    if len(reviewed) != len(queries):
        trace["error"] = f"incomplete result count {len(reviewed)}/{len(queries)}"
        return None, trace
    return reviewed, trace


def _summary_md(profile: CaseProfile, results: list[dict[str, object]]) -> str:
    lines = [
        f"# LLM CSV Audit Summary: {profile.case_name}",
        "",
        "This audit reads structured CSV/JSON outputs and emits discrete labels only.",
        "It does not modify pose, contact points, loss weights, or any result CSV.",
        "",
        "| query | label | blocking | reason |",
        "|---|---|---:|---|",
    ]
    for row in results:
        lines.append(
            f"| {row.get('query_id')} | {row.get('label')} | {row.get('blocking')} | {str(row.get('short_reason', '')).replace('|', '/')} |"
        )
    failures = [r for r in results if r.get("label") != PASS]
    lines.extend(["", "## Failure Labels", ""])
    if not failures:
        lines.append("- none")
    else:
        for row in failures:
            lines.append(f"- `{row.get('label')}` from `{row.get('query_id')}`: {row.get('short_reason')}")
    lines.append("")
    return "\n".join(lines)


def run_llm_csv_audit(profile: CaseProfile, mode: str = "seed") -> dict[str, object]:
    paths = stage_paths(profile)
    summaries = {
        "object_observations": _csv_summary(paths["object_observations"]),
        "object_correspondence": _csv_summary(paths["object_correspondence"]),
        "contact_candidates": _csv_summary(paths["contact_candidates"]),
        "anchor_state": _csv_summary(paths["anchor_state"]),
        "contact_state": _csv_summary(paths["contact_state"]),
        "object_pose_init": _csv_summary(paths["object_pose_init"]),
        "object_pose": _csv_summary(paths["object_pose"]),
        "object_contact_points": _csv_summary(paths["object_contact_points"]),
        "object_phase": _csv_summary(paths["object_phase"]),
        "physical_smooth_residuals": _csv_summary(paths["physical_smooth_residuals"]),
        "motion_regime": _csv_summary(paths["motion_regime"]),
        "pose_jump_audit": _csv_summary(paths["pose_jump_audit"]),
        "optimizer_decisions": _csv_summary(paths["optimizer_decisions"]),
        "stage3_metrics": _metric_json_summary(paths["stage3_metrics"]),
        "stage4_metrics": _metric_json_summary(paths["stage4_metrics"]),
        "stage6_compare_report": _metric_json_summary(paths["compare_report"]),
        "vlm_gates": _csv_summary(paths["vlm_gates"]),
        "loss_summary": _metric_json_summary(paths["loss_summary"]),
    }
    queries = build_queries(profile)
    results = [
        _schema_audit(profile, summaries),
        _stage_consistency_audit(profile, paths, summaries),
        _object_specific_audit(profile, paths),
        _failure_range_audit(profile, paths),
    ]
    llm_trace: dict[str, object] = {"mode": mode, "used": False}
    if mode in ("mistral", "claude"):
        reviewed, llm_trace = _ask_mistral(profile, queries, summaries, results)
        if reviewed is not None:
            results = reviewed
    failures = [row for row in results if row.get("label") != PASS]
    write_json(paths["llm_csv_audit_queries"], {"case_name": profile.case_name, "queries": queries, "summaries": summaries})
    write_json(
        paths["llm_csv_audit_results"],
        {
            "case_name": profile.case_name,
            "mode": "live_llm_csv_audit" if mode in ("mistral", "claude") and llm_trace.get("used") and not llm_trace.get("error") else "deterministic_csv_audit",
            "llm_trace": llm_trace,
            "results": results,
        },
    )
    paths["llm_csv_audit_summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["llm_csv_audit_summary"].write_text(_summary_md(profile, results), encoding="utf-8")
    write_csv(
        paths["llm_csv_audit_failures"],
        [{**row, "evidence": _json_text(row.get("evidence", {}))} for row in failures],
        ["query_id", "label", "blocking", "short_reason", "evidence"],
    )
    return {
        "stage": "stage6.5_llm_csv_audit",
        "case_name": profile.case_name,
        "queries": str(paths["llm_csv_audit_queries"]),
        "results": str(paths["llm_csv_audit_results"]),
        "summary": str(paths["llm_csv_audit_summary"]),
        "failures": str(paths["llm_csv_audit_failures"]),
        "failure_count": len(failures),
        "blocking_count": sum(1 for row in failures if row.get("blocking")),
    }
