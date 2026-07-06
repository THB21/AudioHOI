from __future__ import annotations

import math
from pathlib import Path
import json
from html import escape
from typing import Iterable

from ..base.config import CaseProfile, with_runtime_overrides
from ..base.io import read_csv, write_csv, write_json
from ..base.schema import stage_paths
from .final_evaluator import run_final_evaluator


BENCHMARK_FIELDS = [
    "Case",
    "Method",
    "Result Name",
    "Audio",
    "VLM",
    "LLM",
    "Contact F1 ↑",
    "Contact Proxy ↑",
    "Overlay Proxy ↑",
    "Anchor Drift ↓",
    "Penetration ↓",
    "Floating ↓",
    "Jump ↓",
    "Static Drift ↓",
    "VLM Judge ↑",
    "VLM Stage Action",
    "LLM Stage Action",
    "Pose Δ vs Base",
    "Weight Δ vs Base",
    "Gate Δ vs Base",
    "LLM Audit",
    "Failure Stage",
]

DELTA_FIELDS = ["Case", "Baseline Method", "Method", "Metric", "Baseline Value", "Method Value", "Delta", "Direction"]


def _fmt(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _method_metadata(method: str, *, llm_mode: str) -> dict[str, str]:
    meta = {"Audio": "unknown", "VLM": "unknown", "LLM": llm_mode}
    if method in {"baseline_no_vlm", "no_vlm"}:
        meta["VLM"] = "none"
    if method in {"vlm_gated", "vlm_gated_qwen"}:
        meta["VLM"] = "qwen"
    if method == "no_audio":
        meta["Audio"] = "off"
    if method in {"audio", "audio_enabled", "vlm_gated", "vlm_gated_qwen", "llm_mistral"}:
        meta["Audio"] = "on"
    if method in {"no_llm", "no_llm_seed"}:
        meta["LLM"] = "none"
    if method == "llm_mistral":
        meta["LLM"] = "mistral"
    if method == "oracle_contact":
        meta["VLM"] = "n/a"
        meta["LLM"] = llm_mode
    return meta


def _read_manifest(profile: CaseProfile) -> dict[str, object]:
    path = profile.result_dir / "pipeline_manifest.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_stage0_manifest(profile: CaseProfile) -> dict[str, object]:
    path = profile.result_dir / "stage0_inputs_manifest.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _result_metadata(profile: CaseProfile, method: str, *, llm_mode: str) -> dict[str, str]:
    meta = _method_metadata(method, llm_mode=llm_mode)
    manifest = _read_manifest(profile)
    if manifest.get("vlm_mode"):
        meta["VLM"] = str(manifest.get("vlm_mode"))
    if manifest.get("llm_mode"):
        meta["LLM"] = str(manifest.get("llm_mode"))
    profile_data = manifest.get("profile", {})
    flags: list[str] = []
    if isinstance(profile_data, dict):
        raw_flags = profile_data.get("ablation_flags", [])
        if isinstance(raw_flags, list):
            flags = [str(flag) for flag in raw_flags]
    if "disable_audio_events" in flags:
        meta["Audio"] = "off"
    elif method in {"audio", "audio_enabled", "vlm_gated", "llm_mistral"}:
        meta["Audio"] = "on"
    stage0 = _read_stage0_manifest(profile)
    checks = stage0.get("checks", {})
    if isinstance(checks, dict):
        if checks.get("events_audio_disabled") is True:
            meta["Audio"] = "off"
        elif checks.get("events_audio") is True:
            meta["Audio"] = "on"
    artifact_vlm = _vlm_artifact_mode(profile)
    if artifact_vlm:
        meta["VLM"] = artifact_vlm
    return meta


def _vlm_artifact_mode(profile: CaseProfile) -> str:
    stages = ["stage1", "stage2", "stage4", "stage5"]
    full_qwen = 0
    debug_qwen = 0
    dry_run = 0
    for stage in stages:
        stage_dir = profile.result_dir / "vlm" / stage
        decision = stage_dir / "stage_decision.json"
        if decision.exists():
            try:
                data = json.loads(decision.read_text())
            except Exception:
                data = {}
            if data.get("mode") == "qwen_vl":
                full_qwen += 1
            elif data.get("mode") == "dry_run":
                dry_run += 1
        debug = stage_dir / "stage_decision_qwen_debug.json"
        if debug.exists():
            try:
                data = json.loads(debug.read_text())
            except Exception:
                data = {}
            if data.get("mode") == "qwen_vl":
                debug_qwen += 1
    if full_qwen == len(stages):
        return "qwen"
    if debug_qwen:
        return "qwen_debug"
    if dry_run:
        return "dry-run"
    return ""


def _row_from_summary(summary: dict[str, object]) -> dict[str, object]:
    metrics = summary.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    vlm = summary.get("vlm_score", {})
    if not isinstance(vlm, dict):
        vlm = {}
    method = str(summary.get("method", ""))
    method_meta = summary.get("method_metadata", {})
    if not isinstance(method_meta, dict):
        method_meta = {}
    contact_f1: object = metrics.get("contact_f1")
    if contact_f1 is None:
        contact_f1 = "no_manual_labels"
    overlay_value = metrics.get("overlay_proxy_score")
    if overlay_value is None:
        overlay_value = metrics.get("mask_iou_mean")
    return {
        "Case": summary.get("case", ""),
        "Method": method,
        "Result Name": summary.get("result_name", ""),
        "Audio": method_meta.get("Audio", ""),
        "VLM": method_meta.get("VLM", ""),
        "LLM": method_meta.get("LLM", ""),
        "Contact F1 ↑": _fmt(contact_f1),
        "Contact Proxy ↑": _fmt(metrics.get("contact_proxy_score")),
        "Overlay Proxy ↑": _fmt(overlay_value),
        "Anchor Drift ↓": _fmt(metrics.get("anchor_drift_m_mean")),
        "Penetration ↓": _fmt(metrics.get("penetration_rate")),
        "Floating ↓": _fmt(metrics.get("floating_rate")),
        "Jump ↓": _fmt(metrics.get("jump_count")),
        "Static Drift ↓": _fmt(metrics.get("static_tail_drift_m_max")),
        "VLM Judge ↑": _fmt(vlm.get("score")),
        "VLM Stage Action": "",
        "LLM Stage Action": "",
        "Pose Δ vs Base": "",
        "Weight Δ vs Base": "",
        "Gate Δ vs Base": "",
        "LLM Audit": summary.get("llm_audit_summary", ""),
        "Failure Stage": summary.get("failure_stage_hint", ""),
    }


def _unavailable_row(profile: CaseProfile, method: str, result_name: str, reason: str, *, llm_mode: str) -> dict[str, object]:
    meta = _method_metadata(method, llm_mode=llm_mode)
    return {
        "Case": profile.case_name,
        "Method": method,
        "Result Name": result_name,
        "Audio": meta["Audio"],
        "VLM": meta["VLM"],
        "LLM": meta["LLM"],
        "Contact F1 ↑": f"missing:{reason}",
        "Contact Proxy ↑": "",
        "Overlay Proxy ↑": "",
        "Anchor Drift ↓": "",
        "Penetration ↓": "",
        "Floating ↓": "",
        "Jump ↓": "",
        "Static Drift ↓": "",
        "VLM Judge ↑": "",
        "VLM Stage Action": "",
        "LLM Stage Action": "",
        "Pose Δ vs Base": "",
        "Weight Δ vs Base": "",
        "Gate Δ vs Base": "",
        "LLM Audit": "unavailable",
        "Failure Stage": reason,
    }


def _report(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Generic Pipeline Benchmark",
        "",
        "| " + " | ".join(BENCHMARK_FIELDS) + " |",
        "| " + " | ".join(["---"] * len(BENCHMARK_FIELDS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in BENCHMARK_FIELDS) + " |")
    lines.append("")
    lines.append("Hard metrics, VLM visual judgment, LLM CSV audit, and QA summaries are stored separately under each result's `vlm_trace/06_evaluation` directory.")
    lines.append("GT Contact F1 and proxy contact are separate columns. Missing manual labels are reported as `no_manual_labels`.")
    return "\n".join(lines) + "\n"


def _html_report(rows: list[dict[str, object]]) -> str:
    header = "".join(f"<th>{escape(field)}</th>" for field in BENCHMARK_FIELDS)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{escape(str(row.get(field, '')))}</td>" for field in BENCHMARK_FIELDS) + "</tr>"
        for row in rows
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Generic Pipeline Benchmark</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #40515a; background: #faf4e3; }}
    h1 {{ margin: 0 0 18px; color: #5d6d73; }}
    table {{ border-collapse: collapse; width: 100%; background: rgba(255,255,255,0.24); }}
    th, td {{ border-bottom: 1px solid #cfc5ad; padding: 9px 12px; text-align: left; font-size: 15px; }}
    th {{ color: #53656e; font-weight: 700; }}
    code {{ color: #a62020; background: #f4ead4; padding: 2px 5px; border-radius: 4px; }}
    .note {{ margin-top: 18px; line-height: 1.5; color: #60757d; }}
  </style>
</head>
<body>
  <h1>Generic Pipeline Benchmark</h1>
  <table>
    <thead><tr>{header}</tr></thead>
    <tbody>
{body}
    </tbody>
  </table>
  <p class="note">Hard metrics, VLM visual judgment, LLM CSV audit, and QA summaries are stored separately under each result's <code>vlm_trace/06_evaluation</code> directory.</p>
  <p class="note">GT Contact F1 and proxy contact are separate columns. Missing manual labels are reported as <code>no_manual_labels</code>.</p>
</body>
</html>
"""


def _numeric(value: object) -> float | None:
    try:
        if value in {"", None}:
            return None
        text = str(value)
        if text.startswith("unavailable:") or text.startswith("missing:") or text == "no_manual_labels":
            return None
        return float(text)
    except Exception:
        return None


def _pose_effect(base: CaseProfile, method: CaseProfile) -> str:
    base_rows = {str(row.get("frame", "")): row for row in read_csv(stage_paths(base)["object_pose"])} if stage_paths(base)["object_pose"].exists() else {}
    method_rows = {str(row.get("frame", "")): row for row in read_csv(stage_paths(method)["object_pose"])} if stage_paths(method)["object_pose"].exists() else {}
    frames = sorted(set(base_rows) & set(method_rows), key=lambda value: float(value or 0))
    if not frames:
        return ""
    max_t = 0.0
    max_r = 0.0
    changed = 0
    for frame in frames:
        a = base_rows[frame]
        b = method_rows[frame]
        try:
            dt = math.sqrt(
                sum(
                    (float(b.get(k, "0") or 0) - float(a.get(k, "0") or 0)) ** 2
                    for k in ("tx", "ty", "tz")
                )
            )
            qa = [float(a.get(k, "0") or 0) for k in ("qw", "qx", "qy", "qz")]
            qb = [float(b.get(k, "0") or 0) for k in ("qw", "qx", "qy", "qz")]
            na = math.sqrt(sum(v * v for v in qa)) or 1.0
            nb = math.sqrt(sum(v * v for v in qb)) or 1.0
            dot = abs(sum((x / na) * (y / nb) for x, y in zip(qa, qb)))
            dot = max(-1.0, min(1.0, dot))
            dr = 2.0 * math.acos(dot)
        except Exception:
            continue
        max_t = max(max_t, dt)
        max_r = max(max_r, dr)
        if dt > 1e-5 or dr > 1e-5:
            changed += 1
    return f"frames={changed}/{len(frames)},tmax={max_t:.4f}m,rmax={max_r:.4f}rad"


def _weight_effect(base: CaseProfile, method: CaseProfile) -> str:
    base_path = stage_paths(base)["optimizer_decisions"]
    method_path = stage_paths(method)["optimizer_decisions"]
    if not base_path.exists() or not method_path.exists():
        return ""
    base_rows = {str(row.get("frame", "")): row for row in read_csv(base_path)}
    method_rows = {str(row.get("frame", "")): row for row in read_csv(method_path)}
    frames = sorted(set(base_rows) & set(method_rows), key=lambda value: float(value or 0))
    if not frames:
        return ""
    max_delta = {"trust_weight": 0.0, "smooth_weight": 0.0, "accel_weight": 0.0}
    changed = 0
    reasons: set[str] = set()
    for frame in frames:
        a = base_rows[frame]
        b = method_rows[frame]
        row_changed = False
        for key in max_delta:
            try:
                delta = abs(float(b.get(key, "0") or 0) - float(a.get(key, "0") or 0))
            except Exception:
                delta = 0.0
            max_delta[key] = max(max_delta[key], delta)
            row_changed = row_changed or delta > 1e-6
        for key in ("feedback_reweight_reason", "visual_prior_reason", "overlay_prior_reason", "pose_lock_reason"):
            if str(b.get(key, "")) != str(a.get(key, "")):
                reasons.add(key)
                row_changed = True
        if row_changed:
            changed += 1
    reason_text = ",".join(sorted(reasons)) if reasons else "weights"
    return (
        f"frames={changed}/{len(frames)},"
        f"trust={max_delta['trust_weight']:.4f},"
        f"smooth={max_delta['smooth_weight']:.4f},"
        f"accel={max_delta['accel_weight']:.4f},"
        f"reason={reason_text}"
    )


def _gate_signature(profile: CaseProfile) -> tuple[set[str], int]:
    rows: list[dict[str, str]] = []
    audit_path = stage_paths(profile)["stage_audit_dir"] / "stage_audit_gates.csv"
    if audit_path.exists():
        rows.extend(read_csv(audit_path))
    vlm_root = stage_paths(profile)["vlm_dir"]
    for stage in ("stage1", "stage2", "stage4", "stage5"):
        path = vlm_root / stage / "vlm_gates.csv"
        if path.exists():
            for row in read_csv(path):
                rows.append({"stage": stage, "check_type": "vlm:" + str(row.get("query_type", "")), "pass_gate": str(row.get("pass_gate", "")), "repair_action": str(row.get("affected_constraint", "")), "reason": str(row.get("reason", ""))})
    signature = {
        "|".join(
            [
                str(row.get("stage", "")),
                str(row.get("check_type", "")),
                str(row.get("pass_gate", "")),
                str(row.get("repair_action", "")),
                str(row.get("residual_reweight", "")),
            ]
        )
        for row in rows
    }
    return signature, len(rows)


def _gate_effect(base: CaseProfile, method: CaseProfile) -> str:
    base_sig, base_count = _gate_signature(base)
    method_sig, method_count = _gate_signature(method)
    added = len(method_sig - base_sig)
    removed = len(base_sig - method_sig)
    return f"rows={method_count-base_count:+d},added={added},removed={removed}"


def _vlm_stage_action(profile: CaseProfile) -> str:
    parts: list[str] = []
    root = stage_paths(profile)["vlm_dir"]
    for stage in ("stage1", "stage2", "stage4", "stage5"):
        decision_path = root / stage / "stage_decision.json"
        gate_path = root / stage / "vlm_gates.csv"
        if not decision_path.exists() and not gate_path.exists():
            continue
        mode = ""
        decision = ""
        if decision_path.exists():
            try:
                data = json.loads(decision_path.read_text())
            except Exception:
                data = {}
            mode = str(data.get("mode", ""))
            decision = str(data.get("decision", ""))
        rows = read_csv(gate_path) if gate_path.exists() else []
        effective = [row for row in rows if row.get("is_effective") == "1"]
        bad = [row for row in effective if row.get("pass_gate") in {"reject", "unclear"}]
        parts.append(f"{stage}:{mode or 'gates'}:{decision or 'n/a'}:eff={len(effective)}:bad={len(bad)}")
    return "; ".join(parts)


def _llm_stage_action(profile: CaseProfile) -> str:
    parts: list[str] = []
    root = stage_paths(profile)["stage_audit_dir"]
    for stage in ("stage1", "stage2", "stage4", "stage5"):
        trace_path = root / stage / "llm_audit_trace.json"
        result_path = root / stage / "llm_audit_results.csv"
        query_path = root / stage / "llm_audit_queries.json"
        if not trace_path.exists() and not result_path.exists() and not query_path.exists():
            continue
        trace: dict[str, object] = {}
        if trace_path.exists():
            try:
                trace = json.loads(trace_path.read_text())
            except Exception:
                trace = {}
        rows = read_csv(result_path) if result_path.exists() else []
        labels = "/".join(str(row.get("label", "")) for row in rows) or "none"
        reweight = sum(1 for row in rows if row.get("residual_reweight") == "1")
        disabled = trace.get("disabled") is True
        used = trace.get("used") is True
        if disabled:
            mode = "disabled"
        elif used:
            mode = "live"
        else:
            mode = "deterministic"
        parts.append(f"{stage}:{mode}:labels={labels}:rw={reweight}")
    return "; ".join(parts)


def _attach_effect_columns(case_entries: list[tuple[dict[str, object], CaseProfile]]) -> None:
    baseline = next((profile for row, profile in case_entries if row.get("Method") == "baseline_no_vlm"), None)
    if baseline is None and case_entries:
        baseline = case_entries[0][1]
    if baseline is None:
        return
    for row, method_profile in case_entries:
        row["VLM Stage Action"] = _vlm_stage_action(method_profile)
        row["LLM Stage Action"] = _llm_stage_action(method_profile)
        if method_profile.result_dir.resolve() == baseline.result_dir.resolve():
            row["Pose Δ vs Base"] = "baseline"
            row["Weight Δ vs Base"] = "baseline"
            row["Gate Δ vs Base"] = "baseline"
            continue
        row["Pose Δ vs Base"] = _pose_effect(baseline, method_profile)
        row["Weight Δ vs Base"] = _weight_effect(baseline, method_profile)
        row["Gate Δ vs Base"] = _gate_effect(baseline, method_profile)


def _delta_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = [
        ("Contact F1 ↑", "higher"),
        ("Contact Proxy ↑", "higher"),
        ("Overlay Proxy ↑", "higher"),
        ("Anchor Drift ↓", "lower"),
        ("Penetration ↓", "lower"),
        ("Floating ↓", "lower"),
        ("Jump ↓", "lower"),
        ("Static Drift ↓", "lower"),
        ("VLM Judge ↑", "higher"),
    ]
    out: list[dict[str, object]] = []
    by_case: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_case.setdefault(str(row.get("Case", "")), []).append(row)
    for case, case_rows in by_case.items():
        baseline = next((row for row in case_rows if row.get("Method") == "baseline_no_vlm"), case_rows[0] if case_rows else None)
        if baseline is None:
            continue
        for row in case_rows:
            if row is baseline:
                continue
            for metric, direction in metrics:
                base_value = _numeric(baseline.get(metric))
                method_value = _numeric(row.get(metric))
                if base_value is None or method_value is None:
                    continue
                out.append(
                    {
                        "Case": case,
                        "Baseline Method": baseline.get("Method", ""),
                        "Method": row.get("Method", ""),
                        "Metric": metric,
                        "Baseline Value": _fmt(base_value),
                        "Method Value": _fmt(method_value),
                        "Delta": _fmt(method_value - base_value),
                        "Direction": direction,
                    }
                )
    return out


def run_benchmark(
    profiles: Iterable[CaseProfile],
    *,
    methods: list[str],
    output_dir: Path | None = None,
    method_result_names: dict[str, str] | None = None,
    llm_mode: str = "seed",
    allow_same_result_debug: bool = False,
) -> dict[str, object]:
    profiles = list(profiles)
    if not profiles:
        raise ValueError("run_benchmark requires at least one profile")
    out_dir = output_dir or (profiles[0].result_dir / "benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    method_result_names = method_result_names or {}
    for profile in profiles:
        used_results: dict[Path, str] = {}
        case_entries: list[tuple[dict[str, object], CaseProfile]] = []
        for method in methods:
            method_profile = profile
            mapping_key = f"{profile.case_name}:{method}"
            has_explicit_mapping = mapping_key in method_result_names or method in method_result_names
            result_name = method_result_names.get(mapping_key, method_result_names.get(method, profile.result_name))
            if len(methods) > 1 and not has_explicit_mapping and method != methods[0]:
                row = _unavailable_row(profile, method, "", "missing_result_mapping", llm_mode=llm_mode)
                rows.append(row)
                continue
            if has_explicit_mapping:
                method_profile = with_runtime_overrides(profile, result_name=result_name)
            result_key = method_profile.result_dir.resolve()
            previous_method = used_results.get(result_key)
            if previous_method is not None and previous_method != method and not allow_same_result_debug:
                raise ValueError(
                    f"case {profile.case_name} maps {previous_method!r} and {method!r} to the same result {result_key}; "
                    "provide --method-result for real variants or pass allow_same_result_debug for diagnostics"
                )
            used_results[result_key] = method
            if not method_profile.result_dir.exists():
                row = _unavailable_row(profile, method, result_name, "missing_result", llm_mode=llm_mode)
                rows.append(row)
                continue
            summary = run_final_evaluator(method_profile, method=method, llm_mode=llm_mode)
            summary["method_metadata"] = _result_metadata(method_profile, method, llm_mode=llm_mode)
            summaries.append(summary)
            row = _row_from_summary(summary)
            rows.append(row)
            case_entries.append((row, method_profile))
        _attach_effect_columns(case_entries)
    deltas = _delta_rows(rows)
    write_csv(out_dir / "benchmark_table.csv", rows, BENCHMARK_FIELDS)
    write_csv(out_dir / "benchmark_deltas.csv", deltas, DELTA_FIELDS)
    write_json(
        out_dir / "benchmark_summary.json",
        {
            "rows": len(rows),
            "methods": methods,
            "method_result_names": method_result_names,
            "cases": [p.case_name for p in profiles],
            "summaries": summaries,
            "deltas": deltas,
        },
    )
    (out_dir / "benchmark_report.md").write_text(_report(rows))
    (out_dir / "benchmark_report.html").write_text(_html_report(rows))
    return {"output_dir": str(out_dir), "rows": len(rows), "table": str(out_dir / "benchmark_table.csv")}
