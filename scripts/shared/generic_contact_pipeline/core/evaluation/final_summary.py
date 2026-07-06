from __future__ import annotations

import json
from html import escape
from pathlib import Path

from ..base.config import CaseProfile, with_runtime_overrides
from ..base.io import read_csv, write_csv, write_json
from ..base.schema import stage_paths
from .final_evaluator import run_final_evaluator


FINAL_SUMMARY_FIELDS = [
    "Case",
    "Final Result",
    "Frames",
    "SE3 Pose",
    "Translation",
    "Rotation",
    "Contact GT",
    "Contact F1",
    "Contact Proxy",
    "Overlay Proxy",
    "Anchor Drift Mean ↓",
    "Anchor Drift Max ↓",
    "Penetration Rate ↓",
    "Floating Rate ↓",
    "Jump Count ↓",
    "Static Drift Max ↓",
    "Geometry Spread ↓",
    "VLM Judge ↑",
    "LLM Audit",
    "Failure Stage",
    "Final Pass",
    "Evaluation Summary",
]


def _fmt(value: object, *, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        if value == "":
            return ""
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _fmt_field(field: str, value: object) -> str:
    if field in {"Frames", "Jump Count ↓"}:
        try:
            return str(int(float(value))) if value not in {"", None} else ""
        except Exception:
            return str(value)
    return _fmt(value)


def _pose_schema(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    pose_path = paths["object_pose"]
    rows = read_csv(pose_path) if pose_path.exists() else []
    fields = set(rows[0].keys()) if rows else set()
    translation = {"tx", "ty", "tz"}.issubset(fields)
    rotation = {"qw", "qx", "qy", "qz"}.issubset(fields)
    return {
        "frames": len(rows),
        "se3_pose": translation and rotation,
        "translation": translation,
        "rotation": rotation,
    }


def summarize_final_result(
    profile: CaseProfile,
    *,
    result_name: str = "benchmark_vlm_qwen",
    method: str = "final_vlm_gated",
    llm_mode: str = "seed",
    rerun_evaluator: bool = True,
) -> dict[str, object]:
    final_profile = with_runtime_overrides(profile, result_name=result_name)
    if rerun_evaluator:
        summary = run_final_evaluator(final_profile, method=method, llm_mode=llm_mode)
    else:
        summary_path = final_profile.result_dir / "vlm_trace" / "06_evaluation" / "evaluation_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
        else:
            summary = run_final_evaluator(final_profile, method=method, llm_mode=llm_mode)
    metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
    vlm_score = summary.get("vlm_score", {}) if isinstance(summary, dict) else {}
    schema = _pose_schema(final_profile)
    return {
        "Case": final_profile.case_name,
        "Final Result": final_profile.result_name,
        "Frames": schema["frames"],
        "SE3 Pose": schema["se3_pose"],
        "Translation": schema["translation"],
        "Rotation": schema["rotation"],
        "Contact GT": "manual" if metrics.get("gt_contact_available") else "proxy_only",
        "Contact F1": metrics.get("contact_f1"),
        "Contact Proxy": metrics.get("contact_proxy_score"),
        "Overlay Proxy": metrics.get("overlay_proxy_score"),
        "Anchor Drift Mean ↓": metrics.get("anchor_drift_m_mean"),
        "Anchor Drift Max ↓": metrics.get("anchor_drift_m_max"),
        "Penetration Rate ↓": metrics.get("penetration_rate"),
        "Floating Rate ↓": metrics.get("floating_rate"),
        "Jump Count ↓": metrics.get("jump_count"),
        "Static Drift Max ↓": metrics.get("static_tail_drift_m_max"),
        "Geometry Spread ↓": metrics.get("geometry_length_spread_m"),
        "VLM Judge ↑": vlm_score.get("score") if isinstance(vlm_score, dict) else "",
        "LLM Audit": summary.get("llm_audit_summary", ""),
        "Failure Stage": summary.get("failure_stage_hint", ""),
        "Final Pass": summary.get("final_pass", False),
        "Evaluation Summary": str(final_profile.result_dir / "vlm_trace" / "06_evaluation" / "evaluation_summary.json"),
    }


def _render_markdown(rows: list[dict[str, object]]) -> str:
    fields = FINAL_SUMMARY_FIELDS
    lines = [
        "# Final Result Evaluation Summary",
        "",
        "This table evaluates only the current final result for each case. It is not a benchmark and does not compare methods.",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(field)) for field in fields) + " |")
    lines.append("")
    lines.append("Notes: `Contact F1` is blank unless manual labels exist. `Contact Proxy` is reported separately and must not be read as ground-truth F1.")
    return "\n".join(lines) + "\n"


def _render_html(rows: list[dict[str, object]]) -> str:
    fields = FINAL_SUMMARY_FIELDS
    header = "".join(f"<th>{escape(field)}</th>" for field in fields)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{escape(_fmt(row.get(field)))}</td>" for field in fields) + "</tr>"
        for row in rows
    )
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><title>Final Result Evaluation Summary</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:28px;color:#263238}"
        "table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #cfd8dc;padding:7px;vertical-align:top}"
        "th{background:#eef3f5}caption{text-align:left;font-weight:700;font-size:22px;margin-bottom:12px}</style>"
        "</head><body><table><caption>Final Result Evaluation Summary</caption><thead><tr>"
        + header
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table><p>This is a final-result summary, not a benchmark. Proxy contact and GT contact are separated.</p></body></html>\n"
    )


def write_final_summary(
    profiles: list[CaseProfile],
    *,
    output_dir: Path,
    result_name: str = "benchmark_vlm_qwen",
    method: str = "final_vlm_gated",
    llm_mode: str = "seed",
    rerun_evaluator: bool = True,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = [
        summarize_final_result(
            profile,
            result_name=result_name,
            method=method,
            llm_mode=llm_mode,
            rerun_evaluator=rerun_evaluator,
        )
        for profile in profiles
    ]
    rows = [{field: _fmt_field(field, row.get(field)) for field in FINAL_SUMMARY_FIELDS} for row in raw_rows]
    table = write_csv(output_dir / "final_result_evaluation_summary.csv", rows, FINAL_SUMMARY_FIELDS)
    md = output_dir / "final_result_evaluation_summary.md"
    html = output_dir / "final_result_evaluation_summary.html"
    md.write_text(_render_markdown(rows))
    html.write_text(_render_html(rows))
    manifest = {
        "kind": "final_result_evaluation_summary",
        "result_name": result_name,
        "method": method,
        "llm_mode": llm_mode,
        "rows": len(rows),
        "table": str(table),
        "markdown": str(md),
        "html": str(html),
        "note": "Final-only summary. It intentionally does not compare benchmark methods.",
    }
    write_json(output_dir / "final_result_evaluation_summary_manifest.json", manifest)
    return manifest
