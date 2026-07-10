from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ...base.config import CaseProfile
from ..final_evaluator import run_final_evaluator
from .hoi_contact_metrics import compute_hoi_contact_metrics
from .object_6d_metrics import compute_object_6d_metrics
from .overlay_metrics import compute_overlay_metrics
from .part_metrics import compute_part_metrics
from .penetration_floating_metrics import compute_penetration_floating_metrics
from .schemas import EvaluationPaths
from .temporal_audio_metrics import compute_temporal_audio_metrics
from .utils import write_json, write_rows


DETAIL_FIELDS = [
    "case",
    "result_name",
    "n_frames",
    "se3_valid",
    "translation_valid_rate",
    "rotation_valid_rate",
    "overlay_hard_score",
    "overlay_hard_metric_source",
    "overlay_mask_pair_count",
    "overlay_generated_render_mask_count",
    "overlay_generated_proxy_render_mask_count",
    "overlay_generated_full_geometry_mask_count",
    "overlay_mask_coverage",
    "overlay_render_false_coverage",
    "contact_frame_ratio",
    "contact_gap_mm",
    "contact_proxy",
    "contact_proxy_source",
    "part_correct_ratio",
    "hoi_contact_pair_rows",
    "hoi_contact_interval_count",
    "hoi_observed_contact_rows",
    "hoi_persistent_contact_rows",
    "contact_anchor_drift_mean",
    "contact_anchor_drift_max",
    "human_part_count",
    "human_part_available_count",
    "human_part_contact_coverage",
    "object_part_count",
    "object_part_available_count",
    "object_part_contact_coverage",
    "penetration_frame_ratio",
    "penetration_depth_mean_mm",
    "penetration_depth_max_mm",
    "floating_rate",
    "tradeoff_score",
    "object_jerk",
    "accel_at_events",
    "accel_in_flight",
    "contact_ratio_audio_windows",
    "jump_count",
    "static_tail_drift_m",
    "final_pass",
]


def _final_pass(metrics: dict[str, Any]) -> bool:
    if not metrics.get("se3_valid"):
        return False
    if metrics.get("overlay_hard_score") is not None and float(metrics["overlay_hard_score"]) < 0.5:
        return False
    if metrics.get("penetration_depth_max_mm") is not None and float(metrics["penetration_depth_max_mm"]) > 25.0:
        return False
    if metrics.get("tradeoff_score") is not None and float(metrics["tradeoff_score"]) < 0.2:
        return False
    return True


def run_unified_final_evaluation(profile: CaseProfile) -> dict[str, Any]:
    paths = EvaluationPaths.from_profile(profile)
    paths.evaluation_dir.mkdir(parents=True, exist_ok=True)
    blocks = [
        compute_object_6d_metrics(paths),
        compute_overlay_metrics(paths),
        compute_hoi_contact_metrics(paths),
        compute_part_metrics(paths, profile.data),
        compute_penetration_floating_metrics(paths),
        compute_temporal_audio_metrics(paths),
    ]
    metrics: dict[str, Any] = {}
    artifacts: dict[str, str] = {}
    warnings: list[str] = []
    for block in blocks:
        metrics.update(block.metrics)
        artifacts.update({f"{block.name}.{key}": value for key, value in block.artifacts.items()})
        warnings.extend(block.warnings)
    try:
        qa_summary = run_final_evaluator(profile, method="final_hoi", llm_mode="none")
        qa_dir = profile.result_dir / "vlm_trace" / "06_evaluation"
        qa_artifacts = {
            "qa.metric_scores_json": qa_dir / "metric_scores.json",
            "qa.failure_flags_json": qa_dir / "failure_flags.json",
            "qa.vlm_score_json": qa_dir / "vlm_score.json",
            "qa.vlm_framewise_judgment_csv": qa_dir / "vlm_framewise_judgment.csv",
            "qa.llm_audit_report_json": qa_dir / "llm_audit_report.json",
            "qa.evaluation_summary_json": qa_dir / "evaluation_summary.json",
            "qa.vlm_eval_queries_csv": qa_dir / "vlm_eval_queries.csv",
            "qa.vlm_eval_raw_responses_jsonl": qa_dir / "vlm_eval_raw_responses.jsonl",
            "qa.vlm_eval_parsed_scores_csv": qa_dir / "vlm_eval_parsed_scores.csv",
            "qa.vlm_eval_summary_json": qa_dir / "vlm_eval_summary.json",
            "qa.llm_eval_summary_md": qa_dir / "llm_eval_summary.md",
            "qa.qa_audit_report_html": qa_dir / "qa_audit_report.html",
            "qa.pipeline_qa_summary_csv": qa_dir / "pipeline_qa_summary.csv",
            "qa.pipeline_qa_summary_json": qa_dir / "pipeline_qa_summary.json",
            "qa.pipeline_qa_summary_md": qa_dir / "pipeline_qa_summary.md",
        }
        artifacts.update({name: str(path) for name, path in qa_artifacts.items() if path.exists()})
    except Exception as exc:
        qa_summary = {"status": "failed", "error": str(exc)}
        warnings.append(f"qa_final_evaluator_failed: {exc}")
    metrics["case"] = profile.case_name
    metrics["result_name"] = profile.result_name
    metrics["final_pass"] = _final_pass(metrics)
    summary = {
        "case": profile.case_name,
        "result_name": profile.result_name,
        "metrics": metrics,
        "metric_blocks": [block.name for block in blocks],
        "artifacts": artifacts,
        "warnings": warnings,
        "qa_summary": qa_summary,
        "principle": "hard metrics first; VLM is secondary for perceptual ambiguity",
    }
    write_json(paths.evaluation_dir / "final_evaluation_summary.json", summary)
    write_rows(paths.evaluation_dir / "final_evaluation_detailed.csv", [{field: metrics.get(field, "") for field in DETAIL_FIELDS}], DETAIL_FIELDS)
    _update_pipeline_manifest(profile)
    return summary


def _update_pipeline_manifest(profile: CaseProfile) -> None:
    manifest_path = profile.result_dir / "pipeline_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    except Exception:
        manifest = {}
    eval_dir = profile.result_dir / "evaluation"
    qa_dir = profile.result_dir / "vlm_trace" / "06_evaluation"
    manifest.setdefault("case", profile.case_name)
    manifest.setdefault("result_name", profile.result_name)
    manifest["final_hoi_evaluation"] = {
        "summary_json": str(eval_dir / "final_evaluation_summary.json"),
        "detailed_csv": str(eval_dir / "final_evaluation_detailed.csv"),
        "object_6d_metrics_json": str(eval_dir / "object_6d_metrics.json"),
        "overlay_metrics_json": str(eval_dir / "overlay_metrics.json"),
        "hoi_contact_metrics_csv": str(eval_dir / "hoi_contact_metrics.csv"),
        "part_metrics_csv": str(eval_dir / "part_metrics.csv"),
        "penetration_floating_metrics_json": str(eval_dir / "penetration_floating_metrics.json"),
        "temporal_audio_metrics_json": str(eval_dir / "temporal_audio_metrics.json"),
        "pipeline_qa_summary_csv": str(qa_dir / "pipeline_qa_summary.csv"),
        "qa_audit_report_html": str(qa_dir / "qa_audit_report.html"),
    }
    write_json(manifest_path, manifest)


def _human_readable(rows: list[dict[str, Any]]) -> str:
    fields = ["Case", "Object 6DoF", "Visual Overlay", "Contact/Anchor", "Physical", "Temporal"]
    lines = [
        "# Final HOI Evaluation Human-Readable Summary",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        overlay_raw = row.get("overlay_hard_score", "")
        overlay = _fmt(overlay_raw)
        visual = _status(_num(overlay_raw, 0.0), [(0.75, "ok"), (0.5, "partial")], "weak")
        if overlay not in {"", None}:
            visual = f"{visual} (overlay {overlay})"
        contact_proxy_raw = row.get("contact_proxy", "")
        contact_proxy = _fmt(contact_proxy_raw)
        drift_max = _fmt(row.get("contact_anchor_drift_max", ""), suffix="m")
        contact_status = _status(_num(contact_proxy_raw, 0.0), [(0.75, "strong"), (0.35, "medium")], "weak")
        contact = f"{contact_status} contact"
        if contact_proxy not in {"", None}:
            contact += f", proxy {contact_proxy}"
        if drift_max not in {"", None}:
            contact += f", drift max {drift_max}"
        pen_rate = _fmt(row.get("penetration_frame_ratio", ""))
        float_rate = _fmt(row.get("floating_rate", ""))
        trade_raw = row.get("tradeoff_score", "")
        trade = _fmt(trade_raw)
        physical_status = _status(_num(trade_raw, 0.0), [(0.75, "ok"), (0.4, "mixed")], "weak")
        physical = f"{physical_status}: penetration {pen_rate}, floating {float_rate}, tradeoff {trade}"
        jerk = _fmt(row.get("object_jerk", ""))
        static = _fmt(row.get("static_tail_drift_m", ""), suffix="m")
        jumps = _fmt(row.get("jump_count", ""), decimals=0)
        temporal = f"jump {jumps}, jerk {jerk}, static drift {static}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("case", "")),
                    "yes" if row.get("se3_valid") in {True, "True", "true", "1", "1.0"} else "no",
                    visual,
                    contact,
                    physical,
                    temporal,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _num(value: object, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _status(value: float, thresholds: list[tuple[float, str]], fallback: str) -> str:
    for threshold, label in thresholds:
        if value >= threshold:
            return label
    return fallback


def _fmt(value: object, *, suffix: str = "", decimals: int = 3) -> str:
    if value in {"", None}:
        return "n/a"
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except Exception:
        return f"{value}{suffix}"
    if not math.isfinite(numeric):
        return "n/a"
    if decimals <= 0:
        return f"{numeric:.0f}{suffix}"
    text = f"{numeric:.{decimals}f}".rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return f"{text}{suffix}"


def write_unified_final_summary(profiles: list[CaseProfile], *, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [run_unified_final_evaluation(profile) for profile in profiles]
    rows = [{field: summary["metrics"].get(field, "") for field in DETAIL_FIELDS} for summary in summaries]
    table = write_rows(output_dir / "final_evaluation_detailed.csv", rows, DETAIL_FIELDS)
    md = output_dir / "final_evaluation_human_readable.md"
    md.write_text(_human_readable(rows))
    case_entries = []
    for profile in profiles:
        eval_dir = profile.result_dir / "evaluation"
        qa_dir = profile.result_dir / "vlm_trace" / "06_evaluation"
        case_entries.append(
            {
                "case": profile.case_name,
                "result_name": profile.result_name,
                "result_dir": str(profile.result_dir),
                "evaluation_dir": str(eval_dir),
                "evaluation_summary_json": str(eval_dir / "final_evaluation_summary.json"),
                "detailed_csv": str(eval_dir / "final_evaluation_detailed.csv"),
                "qa_dir": str(qa_dir),
                "pipeline_qa_summary_csv": str(qa_dir / "pipeline_qa_summary.csv"),
                "qa_audit_report_html": str(qa_dir / "qa_audit_report.html"),
            }
        )
    manifest = {
        "kind": "unified_final_hoi_summary",
        "rows": len(rows),
        "table": str(table),
        "human_readable": str(md),
        "cases": case_entries,
        "principle": "hard metrics first; VLM is secondary for perceptual ambiguity",
    }
    write_json(output_dir / "final_evaluation_summary_manifest.json", manifest)
    return manifest
