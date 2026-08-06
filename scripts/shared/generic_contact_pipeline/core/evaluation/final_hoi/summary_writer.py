from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ...base.config import CaseProfile
from ..final_evaluator import run_final_evaluator
from .gate_impact_metrics import compute_gate_impact_metrics
from .hoi_contact_metrics import compute_hoi_contact_metrics
from .object_6d_metrics import compute_object_6d_metrics
from .overlay_metrics import compute_overlay_metrics
from .part_metrics import compute_part_metrics
from .penetration_floating_metrics import compute_penetration_floating_metrics, tradeoff_score
from .schemas import EvaluationPaths
from .temporal_audio_metrics import compute_temporal_audio_metrics
from .temporal_plausibility_metrics import compute_temporal_plausibility_metrics
from .utils import write_json, write_rows


DETAIL_FIELDS = [
    "case",
    "result_name",
    "evaluation_source",
    "final_video",
    "source_video",
    "object_pose_source",
    "gate_trace_source",
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
    "translation_spike_count",
    "rotation_spike_count",
    "event_aligned_spike_count",
    "non_event_spike_count",
    "high_speed_recall",
    "oversmooth_rate",
    "temporal_failure_intervals",
    "gate_impact_status",
    "gate_event_count",
    "gate_active_count",
    "gate_reject_unclear_count",
    "stage_residual_reweight_count",
    "optimizer_reoptimized_frames",
    "optimizer_reweighted_frames",
    "anchor_update_allowed_count",
    "anchor_update_blocked_count",
    "freeze_interpolation_frames",
    "pose_delta_translation_mean_m",
    "pose_delta_translation_max_m",
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


def run_unified_final_evaluation(profile: CaseProfile, *, run_qa: bool = True) -> dict[str, Any]:
    paths = EvaluationPaths.from_profile(profile)
    paths.evaluation_dir.mkdir(parents=True, exist_ok=True)
    blocks = [
        compute_object_6d_metrics(paths),
        compute_overlay_metrics(paths),
        compute_hoi_contact_metrics(paths),
        compute_part_metrics(paths, profile.data),
        compute_penetration_floating_metrics(paths),
        compute_temporal_audio_metrics(paths),
        compute_temporal_plausibility_metrics(paths),
        compute_gate_impact_metrics(paths),
    ]
    metrics: dict[str, Any] = {}
    artifacts: dict[str, str] = {}
    warnings: list[str] = []
    for block in blocks:
        metrics.update(block.metrics)
        artifacts.update({f"{block.name}.{key}": value for key, value in block.artifacts.items()})
        warnings.extend(block.warnings)
    # Object-only evaluations can provide a typed object/tool contact gap
    # without a downstream human-geometry penetration artifact.  Preserve the
    # repository's existing tradeoff formula instead of silently omitting the
    # physical score in that case.
    if metrics.get("tradeoff_score") is None and metrics.get("contact_gap_mm") is not None:
        metrics["tradeoff_score"] = tradeoff_score(
            metrics.get("contact_gap_mm"),
            metrics.get("penetration_depth_mean_mm"),
        )
    try:
        if not run_qa:
            raise RuntimeError("qa_skipped_for_manifest_backed_final_result")
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
        skipped = str(exc) == "qa_skipped_for_manifest_backed_final_result"
        qa_summary = {"status": "skipped" if skipped else "failed", "reason" if skipped else "error": str(exc)}
        if not skipped:
            warnings.append(f"qa_final_evaluator_failed: {exc}")
    source = dict(profile.data.get("evaluation_source", {}))
    metrics["case"] = profile.case_name
    metrics["result_name"] = profile.result_name
    metrics["evaluation_source"] = "final_result_manifest" if source else "pipeline_result_directory"
    metrics["final_video"] = source.get("final_video", "")
    metrics["source_video"] = source.get("source_video", "") or ""
    metrics["object_pose_source"] = source.get("object_pose_csv", "") or ""
    metrics["gate_trace_source"] = source.get("gate_trace_result_dir", "") or ""
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
        "temporal_plausibility_metrics_json": str(eval_dir / "temporal_plausibility_metrics.json"),
        "gate_impact_metrics_json": str(eval_dir / "gate_impact_metrics.json"),
        "pipeline_qa_summary_csv": str(qa_dir / "pipeline_qa_summary.csv"),
        "qa_audit_report_html": str(qa_dir / "qa_audit_report.html"),
    }
    write_json(manifest_path, manifest)


def _human_readable(rows: list[dict[str, Any]]) -> str:
    fields = ["Case", "Object 6DoF", "Visual Overlay", "Contact/Anchor", "Physical", "Temporal"]
    lines = [
        "# Final HOI Evaluation Human-Readable Summary",
        "",
        "Reader-facing table: only metrics with current final-result evidence are shown in the main cells.",
        "Unavailable anchor drift, floating, legacy jump count, and static-tail drift are listed below the table when their source artifacts are absent.",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    unavailable: dict[str, set[str]] = {}
    for row in rows:
        case = str(row.get("case", ""))
        unavailable[case] = set()
        frame_count = _fmt(row.get("n_frames", ""), decimals=0)
        translation_rate = _fmt(row.get("translation_valid_rate", ""))
        rotation_rate = _fmt(row.get("rotation_valid_rate", ""))
        object_6d = (
            f"SE3={'yes' if row.get('se3_valid') in {True, 'True', 'true', '1', '1.0'} else 'no'}; "
            f"frames={frame_count}; T/R valid={translation_rate}/{rotation_rate}"
        )

        overlay = _fmt(row.get("overlay_hard_score", ""))
        mask_coverage = _fmt(row.get("overlay_mask_coverage", ""))
        false_coverage = _fmt(row.get("overlay_render_false_coverage", ""))
        overlay_source = str(row.get("overlay_hard_metric_source", "")).replace("_", " ")
        visual = f"IoU={overlay}; mask coverage={mask_coverage}; false coverage={false_coverage}; source={overlay_source}"

        contact_proxy_raw = row.get("contact_proxy", "")
        contact_proxy = _fmt(contact_proxy_raw)
        contact_gap = _fmt(row.get("contact_gap_mm", ""), suffix="mm")
        contact_frame_ratio = _fmt(row.get("contact_frame_ratio", ""))
        part_correct = _fmt(row.get("part_correct_ratio", ""))
        observed_contacts = _fmt(row.get("hoi_observed_contact_rows", ""), decimals=0)
        contact = (
            f"proxy={contact_proxy}; gap={contact_gap}; contact frames={contact_frame_ratio}; "
            f"observed rows={observed_contacts}; part correct={part_correct}"
        )
        if row.get("contact_anchor_drift_max", "") in {"", None}:
            unavailable[case].add("anchor drift: no stable/observed local anchor coordinates in final contact artifacts")

        pen_rate = _fmt(row.get("penetration_frame_ratio", ""))
        pen_mean = _fmt(row.get("penetration_depth_mean_mm", ""), suffix="mm")
        pen_max = _fmt(row.get("penetration_depth_max_mm", ""), suffix="mm")
        trade_raw = row.get("tradeoff_score", "")
        trade = _fmt(trade_raw)
        physical = f"penetration rate={pen_rate}; depth mean/max={pen_mean}/{pen_max}; contact-physics tradeoff={trade}"
        if row.get("floating_rate", "") in {"", None}:
            unavailable[case].add("floating: no final support-gap/floor-state artifact")

        jerk = _fmt(row.get("object_jerk", ""))
        trans_spikes = _fmt(row.get("translation_spike_count", ""), decimals=0)
        rot_spikes = _fmt(row.get("rotation_spike_count", ""), decimals=0)
        event_spikes = _fmt(row.get("event_aligned_spike_count", ""), decimals=0)
        non_event_spikes = _fmt(row.get("non_event_spike_count", ""), decimals=0)
        high_speed = _fmt(row.get("high_speed_recall", ""))
        oversmooth = _fmt(row.get("oversmooth_rate", ""))
        failure_intervals = str(row.get("temporal_failure_intervals", "") or "[]")
        temporal = (
            f"jerk={jerk}; T/R spikes={trans_spikes}/{rot_spikes}; "
            f"event/non-event spikes={event_spikes}/{non_event_spikes}; "
            f"high-speed recall={high_speed}; oversmooth={oversmooth}; failures={failure_intervals}"
        )
        if row.get("jump_count", "") in {"", None}:
            unavailable[case].add("legacy jump_count: replaced by motion-regime spike metrics")
        if row.get("static_tail_drift_m", "") in {"", None}:
            unavailable[case].add("static-tail drift: no explicit static interval for this final result")

        lines.append(
            "| "
            + " | ".join(
                [
                    case,
                    object_6d,
                    visual,
                    contact,
                    physical,
                    temporal,
                ]
            )
            + " |"
        )
    lines.extend(["", "## Unavailable Evidence Notes", ""])
    for case, notes in unavailable.items():
        if not notes:
            continue
        lines.append(f"- {case}: " + "; ".join(sorted(notes)) + ".")
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


def write_unified_final_summary(
    profiles: list[CaseProfile], *, output_dir: Path, run_qa: bool = True
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [run_unified_final_evaluation(profile, run_qa=run_qa) for profile in profiles]
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
