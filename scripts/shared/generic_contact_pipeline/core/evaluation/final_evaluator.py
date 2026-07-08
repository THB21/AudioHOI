from __future__ import annotations

import math
import json
from html import escape
from pathlib import Path
from typing import Iterable

from ..base.config import CaseProfile
from ..base.io import float_or_none, read_csv, write_csv, write_json
from ..base.schema import stage_paths
from .llm_csv_audit import run_llm_csv_audit
from .vlm_trace import export_vlm_trace


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        return read_csv(path)
    except Exception:
        return []


def _mean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def _max(values: Iterable[float]) -> float | None:
    vals = [v for v in values if math.isfinite(v)]
    return max(vals) if vals else None


def _series(rows: list[dict[str, str]], *keys: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        for key in keys:
            value = float_or_none(row.get(key))
            if value is not None and math.isfinite(value):
                out.append(value)
                break
    return out


def _truthy_series(rows: list[dict[str, str]], *keys: str) -> list[float]:
    return [1.0 if _active(row, *keys) else 0.0 for row in rows]


def _anchor_drift_values(rows: list[dict[str, str]]) -> list[float]:
    out: list[float] = []
    for row in rows:
        direct = None
        for key in ("anchor_drift_m", "hand_object_z_gap_m", "hand_contact_3d_gap_m"):
            direct = float_or_none(row.get(key))
            if direct is not None:
                break
        if direct is not None:
            out.append(abs(direct))
            continue
        sx = float_or_none(row.get("stable_local_x"))
        sy = float_or_none(row.get("stable_local_y"))
        sz = float_or_none(row.get("stable_local_z"))
        ox = float_or_none(row.get("observed_local_x"))
        oy = float_or_none(row.get("observed_local_y"))
        oz = float_or_none(row.get("observed_local_z"))
        if None not in {sx, sy, sz, ox, oy, oz}:
            out.append(math.sqrt((ox - sx) ** 2 + (oy - sy) ** 2 + (oz - sz) ** 2))  # type: ignore[operator]
            continue
        ss = float_or_none(row.get("stable_local_s"))
        os = float_or_none(row.get("observed_local_s"))
        if ss is None:
            ss = float_or_none(row.get("stable_object_local_s"))
        if os is None:
            os = float_or_none(row.get("observed_object_local_s"))
        if ss is not None and os is not None:
            out.append(abs(os - ss))
            continue
        drift_s = float_or_none(row.get("local_s_drift"))
        if drift_s is not None:
            out.append(abs(drift_s))
    return out


def _contact_proxy_score(rows: list[dict[str, str]]) -> float | None:
    if not rows:
        return None
    scores: list[float] = []
    for row in rows:
        active = _active(row, "contact_observed", "contact_active")
        conf = None
        for key in ("contact_confidence", "contact_conf", "anchor_score"):
            conf = float_or_none(row.get(key))
            if conf is not None:
                break
        if conf is None:
            conf = 1.0 if active else 0.0
        conf = max(0.0, min(1.0, conf))
        scores.append(conf if active else 0.0)
    return _mean(scores)


def _overlay_proxy_score(obs_rows: list[dict[str, str]], line_rows: list[dict[str, str]]) -> float | None:
    direct = _series(obs_rows, "mask_iou", "object_mask_iou", "observation_conf", "semantic_conf", "track_geometry_conf", "mask_conf")
    if direct:
        return _mean([max(0.0, min(1.0, value)) for value in direct])
    trusted = _series(line_rows, "line_observation_trusted", "endpoint_track_conf")
    if trusted:
        return _mean([max(0.0, min(1.0, value)) for value in trusted])
    jitter = _series(obs_rows, "proxy_jitter_px", "support_jitter_px")
    if jitter:
        return _mean([1.0 / (1.0 + max(0.0, value) / 20.0) for value in jitter])
    return None


def _signed_depth_offsets(contact_rows: list[dict[str, str]]) -> list[float]:
    return _series(contact_rows, "contact_depth_offset_m", "hand_object_z_gap_m", "depth_offset_m")


def _active(row: dict[str, str], *keys: str) -> bool:
    for key in keys:
        value = str(row.get(key, "")).strip().lower()
        if value in {"1", "1.0", "true", "yes", "active"}:
            return True
    return False


def _frame(row: dict[str, str]) -> int | None:
    try:
        return int(float(row.get("frame", "") or ""))
    except Exception:
        return None


def _contact_f1(pred_rows: list[dict[str, str]], label_rows: list[dict[str, str]]) -> dict[str, object]:
    if not label_rows:
        return {
            "status": "needs_manual_labels",
            "precision": None,
            "recall": None,
            "f1": None,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
    pred = {_frame(row): _active(row, "contact_observed", "contact_active") for row in pred_rows if _frame(row) is not None}
    labels = {_frame(row): _active(row, "contact_observed", "contact_active", "label") for row in label_rows if _frame(row) is not None}
    tp = fp = fn = 0
    for frame, truth in labels.items():
        guess = bool(pred.get(frame, False))
        if guess and truth:
            tp += 1
        elif guess and not truth:
            fp += 1
        elif truth and not guess:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"status": "ok", "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _manual_contact_labels(profile: CaseProfile) -> list[dict[str, str]]:
    path = profile.sample_dir / "annotations" / "evaluation" / "contact_labels.csv"
    return _rows(path)


def _jump_count(rows: list[dict[str, str]]) -> int:
    count = 0
    for row in rows:
        if any(row.get(key) in {"1", "1.0", "true", "True"} for key in ("visual_spike", "contact_spike", "smoothness_spike")):
            count += 1
    return count


def _length_spread(rows: list[dict[str, str]]) -> float | None:
    values = _series(rows, "length_m", "object_length_m", "geometry_length_m", "physical_length_m")
    if len(values) < 2:
        return None
    return max(values) - min(values)


def compute_metric_scores(profile: CaseProfile, *, method: str = "vlm_gated") -> dict[str, object]:
    paths = stage_paths(profile)
    obs = _rows(paths["object_observations"])
    line = _rows(paths["line_correspondence"])
    contact = _rows(paths["contact_candidates"])
    anchors = _rows(paths["anchor_state"])
    residuals = _rows(paths["physical_smooth_residuals"])
    jumps = _rows(paths["pose_jump_audit"])
    pose = _rows(paths["object_pose"])
    labels = _manual_contact_labels(profile)
    contact_metrics = _contact_f1(contact, labels)
    if method == "oracle_contact" and labels:
        contact_metrics = {
            "status": "oracle",
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "tp": sum(1 for row in labels if _active(row, "contact_observed", "contact_active", "label")),
            "fp": 0,
            "fn": 0,
        }
    depth_offsets = _signed_depth_offsets(contact)
    penetration_values = _series(residuals, "penetration_depth_m", "penetration_m", "E_penetration")
    if depth_offsets:
        penetration_values = penetration_values + [abs(value) for value in depth_offsets if value < -1e-6]
    floating_values = _series(residuals, "floating_gap_m", "support_gap_m", "E_support")
    support_gap_px = _series(obs, "support_gap_px")
    floating_values = floating_values + [value / 1000.0 for value in support_gap_px if value > 1e-6]
    if depth_offsets:
        floating_values = floating_values + [value for value in depth_offsets if value > 1e-6]
    static_values = _series(jumps, "static_tail_drift_m", "static_drift_m")
    overlay_values = _series(obs, "overlay_error_px", "residual_px", "line_rmse_px", "proxy_jitter_px")
    if not overlay_values:
        overlay_values = _series(line, "center_jump_px")
    mask_values = _series(obs, "mask_iou", "object_mask_iou")
    if not mask_values:
        trusted = _series(line, "line_observation_trusted", "endpoint_track_conf")
        mask_values = trusted if trusted else []
    anchor_drift = _anchor_drift_values(anchors)
    if not anchor_drift:
        anchor_drift = _anchor_drift_values(contact)
    if not anchor_drift:
        anchor_drift = [abs(value) for value in depth_offsets]
    overlay_proxy = _overlay_proxy_score(obs, line)
    contact_proxy = _contact_proxy_score(contact)
    metrics = {
        "overlay_error_px_mean": _mean(overlay_values),
        "mask_iou_mean": _mean(mask_values),
        "overlay_proxy_score": overlay_proxy,
        "contact_precision": contact_metrics["precision"],
        "contact_recall": contact_metrics["recall"],
        "contact_f1": contact_metrics["f1"],
        "contact_label_status": contact_metrics["status"],
        "contact_proxy_score": contact_proxy,
        "anchor_drift_m_mean": _mean(anchor_drift),
        "anchor_drift_m_max": _max(anchor_drift),
        "penetration_rate": sum(1 for v in penetration_values if v > 1e-4) / len(penetration_values) if penetration_values else 0.0,
        "floating_rate": sum(1 for v in floating_values if v > 1e-3) / len(floating_values) if floating_values else 0.0,
        "velocity_spike_count": sum(1 for v in _series(residuals, "velocity_norm", "velocity_residual") if v > 0.5),
        "acceleration_spike_count": sum(1 for v in _series(residuals, "acceleration_norm", "acceleration_residual") if v > 0.5),
        "jump_count": _jump_count(jumps),
        "static_tail_drift_m_max": _max(static_values) if static_values else 0.0,
        "geometry_length_spread_m": _length_spread(pose) if _length_spread(pose) is not None else _length_spread(line),
    }
    metrics["gt_contact_available"] = bool(labels)
    if method == "oracle_contact" and not labels:
        metrics["method_status"] = "unavailable:no_manual_labels"
    return metrics


def _failure_flags(metrics: dict[str, object]) -> dict[str, object]:
    def num(key: str, default: float = 0.0) -> float:
        value = metrics.get(key)
        try:
            if value in {"", None}:
                return default
            return float(value)  # type: ignore[arg-type]
        except Exception:
            return default

    flags = {
        "overlay_fail": num("overlay_error_px_mean") > 20.0,
        "contact_fail": metrics.get("contact_label_status") == "ok" and num("contact_f1") < 0.8,
        "anchor_drift_fail": num("anchor_drift_m_max") > 0.08,
        "penetration_fail": num("penetration_rate") > 0.05,
        "floating_fail": num("floating_rate") > 0.10,
        "jump_fail": num("jump_count") > 0,
        "static_tail_fail": num("static_tail_drift_m_max") > 0.03,
        "geometry_fail": num("geometry_length_spread_m") > 0.05,
    }
    flags["failure_count"] = sum(1 for value in flags.values() if value is True)
    return flags


def _existing_vlm_visual_judgment(profile: CaseProfile) -> tuple[dict[str, object], list[dict[str, object]]] | None:
    paths = stage_paths(profile)
    visual_rows: list[dict[str, str]] = []
    query_types = {
        "target_mask_check",
        "keypart_identity_check",
        "keypart_visibility_check",
        "track_stability_check",
        "contact_relation_check",
        "overlay_alignment_check",
        "anchor_update_check",
        "temporal_motion_check",
        "post_render_sanity_check",
        "baseline_regression_check",
        "loss_diagnostic_check",
    }
    for stage in ("stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "stage7"):
        for row in _rows(paths["vlm_dir"] / stage / "vlm_gates.csv"):
            if row.get("query_type") in query_types and row.get("pass_gate") not in {"", "not_evaluated"}:
                visual_rows.append({"stage": stage, **row})
    if not visual_rows:
        for row in _rows(paths["vlm_gates"]):
            if row.get("query_type") in query_types and row.get("pass_gate") not in {"", "not_evaluated"}:
                visual_rows.append(row)
    if not visual_rows:
        return None

    score_by_gate = {"pass": 1.0, "unclear": 0.5, "reject": 0.0}
    framewise: list[dict[str, object]] = []
    scores: list[float] = []
    for row in visual_rows:
        gate = row.get("pass_gate", "unclear")
        score = score_by_gate.get(gate, 0.5)
        scores.append(score)
        framewise.append(
            {
                "frame": row.get("frame", ""),
                "question_id": row.get("query_type", ""),
                "question": row.get("query_type", ""),
                "label": "pass" if gate == "pass" else "fail" if gate == "reject" else "unclear",
                "score": score,
                "reason": row.get("reason", row.get("normalized_label", row.get("repair_action", ""))),
            }
        )
    return (
        {
            "mode": "existing_vlm_artifacts",
            "score": sum(scores) / len(scores) if scores else 0.0,
            "checklist_count": len(framewise),
            "pass_count": sum(1 for row in visual_rows if row.get("pass_gate") == "pass"),
            "unclear_count": sum(1 for row in visual_rows if row.get("pass_gate") == "unclear"),
            "reject_count": sum(1 for row in visual_rows if row.get("pass_gate") == "reject"),
            "note": "Scored from existing VLM gate artifacts; no pose coordinates are generated here.",
        },
        framewise,
    )


def _vlm_visual_judgment(profile: CaseProfile, metrics: dict[str, object], flags: dict[str, object]) -> tuple[dict[str, object], list[dict[str, object]]]:
    existing = _existing_vlm_visual_judgment(profile)
    pipeline_score = existing[0] if existing is not None else {}
    pipeline_framewise = existing[1] if existing is not None else []
    paths = stage_paths(profile)
    pose = _rows(paths["object_pose"])
    sample_frames = [_frame(row) for row in pose[:5]]
    sample_frames = [fr for fr in sample_frames if fr is not None] or [""]
    questions = [
        ("object_overlay_alignment", "Does the rendered object align with the visible object?", not flags["overlay_fail"]),
        ("contact_visual_plausibility", "Are contact points visually attached to the expected human/object parts?", not flags["contact_fail"]),
        ("physical_artifacts", "Are floating, penetration, jumps, and object length changes absent?", not (flags["floating_fail"] or flags["penetration_fail"] or flags["jump_fail"] or flags["geometry_fail"])),
        ("static_tail_stability", "Does the static tail stay stable?", not flags["static_tail_fail"]),
    ]
    rows: list[dict[str, object]] = []
    passed = 0
    for frame in sample_frames:
        for question_id, question, ok in questions:
            rows.append(
                {
                    "frame": frame,
                    "question_id": question_id,
                    "question": question,
                    "label": "pass" if ok else "fail",
                    "score": 1.0 if ok else 0.0,
                    "reason": "dry-run evaluator derived from hard metrics; replace with Qwen visual judge in qwen mode",
                }
            )
            passed += 1 if ok else 0
    score = passed / len(rows) if rows else 0.0
    return (
        {
            "mode": "dry-run_metric_grounded",
            "score": score,
            "checklist_count": len(rows),
            "pipeline_vlm_gate_score": pipeline_score,
            "pipeline_vlm_gate_framewise_count": len(pipeline_framewise),
            "note": "VLM visual judge artifact is separated from hard metrics; no pose coordinates are generated here.",
        },
        rows,
    )


def _llm_csv_audit(profile: CaseProfile, metrics: dict[str, object], flags: dict[str, object], *, llm_mode: str) -> dict[str, object]:
    if llm_mode in {"seed", "mistral"}:
        run_result = run_llm_csv_audit(profile, mode=llm_mode)
        paths = stage_paths(profile)
        audit_results_path = paths["llm_csv_audit_results"]
        if audit_results_path.exists():
            try:
                import json

                data = json.loads(audit_results_path.read_text())
            except Exception:
                data = {}
            return {
                "mode": data.get("mode", "deterministic_csv_audit"),
                "source_path": str(audit_results_path),
                "failure_count": run_result.get("failure_count", 0),
                "blocking_count": run_result.get("blocking_count", 0),
                "summary": "pass" if run_result.get("failure_count", 0) == 0 else "review_required",
                "results": data.get("results", []),
                "llm_trace": data.get("llm_trace", {}),
            }
    paths = stage_paths(profile)
    required = {
        "anchor_state": paths["anchor_state"],
        "gate_timeline": profile.result_dir / "vlm_trace" / "04_gating" / "gate_timeline.csv",
        "physical_smooth_residuals": paths["physical_smooth_residuals"],
        "pose_jump_audit": paths["pose_jump_audit"],
        "optimizer_decisions": paths["optimizer_decisions"],
    }
    missing = [name for name, path in required.items() if not path.exists()]
    failure_sources: list[str] = []
    if flags["overlay_fail"] or flags["geometry_fail"]:
        failure_sources.append("stage1_observation_or_stage3_pose_init")
    if flags["contact_fail"] or flags["anchor_drift_fail"]:
        failure_sources.append("stage2_contact_anchor")
    if flags["jump_fail"] or flags["static_tail_fail"] or flags["floating_fail"] or flags["penetration_fail"]:
        failure_sources.append("stage4_sequence_optimizer")
    if missing:
        failure_sources.append("trace_or_schema")
    return {
        "mode": "deterministic_csv_audit",
        "missing_inputs": missing,
        "failure_sources": sorted(set(failure_sources)),
        "gate_effectiveness_checked": (profile.result_dir / "vlm_trace" / "04_gating" / "gate_timeline.csv").exists(),
        "summary": "pass" if not failure_sources else "review_required",
        "metrics_used": sorted(metrics.keys()),
    }


def _num(value: object, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _bounded_score(value: float) -> int:
    return max(1, min(5, int(round(value))))


def _failure_stage_hint(flags: dict[str, object]) -> str:
    if flags.get("overlay_fail") or flags.get("geometry_fail"):
        return "stage1_observation|stage3_pose_init"
    if flags.get("contact_fail") or flags.get("anchor_drift_fail"):
        return "stage2_contact"
    if flags.get("penetration_fail") or flags.get("floating_fail") or flags.get("jump_fail") or flags.get("static_tail_fail"):
        return "stage4_optimizer"
    return "pass"


def _checklist_scores(metrics: dict[str, object], flags: dict[str, object]) -> dict[str, object]:
    overlay = metrics.get("overlay_proxy_score")
    if overlay is None:
        overlay = metrics.get("mask_iou_mean")
    contact = metrics.get("contact_f1")
    if contact is None:
        contact = metrics.get("contact_proxy_score")
    visibility = _bounded_score(1.0 + 4.0 * _num(overlay, 0.5))
    contact_correctness = _bounded_score(1.0 + 4.0 * _num(contact, 0.5))
    support_consistency = _bounded_score(5.0 - 4.0 * _num(metrics.get("floating_rate"), 0.0))
    penetration_absence = _bounded_score(5.0 - 4.0 * _num(metrics.get("penetration_rate"), 0.0))
    temporal_plausibility = _bounded_score(5.0 - min(4.0, _num(metrics.get("jump_count"), 0.0)))
    overall = _bounded_score(
        (visibility + contact_correctness + support_consistency + penetration_absence + temporal_plausibility) / 5.0
    )
    return {
        "visibility": visibility,
        "contact_correctness": contact_correctness,
        "support_consistency": support_consistency,
        "penetration_absence": penetration_absence,
        "temporal_plausibility": temporal_plausibility,
        "overall_plausibility": overall,
        "short_reason": "metric-grounded final evaluator checklist; replace source with Qwen final judge when available",
        "failure_stage_hint": _failure_stage_hint(flags),
    }


def _representative_frames(profile: CaseProfile) -> list[str]:
    rows = _rows(stage_paths(profile)["object_pose"])
    frames = [str(_frame(row)) for row in rows if _frame(row) is not None]
    if not frames:
        return [""]
    picks = [frames[0], frames[len(frames) // 2], frames[-1]]
    return list(dict.fromkeys(picks))


def _write_final_eval_qa(
    profile: CaseProfile,
    metrics: dict[str, object],
    flags: dict[str, object],
    vlm_score: dict[str, object],
    framewise: list[dict[str, object]],
    llm_report: dict[str, object],
) -> dict[str, object]:
    eval_dir = profile.result_dir / "vlm_trace" / "06_evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    trace_root = profile.result_dir / "vlm_trace"
    evidence_by_stage = {
        "stage1_observation": trace_root / "01_observation" / "object_mask_overlay.mp4",
        "stage2_contact": trace_root / "02_contact_candidates" / "contact_candidate_overlay.mp4",
        "stage4_optimizer": trace_root / "05_optimization" / "before_after_overlay.mp4",
        "stage5_render": trace_root / "06_evaluation" / "final_render_review.mp4",
    }
    question = (
        "Judge the final HOI reconstruction for visibility, contact correctness, support consistency, "
        "penetration absence, temporal plausibility, and overall plausibility. Return the fixed JSON schema."
    )
    schema = {
        "visibility": "1-5",
        "contact_correctness": "1-5",
        "support_consistency": "1-5",
        "penetration_absence": "1-5",
        "temporal_plausibility": "1-5",
        "overall_plausibility": "1-5",
        "short_reason": "string",
        "failure_stage_hint": "stage2_contact|stage4_optimizer|stage5_render|unclear|pass",
    }
    scores = _checklist_scores(metrics, flags)
    frames = _representative_frames(profile)
    queries: list[dict[str, object]] = []
    parsed_rows: list[dict[str, object]] = []
    raw_lines: list[str] = []
    affected = {
        "stage1_observation": "visual_overlay_residual",
        "stage2_contact": "contact_anchor_residual",
        "stage4_optimizer": "velocity_acceleration_static_residual",
        "stage5_render": "render_acceptance_gate",
    }
    for idx, frame in enumerate(frames):
        stage = "stage4_optimizer" if scores["failure_stage_hint"] != "stage2_contact" else "stage2_contact"
        evidence_path = evidence_by_stage.get(stage, evidence_by_stage["stage4_optimizer"])
        query_id = f"final_eval_{idx:03d}"
        query = {
            "query_id": query_id,
            "stage": stage,
            "frame": frame,
            "window": "representative_interval",
            "evidence_path": str(evidence_path),
            "question": question,
            "json_schema": json.dumps(schema, sort_keys=True),
            "source": "final_evaluator",
        }
        raw_answer = dict(scores)
        raw_answer["source"] = vlm_score.get("mode", "metric_grounded_surrogate")
        parsed = {
            "query_id": query_id,
            "stage": stage,
            "frame": frame,
            "window": "representative_interval",
            "evidence_path": str(evidence_path),
            "question": question,
            "raw_answer": json.dumps(raw_answer, sort_keys=True),
            "affected_constraint": affected.get(stage, "evaluation_report"),
            "changed_optimizer_behavior": "0",
            **scores,
        }
        queries.append(query)
        parsed_rows.append(parsed)
        raw_lines.append(json.dumps({"query": query, "raw_answer": raw_answer}, sort_keys=True))
    write_csv(eval_dir / "vlm_eval_queries.csv", queries)
    (eval_dir / "vlm_eval_raw_responses.jsonl").write_text("\n".join(raw_lines) + ("\n" if raw_lines else ""))
    write_csv(eval_dir / "vlm_eval_parsed_scores.csv", parsed_rows)
    summary = {
        "mode": vlm_score.get("mode", "metric_grounded_surrogate"),
        "query_count": len(parsed_rows),
        "mean_overall_plausibility": _mean([_num(row.get("overall_plausibility")) for row in parsed_rows]),
        "failure_stage_hint": scores["failure_stage_hint"],
        "scores": scores,
        "vlm_pipeline_score": vlm_score,
        "framewise_pipeline_judgment_count": len(framewise),
    }
    write_json(eval_dir / "vlm_eval_summary.json", summary)
    llm_md = [
        "# LLM CSV Audit Summary",
        "",
        f"- Mode: {llm_report.get('mode', 'deterministic_csv_audit')}",
        f"- Summary: {llm_report.get('summary', '')}",
        f"- Failure stage hint: {scores['failure_stage_hint']}",
        f"- Missing inputs: {', '.join(llm_report.get('missing_inputs', []) or [])}",
        "",
        "The audit reads CSV/metric artifacts and reports whether gate, anchor, smooth, and static-tail evidence is consistent. It does not generate pose.",
    ]
    (eval_dir / "llm_eval_summary.md").write_text("\n".join(llm_md) + "\n")
    rows_html = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{escape(str(row.get(key, '')))}</td>"
            for key in (
                "query_id",
                "stage",
                "frame",
                "question",
                "overall_plausibility",
                "failure_stage_hint",
                "affected_constraint",
                "changed_optimizer_behavior",
                "evidence_path",
            )
        )
        + "</tr>"
        for row in parsed_rows
    )
    (eval_dir / "qa_audit_report.html").write_text(
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>QA Audit Report</title>
<style>body{font-family:system-ui,sans-serif;margin:28px;color:#263238}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccd6dd;padding:7px;vertical-align:top;font-size:13px}th{background:#eef3f5}</style>
</head><body><h1>VLM/LLM QA Audit Report</h1><table><thead><tr>
<th>Query</th><th>Stage</th><th>Frame</th><th>Question</th><th>Overall</th><th>Failure Stage</th><th>Affected Constraint</th><th>Changed Optimizer</th><th>Evidence</th>
</tr></thead><tbody>"""
        + rows_html
        + "</tbody></table></body></html>\n"
    )
    return summary


def run_final_evaluator(profile: CaseProfile, *, method: str = "vlm_gated", llm_mode: str = "seed") -> dict[str, object]:
    export_vlm_trace(profile)
    eval_dir = profile.result_dir / "vlm_trace" / "06_evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metric_scores(profile, method=method)
    flags = _failure_flags(metrics)
    vlm_score, framewise = _vlm_visual_judgment(profile, metrics, flags)
    llm_report = _llm_csv_audit(profile, metrics, flags, llm_mode=llm_mode)
    qa_summary = _write_final_eval_qa(profile, metrics, flags, vlm_score, framewise, llm_report)
    final_pass = not any(value is True for value in flags.values())
    summary = {
        "case": profile.case_name,
        "method": method,
        "result_name": profile.result_name,
        "method_status": metrics.get("method_status", "ok"),
        "final_pass": final_pass,
        "metrics": metrics,
        "failure_flags": flags,
        "vlm_score": vlm_score,
        "llm_audit_summary": llm_report["summary"],
        "failure_stage_hint": qa_summary.get("failure_stage_hint", _failure_stage_hint(flags)),
        "vlm_eval_summary": qa_summary,
    }
    write_json(eval_dir / "metric_scores.json", metrics)
    write_json(eval_dir / "failure_flags.json", flags)
    write_json(profile.result_dir / "vlm_trace" / "05_optimization" / "failure_flags.json", flags)
    write_json(eval_dir / "vlm_score.json", vlm_score)
    write_csv(eval_dir / "vlm_framewise_judgment.csv", framewise, ["frame", "question_id", "question", "label", "score", "reason"])
    write_csv(eval_dir / "vlm_pairwise_preference.csv", [])
    write_json(eval_dir / "llm_audit_report.json", llm_report)
    write_json(eval_dir / "evaluation_summary.json", summary)
    return summary
