from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ...base.config import CaseProfile
from .ablation_registry import MethodVariant, resolve_variant_profile, validate_method_result_mapping
from .summary_writer import run_unified_final_evaluation
from .utils import write_json, write_rows


ABLATION_FIELDS = [
    "case",
    "method",
    "result_name",
    "result_dir",
    "method_status",
    "audio",
    "vlm",
    "llm",
    "ablation_flags",
    "actual_vlm_mode",
    "actual_llm_mode",
    "actual_ablation_flags",
    "method_manifest_valid",
    "method_manifest_mismatch_reason",
    "pose_sha256",
    "same_pose_as_baseline",
    "metrics_identical_to_baseline",
    "overlay_hard_score",
    "overlay_hard_metric_source",
    "contact_gap_mm",
    "contact_proxy",
    "contact_proxy_source",
    "part_correct_ratio",
    "penetration_frame_ratio",
    "penetration_depth_max_mm",
    "floating_rate",
    "tradeoff_score",
    "object_jerk",
    "translation_spike_count",
    "rotation_spike_count",
    "event_aligned_spike_count",
    "non_event_spike_count",
    "high_speed_recall",
    "oversmooth_rate",
    "contact_ratio_audio_windows",
    "jump_count",
    "static_tail_drift_m",
    "gate_impact_status",
    "gate_timeline_source",
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
    "pose_delta_rotation_mean_rad",
    "pose_delta_rotation_max_rad",
    "final_pass",
]

COMPARE_FIELDS = [
    "overlay_hard_score",
    "contact_gap_mm",
    "contact_proxy",
    "part_correct_ratio",
    "penetration_frame_ratio",
    "penetration_depth_max_mm",
    "floating_rate",
    "tradeoff_score",
    "object_jerk",
    "translation_spike_count",
    "rotation_spike_count",
    "event_aligned_spike_count",
    "non_event_spike_count",
    "high_speed_recall",
    "oversmooth_rate",
    "contact_ratio_audio_windows",
    "jump_count",
    "static_tail_drift_m",
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
    "pose_delta_rotation_mean_rad",
    "pose_delta_rotation_max_rad",
    "final_pass",
]

REGISTRY_FIELDS = [
    "case",
    "method",
    "result_name",
    "result_dir",
    "result_exists",
    "audio",
    "vlm",
    "llm",
    "ablation_flags",
    "actual_vlm_mode",
    "actual_llm_mode",
    "actual_ablation_flags",
    "method_manifest_valid",
    "method_manifest_mismatch_reason",
]


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_pipeline_manifest(result_dir: Path) -> dict[str, Any]:
    path = result_dir / "pipeline_manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"_manifest_parse_error": True}


def _manifest_audit(result_dir: Path, variant: MethodVariant) -> dict[str, Any]:
    manifest = _load_pipeline_manifest(result_dir)
    if not manifest:
        return {
            "actual_vlm_mode": "",
            "actual_llm_mode": "",
            "actual_ablation_flags": "",
            "method_manifest_valid": False,
            "method_manifest_mismatch_reason": "missing_pipeline_manifest",
        }
    if manifest.get("_manifest_parse_error"):
        return {
            "actual_vlm_mode": "",
            "actual_llm_mode": "",
            "actual_ablation_flags": "",
            "method_manifest_valid": False,
            "method_manifest_mismatch_reason": "invalid_pipeline_manifest_json",
        }

    actual_vlm = str(manifest.get("vlm_mode", "") or "")
    actual_llm = str(manifest.get("llm_mode", "") or "")
    profile = manifest.get("profile", {}) if isinstance(manifest.get("profile"), dict) else {}
    actual_flags = [str(flag) for flag in profile.get("ablation_flags", []) or []]
    actual_flag_set = set(actual_flags)
    expected_vlm = str(variant.vlm or "")
    expected_llm = str(variant.llm or "")
    expected_flags = set(variant.ablation_flags)
    mismatches: list[str] = []

    if expected_vlm and actual_vlm != expected_vlm:
        mismatches.append(f"vlm_mode expected {expected_vlm} got {actual_vlm or 'missing'}")
    if expected_llm and actual_llm != expected_llm:
        mismatches.append(f"llm_mode expected {expected_llm} got {actual_llm or 'missing'}")
    if variant.audio is False and "disable_audio_events" not in actual_flag_set:
        mismatches.append("audio expected disabled via disable_audio_events")
    if variant.audio is True and "disable_audio_events" in actual_flag_set:
        mismatches.append("audio expected enabled but disable_audio_events is present")
    missing_flags = sorted(expected_flags - actual_flag_set)
    if variant.method == "no_audio":
        missing_flags = [flag for flag in missing_flags if flag != "no_audio"]
    if missing_flags:
        mismatches.append("missing ablation flags " + ",".join(missing_flags))

    return {
        "actual_vlm_mode": actual_vlm,
        "actual_llm_mode": actual_llm,
        "actual_ablation_flags": "|".join(actual_flags),
        "method_manifest_valid": not mismatches,
        "method_manifest_mismatch_reason": "; ".join(mismatches),
    }


def _row(profile: CaseProfile, variant: MethodVariant) -> dict[str, Any]:
    vprofile = resolve_variant_profile(profile, variant)
    result_dir = vprofile.result_dir
    base = {
        "case": profile.case_name,
        "method": variant.method,
        "result_name": variant.result_name,
        "result_dir": str(result_dir),
        "audio": variant.audio,
        "vlm": variant.vlm or "",
        "llm": variant.llm or "",
        "ablation_flags": "|".join(variant.ablation_flags),
    }
    if not result_dir.exists() or not (result_dir / "object_pose.csv").exists():
        return {
            **base,
            "method_status": "missing_result",
            "actual_vlm_mode": "",
            "actual_llm_mode": "",
            "actual_ablation_flags": "",
            "method_manifest_valid": "",
            "method_manifest_mismatch_reason": "",
        }
    manifest_audit = _manifest_audit(result_dir, variant)
    if not manifest_audit["method_manifest_valid"]:
        return {
            **base,
            "method_status": "invalid_manifest",
            "pose_sha256": _sha256(result_dir / "object_pose.csv"),
            **manifest_audit,
        }
    summary = run_unified_final_evaluation(vprofile, run_qa=False)
    metrics = summary["metrics"]
    if (variant.vlm or "").lower() == "none" and (variant.llm or "").lower() == "none":
        metrics = {
            **metrics,
            "gate_impact_status": "disabled_by_ablation",
            "gate_timeline_source": "disabled_by_ablation",
            "gate_event_count": 0,
            "gate_active_count": 0,
            "gate_reject_unclear_count": 0,
            "stage_residual_reweight_count": 0,
        }
    fixed_fields = set(base) | set(manifest_audit) | {"method_status", "pose_sha256", "same_pose_as_baseline", "metrics_identical_to_baseline"}
    return {
        **base,
        "method_status": "ok",
        "pose_sha256": _sha256(result_dir / "object_pose.csv"),
        **manifest_audit,
        **{field: metrics.get(field, "") for field in ABLATION_FIELDS if field not in fixed_fields},
    }


def _annotate_effectiveness(rows: list[dict[str, Any]], baseline_method: str = "full_audio_vlm_llm") -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case"]), {})[str(row["method"])] = row
    for case_rows in by_case.values():
        baseline = case_rows.get(baseline_method)
        if not baseline or baseline.get("method_status") != "ok":
            for row in case_rows.values():
                row.setdefault("same_pose_as_baseline", "")
                row.setdefault("metrics_identical_to_baseline", "")
            continue
        base_hash = str(baseline.get("pose_sha256", ""))
        for row in case_rows.values():
            if row.get("method_status") != "ok":
                row["same_pose_as_baseline"] = ""
                row["metrics_identical_to_baseline"] = ""
                continue
            row["same_pose_as_baseline"] = str(row.get("pose_sha256", "")) == base_hash
            row["metrics_identical_to_baseline"] = all(str(row.get(field, "")) == str(baseline.get(field, "")) for field in COMPARE_FIELDS)
    return rows


def _delta_rows(rows: list[dict[str, Any]], baseline_method: str = "full_audio_vlm_llm") -> list[dict[str, Any]]:
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case"], {})[row["method"]] = row
    numeric_fields = [
        "overlay_hard_score",
        "contact_gap_mm",
        "contact_proxy",
        "part_correct_ratio",
        "penetration_frame_ratio",
        "penetration_depth_max_mm",
        "floating_rate",
        "tradeoff_score",
        "object_jerk",
        "translation_spike_count",
        "rotation_spike_count",
        "event_aligned_spike_count",
        "non_event_spike_count",
        "high_speed_recall",
        "oversmooth_rate",
        "contact_ratio_audio_windows",
        "jump_count",
        "static_tail_drift_m",
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
        "pose_delta_rotation_mean_rad",
        "pose_delta_rotation_max_rad",
    ]
    out = []
    for case, methods in by_case.items():
        base = methods.get(baseline_method)
        if not base or base.get("method_status") != "ok":
            continue
        for method, row in methods.items():
            if method == baseline_method or row.get("method_status") != "ok":
                continue
            rec = {"case": case, "baseline_method": baseline_method, "method": method}
            for field in numeric_fields:
                try:
                    rec[f"delta_{field}"] = float(row.get(field, "")) - float(base.get(field, ""))
                except Exception:
                    rec[f"delta_{field}"] = ""
            rec.update(_causal_ablation_status(row, rec))
            out.append(rec)
    return out


def _nonzero(value: object, eps: float = 1e-9) -> bool:
    try:
        return abs(float(value)) > eps
    except Exception:
        return False


def _causal_ablation_status(row: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    method = str(row.get("method", ""))
    intervention_valid = row.get("method_status") == "ok"
    if method == "no_audio":
        intervention_valid = intervention_valid and str(row.get("audio")) == "False"
    elif method == "no_vlm_llm":
        intervention_valid = (
            intervention_valid
            and str(row.get("vlm")) == "none"
            and str(row.get("llm")) == "none"
            and str(row.get("gate_impact_status")) == "disabled_by_ablation"
        )

    mechanism_fields = [
        "delta_gate_event_count",
        "delta_gate_active_count",
        "delta_stage_residual_reweight_count",
        "delta_optimizer_reweighted_frames",
        "delta_anchor_update_allowed_count",
        "delta_anchor_update_blocked_count",
        "delta_freeze_interpolation_frames",
    ]
    outcome_fields = [
        "delta_pose_delta_translation_max_m",
        "delta_pose_delta_rotation_max_rad",
        "delta_translation_spike_count",
        "delta_rotation_spike_count",
        "delta_event_aligned_spike_count",
        "delta_non_event_spike_count",
        "delta_high_speed_recall",
        "delta_oversmooth_rate",
        "delta_contact_proxy",
        "delta_overlay_hard_score",
        "delta_contact_ratio_audio_windows",
    ]
    mechanism_changed = any(_nonzero(delta.get(field)) for field in mechanism_fields)
    outcome_changed = any(_nonzero(delta.get(field)) for field in outcome_fields)

    if not intervention_valid:
        interpretation = "intervention_invalid"
    elif not mechanism_changed:
        interpretation = "mechanism_not_connected"
    elif not outcome_changed:
        interpretation = "mechanism_connected_but_outcome_unchanged"
    else:
        interpretation = "measurable_downstream_effect"

    return {
        "intervention_valid": intervention_valid,
        "mechanism_changed": mechanism_changed,
        "outcome_changed": outcome_changed,
        "causal_interpretation": interpretation,
    }


def _registry_rows(profiles: list[CaseProfile], variants: list[MethodVariant]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        for variant in variants:
            vprofile = resolve_variant_profile(profile, variant)
            result_dir = vprofile.result_dir
            rows.append(
                {
                    "case": profile.case_name,
                    "method": variant.method,
                    "result_name": variant.result_name,
                    "result_dir": str(result_dir),
                    "result_exists": result_dir.exists(),
                    "audio": variant.audio,
                    "vlm": variant.vlm or "",
                    "llm": variant.llm or "",
                    "ablation_flags": "|".join(variant.ablation_flags),
                    **(_manifest_audit(result_dir, variant) if result_dir.exists() else {
                        "actual_vlm_mode": "",
                        "actual_llm_mode": "",
                        "actual_ablation_flags": "",
                        "method_manifest_valid": "",
                        "method_manifest_mismatch_reason": "",
                    }),
                }
            )
    return rows


def _write_report(output_dir: Path, rows: list[dict[str, Any]], deltas: list[dict[str, Any]]) -> Path:
    ok_rows = [row for row in rows if row.get("method_status") == "ok"]
    missing_rows = [row for row in rows if row.get("method_status") != "ok"]
    same_pose = sum(1 for row in ok_rows if str(row.get("same_pose_as_baseline")) == "True" and row.get("method") != "full_audio_vlm_llm")
    same_metrics = sum(1 for row in ok_rows if str(row.get("metrics_identical_to_baseline")) == "True" and row.get("method") != "full_audio_vlm_llm")
    lines = [
        "# Ablation Evaluation Report",
        "",
        "This report compares real result directories. It does not reuse one result under multiple method labels.",
        "",
        "## Summary",
        "",
        f"- rows: {len(rows)}",
        f"- ok rows: {len(ok_rows)}",
        f"- missing rows: {len(missing_rows)}",
        f"- non-baseline rows with identical object_pose.csv hash: {same_pose}",
        f"- non-baseline rows with identical selected metrics: {same_metrics}",
        f"- delta rows: {len(deltas)}",
        "",
        "## How to read",
        "",
        "- `same_pose_as_baseline=True` means the variant's `object_pose.csv` is byte-identical to `full_audio_vlm_llm` for that case.",
        "- `metrics_identical_to_baseline=True` means the selected final metrics are identical to the baseline, even if files differ.",
        "- If pose differs but metrics are identical, the current metrics are not sensitive to that variant or shared aggregate HOI metrics dominate the table.",
        "- `audio`, `VLM`, `LLM`, and `flags` show the intended variant configuration; this is what prevents the table from silently reusing one result under several method labels.",
        "- `gate status=ok` uses the frame-level gate timeline.",
        "- `gate status=ok:stage_audit_fallback` means the frame-level timeline was empty, so the report uses stage-audit gate records instead of treating the row as missing.",
        "- `gate status=disabled_by_ablation` is expected for the VLM+LLM-off variant.",
        "",
        "## Variant audit",
        "",
        "| case | method | status | result | audio | VLM | LLM | flags | same pose | same metrics | contact proxy | overlay IoU | overlay source | gate status | gate source | gate events | gates active | reweight | pose delta max | final pass |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("case", "")),
                    str(row.get("method", "")),
                    str(row.get("method_status", "")),
                    str(row.get("result_name", "")),
                    str(row.get("audio", "")),
                    str(row.get("vlm", "")),
                    str(row.get("llm", "")),
                    str(row.get("ablation_flags", "")),
                    str(row.get("same_pose_as_baseline", "")),
                    str(row.get("metrics_identical_to_baseline", "")),
                    _fmt(row.get("contact_proxy", "")),
                    _fmt(row.get("overlay_hard_score", "")),
                    str(row.get("overlay_hard_metric_source", "")),
                    str(row.get("gate_impact_status", "")),
                    str(row.get("gate_timeline_source", "")),
                    _fmt(row.get("gate_event_count", ""), decimals=0),
                    _fmt(row.get("gate_active_count", ""), decimals=0),
                    _fmt(row.get("optimizer_reweighted_frames", ""), decimals=0),
                    _fmt(row.get("pose_delta_translation_max_m", "")),
                    str(row.get("final_pass", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Focused Ablation Deltas",
            "",
            "`delta = method - full_audio_vlm_llm`. For error-like metrics, positive is worse. For recall-like metrics, negative is worse.",
            "",
            "| case | method | intervention valid | mechanism changed | outcome changed | interpretation | Δ contact proxy | Δ overlay | Δ high-speed recall | Δ oversmooth | Δ gate events | Δ gate active | Δ reweight frames | Δ pose delta max | Δ anchor updates |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in deltas:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("case", "")),
                    str(row.get("method", "")),
                    str(row.get("intervention_valid", "")),
                    str(row.get("mechanism_changed", "")),
                    str(row.get("outcome_changed", "")),
                    str(row.get("causal_interpretation", "")),
                    _fmt(row.get("delta_contact_proxy", "")),
                    _fmt(row.get("delta_overlay_hard_score", "")),
                    _fmt(row.get("delta_high_speed_recall", "")),
                    _fmt(row.get("delta_oversmooth_rate", "")),
                    _fmt(row.get("delta_gate_event_count", ""), decimals=0),
                    _fmt(row.get("delta_gate_active_count", ""), decimals=0),
                    _fmt(row.get("delta_optimizer_reweighted_frames", ""), decimals=0),
                    _fmt(row.get("delta_pose_delta_translation_max_m", "")),
                    _fmt(row.get("delta_anchor_update_allowed_count", ""), decimals=0),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Current Interpretation",
            "",
            "- `no_audio` tests whether audio timing/contact evidence changes the optimizer while VLM+LLM remain enabled.",
            "- `no_vlm_llm` tests whether the VLM+LLM gate/audit path changes the result while audio remains enabled.",
            "- If contact proxy and overlay remain unchanged but gate/pose/temporal deltas change, the current hard metrics are too coarse to show visual improvement and the gate-impact metrics should be used as the evidence.",
            "- If a future `no_vlm_llm` row has zero pose/temporal delta, inspect `intervention_valid`, `mechanism_changed`, and `outcome_changed` before blaming the model or the evaluator.",
        ]
    )
    path = output_dir / "ablation_report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def _fmt(value: object, decimals: int = 3) -> str:
    if value in {"", None}:
        return ""
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except Exception:
        return str(value)
    if not math.isfinite(numeric):
        return ""
    text = f"{numeric:.{decimals}f}"
    if "." not in text:
        return text
    return text.rstrip("0").rstrip(".")


def run_ablation_evaluation(
    profiles: list[CaseProfile],
    *,
    variants: list[MethodVariant],
    output_dir: Path,
    allow_same_result_debug: bool = False,
    require_existing: bool = False,
) -> dict[str, Any]:
    validate_method_result_mapping(
        profiles,
        variants,
        allow_same_result_debug=allow_same_result_debug,
        require_existing=require_existing,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _annotate_effectiveness([_row(profile, variant) for profile in profiles for variant in variants])
    table = write_rows(output_dir / "ablation_table.csv", rows, ABLATION_FIELDS)
    deltas = _delta_rows(rows)
    delta_table = write_rows(output_dir / "ablation_delta_table.csv", deltas)
    registry_table = write_rows(output_dir / "ablation_method_registry.csv", _registry_rows(profiles, variants), REGISTRY_FIELDS)
    report = _write_report(output_dir, rows, deltas)
    manifest = {
        "rows": len(rows),
        "delta_rows": len(deltas),
        "table": str(table),
        "delta_table": str(delta_table),
        "registry_table": str(registry_table),
        "report": str(report),
        "missing_results": sum(1 for row in rows if row.get("method_status") == "missing_result"),
        "identical_pose_rows": sum(1 for row in rows if row.get("method") != "full_audio_vlm_llm" and str(row.get("same_pose_as_baseline")) == "True"),
        "identical_metric_rows": sum(1 for row in rows if row.get("method") != "full_audio_vlm_llm" and str(row.get("metrics_identical_to_baseline")) == "True"),
    }
    write_json(
        output_dir / "ablation_method_registry_manifest.json",
        {
            "rows": len(profiles) * len(variants),
            "table": str(registry_table),
            "allow_same_result_debug": allow_same_result_debug,
            "require_existing": require_existing,
        },
    )
    write_json(output_dir / "ablation_evaluation_manifest.json", manifest)
    return manifest
