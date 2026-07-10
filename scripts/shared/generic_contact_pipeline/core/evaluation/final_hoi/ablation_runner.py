from __future__ import annotations

import hashlib
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
    "pose_sha256",
    "same_pose_as_baseline",
    "metrics_identical_to_baseline",
    "overlay_hard_score",
    "contact_gap_mm",
    "contact_proxy",
    "contact_proxy_source",
    "part_correct_ratio",
    "penetration_frame_ratio",
    "penetration_depth_max_mm",
    "floating_rate",
    "tradeoff_score",
    "object_jerk",
    "contact_ratio_audio_windows",
    "jump_count",
    "static_tail_drift_m",
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
    "contact_ratio_audio_windows",
    "jump_count",
    "static_tail_drift_m",
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
]


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        return {**base, "method_status": "missing_result"}
    summary = run_unified_final_evaluation(vprofile)
    metrics = summary["metrics"]
    fixed_fields = set(base) | {"method_status", "pose_sha256", "same_pose_as_baseline", "metrics_identical_to_baseline"}
    return {
        **base,
        "method_status": "ok",
        "pose_sha256": _sha256(result_dir / "object_pose.csv"),
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
        "contact_ratio_audio_windows",
        "jump_count",
        "static_tail_drift_m",
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
            out.append(rec)
    return out


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
        "",
        "## Variant audit",
        "",
        "| case | method | status | result | audio | VLM | LLM | flags | same pose | same metrics | contact proxy | overlay | final pass |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
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
                    str(row.get("final_pass", "")),
                ]
            )
            + " |"
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
    return f"{numeric:.{decimals}f}".rstrip("0").rstrip(".")


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
