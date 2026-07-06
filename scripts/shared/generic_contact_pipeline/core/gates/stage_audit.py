from __future__ import annotations

import json
from pathlib import Path

import requests

from ..base.config import CaseProfile
from ..base.io import read_csv, write_csv, write_json
from ..base.schema import stage_paths
from .llm_provider import load_llm_provider


AUDIT_FIELDS = [
    "case_name",
    "stage",
    "check_type",
    "pass_gate",
    "repair_action",
    "rerun_stage",
    "allow_continue",
    "residual_reweight",
    "reason",
]


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        return read_csv(path)
    except Exception:
        return []


def _exists_rows(path: Path) -> bool:
    return path.exists() and len(_rows(path)) > 0


def _gate_row(profile: CaseProfile, stage: str, check_type: str, gate: str, action: str, reason: str, *, rerun: bool = False, reweight: bool = False) -> dict[str, object]:
    return {
        "case_name": profile.case_name,
        "stage": stage,
        "check_type": check_type,
        "pass_gate": gate,
        "repair_action": action,
        "rerun_stage": "1" if rerun else "0",
        "allow_continue": "1" if gate in {"pass", "unclear"} else "0",
        "residual_reweight": "1" if reweight else "0",
        "reason": reason,
    }


def _safe_int(value: str | object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except Exception:
        return default


def _safe_float(value: str | object, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except Exception:
        return default


def _vlm_gate_rows(profile: CaseProfile, stage: str) -> list[dict[str, str]]:
    return _rows(stage_paths(profile)["vlm_dir"] / stage / "vlm_gates.csv")


def _vlm_audit(profile: CaseProfile, stage: str) -> list[dict[str, object]]:
    gates = [row for row in _vlm_gate_rows(profile, stage) if row.get("is_effective") == "1"]
    if not gates:
        return [_gate_row(profile, stage, "vlm_gate", "pass", "no_effective_vlm_gate", "no effective VLM rejection/unclear gate")]
    rejects = [row for row in gates if row.get("pass_gate") == "reject" and row.get("report_only") != "1"]
    unclear = [row for row in gates if row.get("pass_gate") == "unclear"]
    if rejects:
        return [_gate_row(profile, stage, "vlm_gate", "reject", "rerun_stage_with_gate", f"VLM rejected {len(rejects)} frame/query rows", rerun=stage in {"stage2", "stage3", "stage4"}, reweight=stage == "stage4")]
    if unclear:
        return [_gate_row(profile, stage, "vlm_gate", "unclear", "rerun_stage_with_lower_trust", f"VLM marked {len(unclear)} frame/query rows unclear", rerun=stage in {"stage2", "stage3", "stage4"}, reweight=stage == "stage4")]
    return [_gate_row(profile, stage, "vlm_gate", "pass", "keep_outputs", "all effective VLM gates pass")]


def _stage_artifact_audit(profile: CaseProfile, stage: str) -> list[dict[str, object]]:
    paths = stage_paths(profile)
    checks: list[tuple[str, Path]] = []
    if stage == "stage0":
        checks = [("stage0_inputs_manifest", paths["stage0_inputs_manifest"])]
    elif stage == "stage1":
        checks = [
            ("object_observations", paths["object_observations"]),
            ("object_correspondence", paths["object_correspondence"]),
        ]
    elif stage == "stage2":
        checks = [
            ("contact_candidates", paths["contact_candidates"]),
            ("anchor_state", paths["anchor_state"]),
        ]
    elif stage == "stage3":
        checks = [("object_pose_init", paths["object_pose_init"])]
    elif stage == "stage4":
        checks = [
            ("object_pose", paths["object_pose"]),
            ("motion_regime", paths["motion_regime"]),
            ("optimizer_decisions", paths["optimizer_decisions"]),
            ("physical_smooth_residuals", paths["physical_smooth_residuals"]),
            ("pose_jump_audit", paths["pose_jump_audit"]),
        ]
    elif stage == "stage5":
        checks = [("render_manifest", paths["render_manifest"])]

    rows: list[dict[str, object]] = []
    for name, path in checks:
        if not _exists_rows(path) and path.suffix == ".csv":
            rows.append(_gate_row(profile, stage, f"artifact:{name}", "reject", "rerun_stage_or_block", f"{name} is missing or empty", rerun=stage in {"stage1", "stage2", "stage3", "stage4"}))
        elif not path.exists():
            rows.append(_gate_row(profile, stage, f"artifact:{name}", "reject", "rerun_stage_or_block", f"{name} is missing", rerun=stage in {"stage1", "stage2", "stage3", "stage4"}))
        else:
            rows.append(_gate_row(profile, stage, f"artifact:{name}", "pass", "keep_outputs", f"{name} exists"))

    if stage == "stage3" and paths["object_pose_init"].exists():
        pose_rows = _rows(paths["object_pose_init"])
        missing = []
        if pose_rows:
            for col in ("tx", "ty", "tz", "qw", "qx", "qy", "qz"):
                if col not in pose_rows[0]:
                    missing.append(col)
        if missing:
            rows.append(_gate_row(profile, stage, "se3_schema", "reject", "rerun_pose_init", f"object_pose_init missing SE3 columns: {missing}", rerun=True))
        elif pose_rows:
            rows.append(_gate_row(profile, stage, "se3_schema", "pass", "keep_outputs", "object_pose_init has SE3 schema"))

    if stage == "stage4" and paths["pose_jump_audit"].exists():
        audit_rows = _rows(paths["pose_jump_audit"])
        decision_rows = _rows(paths["optimizer_decisions"])
        trusted_pose_lock = bool(decision_rows) and all(row.get("pose_lock_reason") for row in decision_rows)
        spike_rows = [
            row
            for row in audit_rows
            if row.get("visual_spike") == "1" or row.get("contact_spike") == "1" or row.get("smoothness_spike") == "1"
        ]
        if spike_rows and trusted_pose_lock:
            rows.append(_gate_row(profile, stage, "pose_jump_audit", "pass", "report_locked_solved_pose_spikes_only", f"{len(spike_rows)} spike rows are reported only because trusted solved pose lock is active"))
        elif spike_rows:
            rows.append(_gate_row(profile, stage, "pose_jump_audit", "unclear", "reweight_and_rerun_optimizer", f"{len(spike_rows)} spike rows require optimizer reweight/recheck", rerun=True, reweight=True))
        else:
            rows.append(_gate_row(profile, stage, "pose_jump_audit", "pass", "keep_outputs", "no pose/contact/smoothness spike rows"))
    return rows


def _llm_stage_queries(profile: CaseProfile, stage: str) -> list[dict[str, object]]:
    common = {
        "case_name": profile.case_name,
        "stage": stage,
        "object_family": profile.data.get("object_family", ""),
        "role": "CSV/metrics LLM audit gate; labels only, no coordinates or pose generation",
    }
    if stage == "stage0":
        return [
            {
                **common,
                "query_id": "stage0_inputs",
                "question": "Are preprocessing inputs present or explicitly marked with a reusable/generatable status?",
                "allowed_labels": ["pass", "schema_missing", "stage_inconsistent", "unclear"],
            }
        ]
    if stage == "stage1":
        return [
            {
                **common,
                "query_id": "observation_contract",
                "question": "Do object observations and correspondence rows exist with frame-aligned evidence?",
                "allowed_labels": ["pass", "schema_missing", "stage_inconsistent", "unclear"],
            }
        ]
    if stage == "stage2":
        return [
            {
                **common,
                "query_id": "contact_anchor_contract",
                "question": "Do contact candidates and anchor_state expose observed/persistent/update/pose-anchor fields consistently?",
                "allowed_labels": ["pass", "schema_missing", "contact_empty", "stage_inconsistent", "unclear"],
            }
        ]
    if stage == "stage3":
        return [
            {
                **common,
                "query_id": "se3_schema",
                "question": "Does the initialized object pose use the fixed SE3 schema for every frame?",
                "allowed_labels": ["pass", "schema_missing", "stage_inconsistent", "unclear"],
            }
        ]
    if stage == "stage4":
        return [
            {
                **common,
                "query_id": "sequence_optimizer_contract",
                "question": "Did the fixed sequence SE3 optimizer run and emit residual, regime, jump, and decision artifacts?",
                "allowed_labels": ["pass", "schema_missing", "stage_inconsistent", "unclear"],
            },
            {
                **common,
                "query_id": "pose_jump_audit",
                "question": "Do pose/contact/smoothness spike metrics require reweighting and optimizer rerun?",
                "allowed_labels": ["pass", "rotation_jump", "static_drift", "unclear"],
            },
        ]
    if stage == "stage5":
        return [
            {
                **common,
                "query_id": "render_manifest",
                "question": "Did render/eval produce a manifest that can be checked by VLM post-render gates?",
                "allowed_labels": ["pass", "schema_missing", "stage_inconsistent", "unclear"],
            },
            {
                **common,
                "query_id": "render_temporal_consistency",
                "question": "Do machine temporal spike metrics and VLM render-temporal gates agree, or does the case need manual/optimizer review?",
                "allowed_labels": ["pass", "temporal_spike_not_caught_by_vlm", "stage_inconsistent", "unclear"],
            },
        ]
    return [
        {
            **common,
            "query_id": "stage_contract",
            "question": "Do this stage's machine metrics and gate artifacts allow the pipeline to continue?",
            "allowed_labels": ["pass", "schema_missing", "stage_inconsistent", "unclear"],
        }
    ]


def _llm_result_row(query: dict[str, object], label: str, reason: str, *, blocking: bool = False, rerun: bool = False, reweight: bool = False) -> dict[str, object]:
    return {
        "query_id": query.get("query_id", ""),
        "label": label,
        "blocking": "1" if blocking else "0",
        "rerun_stage": "1" if rerun else "0",
        "residual_reweight": "1" if reweight else "0",
        "short_reason": reason,
    }


def _stage_llm_results(profile: CaseProfile, stage: str, queries: list[dict[str, object]]) -> list[dict[str, object]]:
    paths = stage_paths(profile)
    by_id = {str(q["query_id"]): q for q in queries}
    rows: list[dict[str, object]] = []
    if stage == "stage0":
        manifest = paths["stage0_inputs_manifest"]
        label = "pass" if manifest.exists() else "schema_missing"
        rows.append(_llm_result_row(by_id["stage0_inputs"], label, "stage0 input manifest exists" if label == "pass" else "stage0 input manifest missing", blocking=label != "pass"))
    elif stage == "stage1":
        obs = _rows(paths["object_observations"])
        corr = _rows(paths["object_correspondence"])
        label = "pass" if obs and corr else "schema_missing"
        rows.append(_llm_result_row(by_id["observation_contract"], label, f"object_observations={len(obs)} object_correspondence={len(corr)}", blocking=label != "pass", rerun=label != "pass"))
    elif stage == "stage2":
        candidates = _rows(paths["contact_candidates"])
        anchors = _rows(paths["anchor_state"])
        missing_cols = []
        if anchors:
            for col in ("contact_observed", "contact_persistent", "anchor_update_allowed", "pose_anchor_allowed"):
                if col not in anchors[0]:
                    missing_cols.append(col)
        if not candidates or not anchors or missing_cols:
            label = "schema_missing" if missing_cols or not anchors else "contact_empty"
            reason = f"contact_candidates={len(candidates)} anchor_state={len(anchors)} missing_cols={missing_cols}"
            rows.append(_llm_result_row(by_id["contact_anchor_contract"], label, reason, blocking=True, rerun=True))
        else:
            rows.append(_llm_result_row(by_id["contact_anchor_contract"], "pass", f"contact_candidates={len(candidates)} anchor_state={len(anchors)}"))
    elif stage == "stage3":
        pose = _rows(paths["object_pose_init"])
        required = {"tx", "ty", "tz", "qw", "qx", "qy", "qz"}
        missing = sorted(required - set(pose[0])) if pose else sorted(required)
        label = "pass" if pose and not missing else "schema_missing"
        rows.append(_llm_result_row(by_id["se3_schema"], label, "object_pose_init has SE3 schema" if label == "pass" else f"missing {missing}", blocking=label != "pass", rerun=label != "pass"))
    elif stage == "stage4":
        required_paths = [
            paths["object_pose"],
            paths["motion_regime"],
            paths["optimizer_decisions"],
            paths["physical_smooth_residuals"],
            paths["pose_jump_audit"],
        ]
        missing = [p.name for p in required_paths if not _exists_rows(p)]
        if missing:
            rows.append(_llm_result_row(by_id["sequence_optimizer_contract"], "schema_missing", f"missing/empty optimizer artifacts: {missing}", blocking=True, rerun=True))
        else:
            decisions = _rows(paths["optimizer_decisions"])
            enabled = any(row.get("sequence_se3_optimizer_enabled") == "1" for row in decisions)
            label = "pass" if enabled else "stage_inconsistent"
            rows.append(_llm_result_row(by_id["sequence_optimizer_contract"], label, "sequence optimizer decisions enabled" if enabled else "sequence optimizer not enabled", blocking=not enabled, rerun=not enabled))
        audit = _rows(paths["pose_jump_audit"])
        decisions = _rows(paths["optimizer_decisions"])
        locked = bool(decisions) and all(row.get("pose_lock_reason") for row in decisions)
        spikes = [
            row
            for row in audit
            if row.get("visual_spike") == "1" or row.get("contact_spike") == "1" or row.get("smoothness_spike") == "1"
        ]
        if spikes and locked:
            rows.append(_llm_result_row(by_id["pose_jump_audit"], "pass", f"{len(spikes)} spikes reported under trusted pose lock"))
        elif spikes:
            rows.append(_llm_result_row(by_id["pose_jump_audit"], "unclear", f"{len(spikes)} spike rows request feedback reweight", rerun=True, reweight=True))
        else:
            rows.append(_llm_result_row(by_id["pose_jump_audit"], "pass", "no pose/contact/smoothness spikes"))
    elif stage == "stage5":
        label = "pass" if paths["render_manifest"].exists() else "schema_missing"
        rows.append(_llm_result_row(by_id["render_manifest"], label, "render manifest exists" if label == "pass" else "render manifest missing", blocking=label != "pass"))
        spike_rows = [
            row
            for row in _rows(paths["pose_jump_audit"])
            if row.get("visual_spike") == "1" or row.get("contact_spike") == "1" or row.get("smoothness_spike") == "1"
        ]
        decisions = _rows(paths["optimizer_decisions"])
        locked = bool(decisions) and all(row.get("pose_lock_reason") for row in decisions)
        vlm_rows = _vlm_gate_rows(profile, stage)
        temporal_rows = [row for row in vlm_rows if row.get("query_type") == "temporal_motion_check" and row.get("is_effective") == "1"]
        temporal_bad = [row for row in temporal_rows if row.get("pass_gate") in {"reject", "unclear"}]
        if spike_rows and locked:
            rows.append(
                _llm_result_row(
                    by_id["render_temporal_consistency"],
                    "pass",
                    f"{len(spike_rows)} spike rows are report-only under trusted solved pose lock",
                )
            )
        elif spike_rows and temporal_rows and not temporal_bad:
            rows.append(
                _llm_result_row(
                    by_id["render_temporal_consistency"],
                    "temporal_spike_not_caught_by_vlm",
                    f"{len(spike_rows)} machine spike rows but {len(temporal_rows)} temporal VLM rows all pass",
                    blocking=False,
                )
            )
        elif spike_rows and not temporal_rows:
            rows.append(
                _llm_result_row(
                    by_id["render_temporal_consistency"],
                    "unclear",
                    f"{len(spike_rows)} machine spike rows but no effective temporal VLM rows",
                    blocking=False,
                )
            )
        else:
            rows.append(_llm_result_row(by_id["render_temporal_consistency"], "pass", "temporal VLM/machine audit has no mismatch"))
    else:
        rows.append(_llm_result_row(queries[0], "pass", "no stage-specific CSV audit rule"))
    return rows


LIVE_LLM_SYSTEM_PROMPT = """You are an AudioHOI stage CSV/metrics auditor.
You only inspect structured evidence and answer with labels/actions.
Do not generate coordinates, object pose, contact points, or continuous values.
Return strict JSON with this shape:
{"results":[{"query_id":"...","label":"...","blocking":false,"rerun_stage":false,"residual_reweight":false,"short_reason":"..."}]}
Use only labels listed in each query's allowed_labels."""


def _live_stage_summaries(profile: CaseProfile, stage: str) -> dict[str, object]:
    paths = stage_paths(profile)
    def csv_summary(name: str, path: Path, limit: int = 3) -> dict[str, object]:
        rows = _rows(path)
        return {
            "name": name,
            "path": str(path),
            "exists": path.exists(),
            "row_count": len(rows),
            "columns": sorted(rows[0].keys()) if rows else [],
            "sample_rows": rows[:limit],
        }

    out: dict[str, object] = {
        "case_name": profile.case_name,
        "stage": stage,
        "object_family": profile.data.get("object_family", ""),
    }
    if stage == "stage1":
        out["artifacts"] = [
            csv_summary("object_observations", paths["object_observations"]),
            csv_summary("object_correspondence", paths["object_correspondence"]),
        ]
    elif stage == "stage2":
        out["artifacts"] = [
            csv_summary("contact_candidates", paths["contact_candidates"]),
            csv_summary("anchor_state", paths["anchor_state"]),
        ]
    elif stage == "stage3":
        out["artifacts"] = [csv_summary("object_pose_init", paths["object_pose_init"])]
    elif stage == "stage4":
        out["artifacts"] = [
            csv_summary("object_pose", paths["object_pose"]),
            csv_summary("motion_regime", paths["motion_regime"]),
            csv_summary("optimizer_decisions", paths["optimizer_decisions"]),
            csv_summary("physical_smooth_residuals", paths["physical_smooth_residuals"]),
            csv_summary("pose_jump_audit", paths["pose_jump_audit"]),
        ]
        audit_rows = _rows(paths["pose_jump_audit"])
        out["spike_counts"] = {
            "visual_spike": sum(1 for row in audit_rows if row.get("visual_spike") == "1"),
            "contact_spike": sum(1 for row in audit_rows if row.get("contact_spike") == "1"),
            "smoothness_spike": sum(1 for row in audit_rows if row.get("smoothness_spike") == "1"),
        }
    elif stage == "stage5":
        out["artifacts"] = [
            {"name": "render_manifest", "path": str(paths["render_manifest"]), "exists": paths["render_manifest"].exists()},
            csv_summary("pose_jump_audit", paths["pose_jump_audit"]),
            csv_summary("stage5_vlm_gates", paths["vlm_dir"] / "stage5" / "vlm_gates.csv"),
        ]
    else:
        out["artifacts"] = [{"name": "stage0_inputs_manifest", "path": str(paths["stage0_inputs_manifest"]), "exists": paths["stage0_inputs_manifest"].exists()}]
    return out


def _parse_json_object(text: str) -> dict[str, object] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _call_live_stage_llm(
    profile: CaseProfile,
    stage: str,
    queries: list[dict[str, object]],
    deterministic_results: list[dict[str, object]],
) -> tuple[list[dict[str, object]] | None, dict[str, object]]:
    provider = load_llm_provider()
    trace: dict[str, object] = {"provider": provider.doctor(), "used": False, "error": ""}
    if not provider.api_key:
        trace["error"] = f"missing API key {provider.api_key_env}"
        return None, trace
    summaries = _live_stage_summaries(profile, stage)
    prompt = (
        "Review this AudioHOI pipeline stage. Decide whether the stage can continue, "
        "needs rerun, or needs residual reweighting. You may only emit the allowed labels.\n\n"
        f"Queries:\n{json.dumps(queries, ensure_ascii=False, indent=2)}\n\n"
        f"Structured evidence:\n{json.dumps(summaries, ensure_ascii=False, indent=2)}\n\n"
        f"Deterministic fallback results:\n{json.dumps(deterministic_results, ensure_ascii=False, indent=2)}"
    )
    payload = {
        "model": provider.model_id,
        "messages": [
            {"role": "system", "content": LIVE_LLM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": provider.temperature,
        "max_tokens": provider.max_new_tokens,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {provider.api_key}", "Content-Type": "application/json"}
    try:
        resp = requests.post(provider.base_url.rstrip("/") + "/chat/completions", headers=headers, json=payload, timeout=provider.timeout_seconds)
        if resp.status_code >= 400:
            trace["error"] = f"mistral status {resp.status_code}: {resp.text[:500]}"
            return None, trace
        data = resp.json()
    except Exception as exc:
        trace["error"] = str(exc)
        return None, trace
    text = str(data.get("choices", [{}])[0].get("message", {}).get("content", ""))
    trace.update({"used": True, "response_id": data.get("id", ""), "usage": data.get("usage", {}), "raw_text": text[:4000]})
    parsed = _parse_json_object(text)
    if not parsed or not isinstance(parsed.get("results"), list):
        trace["error"] = "response did not contain a results list"
        return None, trace

    allowed = {str(q["query_id"]): {str(label) for label in q["allowed_labels"]} for q in queries}
    fallback = {str(row.get("query_id", "")): row for row in deterministic_results}
    reviewed: list[dict[str, object]] = []
    for item in parsed["results"]:
        if not isinstance(item, dict):
            continue
        qid = str(item.get("query_id", ""))
        if qid not in allowed:
            continue
        label = str(item.get("label", "unclear"))
        if label not in allowed[qid]:
            label = "unclear"
        base = dict(fallback.get(qid, {"query_id": qid}))
        blocking = bool(item.get("blocking", base.get("blocking") == "1"))
        rerun = bool(item.get("rerun_stage", base.get("rerun_stage") == "1"))
        reweight = bool(item.get("residual_reweight", base.get("residual_reweight") == "1"))
        if stage == "stage4" and label in {"rotation_jump", "static_drift", "unclear"}:
            rerun = True
            reweight = True
        base.update(
            {
                "query_id": qid,
                "label": label,
                "blocking": "1" if blocking else "0",
                "rerun_stage": "1" if rerun else "0",
                "residual_reweight": "1" if reweight else "0",
                "short_reason": str(item.get("short_reason", ""))[:260] or str(base.get("short_reason", "")),
                "llm_reviewed": "1",
            }
        )
        reviewed.append(base)
    if len(reviewed) != len(queries):
        trace["error"] = f"incomplete result count {len(reviewed)}/{len(queries)}"
        return None, trace
    return reviewed, trace


def _llm_csv_audit(profile: CaseProfile, stage: str, *, llm_mode: str) -> list[dict[str, object]]:
    paths = stage_paths(profile)
    out_dir = paths["stage_audit_dir"] / stage
    queries = _llm_stage_queries(profile, stage)
    if llm_mode == "none":
        write_json(
            out_dir / "llm_audit_queries.json",
            {
                "case_name": profile.case_name,
                "stage": stage,
                "llm_mode": llm_mode,
                "disabled": True,
                "note": "LLM stage audit is disabled for this run; only hard artifact/VLM gates are active.",
                "queries": queries,
            },
        )
        write_csv(
            out_dir / "llm_audit_results.csv",
            [],
            ["query_id", "label", "blocking", "rerun_stage", "residual_reweight", "short_reason"],
        )
        write_json(out_dir / "llm_audit_trace.json", {"mode": llm_mode, "used": False, "disabled": True})
        return []
    result_rows = _stage_llm_results(profile, stage, queries)
    llm_trace: dict[str, object] = {"mode": llm_mode, "used": False}
    if llm_mode == "mistral":
        reviewed, llm_trace = _call_live_stage_llm(profile, stage, queries, result_rows)
        if reviewed is not None:
            result_rows = reviewed
    write_json(
        out_dir / "llm_audit_queries.json",
        {
            "case_name": profile.case_name,
            "stage": stage,
            "llm_mode": llm_mode,
            "note": "LLM audit operates on CSV/metrics evidence and may only emit discrete gates/actions.",
            "queries": queries,
        },
    )
    write_json(out_dir / "llm_audit_trace.json", llm_trace)
    write_csv(
        out_dir / "llm_audit_results.csv",
        result_rows,
        ["query_id", "label", "blocking", "rerun_stage", "residual_reweight", "short_reason", "llm_reviewed"],
    )
    gate_rows: list[dict[str, object]] = []
    for row in result_rows:
        label = str(row.get("label", "unclear"))
        gate = "pass" if label == "pass" else ("reject" if row.get("blocking") == "1" else "unclear")
        qid = str(row.get("query_id", "stage_contract"))
        if row.get("residual_reweight") == "1":
            action = "reweight_and_rerun_optimizer"
        elif row.get("rerun_stage") == "1":
            action = "rerun_stage_with_llm_csv_gate"
        elif gate == "pass":
            action = "keep_outputs"
        else:
            action = "report_unclear_csv_audit"
        gate_rows.append(
            _gate_row(
                profile,
                stage,
                f"llm_csv:{qid}",
                gate,
                action,
                str(row.get("short_reason", "")),
                rerun=row.get("rerun_stage") == "1",
                reweight=row.get("residual_reweight") == "1",
            )
        )
    return gate_rows


def write_stage_audit(profile: CaseProfile, stage: str, *, llm_mode: str = "seed") -> dict[str, object]:
    paths = stage_paths(profile)
    out_dir = paths["stage_audit_dir"] / stage
    rows = _stage_artifact_audit(profile, stage)
    rows.extend(_vlm_audit(profile, stage))
    rows.extend(_llm_csv_audit(profile, stage, llm_mode=llm_mode))

    blocking = [row for row in rows if row["pass_gate"] == "reject" and row["allow_continue"] != "1"]
    reweight = [row for row in rows if row["residual_reweight"] == "1"]
    rerun = [row for row in rows if row["rerun_stage"] == "1"]
    decision = "reject" if blocking else ("unclear" if any(row["pass_gate"] == "unclear" for row in rows) else "pass")
    summary = {
        "case_name": profile.case_name,
        "stage": stage,
        "decision": decision,
        "llm_mode": llm_mode,
        "llm_role": "CSV/metrics audit gate; deterministic fallback is used when no live LLM stage auditor is configured",
        "gate_rows": len(rows),
        "blocking_count": len(blocking),
        "rerun_stage": bool(rerun),
        "residual_reweight": bool(reweight),
        "repair_actions": sorted({str(row["repair_action"]) for row in rows if row["repair_action"] != "keep_outputs"}),
        "audit_csv": str(out_dir / "stage_audit_gates.csv"),
        "llm_audit_queries": str(out_dir / "llm_audit_queries.json"),
        "llm_audit_results": str(out_dir / "llm_audit_results.csv"),
    }
    write_csv(out_dir / "stage_audit_gates.csv", rows, AUDIT_FIELDS)
    write_json(out_dir / "stage_audit_decision.json", summary)
    aggregate: list[dict[str, str]] = []
    for path in sorted(paths["stage_audit_dir"].glob("*/stage_audit_gates.csv")):
        aggregate.extend(_rows(path))
    write_csv(paths["stage_audit_dir"] / "stage_audit_gates.csv", aggregate, AUDIT_FIELDS)
    write_json(paths["stage_audit_dir"] / "stage_audit_latest.json", summary)
    return summary
