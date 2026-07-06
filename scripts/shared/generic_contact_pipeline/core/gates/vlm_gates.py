from __future__ import annotations

import json
from pathlib import Path

from ..base.config import CaseProfile
from ..base.io import read_csv, write_csv, write_json
from ..base.schema import stage_paths
from .vlm import GATE_PASS, GATE_REJECT, GATE_UNCLEAR, GATE_NOT_EVALUATED, STAGE_QUERY_TYPES, write_aggregate


GATE_FIELDS = [
    "case_name",
    "stage",
    "frame",
    "query_type",
    "normalized_label",
    "pass_gate",
    "repair_action",
    "allow_anchor_update",
    "allow_contact_residual",
    "allow_pose_refine",
    "report_only",
    "is_effective",
]


def _frame(row: dict[str, str]) -> int:
    try:
        return int(float(row.get("frame", "-1") or -1))
    except Exception:
        return -1


def _bool_str(value: bool) -> str:
    return "1" if value else "0"


def gate_row_from_result(row: dict[str, str]) -> dict[str, object]:
    gate = row.get("pass_gate", "")
    qtype = row.get("query_type", "")
    action = row.get("repair_action", "")
    is_effective = gate in {GATE_PASS, GATE_REJECT, GATE_UNCLEAR}
    report_only = action == "mark_for_report_only" or qtype in {"post_render_sanity_check", "baseline_regression_check", "loss_diagnostic_check"}

    allow_anchor = gate == GATE_PASS and qtype in {
        "keypart_identity_check",
        "track_stability_check",
        "keypart_visibility_check",
        "anchor_update_check",
    }
    allow_contact = gate == GATE_PASS and qtype == "contact_relation_check"
    allow_pose_refine = gate == GATE_PASS or (qtype == "overlay_alignment_check" and action == "allow_small_se3_refine")

    if gate in {GATE_REJECT, GATE_UNCLEAR, GATE_NOT_EVALUATED}:
        allow_anchor = False
        allow_contact = False
    if gate in {GATE_REJECT, GATE_NOT_EVALUATED} and not report_only:
        allow_pose_refine = False

    return {
        "case_name": row.get("case_name", ""),
        "stage": row.get("stage", ""),
        "frame": row.get("frame", ""),
        "query_type": qtype,
        "normalized_label": row.get("normalized_label", ""),
        "pass_gate": gate,
        "repair_action": action,
        "allow_anchor_update": _bool_str(allow_anchor),
        "allow_contact_residual": _bool_str(allow_contact),
        "allow_pose_refine": _bool_str(allow_pose_refine),
        "report_only": _bool_str(report_only),
        "is_effective": _bool_str(is_effective),
    }


def write_stage_gates(profile: CaseProfile, stage: str) -> dict[str, object]:
    paths = stage_paths(profile)
    out_dir = paths["vlm_dir"] / stage
    results_path = out_dir / "vlm_results.csv"
    rows = read_csv(results_path) if results_path.exists() else []
    gates = [gate_row_from_result(row) for row in rows]
    write_csv(out_dir / "vlm_gates.csv", gates, GATE_FIELDS)

    blocked = [row for row in gates if row["is_effective"] == "1" and row["pass_gate"] == GATE_REJECT and row["report_only"] != "1"]
    unclear = [row for row in gates if row["is_effective"] == "1" and row["pass_gate"] == GATE_UNCLEAR]
    accepted_contact = [row for row in gates if row["allow_contact_residual"] == "1"]
    accepted_anchor = [row for row in gates if row["allow_anchor_update"] == "1"]
    decision = {
        "case_name": profile.case_name,
        "stage": stage,
        "gate_rows": len(gates),
        "effective_rows": sum(1 for row in gates if row["is_effective"] == "1"),
        "blocked_frames": sorted({_frame(row) for row in blocked if _frame(row) >= 0}),
        "unclear_frames": sorted({_frame(row) for row in unclear if _frame(row) >= 0}),
        "accepted_contact_frames": sorted({_frame(row) for row in accepted_contact if _frame(row) >= 0}),
        "accepted_anchor_frames": sorted({_frame(row) for row in accepted_anchor if _frame(row) >= 0}),
        "gate_csv": str(out_dir / "vlm_gates.csv"),
    }
    write_json(out_dir / "vlm_gate_decision.json", decision)
    write_aggregate_gates(profile)
    return decision


def write_aggregate_gates(profile: CaseProfile) -> None:
    paths = stage_paths(profile)
    rows: list[dict[str, object]] = []
    for stage in STAGE_QUERY_TYPES:
        path = paths["vlm_dir"] / stage / "vlm_gates.csv"
        if path.exists():
            rows.extend(read_csv(path))
    write_csv(paths["vlm_gates"], rows, GATE_FIELDS)


def _read_stage_gates(profile: CaseProfile, stage: str) -> list[dict[str, str]]:
    path = stage_paths(profile)["vlm_dir"] / stage / "vlm_gates.csv"
    if not path.exists():
        return []
    return read_csv(path)


def disabled_contact_frames(profile: CaseProfile, stage: str = "stage2") -> set[int]:
    frames: set[int] = set()
    for row in _read_stage_gates(profile, stage):
        if row.get("query_type") != "contact_relation_check":
            continue
        if row.get("is_effective") != "1":
            continue
        if row.get("pass_gate") in {GATE_REJECT, GATE_UNCLEAR} or row.get("allow_contact_residual") != "1":
            fr = _frame(row)
            if fr >= 0:
                frames.add(fr)
    return frames


def disabled_anchor_frames(profile: CaseProfile, stage: str = "stage1") -> set[int]:
    frames: set[int] = set()
    for row in _read_stage_gates(profile, stage):
        if row.get("query_type") not in {"keypart_identity_check", "track_stability_check", "keypart_visibility_check", "anchor_update_check"}:
            continue
        if row.get("is_effective") != "1":
            continue
        if row.get("pass_gate") == GATE_REJECT:
            fr = _frame(row)
            if fr >= 0:
                frames.add(fr)
    return frames


def apply_contact_gate_to_rows(profile: CaseProfile, rows: list[dict[str, str]], *, stage: str = "stage2") -> tuple[list[dict[str, str]], dict[str, object]]:
    if "disable_vlm_contact_gate" in set(profile.data.get("ablation_flags", [])):
        out = []
        for row in rows:
            new = dict(row)
            new.setdefault("vlm_contact_gate", "ablation_disabled")
            out.append(new)
        return out, {"vlm_contact_gate": "ablation_disabled", "disabled_frames": [], "rows_disabled": 0}
    disabled = disabled_contact_frames(profile, stage)
    if not disabled:
        return rows, {"vlm_contact_gate": "no_effective_disabled_frames", "disabled_frames": []}
    out: list[dict[str, str]] = []
    changed = 0
    for row in rows:
        new = dict(row)
        fr = _frame(new)
        if fr in disabled:
            for key in ("contact_active", "anchor_score", "contact_confidence", "score"):
                if key in new:
                    new[key] = "0"
            for key in (
                "use_this_point_for_hand_attachment",
                "use_previous_grasp_for_hand_attachment",
                "rim_drinking_contact",
                "hand_depth_alignment_needed",
            ):
                if key in new:
                    new[key] = "0"
            if "vlm_handle_contact" in new:
                new["vlm_handle_contact"] = "False"
            for key in ("stable_grasp_conf", "hand_to_object_contact_px", "hand_contact_3d_gap_m", "hand_object_z_gap_m"):
                if key in new:
                    new[key] = "0"
            for key in ("active_label", "object_contact_event", "hand_mug_contact_state"):
                if key in new:
                    new[key] = "disabled_by_vlm"
            new["vlm_contact_gate"] = "disabled_by_vlm"
            changed += 1
        else:
            new.setdefault("vlm_contact_gate", "unchanged")
        out.append(new)
    return out, {"vlm_contact_gate": "applied", "disabled_frames": sorted(disabled), "rows_disabled": changed}


def append_gate_summary_to_vlm_summary(profile: CaseProfile) -> None:
    paths = stage_paths(profile)
    if not paths["vlm_summary"].exists():
        write_aggregate(profile)
    aggregate = paths["vlm_gates"]
    if not aggregate.exists():
        return
    rows = read_csv(aggregate)
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row.get('stage')}:{row.get('query_type')}:{row.get('pass_gate')}"
        counts[key] = counts.get(key, 0) + 1
    lines = ["", "## Gate Artifact Summary", ""]
    for key in sorted(counts):
        lines.append(f"- {key}: {counts[key]}")
    with Path(paths["vlm_summary"]).open("a") as f:
        f.write("\n".join(lines) + "\n")
