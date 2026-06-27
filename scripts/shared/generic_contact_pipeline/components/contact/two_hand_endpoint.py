from __future__ import annotations

import subprocess
from pathlib import Path

from ...core.base.config import CaseProfile
from ...core.base.io import REPO, read_csv, repo_path, write_csv, write_json
from ...core.base.runtime import runtime_python
from ...core.base.schema import stage_paths


FIELDS = [
    "frame",
    "time",
    "contact_active",
    "human_part",
    "human_side",
    "object_part",
    "object_local_id",
    "contact_u",
    "contact_v",
    "contact_depth_offset_m",
    "anchor_score",
    "source",
    "object_local_x",
    "object_local_y",
    "object_local_z",
]


def _active(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _endpoint_row(obs: dict[str, str], *, side: str, state: dict[str, str], role: str) -> dict[str, object]:
    if side == "right_hand":
        u = obs.get("top_rail_left_u", "")
        v = obs.get("top_rail_left_v", "")
        local_id = "backrest_top_left"
        local_x = "-0.191000"
    else:
        u = obs.get("top_rail_right_u", "")
        v = obs.get("top_rail_right_v", "")
        local_id = "backrest_top_right"
        local_x = "0.191000"
    prefix = "primary" if side == "right_hand" else "secondary"
    return {
        "frame": obs.get("frame", ""),
        "time": obs.get("time", ""),
        "contact_active": "1",
        "human_part": "palm",
        "human_side": side,
        "object_part": "top_rail",
        "object_local_id": local_id,
        "contact_u": u,
        "contact_v": v,
        "contact_depth_offset_m": "",
        "anchor_score": state.get(f"{prefix}_dist_px", ""),
        "source": f"two_hand_endpoint_from_gvhmr_qwen_{role}_observed_toprail_endpoint",
        "object_local_x": local_x,
        "object_local_y": "0.130000",
        "object_local_z": "0.751000",
    }


def _inactive_row(obs: dict[str, str]) -> dict[str, object]:
    return {
        "frame": obs.get("frame", ""),
        "time": obs.get("time", ""),
        "contact_active": "0",
        "human_part": "none",
        "human_side": "none",
        "object_part": "none",
        "object_local_id": "",
        "contact_u": "",
        "contact_v": "",
        "contact_depth_offset_m": "",
        "anchor_score": "",
        "source": "two_hand_endpoint_no_active_gvhmr_qwen_contact",
        "object_local_x": "",
        "object_local_y": "",
        "object_local_z": "",
    }


def build(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    obs_rows = read_csv(paths["object_observations"])
    state_dir = profile.result_dir / "chair_grasp_anchor_state"
    state_csv = state_dir / "chair_grasp_anchor_state.csv"
    state_summary = state_dir / "chair_grasp_anchor_state_summary.json"
    qwen_csv = profile.sample_dir / "annotations/vlm_chair_semantic_candidates/chair_semantic_candidate_annotations.csv"
    script = REPO / "scripts/shared/generic_contact_pipeline/components/contact/chair_grasp_anchor_state.py"
    cmd = [
        runtime_python("audiohoi", override_env="AUDIOHOI_PYTHON"),
        str(script),
        "--sample-dir",
        str(profile.sample_dir),
        "--obs-csv",
        str(repo_path(paths["object_observations"])),
        "--qwen-csv",
        str(qwen_csv),
        "--out-csv",
        str(state_csv),
        "--out-summary",
        str(state_summary),
    ]
    subprocess.run(cmd, cwd=REPO, check=True)

    state_by_frame = {row["frame"]: row for row in read_csv(state_csv)}
    window = profile.data.get("contact_window", {})
    start_frame = int(window.get("start_frame", 1)) if isinstance(window, dict) else 1
    end_frame = int(window.get("end_frame", 10**9)) if isinstance(window, dict) else 10**9
    rows: list[dict[str, object]] = []
    for obs in obs_rows:
        frame = int(float(obs.get("frame", "0") or 0))
        state = state_by_frame.get(obs["frame"], {})
        wrote = False
        within_window = start_frame <= frame <= end_frame
        if within_window and _active(state.get("primary_active")):
            rows.append(_endpoint_row(obs, side="right_hand", state=state, role="right_to_left"))
            wrote = True
        if within_window and _active(state.get("secondary_active")):
            rows.append(_endpoint_row(obs, side="left_hand", state=state, role="left_to_right"))
            wrote = True
        if not wrote:
            rows.append(_inactive_row(obs))
    out = write_csv(paths["contact_candidates"], rows, FIELDS)
    active_rows = sum(1 for row in rows if row.get("contact_active") == "1")
    metrics = {
        "component": "two_hand_endpoint",
        "rows": len(rows),
        "active_rows": active_rows,
        "contact_candidates": str(out),
        "grasp_anchor_state": str(state_csv),
        "grasp_anchor_summary": str(state_summary),
        "contact_window": {"start_frame": start_frame, "end_frame": end_frame, "source": window.get("source", "") if isinstance(window, dict) else ""},
        "policy": "gvhmr/qwen gates contact frames; observed top-rail endpoints define right_hand->left and left_hand->right object anchors",
    }
    write_json(paths["stage2_metrics"], metrics)
    return metrics
