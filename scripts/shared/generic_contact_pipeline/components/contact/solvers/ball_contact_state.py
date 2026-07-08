from __future__ import annotations

import subprocess
import os
from pathlib import Path

from ....core.base.config import CaseProfile
from ....core.base.io import REPO
from ....core.base.io import read_csv, write_csv, write_json
from ....core.base.runtime import runtime_python
from ....core.base.schema import stage_paths


STATE_FIELDS = [
    "frame",
    "time",
    "proposal_score",
    "anchor_score",
    "floor_score",
    "anchor_contact_state",
    "human_contact_state",
    "floor_contact_state",
    "transition_contact_state",
    "multi_contact_state",
    "confidence_level",
    "contact_part",
    "contact_label",
    "contact_side",
    "anchor_type",
    "min_object_boundary_gap_px",
    "active_part_distance_px",
    "contact_depth_offset_m",
    "object_acceleration",
    "support_surface_type",
    "support_source",
    "signed_support_gap_px",
    "support_conf",
    "active_object_point_id",
    "active_object_u",
    "active_object_v",
    "state_active_part_u",
    "state_active_part_v",
]


def _by_frame(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            out[int(float(row["frame"]))] = row
        except Exception:
            continue
    return out


def _s(row: dict[str, str], key: str, default: str = "") -> str:
    value = row.get(key, default)
    return default if value is None else str(value)


def _is_on(row: dict[str, str], key: str) -> bool:
    try:
        return int(float(row.get(key, "0") or 0)) == 1
    except Exception:
        return False


def _label(default_part: str, side: str, human_on: bool, floor_on: bool) -> tuple[str, str, str]:
    if floor_on and not human_on:
        return "floor", "floor", ""
    side = side.strip()
    if side:
        return default_part, f"{side}_{default_part}", side
    return default_part, default_part, ""


def _build_ball_proxy_depth(profile: CaseProfile) -> str:
    paths = stage_paths(profile)
    out_csv = profile.result_dir / "ball_proxy_depth.csv"
    builder = REPO / "scripts/shared/generic_contact_pipeline/components/contact/solvers/ball_proxy_depth_builder.py"
    cmd = [
        runtime_python("audiohoi"),
        str(builder),
        "--sample-dir",
        str(profile.sample_dir),
        "--object-observation-csv",
        str(paths["object_observations"]),
        "--output-csv",
        str(out_csv),
    ]
    env = os.environ.copy()
    if "disable_audio_events" in set(profile.data.get("ablation_flags", [])):
        env["AUDIOHOI_DISABLE_AUDIO_EVENTS"] = "1"
    result = subprocess.run(cmd, cwd=str(REPO), text=True, capture_output=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            "Generic ball proxy-depth builder failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return str(out_csv)


def _build_ball_contact_candidates(profile: CaseProfile, proxy_csv: str) -> str:
    out_dir = profile.result_dir / "contact_candidates_internal"
    builder = REPO / "scripts/shared/generic_contact_pipeline/components/contact/solvers/ball_contact_candidate_builder.py"
    cmd = [
        runtime_python("audiohoi"),
        str(builder),
        "--sample-dir",
        str(profile.sample_dir),
        "--object-proxy-observation-csv",
        proxy_csv,
        "--out-dir",
        str(out_dir),
    ]
    env = os.environ.copy()
    if "disable_audio_events" in set(profile.data.get("ablation_flags", [])):
        env["AUDIOHOI_DISABLE_AUDIO_EVENTS"] = "1"
    result = subprocess.run(cmd, cwd=str(REPO), text=True, capture_output=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            "Generic ball contact-candidate builder failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return str(out_dir)


def build(profile: CaseProfile, *, default_part: str, source_name: str) -> dict[str, object]:
    paths = stage_paths(profile)
    proxy_csv = _build_ball_proxy_depth(profile)
    root = Path(_build_ball_contact_candidates(profile, proxy_csv))
    anchor_csv = root / "anchor_contact_candidates.csv"
    floor_csv = root / "floor_contact_candidates.csv"
    if not anchor_csv.exists():
        raise FileNotFoundError(f"Missing anchor contact candidates: {anchor_csv}")
    if not floor_csv.exists():
        raise FileNotFoundError(f"Missing floor contact candidates: {floor_csv}")
    if not Path(proxy_csv).exists():
        raise FileNotFoundError(f"Missing object proxy observations for contact depth offsets: {proxy_csv}")

    anchor_by_frame = _by_frame(read_csv(anchor_csv))
    floor_by_frame = _by_frame(read_csv(floor_csv))
    proxy_by_frame = _by_frame(read_csv(proxy_csv))
    frames = sorted(set(anchor_by_frame) | set(floor_by_frame) | set(proxy_by_frame))

    state_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for fr in frames:
        anchor = anchor_by_frame.get(fr, {})
        floor = floor_by_frame.get(fr, {})
        proxy = proxy_by_frame.get(fr, {})
        human_on = _is_on(anchor, "anchor_contact_state")
        floor_on = _is_on(floor, "floor_contact_state")
        active = int(human_on or floor_on)
        contact_part, contact_label, contact_side = _label(default_part, _s(anchor, "active_side"), human_on, floor_on)
        confidence = _s(anchor, "confidence_level") or _s(floor, "confidence_level")
        if floor_on and not human_on:
            confidence = _s(floor, "confidence_level", confidence)
        proposal_score = _s(anchor, "proposal_score") or _s(floor, "proposal_score")
        anchor_score = _s(anchor, "anchor_score", "0.000000")
        floor_score = _s(floor, "floor_score", "0.000000")
        offset = _s(proxy, "contact_depth_offset_m")
        contact_u = _s(anchor, "active_object_u") or _s(proxy, "contact_u")
        contact_v = _s(anchor, "active_object_v") or _s(proxy, "contact_v")
        state = {
            "frame": str(fr),
            "time": _s(anchor, "time") or _s(floor, "time") or _s(proxy, "time"),
            "proposal_score": proposal_score,
            "anchor_score": anchor_score,
            "floor_score": floor_score,
            "anchor_contact_state": int(human_on),
            "human_contact_state": int(human_on),
            "floor_contact_state": int(floor_on),
            "transition_contact_state": int(active),
            "multi_contact_state": int(human_on and floor_on),
            "confidence_level": confidence,
            "contact_part": contact_part,
            "contact_label": contact_label,
            "contact_side": contact_side,
            "anchor_type": default_part,
            "min_object_boundary_gap_px": _s(anchor, "min_object_boundary_gap_px"),
            "active_part_distance_px": _s(anchor, "active_part_distance_px"),
            "contact_depth_offset_m": offset,
            "object_acceleration": "",
            "support_surface_type": "floor" if profile.case_name == "basketball" else "unknown_plane",
            "support_source": "generic_floor_candidates",
            "signed_support_gap_px": _s(floor, "signed_gap"),
            "support_conf": floor_score,
            "active_object_point_id": _s(anchor, "active_object_point_id") or _s(proxy, "contact_proxy_name"),
            "active_object_u": contact_u,
            "active_object_v": contact_v,
            "state_active_part_u": _s(proxy, "active_part_u"),
            "state_active_part_v": _s(proxy, "active_part_v"),
        }
        state_rows.append(state)
        candidate_rows.append(
            {
                "frame": str(fr),
                "time": state["time"],
                "contact_active": active,
                "human_part": default_part if human_on else ("floor" if floor_on else "none"),
                "human_side": contact_side,
                "object_part": "ball_boundary" if human_on else ("floor_support" if floor_on else "none"),
                "object_local_id": state["active_object_point_id"],
                "contact_u": contact_u,
                "contact_v": contact_v,
                "contact_depth_offset_m": offset,
                "contact_conf": _s(proxy, "contact_conf") or anchor_score,
                "anchor_score": anchor_score if human_on else floor_score,
                "source": source_name,
            }
        )

    state_out = write_csv(paths["contact_state"], state_rows, STATE_FIELDS)
    candidates_out = write_csv(paths["contact_candidates"], candidate_rows)
    metrics = {
        "component": source_name,
        "rows": len(candidate_rows),
        "contact_candidates": str(candidates_out),
        "contact_state": str(state_out),
        "anchor_candidate_source": str(anchor_csv),
        "floor_candidate_source": str(floor_csv),
        "depth_offset_source": str(proxy_csv),
        "policy": "generic fusion of generic anchor/floor candidate gates; generic ball_proxy_depth supplies GVHMR+mesh+DA3 contact-depth offsets",
    }
    write_json(paths["stage2_metrics"], metrics)
    return metrics
