from __future__ import annotations

from collections import defaultdict

from ...core.base.config import CaseProfile
from ...core.base.io import read_csv, write_csv, write_json
from ...core.base.schema import stage_paths
from ...core.plugins.registry import invoke_selected_plugin
from ..line_object.mainline import write_anchor_state


ANCHOR_FIELDS = [
    "frame",
    "time",
    "contact_id",
    "human_part",
    "human_side",
    "object_part",
    "contact_observed",
    "contact_persistent",
    "anchor_update_allowed",
    "pose_anchor_allowed",
    "stable_local_x",
    "stable_local_y",
    "stable_local_z",
    "stable_local_s",
    "observed_local_x",
    "observed_local_y",
    "observed_local_z",
    "observed_local_s",
    "contact_confidence",
    "anchor_action",
    "source",
]


def _legacy_contact_adapter(profile: CaseProfile) -> dict[str, object]:
    name = profile.component("contact_policy")
    result, plugin = invoke_selected_plugin(profile, "contact", name)
    result = dict(result)
    result["adapter_component"] = name
    result["capability_plugin"] = plugin
    return result


def _active(row: dict[str, str]) -> bool:
    try:
        return int(float(row.get("contact_active", "0") or 0)) == 1
    except Exception:
        return False


def _confidence(row: dict[str, str]) -> str:
    for key in ("contact_conf", "anchor_score", "proposal_score", "contact_confidence"):
        value = row.get(key, "")
        if value not in {"", None}:
            return str(value)
    return "1.000000" if _active(row) else "0.000000"


def _generic_anchor_rows(contact_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_id: defaultdict[str, int] = defaultdict(int)
    out: list[dict[str, object]] = []
    for row in contact_rows:
        observed = _active(row)
        contact_id = (
            row.get("contact_label")
            or "_".join(part for part in [row.get("human_side", ""), row.get("human_part", ""), row.get("object_part", "")] if part)
            or "object_contact"
        )
        if observed:
            by_id[contact_id] += 1
        else:
            by_id[contact_id] = 0
        persistent = observed and by_id[contact_id] >= 2
        local_s = row.get("object_local_s", row.get("stable_object_local_s", ""))
        out.append(
            {
                "frame": row.get("frame", ""),
                "time": row.get("time", ""),
                "contact_id": contact_id,
                "human_part": row.get("human_part", ""),
                "human_side": row.get("human_side", ""),
                "object_part": row.get("object_part", ""),
                "contact_observed": "1" if observed else "0",
                "contact_persistent": "1" if persistent else "0",
                "anchor_update_allowed": "1" if observed else "0",
                "pose_anchor_allowed": "1" if observed else "0",
                "stable_local_x": row.get("object_local_x", ""),
                "stable_local_y": row.get("object_local_y", ""),
                "stable_local_z": row.get("object_local_z", ""),
                "stable_local_s": row.get("stable_object_local_s", local_s),
                "observed_local_x": row.get("object_local_x", ""),
                "observed_local_y": row.get("object_local_y", ""),
                "observed_local_z": row.get("object_local_z", ""),
                "observed_local_s": local_s,
                "contact_confidence": _confidence(row),
                "anchor_action": "update" if observed else "inactive",
                "source": row.get("source", "generic_contact_anchor_adapter"),
            }
        )
    return out


def build(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    adapter = _legacy_contact_adapter(profile)
    if profile.data.get("line_object"):
        anchor_metrics = write_anchor_state(profile)
        adapter["generic_anchor_state"] = anchor_metrics
        rows = read_csv(paths["anchor_state"]) if paths["anchor_state"].exists() else []
    else:
        contact_rows = read_csv(paths["contact_candidates"]) if paths["contact_candidates"].exists() else []
        rows = _generic_anchor_rows(contact_rows)
        write_csv(paths["anchor_state"], rows, ANCHOR_FIELDS)

    metrics = {
        "component": "generic_contact_anchor_mainline",
        "case_name": profile.case_name,
        "adapter": adapter,
        "contact_candidates": str(paths["contact_candidates"]),
        "anchor_state": str(paths["anchor_state"]),
        "anchor_rows": len(rows),
        "policy": "fixed generic contact/anchor contract; case contact code is compatibility candidate builder only",
    }
    write_json(paths["stage2_metrics"], metrics)
    return metrics
