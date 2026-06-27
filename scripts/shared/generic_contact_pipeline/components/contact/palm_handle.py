from __future__ import annotations

from ...core.base.config import CaseProfile
from ...core.base.io import read_csv, write_csv, write_json
from ...core.base.schema import stage_paths


def build(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    rows = []
    for row in read_csv(paths["object_local_points"]):
        object_part = row.get("semantic_contact_part", row.get("nearest_articraft_part", ""))
        update = row.get("use_this_point_for_hand_attachment", "")
        keep = row.get("use_previous_grasp_for_hand_attachment", "")
        active = 1 if row.get("object_contact_event", "") not in {"", "none", "no_contact"} else 0
        rows.append(
            {
                "frame": row["frame"],
                "time": row["time"],
                "contact_active": active,
                "human_part": "palm",
                "human_side": row.get("active_label", ""),
                "object_part": object_part,
                "object_local_id": row.get("nearest_vertex_index", ""),
                "contact_u": row.get("contact_u", ""),
                "contact_v": row.get("contact_v", ""),
                "contact_depth_offset_m": row.get("hand_object_z_gap_m", ""),
                "anchor_score": row.get("contact_conf", ""),
                "source": "palm_handle_from_stable_grasp_anchor",
                "stable_local_x": row.get("mug_local_x", ""),
                "stable_local_y": row.get("mug_local_y", ""),
                "stable_local_z": row.get("mug_local_z", ""),
                "visibility": row.get("vlm_visibility", ""),
                "anchor_update": update,
                "keep_previous": keep,
            }
        )
    out = write_csv(paths["contact_candidates"], rows)
    metrics = {"component": "palm_handle", "rows": len(rows), "contact_candidates": str(out), "source": str(paths["object_local_points"])}
    write_json(paths["stage2_metrics"], metrics)
    return metrics
