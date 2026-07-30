from __future__ import annotations

from typing import Any


def enforce_declared_static_tail(
    profile: Any,
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Preserve a compatibility policy's explicit static-tail postcondition."""
    if "table_freeze" not in profile.refinement_policies() or not rows:
        return rows, {"enabled": False, "reason": "table_freeze_not_requested"}
    static_values = {
        int(float(row["m45_static_frame"]))
        for row in rows
        if row.get("m45_static_frame", "")
    }
    if len(static_values) != 1:
        return rows, {"enabled": False, "reason": "missing_or_inconsistent_declared_static_frame"}
    static_frame = next(iter(static_values))
    reference = next((row for row in rows if int(float(row.get("frame", "0") or 0)) == static_frame), None)
    if reference is None:
        return rows, {"enabled": False, "reason": "declared_static_frame_not_found", "static_frame": static_frame}
    fields = (
        "tx", "ty", "tz", "qw", "qx", "qy", "qz",
        "x", "y", "z", "yaw", "pitch", "roll", "scale",
        "yaw_deg", "pitch_deg", "roll_deg",
    )
    out = [dict(row) for row in rows]
    frozen = 0
    for row in out:
        if int(float(row.get("frame", "0") or 0)) <= static_frame:
            continue
        for field in fields:
            if field in reference:
                row[field] = reference[field]
        row["static_tail_postcondition"] = "table_freeze_after_sequence_optimizer"
        row["source"] = (row.get("source", "") + "+table_freeze_postcondition").strip("+")
        frozen += 1
    return out, {
        "enabled": True,
        "static_frame": static_frame,
        "frozen_rows": frozen,
        "policy": "reapply declared table-freeze postcondition after generic sequence smoothing",
    }
