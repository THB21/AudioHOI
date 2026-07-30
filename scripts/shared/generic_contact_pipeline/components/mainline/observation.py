from __future__ import annotations

from pathlib import Path

from ...core.base.config import CaseProfile
from ...core.base.io import read_csv, write_csv, write_json
from ...core.base.schema import stage_paths
from ...core.plugins.registry import invoke_selected_plugin
from ..line_object.mainline import write_line_correspondence


def _copy_or_empty(src: Path, dst: Path, fields: list[str]) -> tuple[str, int]:
    if src.exists():
        rows = read_csv(src)
        write_csv(dst, rows)
        return str(dst), len(rows)
    write_csv(dst, [], fields=fields)
    return str(dst), 0


def _legacy_observation_adapter(profile: CaseProfile) -> dict[str, object]:
    name = profile.component("observation_model")
    result, plugin = invoke_selected_plugin(profile, "observation", name)
    result = dict(result)
    result["adapter_component"] = name
    result["capability_plugin"] = plugin
    return result


def build(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    adapter = _legacy_observation_adapter(profile)
    obs_rows = read_csv(paths["object_observations"]) if paths["object_observations"].exists() else []

    correspondence_rows: list[dict[str, object]] = []
    if profile.data.get("line_object"):
        line_metrics = write_line_correspondence(profile)
        adapter["generic_line_correspondence"] = line_metrics
        if paths["line_correspondence"].exists():
            correspondence_rows = read_csv(paths["line_correspondence"])
    else:
        for row in obs_rows:
            correspondence_rows.append(
                {
                    "frame": row.get("frame", ""),
                    "time": row.get("time", ""),
                    "correspondence_id": row.get("ref_type", "object_ref") or "object_ref",
                    "u": row.get("ref_u", row.get("u", "")),
                    "v": row.get("ref_v", row.get("v", "")),
                    "track_confidence": row.get("ref_conf", row.get("observation_conf", "1.000000")),
                    "visible_fraction": row.get("visible_fraction", "1.000000"),
                    "occlusion_state": row.get("occlusion_state", "visible"),
                    "source": row.get("ref_source", adapter.get("adapter_component", "generic_observation_adapter")),
                }
            )
    write_csv(
        paths["object_correspondence"],
        correspondence_rows,
        fields=[
            "frame",
            "time",
            "correspondence_id",
            "u",
            "v",
            "track_confidence",
            "visible_fraction",
            "occlusion_state",
            "source",
        ],
    )
    surface_path, surface_rows = _copy_or_empty(
        paths["object_local_points"],
        paths["object_surface_points"],
        ["point_id", "part", "role", "local_x", "local_y", "local_z", "source"],
    )
    semantic_path, semantic_rows = _copy_or_empty(
        paths["object_local_segments"],
        paths["object_semantic_points"],
        ["point_id", "part", "role", "local_x", "local_y", "local_z", "source"],
    )

    metrics = {
        "component": "generic_observation_mainline",
        "case_name": profile.case_name,
        "adapter": adapter,
        "object_observations": str(paths["object_observations"]),
        "object_correspondence": str(paths["object_correspondence"]),
        "object_surface_points": surface_path,
        "object_semantic_points": semantic_path,
        "rows": len(obs_rows),
        "correspondence_rows": len(correspondence_rows),
        "surface_rows": surface_rows,
        "semantic_rows": semantic_rows,
        "policy": "fixed generic observation contract; case-specific observation code is compatibility adapter only",
    }
    write_json(paths["stage1_metrics"], metrics)
    return metrics
