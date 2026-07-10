from __future__ import annotations

from collections import defaultdict
from typing import Any

from .schemas import EvaluationPaths, MetricBlock
from .utils import mean, read_rows, write_json, write_rows


DEFAULT_HUMAN_PARTS = ["left_hand", "right_hand", "left_foot", "right_foot", "torso", "hip", "back", "head", "mouth"]
IGNORED_OBJECT_PARTS = {"background", "unclear", "none", ""}
HAND_ALIASES = {"palm", "hand", "fingertip"}
FOOT_ALIASES = {"foot"}

OBJECT_PART_ALIASES = {
    "ball_boundary": "surface",
    "ball_center": "center",
    "ball_bottom": "support_region",
    "contact_patch": "surface",
    "floor_support": "support_region",
    "handle": "handle",
    "handle_loop": "handle",
    "mug_handle": "handle",
    "rim": "rim",
    "rim_ring": "rim",
    "rim_mouth": "rim",
    "cup_body": "cup_body",
    "mug_body": "cup_body",
    "body": "cup_body",
    "body_shell": "cup_body",
    "body_shell_or_occluded_handle_region": "cup_body",
    "bottom": "bottom",
    "bottom_disk": "bottom",
    "front_leg": "legs",
    "rear_leg": "legs",
    "leg": "legs",
    "legs": "legs",
    "feet": "feet",
    "seat": "seat",
    "seat_slat": "seat",
    "seat_edge": "seat",
    "backrest": "back",
    "backrest_board": "back",
    "back": "back",
    "top_rail": "top_rail",
    "top_rail_endpoint": "top_rail",
    "front_stretcher": "stretcher",
    "rear_stretcher": "stretcher",
    "stretcher": "stretcher",
    "hole": "hole",
    "main_body": "shaft",
    "shaft": "shaft",
    "contact_region": "grip_region",
    "grip_region": "grip_region",
    "grip_regions": "grip_region",
    "support_region": "support_region",
    "left_palm_line_anchor": "grip_region",
    "right_palm_line_anchor": "grip_region",
    "endpoint": "endpoints",
    "endpoints": "endpoints",
}


def _norm_human_part(part: str, side: str = "") -> str:
    part = str(part or "").strip()
    side = str(side or "").strip()
    if part in HAND_ALIASES:
        if "left" in side:
            return "left_hand"
        if "right" in side:
            return "right_hand"
        return "left_hand" if side == "" else side
    if part in FOOT_ALIASES:
        if "left" in side:
            return "left_foot"
        if "right" in side:
            return "right_foot"
        return "left_foot" if side == "" else side
    if part in {"body", "torso"}:
        return "torso"
    return part


def _clean_part(part: str) -> str:
    return str(part or "").strip().lower()


def _norm_object_part(part: str) -> str:
    cleaned = _clean_part(part)
    if cleaned in IGNORED_OBJECT_PARTS:
        return ""
    return OBJECT_PART_ALIASES.get(cleaned, cleaned)


def _object_parts(config: dict[str, Any], contact_pairs: list[dict[str, str]], surface_rows: list[dict[str, str]]) -> list[str]:
    parts: list[str] = []
    for part in ((config.get("vlm") or {}).get("parts") or []):
        p = _norm_object_part(str(part))
        if p not in IGNORED_OBJECT_PARTS and p not in parts:
            parts.append(p)
    for row in contact_pairs:
        p = _norm_object_part(str(row.get("object_part", "")))
        if p not in IGNORED_OBJECT_PARTS and p not in parts:
            parts.append(p)
    for row in surface_rows:
        p = _norm_object_part(str(row.get("part") or row.get("object_part") or row.get("semantic_part") or ""))
        if p not in IGNORED_OBJECT_PARTS and p not in parts:
            parts.append(p)
    return parts


def _human_evidence(paths: EvaluationPaths) -> dict[str, bool]:
    base = paths.sample_dir / "results"
    has_body = (base / "gvhmr" / "result.pkl").exists() or (base / "human_gvhmr" / "result.pkl").exists()
    has_hands = (
        (base / "hands" / "hand_keypoints_3d.csv").exists()
        or (base / "human_hands" / "hand_keypoints_3d.csv").exists()
        or (base / "hands" / "stitched_smplx_params.pkl").exists()
        or (base / "human_hands" / "stitched_smplx_params.pkl").exists()
    )
    return {
        "left_hand": has_hands,
        "right_hand": has_hands,
        "left_foot": has_body,
        "right_foot": has_body,
        "torso": has_body,
        "hip": has_body,
        "back": has_body,
        "head": has_body,
        "mouth": has_body or has_hands,
    }


def compute_part_metrics(paths: EvaluationPaths, config: dict[str, Any]) -> MetricBlock:
    contact_pairs = read_rows(paths.evaluation_dir / "hoi_contact_pairs.csv")
    surface_rows = read_rows(paths.result_dir / "object_surface_points.csv")
    semantic_rows = read_rows(paths.result_dir / "object_semantic_points.csv")
    surface_rows = surface_rows or semantic_rows
    human_evidence = _human_evidence(paths)
    human_contact_counts = {part: 0 for part in DEFAULT_HUMAN_PARTS}
    object_contact_counts: dict[str, int] = {}
    raw_by_canonical: dict[str, set[str]] = defaultdict(set)
    vocab_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in contact_pairs:
        if str(row.get("observed", "")) not in {"1", "1.0", "true", "True"}:
            continue
        hp = _norm_human_part(str(row.get("human_part", "")), str(row.get("human_side", "")))
        if hp not in human_contact_counts:
            human_contact_counts[hp] = 0
        human_contact_counts[hp] += 1
        raw_op = _clean_part(str(row.get("object_part", "")))
        op = _norm_object_part(raw_op)
        if op not in IGNORED_OBJECT_PARTS:
            object_contact_counts[op] = object_contact_counts.get(op, 0) + 1
            raw_by_canonical[op].add(raw_op)
            vocab_sources[(raw_op, op)].add("hoi_contact_pairs")

    human_parts = sorted(set(DEFAULT_HUMAN_PARTS) | {p for p, n in human_contact_counts.items() if n})
    human_rows = [
        {
            "part": part,
            "available": int(bool(human_evidence.get(part, False) or human_contact_counts.get(part, 0))),
            "contact_evidence_rows": human_contact_counts.get(part, 0),
            "source": "body_or_hand_artifact+hoi_contact_pairs",
        }
        for part in human_parts
    ]

    object_parts = _object_parts(config, contact_pairs, surface_rows)
    surface_counts: dict[str, int] = {}
    for part in ((config.get("vlm") or {}).get("parts") or []):
        raw = _clean_part(str(part))
        canonical = _norm_object_part(raw)
        if canonical not in IGNORED_OBJECT_PARTS:
            raw_by_canonical[canonical].add(raw)
            vocab_sources[(raw, canonical)].add("case_config")
    for row in surface_rows:
        raw = _clean_part(str(row.get("part") or row.get("object_part") or row.get("semantic_part") or ""))
        p = _norm_object_part(raw)
        if p not in IGNORED_OBJECT_PARTS:
            surface_counts[p] = surface_counts.get(p, 0) + 1
            raw_by_canonical[p].add(raw)
            vocab_sources[(raw, p)].add("object_surface_points")
    for row in contact_pairs:
        raw = _clean_part(str(row.get("object_part", "")))
        canonical = _norm_object_part(raw)
        if canonical not in IGNORED_OBJECT_PARTS:
            raw_by_canonical[canonical].add(raw)
            vocab_sources[(raw, canonical)].add("hoi_contact_pairs")
    object_rows = [
        {
            "part": part,
            "raw_parts": "|".join(sorted(raw_by_canonical.get(part, {part}))),
            "available": int(bool(surface_counts.get(part, 0) or object_contact_counts.get(part, 0))),
            "surface_point_rows": surface_counts.get(part, 0),
            "contact_evidence_rows": object_contact_counts.get(part, 0),
            "source": "case_config+object_surface_points+hoi_contact_pairs+vocabulary_normalization",
        }
        for part in object_parts
    ]
    vocab_rows = [
        {
            "raw_part": raw,
            "canonical_part": canonical,
            "source": "+".join(sorted(sources)),
        }
        for (raw, canonical), sources in sorted(vocab_sources.items())
        if raw and canonical
    ]

    relevant_human = sorted({part for part, count in human_contact_counts.items() if count} | ({"left_hand", "right_hand"} if human_evidence.get("left_hand") or human_evidence.get("right_hand") else set()))
    human_contact_coverage = mean(1.0 if human_contact_counts.get(part, 0) else 0.0 for part in relevant_human)
    object_contact_coverage = mean(1.0 if object_contact_counts.get(row["part"], 0) else 0.0 for row in object_rows)
    metrics = {
        "human_part_count": len(human_rows),
        "human_part_available_count": sum(int(row["available"]) for row in human_rows),
        "human_part_contact_coverage": human_contact_coverage,
        "object_part_count": len(object_rows),
        "object_part_available_count": sum(int(row["available"]) for row in object_rows),
        "object_part_contact_coverage": object_contact_coverage,
    }
    human_csv = write_rows(paths.evaluation_dir / "human_parts.csv", human_rows)
    object_csv = write_rows(paths.evaluation_dir / "object_parts.csv", object_rows)
    vocab_csv = write_rows(paths.evaluation_dir / "object_part_vocab_map.csv", vocab_rows)
    metrics_csv = write_rows(paths.evaluation_dir / "part_metrics.csv", [metrics])
    metrics_json = write_json(paths.evaluation_dir / "part_metrics.json", metrics)
    return MetricBlock(
        "part_metrics",
        metrics,
        {
            "human_parts_csv": str(human_csv),
            "object_parts_csv": str(object_csv),
            "object_part_vocab_map_csv": str(vocab_csv),
            "csv": str(metrics_csv),
            "json": str(metrics_json),
        },
    )
