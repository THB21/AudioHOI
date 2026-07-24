#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.base.config import load_case_profile, with_runtime_overrides
from scripts.shared.generic_contact_pipeline.core.provenance.seed_dependencies import audit_seed_dependencies


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(name: str, canonical: object, fresh: object, relation: str) -> dict[str, object]:
    passed = {
        "equal": fresh == canonical,
        "greater_or_equal": float(fresh) >= float(canonical),
        "less_or_equal": float(fresh) <= float(canonical),
    }[relation]
    return {"name": name, "canonical": canonical, "fresh": fresh, "relation": relation, "pass": passed}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare an observation-derived mug rerun with the canonical result.")
    parser.add_argument("--canonical-result", default="benchmark_vlm_qwen")
    parser.add_argument("--fresh-result", required=True)
    parser.add_argument("--output", type=Path, help="Optional JSON evidence path.")
    args = parser.parse_args()

    sample = REPO / "samples_known_object/02_mug/results"
    canonical_root = sample / args.canonical_result
    fresh_root = sample / args.fresh_result
    canonical = _json(canonical_root / "evaluation/final_evaluation_summary.json")["metrics"]
    fresh = _json(fresh_root / "evaluation/final_evaluation_summary.json")["metrics"]
    checks = [
        {"name": "fresh_final_pass", "canonical": canonical.get("final_pass"), "fresh": fresh.get("final_pass"), "relation": "is_true", "pass": fresh.get("final_pass") is True},
        _check("overlay_hard_score", canonical["overlay_hard_score"], fresh["overlay_hard_score"], "greater_or_equal"),
        _check("overlay_mask_coverage", canonical["overlay_mask_coverage"], fresh["overlay_mask_coverage"], "greater_or_equal"),
        _check("overlay_render_false_coverage", canonical["overlay_render_false_coverage"], fresh["overlay_render_false_coverage"], "less_or_equal"),
        _check("contact_frame_ratio", canonical["contact_frame_ratio"], fresh["contact_frame_ratio"], "equal"),
        _check("contact_gap_mm", canonical["contact_gap_mm"], fresh["contact_gap_mm"], "less_or_equal"),
        _check("contact_proxy", canonical["contact_proxy"], fresh["contact_proxy"], "greater_or_equal"),
        _check("part_correct_ratio", canonical["part_correct_ratio"], fresh["part_correct_ratio"], "greater_or_equal"),
        _check("contact_ratio_audio_windows", canonical["contact_ratio_audio_windows"], fresh["contact_ratio_audio_windows"], "greater_or_equal"),
        _check("human_part_contact_coverage", canonical["human_part_contact_coverage"], fresh["human_part_contact_coverage"], "greater_or_equal"),
        _check("object_part_contact_coverage", canonical["object_part_contact_coverage"], fresh["object_part_contact_coverage"], "greater_or_equal"),
        _check("penetration_frame_ratio", canonical["penetration_frame_ratio"], fresh["penetration_frame_ratio"], "less_or_equal"),
        _check("non_collision_ratio", canonical["non_collision_ratio"], fresh["non_collision_ratio"], "greater_or_equal"),
        _check(
            "total_temporal_spikes",
            int(canonical["translation_spike_count"]) + int(canonical["rotation_spike_count"]),
            int(fresh["translation_spike_count"]) + int(fresh["rotation_spike_count"]),
            "less_or_equal",
        ),
        _check("jump_count", canonical["jump_count"], fresh["jump_count"], "less_or_equal"),
        _check("static_tail_drift_m", canonical["static_tail_drift_m"], fresh["static_tail_drift_m"], "less_or_equal"),
        _check("hoi_persistent_contact_rows", canonical["hoi_persistent_contact_rows"], fresh["hoi_persistent_contact_rows"], "greater_or_equal"),
    ]

    semantic_fields = ("vlm_hand_contact_part", "semantic_contact_part", "object_contact_event")
    semantic_counts: dict[str, object] = {}
    canonical_rows = _rows(canonical_root / "object_local_points.csv")
    fresh_rows = _rows(fresh_root / "object_local_points.csv")
    for field in semantic_fields:
        canonical_count = dict(sorted(Counter(row.get(field, "") for row in canonical_rows).items()))
        fresh_count = dict(sorted(Counter(row.get(field, "") for row in fresh_rows).items()))
        semantic_counts[field] = {
            "canonical": canonical_count,
            "fresh": fresh_count,
            "exact": canonical_count == fresh_count,
        }
    for label in ("handle_loop", "rim_ring"):
        canonical_count = sum(row.get("semantic_contact_part") == label for row in canonical_rows)
        fresh_count = sum(row.get("semantic_contact_part") == label for row in fresh_rows)
        checks.append(_check(f"semantic_contact_part:{label}", canonical_count, fresh_count, "equal"))
    checks.append(
        {
            "name": "vlm_hand_contact_part_distribution",
            "canonical": semantic_counts["vlm_hand_contact_part"]["canonical"],
            "fresh": semantic_counts["vlm_hand_contact_part"]["fresh"],
            "relation": "equal",
            "pass": semantic_counts["vlm_hand_contact_part"]["exact"],
        }
    )

    profile = with_runtime_overrides(load_case_profile("mug"), result_name=args.fresh_result)
    seed_audit = audit_seed_dependencies(profile)
    checks.append(
        {
            "name": "no_solved_seed_dependency",
            "canonical": True,
            "fresh": not seed_audit["has_solved_seed_dependency"],
            "relation": "is_true",
            "pass": not seed_audit["has_solved_seed_dependency"] and seed_audit["rerun_readiness"] == "ready",
        }
    )

    render_hashes = {}
    for scope in ("object_only", "with_human"):
        for view in ("overlay", "camera3d", "side_yz"):
            path = profile.render_dir / scope / f"{view}.mp4"
            render_hashes[f"{scope}/{view}.mp4"] = {
                "size_bytes": path.stat().st_size,
                "sha256_file": _sha256(path),
            }
    canonical_manifest = _json(canonical_root / "pipeline_manifest.json")
    fresh_manifest = _json(fresh_root / "pipeline_manifest.json")
    payload = {
        "schema_version": 1,
        "case_name": "mug",
        "canonical_result": args.canonical_result,
        "fresh_result": args.fresh_result,
        "pass": all(item["pass"] for item in checks),
        "checks": checks,
        "semantic_distributions": semantic_counts,
        "seed_audit": seed_audit,
        "gate_execution": {
            "canonical_vlm_mode": canonical_manifest.get("vlm_mode"),
            "fresh_vlm_mode": fresh_manifest.get("vlm_mode"),
            "comparable": canonical_manifest.get("vlm_mode") == fresh_manifest.get("vlm_mode"),
            "note": "Contact VLM annotations are compared above; pipeline VLM gate timelines require matching execution modes.",
        },
        "fresh_render_hashes": render_hashes,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
