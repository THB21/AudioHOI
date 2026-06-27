from __future__ import annotations

from pathlib import Path

from ..core.config import CaseProfile
from ..core.io import copy_file, repo_path, write_json
from ..core.schema import stage_paths


def run(profile: CaseProfile) -> dict[str, object]:
    sample = profile.sample_dir
    checks = {
        "sample_dir": sample.exists(),
        "metadata": (sample / "metadata.json").exists(),
        "segmentation_masks": (sample / "results/segmentation/masks").exists(),
        "tracking_dir": (sample / "results/tracking").exists(),
        "da3_scene_depth": (sample / "results/da3/scene_depth/index.csv").exists(),
        "events_audio": (sample / "results/events/audio_events.csv").exists(),
        "gvhmr_available": any((sample / "results").glob("**/*gvhmr*")),
    }
    baseline_exists = {key: repo_path(value).exists() for key, value in profile.baseline.items() if isinstance(value, str)}
    provenance_snapshots: dict[str, dict[str, object]] = {}
    snapshot_specs = {}.get(profile.case_name, {})
    provenance_dir = profile.result_dir / "provenance_snapshots"
    if snapshot_specs:
        provenance_dir.mkdir(parents=True, exist_ok=True)
    for baseline_key, filename in snapshot_specs.items():
        src_raw = profile.baseline.get(baseline_key)
        if not src_raw:
            continue
        src = repo_path(src_raw)
        dst = provenance_dir / filename
        if src.exists():
            copy_file(src, dst)
        provenance_snapshots[baseline_key] = {
            "source": str(src),
            "snapshot": str(dst),
            "exists": dst.exists(),
            "note": "preserved solved provenance boundary; upstream historical inputs are not fully available for lossless regeneration",
        }

    metrics = {
        "stage": "stage0_preprocess",
        "case_name": profile.case_name,
        "checks": checks,
        "baseline_paths_exist": baseline_exists,
        "provenance_snapshots": provenance_snapshots,
        "note": "v1 validates and reuses existing preprocess artifacts; it does not overwrite them.",
    }
    manifest = {
        "stage": "stage0_preprocess",
        "case_name": profile.case_name,
        "sample_dir": str(sample),
        "result_dir": str(profile.result_dir),
        "render_dir": str(profile.render_dir),
        "camera": profile.camera,
        "prepared_inputs": {
            "sam2_masks": str(sample / "results/segmentation/masks"),
            "cotracker": str(sample / "results/tracking"),
            "da3_scene_depth_index": str(sample / "results/da3/scene_depth/index.csv"),
            "audio_events": str(sample / "results/events/audio_events.csv"),
            "gvhmr_search_root": str(sample / "results"),
        },
        "checks": checks,
        "note": "Stage0 is non-destructive: it validates existing SAM2/CoTracker/DA3/GVHMR/audio artifacts for downstream generic stages.",
    }
    write_json(stage_paths(profile)["stage0_inputs_manifest"], manifest)
    write_json(stage_paths(profile)["stage0_metrics"], metrics)
    return metrics
