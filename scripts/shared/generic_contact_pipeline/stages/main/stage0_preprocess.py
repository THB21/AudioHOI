from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ...core.base.config import CaseProfile
from ...core.base.io import REPO, copy_file, repo_path, write_json
from ...core.base.runtime import runtime_python
from ...core.base.schema import stage_paths


def _frame_count(sample: Path) -> int:
    frames = sorted((sample / "frames").glob("*.png")) or sorted((sample / "frames").glob("*.jpg"))
    return len(frames)


def _artifact(path: Path, *, kind: str, runner: str = "") -> dict[str, object]:
    return {
        "kind": kind,
        "path": str(path),
        "exists": path.exists(),
        "status": "reused" if path.exists() else "missing",
        "runner": runner,
        "frames": 0,
        "error": "",
    }


def _run_json(cmd: list[str]) -> tuple[dict[str, object], str]:
    result = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True)
    if result.returncode != 0:
        return {}, (result.stderr or result.stdout or f"command exited {result.returncode}")[:2000]
    text = result.stdout.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}, ""
        except Exception as exc:
            return {}, f"could not parse json output: {exc}"
    return {}, ""


def _maybe_run_sam2(profile: CaseProfile, entry: dict[str, object], *, auto_run: bool, fps: float) -> None:
    if entry["exists"]:
        return
    preprocess = profile.data.get("preprocess", {}) if isinstance(profile.data.get("preprocess", {}), dict) else {}
    if not auto_run:
        entry["status"] = "skipped"
        entry["error"] = "preprocess.auto_run is false"
        return
    if not preprocess.get("detector_text") and not preprocess.get("first_frame_box"):
        entry["status"] = "missing"
        entry["error"] = "missing preprocess.detector_text or preprocess.first_frame_box for DINO/SAM2"
        return
    case_config = REPO / "scripts/shared/generic_contact_pipeline/configs/cases" / f"{profile.case_name}.yaml"
    cmd = [
        runtime_python("audiohoi", override_env="AUDIOHOI_PYTHON"),
        str(REPO / "scripts/shared/generic_contact_pipeline/tools/run_sam2_object.py"),
        "--case-config",
        str(case_config),
        "--sample-dir",
        str(profile.sample_dir),
        "--fps",
        str(fps),
    ]
    if preprocess.get("first_frame_box"):
        cmd.extend(["--box", str(preprocess["first_frame_box"])])
    parsed, error = _run_json(cmd)
    entry["status"] = "generated" if not error and Path(str(entry["path"])).exists() else "failed"
    entry["exists"] = Path(str(entry["path"])).exists()
    entry["runner_summary"] = parsed
    entry["error"] = error


def _maybe_run_cotracker(profile: CaseProfile, entry: dict[str, object], *, auto_run: bool, fps: float) -> None:
    if entry["exists"]:
        return
    masks_dir = profile.sample_dir / "results/segmentation/masks"
    if not auto_run:
        entry["status"] = "skipped"
        entry["error"] = "preprocess.auto_run is false"
        return
    if not masks_dir.exists():
        entry["status"] = "missing"
        entry["error"] = "SAM2 masks missing; CoTracker mask query points cannot run"
        return
    case_config = REPO / "scripts/shared/generic_contact_pipeline/configs/cases" / f"{profile.case_name}.yaml"
    preprocess = profile.data.get("preprocess", {}) if isinstance(profile.data.get("preprocess", {}), dict) else {}
    cmd = [
        runtime_python("audiohoi", override_env="AUDIOHOI_PYTHON"),
        str(REPO / "scripts/shared/generic_contact_pipeline/tools/run_cotracker_object_points.py"),
        "--case-config",
        str(case_config),
        "--sample-dir",
        str(profile.sample_dir),
        "--fps",
        str(fps),
        "--object-family",
        str(profile.data.get("object_family", "generic_object")),
    ]
    if preprocess.get("tracker_resize_width"):
        cmd.extend(["--resize-width", str(preprocess["tracker_resize_width"])])
    parsed, error = _run_json(cmd)
    entry["status"] = "generated" if not error and Path(str(entry["path"])).exists() else "failed"
    entry["exists"] = Path(str(entry["path"])).exists()
    entry["runner_summary"] = parsed
    entry["error"] = error


def run(profile: CaseProfile) -> dict[str, object]:
    sample = profile.sample_dir
    preprocess = profile.data.get("preprocess", {}) if isinstance(profile.data.get("preprocess", {}), dict) else {}
    flags = set(profile.data.get("ablation_flags", []))
    audio_disabled = "disable_audio_events" in flags
    auto_run = bool(preprocess.get("auto_run", True))
    fps = float(preprocess.get("fps", 24.0))
    frame_count = _frame_count(sample)
    artifacts = {
        "sam2_masks": _artifact(sample / "results/segmentation/masks", kind="directory", runner="run_sam2_object.py"),
        "sam2_trajectory": _artifact(sample / "results/tracking/object_trajectory.csv", kind="csv", runner="run_sam2_object.py"),
        "cotracker_center": _artifact(sample / "results/tracking/object_center_trajectory.csv", kind="csv", runner="run_cotracker_object_points.py"),
        "cotracker_points": _artifact(sample / "results/tracking/object_points.csv", kind="csv", runner="run_cotracker_object_points.py"),
        "cotracker_mesh_tracks": _artifact(sample / "results/tracking/object_mesh_tracks_test.csv", kind="csv", runner="run_cotracker_object_points.py"),
        "da3_scene_depth": _artifact(sample / "results/da3/scene_depth/index.csv", kind="csv", runner="external_da3_required"),
        "audio_events": _artifact(sample / "results/events/audio_events.csv", kind="csv", runner="external_audio_events_required"),
        "gvhmr": _artifact(sample / "results/gvhmr/result.pkl", kind="pkl", runner="external_gvhmr_required"),
    }
    _maybe_run_sam2(profile, artifacts["sam2_masks"], auto_run=auto_run, fps=fps)
    artifacts["sam2_trajectory"]["exists"] = Path(str(artifacts["sam2_trajectory"]["path"])).exists()
    if artifacts["sam2_trajectory"]["exists"] and artifacts["sam2_trajectory"]["status"] == "missing":
        artifacts["sam2_trajectory"]["status"] = "reused"
    _maybe_run_cotracker(profile, artifacts["cotracker_mesh_tracks"], auto_run=auto_run, fps=fps)
    for key in ("cotracker_center", "cotracker_points"):
        artifacts[key]["exists"] = Path(str(artifacts[key]["path"])).exists()
        if artifacts[key]["exists"] and artifacts[key]["status"] == "missing":
            artifacts[key]["status"] = "reused"
    for entry in artifacts.values():
        entry["frames"] = frame_count
        if not entry["exists"] and not entry["error"] and str(entry["runner"]).startswith("external_"):
            entry["error"] = "required external preprocessing artifact is missing; no generic in-pipeline runner is registered yet"
    if audio_disabled:
        artifacts["audio_events"]["status"] = "disabled"
        artifacts["audio_events"]["exists"] = False
        artifacts["audio_events"]["error"] = "disabled by ablation flag disable_audio_events"

    checks = {
        "sample_dir": sample.exists(),
        "metadata": (sample / "metadata.json").exists(),
        "segmentation_masks": bool(artifacts["sam2_masks"]["exists"]),
        "tracking_dir": (sample / "results/tracking").exists(),
        "da3_scene_depth": bool(artifacts["da3_scene_depth"]["exists"]),
        "events_audio": bool(artifacts["audio_events"]["exists"]),
        "events_audio_disabled": audio_disabled,
        "gvhmr_available": bool(artifacts["gvhmr"]["exists"]) or any((sample / "results").glob("**/*gvhmr*")),
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
        "artifacts": artifacts,
        "note": "Stage0 reuses existing artifacts and generates missing SAM2/CoTracker object artifacts with generic tools when possible.",
    }
    manifest = {
        "stage": "stage0_preprocess",
        "case_name": profile.case_name,
        "sample_dir": str(sample),
        "result_dir": str(profile.result_dir),
        "render_dir": str(profile.render_dir),
        "camera": profile.camera,
        "prepared_inputs": artifacts,
        "checks": checks,
        "auto_run": auto_run,
        "ablation_flags": sorted(flags),
        "note": "Stage0 is non-destructive for existing artifacts; missing SAM2/CoTracker artifacts are generated by generic tools, while DA3/GVHMR/audio missing artifacts are reported with failure reasons.",
    }
    write_json(stage_paths(profile)["stage0_inputs_manifest"], manifest)
    write_json(stage_paths(profile)["stage0_metrics"], metrics)
    return metrics
