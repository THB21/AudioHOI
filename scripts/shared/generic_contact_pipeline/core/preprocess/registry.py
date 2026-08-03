"""Capability-driven registration of the fixed case-ingestion DAG."""
from __future__ import annotations

import csv
import json
import os
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np

from ..base.config import CONFIG_DIR, CaseProfile
from ..base.io import REPO
from ..base.runtime import runtime_python
from .types import ArtifactSpec, PreprocessTask


def _images(path: Path) -> list[Path]:
    return sorted(path.glob("*.png")) or sorted(path.glob("*.jpg"))


def _validate_frames(path: Path, expected: int) -> None:
    count = len(_images(path))
    if count != expected:
        raise ValueError(f"frame extraction produced {count} frames; expected {expected}")


def _validate_masks(path: Path, expected: int) -> None:
    count = len(list(path.glob("*_mask.png")))
    if count != expected:
        raise ValueError(f"SAM2 produced {count} masks; expected {expected}")


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_frame_csv(path: Path, expected: int, *, allow_multiple: bool = False) -> None:
    rows = _csv_rows(path)
    frames = {int(float(row["frame"])) for row in rows}
    if len(frames) != expected:
        raise ValueError(f"{path.name} covers {len(frames)} frames; expected {expected}")
    if not allow_multiple and len(rows) != expected:
        raise ValueError(f"{path.name} has {len(rows)} rows; expected {expected}")


def _validate_da3(path: Path, expected: int) -> None:
    index = path / "index.csv"
    _validate_frame_csv(index, expected)
    for row in _csv_rows(index):
        depth = path / row["file"]
        values = np.load(depth, allow_pickle=False)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError(f"invalid DA3 depth frame: {depth}")


def _validate_gvhmr(path: Path, expected: int) -> None:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    required = {"smpl_params_global", "smpl_params_incam", "K_fullimg"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"GVHMR result is missing fields: {sorted(required - set(payload or {}))}")
    incam = payload["smpl_params_incam"]
    if not isinstance(incam, dict) or "transl" not in incam:
        raise ValueError("GVHMR result has no incam translation")
    if int(np.asarray(incam["transl"]).shape[0]) != expected:
        raise ValueError("GVHMR result frame count does not match video")


def _validate_audio_events(path: Path, _expected: int) -> None:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        rows = list(reader)
    required = {"event", "audio_time", "audio_frame", "peak", "prominence", "audio_score"}
    if not required.issubset(fields):
        raise ValueError(f"audio events are missing columns: {sorted(required - fields)}")
    for row in rows:
        if not all(np.isfinite(float(row[field])) for field in required - {"event"}):
            raise ValueError("audio event contains a non-finite numeric value")


def _validate_object_depth_prior(path: Path, expected: int) -> None:
    rows = _csv_rows(path)
    if len(rows) != expected or {int(row["frame"]) for row in rows} != set(range(1, expected + 1)):
        raise ValueError(f"object depth prior has {len(rows)} rows; expected {expected}")
    required = {"u", "v", "da3_depth_raw", "da3_depth_smooth", "object_depth_confidence"}
    for row in rows:
        if not all(np.isfinite(float(row[field])) for field in required):
            raise ValueError("object depth prior contains a non-finite numeric value")


def _runtime_command(environment: str, script: Path, *arguments: str) -> tuple[str, ...]:
    return (runtime_python(environment), str(script), *arguments)


def _task(
    task_id: str,
    environment: str,
    dependencies: Iterable[str],
    inputs: Iterable[ArtifactSpec],
    outputs: Iterable[ArtifactSpec],
    command: tuple[str, ...],
    *,
    config: dict[str, object],
    model: dict[str, object] | None = None,
    required: bool = True,
    human_state_role: str | None = None,
    validator=None,
) -> PreprocessTask:
    return PreprocessTask(
        task_id=task_id,
        runtime_env=environment,
        dependencies=tuple(dependencies),
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        command_builder=lambda command=command: command,
        config_fingerprint=config,
        model_identity=model or {},
        required=required,
        human_state_role=human_state_role,
        output_validator=validator,
    )


def build_preprocess_tasks(profile: CaseProfile) -> tuple[PreprocessTask, ...]:
    sample = profile.sample_dir
    results = sample / "results"
    preprocess = profile.data.get("preprocess", {})
    preprocess = dict(preprocess) if isinstance(preprocess, dict) else {}
    fps = float(preprocess.get("fps", 0.0))
    audio_disabled = "disable_audio_events" in set(profile.data.get("ablation_flags", ()))
    config_path = CONFIG_DIR / f"{profile.case_name}.yaml"
    media_tool = REPO / "scripts/shared/generic_contact_pipeline/tools/run_media_extract.py"
    sam2_tool = REPO / "scripts/shared/generic_contact_pipeline/tools/run_sam2_object.py"
    cotracker_tool = REPO / "scripts/shared/generic_contact_pipeline/tools/run_cotracker_object_points.py"
    da3_tool = REPO / "scripts/shared/generic_contact_pipeline/tools/run_da3_scene_depth.py"
    depth_prior_tool = REPO / "scripts/shared/generic_contact_pipeline/tools/run_object_depth_prior.py"
    audio_tool = REPO / "scripts/shared/generic_contact_pipeline/tools/run_audio_event_extract.py"
    gvhmr_tool = REPO / "scripts/shared/human/gvhmr/run_gvhmr.py"
    da3_root = Path(os.environ.get("AUDIOHOI_DA3_ROOT", REPO / "third-party/Depth-Anything-3"))
    da3_cli = da3_root / "src/depth_anything_3/cli.py"
    da3_model = str(preprocess.get("da3_model", "depth-anything/DA3METRIC-LARGE"))
    da3_chunk_size = int(preprocess.get("da3_chunk_size", 16))
    gvhmr_checkpoint = REPO / "third-party/GVHMR/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"
    frame_args = ("--sample-dir", str(sample), "--kind", "frames")
    audio_args = ("--sample-dir", str(sample), "--kind", "audio")
    sam2_args = ["--case-config", str(config_path), "--sample-dir", str(sample)]
    if fps > 0.0:
        sam2_args.extend(("--fps", str(fps)))
    if preprocess.get("first_frame_box"):
        sam2_args.extend(("--box", str(preprocess["first_frame_box"])))
    cotracker_args = [
        "--case-config",
        str(config_path),
        "--sample-dir",
        str(sample),
        "--object-family",
        str(profile.data.get("object_family", "generic_object")),
    ]
    if fps > 0.0:
        cotracker_args.extend(("--fps", str(fps)))
    if preprocess.get("tracker_resize_width"):
        cotracker_args.extend(("--resize-width", str(preprocess["tracker_resize_width"])))
    tracker_sequence_mode = str(preprocess.get("tracker_sequence_mode", "chunked_legacy"))
    tracker_grid_size = int(preprocess.get("tracker_grid_size", 12))
    cotracker_args.extend(("--sequence-mode", tracker_sequence_mode, "--grid-size", str(tracker_grid_size)))
    da3_args = ["--sample-dir", str(sample), "--da3-root", str(da3_root)]
    da3_args.extend(("--model-dir", da3_model, "--chunk-size", str(da3_chunk_size)))
    if preprocess.get("da3_process_res"):
        da3_args.extend(("--process-res", str(preprocess["da3_process_res"])))
    gvhmr_args = ["--sample-dir", str(sample), "--fps", str(int(round(fps or 30.0)))]
    if preprocess.get("static_camera"):
        gvhmr_args.append("--static-cam")
    if preprocess.get("person_index") is not None:
        gvhmr_args.extend(("--person", str(preprocess["person_index"])))

    video = ArtifactSpec("video", sample / "video.mp4")
    frames = ArtifactSpec("frames", sample / "frames", "directory")
    audio = ArtifactSpec("audio", sample / "audio.wav", required=not audio_disabled)
    masks = ArtifactSpec("sam2_masks", results / "segmentation/masks", "directory")
    trajectory = ArtifactSpec("sam2_trajectory", results / "tracking/object_trajectory.csv")
    center = ArtifactSpec("cotracker_center", results / "tracking/object_center_trajectory.csv")
    points = ArtifactSpec("cotracker_points", results / "tracking/object_points.csv")
    mesh = ArtifactSpec("cotracker_mesh_tracks", results / "tracking/object_mesh_tracks_test.csv")
    persistent_tracking = tracker_sequence_mode in {"persistent_offline", "persistent_online"}
    rigid_tracks = ArtifactSpec(
        "cotracker_rigid_tracks",
        results / "tracking/rigid_point_tracks.csv",
        required=persistent_tracking,
    )
    rigid_tracks_manifest = ArtifactSpec(
        "cotracker_rigid_tracks_manifest",
        results / "tracking/rigid_point_tracks_manifest.json",
        required=persistent_tracking,
    )
    depth = ArtifactSpec("da3_scene_depth", results / "da3/scene_depth", "directory")
    depth_prior = ArtifactSpec("object_depth_prior", results / "da3/priors/object_depth_prior.csv")
    gvhmr = ArtifactSpec("gvhmr_result", results / "gvhmr/result.pkl")
    events = ArtifactSpec(
        "audio_events", results / "events/audio_events.csv", required=not audio_disabled
    )
    return (
        _task(
            "frame_extract", "audiohoi", (), (video,), (frames,),
            _runtime_command("audiohoi", media_tool, *frame_args),
            config={"kind": "frames"}, validator=lambda count: _validate_frames(frames.path, count),
        ),
        _task(
            "audio_extract", "audiohoi", (), (video,), (audio,),
            _runtime_command("audiohoi", media_tool, *audio_args),
            config={"kind": "audio", "sample_rate": 16000}, required=not audio_disabled,
        ),
        _task(
            "sam2", "audiohoi", ("frame_extract",), (frames, ArtifactSpec("case_config", config_path)),
            (masks, trajectory), _runtime_command("audiohoi", sam2_tool, *sam2_args),
            config=preprocess,
            model={"detector": preprocess.get("detector_model", "default"), "segmenter": preprocess.get("segmenter_model", "default")},
            validator=lambda count: (_validate_masks(masks.path, count), _validate_frame_csv(trajectory.path, count)),
        ),
        _task(
            "cotracker", "audiohoi", ("frame_extract", "sam2"), (frames, masks),
            (center, points, mesh, rigid_tracks, rigid_tracks_manifest), _runtime_command("audiohoi", cotracker_tool, *cotracker_args),
            config={
                "object_family": profile.data.get("object_family", "generic_object"),
                "tracker": preprocess.get("tracker", "cotracker3_offline"),
                "sequence_mode": tracker_sequence_mode,
                "query_policy": "sam2_mask_interior_grid_plus_legacy_anchors" if persistent_tracking else "chunk_local_legacy_anchors",
                "grid_size": tracker_grid_size,
            },
            model={"tracker": preprocess.get("tracker", "cotracker3_offline")},
            validator=lambda count: (
                _validate_frame_csv(center.path, count),
                _validate_frame_csv(points.path, count),
                _validate_frame_csv(mesh.path, count, allow_multiple=True),
                _validate_frame_csv(rigid_tracks.path, count, allow_multiple=True) if persistent_tracking else None,
                json.loads(rigid_tracks_manifest.path.read_text()) if persistent_tracking else None,
            ),
        ),
        _task(
            "da3", "da3", ("frame_extract",), (frames, ArtifactSpec("da3_cli", da3_cli)), (depth,),
            _runtime_command("da3", da3_tool, *da3_args),
            config={"process_res": preprocess.get("da3_process_res", 504), "chunk_size": da3_chunk_size, "da3_root": str(da3_root)},
            model={"model": da3_model}, validator=lambda count: _validate_da3(depth.path, count),
        ),
        _task(
            "object_depth_prior", "audiohoi", ("sam2", "cotracker", "da3"),
            (depth, masks, trajectory, center), (depth_prior,),
            _runtime_command("audiohoi", depth_prior_tool, "--sample-dir", str(sample)),
            config={"reducer": "sam2_mask_median_with_missing_mask_interpolation", "smooth_window": 7},
            validator=lambda count: _validate_object_depth_prior(depth_prior.path, count),
        ),
        _task(
            "gvhmr", "gvhmr", ("frame_extract",),
            (video, frames, ArtifactSpec("gvhmr_checkpoint", gvhmr_checkpoint)), (gvhmr,),
            _runtime_command("gvhmr", gvhmr_tool, *gvhmr_args),
            config={"fps": fps or 30.0, "static_camera": bool(preprocess.get("static_camera", False)), "person_index": int(preprocess.get("person_index", 0))},
            model={"checkpoint": "gvhmr_siga24_release.ckpt"},
            human_state_role="read_only_observed", validator=lambda count: _validate_gvhmr(gvhmr.path, count),
        ),
        _task(
            "audio_events", "audiohoi", ("audio_extract", "frame_extract", "cotracker", "gvhmr"),
            (audio, frames, gvhmr, center), (events,),
            _runtime_command("audiohoi", audio_tool, "--sample-dir", str(sample)),
            config={"detector": "combined", "classifier": "rule"}, required=not audio_disabled,
            validator=lambda count: _validate_audio_events(events.path, count),
        ),
    )


def validate_task_graph(tasks: Iterable[PreprocessTask]) -> tuple[PreprocessTask, ...]:
    ordered = tuple(tasks)
    ids = [task.task_id for task in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("case-ingestion task ids must be unique")
    seen: set[str] = set()
    for task in ordered:
        missing = set(task.dependencies) - seen
        if missing:
            raise ValueError(f"task {task.task_id} has unresolved dependencies: {sorted(missing)}")
        seen.add(task.task_id)
    return ordered
