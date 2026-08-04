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


def _validate_rigid_pose_hypotheses(
    path: Path, manifest_path: Path, requested_frames: tuple[int, ...]
) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    manifest = json.loads(manifest_path.read_text())
    successful = tuple(sorted(int(frame) for frame in manifest.get("successful_frames", ())))
    if successful != requested_frames or manifest.get("failures"):
        raise ValueError("external rigid pose provider did not complete every requested keyframe")
    covered = {int(row["frame"]) for row in rows}
    if covered != set(requested_frames):
        raise ValueError("external rigid pose hypotheses do not cover the requested keyframes")
    for row in rows:
        required = {
            "frame", "tx_m", "ty_m", "tz_m", "qx", "qy", "qz", "qw",
            "provider_status", "official_render_mask_iou", "visual_geometry_rank",
        }
        if not required.issubset(row):
            raise ValueError(f"external rigid pose row is missing fields: {sorted(required - set(row))}")


def _validate_rigid_feature_measurements(path: Path, manifest_path: Path) -> None:
    rows = _csv_rows(path)
    manifest = json.loads(manifest_path.read_text())
    if not rows or int(manifest.get("measurement_count", -1)) != len(rows):
        raise ValueError("rigid feature binding produced no rows or a mismatched manifest")
    required = {
        "frame", "time", "u", "v", "geometry_feature_id", "semantic_role",
        "track_id", "confidence", "source_anchor_frames",
    }
    if not required.issubset(rows[0]):
        raise ValueError(f"rigid feature rows are missing columns: {sorted(required - set(rows[0]))}")
    expected_roles = {"rigid_body_corner", "rigid_wheel_center", "rigid_rail_endpoint"}
    roles = {row["semantic_role"] for row in rows}
    if not expected_roles.issubset(roles):
        raise ValueError(f"rigid feature coverage is missing roles: {sorted(expected_roles - roles)}")
    for row in rows:
        if not row["geometry_feature_id"] or not row["track_id"]:
            raise ValueError("rigid feature row has an empty typed identity")
        numeric = (float(row["time"]), float(row["u"]), float(row["v"]), float(row["confidence"]))
        if not all(np.isfinite(value) for value in numeric) or not 0.0 <= numeric[-1] <= 1.0:
            raise ValueError("rigid feature row contains an invalid numeric value")
    if manifest.get("baseline_pose_read") is not False or manifest.get("human_state_optimized") is not False:
        raise ValueError("rigid feature binding violated the object-only input boundary")


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
    sam_pt_tool = REPO / "scripts/shared/generic_contact_pipeline/tools/run_sam_pt_mask_refine.py"
    rigid_mesh_export_tool = REPO / "scripts/shared/generic_contact_pipeline/tools/export_rigid_asset_mesh.py"
    megapose_tool = REPO / "scripts/shared/generic_contact_pipeline/tools/run_megapose_rigid_pose.py"
    rigid_feature_binding_tool = REPO / "scripts/shared/generic_contact_pipeline/tools/bind_rigid_feature_tracks.py"
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
    sam_pt_enabled = bool(preprocess.get("sam_pt_mask_refine", False))
    sam_pt_prompt_stride = int(preprocess.get("sam_pt_prompt_stride", 24))
    sam_pt_minimum_visible = int(preprocess.get("sam_pt_minimum_visible_points", 16))
    sam_pt_maximum_prompts = int(preprocess.get("sam_pt_maximum_prompt_points", 16))
    rigid_pose_provider = str(preprocess.get("rigid_pose_provider", "none"))
    megapose_enabled = rigid_pose_provider == "megapose_rgb"
    megapose_keyframes = tuple(sorted({int(frame) for frame in preprocess.get("rigid_pose_keyframes", ())}))
    megapose_data_dir = Path(
        str(preprocess.get("megapose_data_dir", REPO / "third-party/megapose6d/local_data"))
    )
    megapose_display = str(preprocess.get("megapose_display", ":1"))
    asset_descriptor_raw = profile.data.get("geometry_asset_descriptor")
    asset_descriptor = (
        Path(str(asset_descriptor_raw)) if asset_descriptor_raw else REPO / "missing_asset_descriptor.json"
    )
    if not asset_descriptor.is_absolute():
        asset_descriptor = REPO / asset_descriptor
    sam_pt_args = [
        "--sample-dir",
        str(sample),
        "--prompt-stride",
        str(sam_pt_prompt_stride),
        "--minimum-visible-points",
        str(sam_pt_minimum_visible),
        "--maximum-prompt-points",
        str(sam_pt_maximum_prompts),
    ]
    rigid_asset_mesh_path = sample / "articraft/megapose/fixed_rigid_asset_mm.ply"
    rigid_mesh_args = [
        "--asset-descriptor", str(asset_descriptor),
        "--output", str(rigid_asset_mesh_path),
    ]
    megapose_output_path = results / "megapose/rigid_pose_hypotheses.jsonl"
    megapose_overlay_path = results / "megapose/official_overlays"
    megapose_args = [
        "--case-config", str(config_path),
        "--sample-dir", str(sample),
        "--asset-mesh", str(rigid_asset_mesh_path),
        "--mask-dir", str(results / "segmentation/masks"),
        "--track-artifact", str(results / "tracking/rigid_point_tracks.csv"),
        "--frames", ",".join(str(frame) for frame in megapose_keyframes),
        "--output", str(megapose_output_path),
        "--overlay-dir", str(megapose_overlay_path),
        "--megapose-data-dir", str(megapose_data_dir),
        "--display", megapose_display,
        "--renderer-workers", "0",
        "--batch-size", "64",
    ]
    supplemental = profile.data.get("supplemental_measurements", ())
    rigid_feature_binding_enabled = megapose_enabled and any(
        isinstance(spec, dict) and spec.get("adapter") == "rigid_feature_points_v1"
        for spec in supplemental
    )
    rigid_feature_measurements_path = profile.result_dir / "rigid_feature_measurements.csv"
    rigid_feature_manifest_path = profile.result_dir / "rigid_feature_measurements_manifest.json"
    rigid_feature_binding_args = [
        "--case", profile.case_name,
        "--result-name", profile.result_name,
        "--track-artifact", str(results / "tracking/rigid_point_tracks.csv"),
        "--pose-hypotheses", str(megapose_output_path),
        "--output-csv", str(rigid_feature_measurements_path),
        "--output-manifest", str(rigid_feature_manifest_path),
        "--maximum-anchor-error-px", str(preprocess.get("rigid_feature_maximum_anchor_error_px", 18.0)),
        "--minimum-track-visibility", str(preprocess.get("rigid_feature_minimum_track_visibility", 0.5)),
        "--minimum-pose-mask-iou", str(preprocess.get("rigid_feature_minimum_pose_mask_iou", 0.60)),
    ]
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
    sam_pt_masks = ArtifactSpec(
        "sam_pt_candidate_masks",
        results / "segmentation/sam_pt_candidate_masks",
        "directory",
        required=sam_pt_enabled,
    )
    sam_pt_manifest = ArtifactSpec(
        "sam_pt_candidate_manifest",
        results / "segmentation/sam_pt_candidate_manifest.json",
        required=sam_pt_enabled,
    )
    rigid_asset_mesh = ArtifactSpec(
        "rigid_provider_mesh", rigid_asset_mesh_path, required=megapose_enabled
    )
    rigid_asset_mesh_manifest = ArtifactSpec(
        "rigid_provider_mesh_manifest",
        rigid_asset_mesh_path.with_suffix(rigid_asset_mesh_path.suffix + ".json"),
        required=megapose_enabled,
    )
    megapose_hypotheses = ArtifactSpec(
        "external_rigid_pose_hypotheses", megapose_output_path, required=megapose_enabled
    )
    megapose_manifest = ArtifactSpec(
        "external_rigid_pose_manifest",
        megapose_output_path.with_suffix(megapose_output_path.suffix + ".manifest.json"),
        required=megapose_enabled,
    )
    megapose_overlays = ArtifactSpec(
        "external_rigid_pose_overlays",
        megapose_overlay_path,
        "directory",
        required=megapose_enabled,
    )
    rigid_feature_measurements = ArtifactSpec(
        "rigid_feature_measurements",
        rigid_feature_measurements_path,
        required=rigid_feature_binding_enabled,
    )
    rigid_feature_manifest = ArtifactSpec(
        "rigid_feature_measurements_manifest",
        rigid_feature_manifest_path,
        required=rigid_feature_binding_enabled,
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
            "sam_pt_mask_candidate",
            "audiohoi",
            ("frame_extract", "sam2", "cotracker"),
            (frames, masks, rigid_tracks),
            (sam_pt_masks, sam_pt_manifest),
            _runtime_command("audiohoi", sam_pt_tool, *sam_pt_args),
            config={
                "method": "sam_pt_compatible_persistent_point_prompting",
                "prompt_stride": sam_pt_prompt_stride,
                "minimum_visible_points": sam_pt_minimum_visible,
                "maximum_prompt_points": sam_pt_maximum_prompts,
                "canonical_masks_overwritten": False,
            },
            model={"segmenter": preprocess.get("segmenter_model", "facebook/sam2.1-hiera-tiny")},
            required=sam_pt_enabled,
            validator=lambda count: (
                _validate_masks(sam_pt_masks.path, count),
                json.loads(sam_pt_manifest.path.read_text()),
            ),
        ),
        _task(
            "rigid_asset_mesh_export",
            "megapose",
            ("frame_extract",),
            (ArtifactSpec("rigid_asset_descriptor", asset_descriptor),),
            (rigid_asset_mesh, rigid_asset_mesh_manifest),
            _runtime_command("megapose", rigid_mesh_export_tool, *rigid_mesh_args),
            config={"geometry_contract": "fixed_rigid_mesh_mm", "minimum_vertices": 4096},
            required=megapose_enabled,
            validator=lambda _count: json.loads(rigid_asset_mesh_manifest.path.read_text()),
        ),
        _task(
            "external_rigid_pose_provider",
            "megapose",
            ("frame_extract", "sam2", "cotracker", "rigid_asset_mesh_export"),
            (frames, masks, rigid_tracks, rigid_asset_mesh, rigid_asset_mesh_manifest),
            (megapose_hypotheses, megapose_manifest, megapose_overlays),
            _runtime_command("megapose", megapose_tool, *megapose_args),
            config={
                "provider": rigid_pose_provider,
                "keyframes": list(megapose_keyframes),
                "selection": "official_render_mask_iou_with_persistent_track_visibility",
                "accepted_pose_publication": False,
            },
            model={"model": "megapose-1.0-RGB-multi-hypothesis"},
            required=megapose_enabled,
            validator=lambda _count: _validate_rigid_pose_hypotheses(
                megapose_hypotheses.path, megapose_manifest.path, megapose_keyframes
            ),
        ),
        _task(
            "bind_rigid_feature_tracks",
            "audiohoi",
            ("cotracker", "external_rigid_pose_provider"),
            (
                rigid_tracks,
                rigid_tracks_manifest,
                megapose_hypotheses,
                megapose_manifest,
                ArtifactSpec("rigid_asset_descriptor", asset_descriptor),
            ),
            (rigid_feature_measurements, rigid_feature_manifest),
            _runtime_command("audiohoi", rigid_feature_binding_tool, *rigid_feature_binding_args),
            config={
                "association": "global_unique_best_reliable_pose_anchor",
                "maximum_anchor_error_px": preprocess.get("rigid_feature_maximum_anchor_error_px", 18.0),
                "minimum_track_visibility": preprocess.get("rigid_feature_minimum_track_visibility", 0.5),
                "minimum_pose_mask_iou": preprocess.get("rigid_feature_minimum_pose_mask_iou", 0.60),
                "baseline_pose_read": False,
                "human_state_optimized": False,
            },
            required=rigid_feature_binding_enabled,
            validator=lambda _count: _validate_rigid_feature_measurements(
                rigid_feature_measurements.path, rigid_feature_manifest.path
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
