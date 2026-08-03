#!/usr/bin/env python3
"""Generate isolated SAM2 masks conditioned by persistent CoTracker points.

This is a compatibility adapter for the SAM-PT data flow. It uses the current
official SAM2.1 predictor and official CoTracker3 outputs already produced by
Stage 0; it does not reimplement either model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from run_sam2_object import DEFAULT_SAM2_MODEL, prepare_jpg_frames


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tracks(path: Path) -> dict[int, list[dict[str, str]]]:
    rows_by_frame: dict[int, list[dict[str, str]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows_by_frame.setdefault(int(row["frame"]), []).append(row)
    if not rows_by_frame:
        raise ValueError(f"No persistent tracks in {path}")
    return rows_by_frame


def mask_bbox(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Cannot prompt SAM2 from an empty mask")
    return np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def visible_object_points(
    rows: list[dict[str, str]],
    mask: np.ndarray,
    *,
    visibility_threshold: float,
    maximum_points: int,
) -> tuple[np.ndarray, list[str]]:
    expanded = cv2.dilate((mask > 0).astype(np.uint8), np.ones((15, 15), np.uint8))
    candidates: list[tuple[str, float, float]] = []
    for row in rows:
        if float(row["visible"]) < visibility_threshold:
            continue
        x = float(row["x"])
        y = float(row["y"])
        xi = int(np.clip(round(x), 0, mask.shape[1] - 1))
        yi = int(np.clip(round(y), 0, mask.shape[0] - 1))
        if expanded[yi, xi] > 0:
            candidates.append((row["track_id"], x, y))
    if len(candidates) > maximum_points:
        indices = np.linspace(0, len(candidates) - 1, maximum_points).round().astype(int)
        candidates = [candidates[index] for index in indices]
    points = np.asarray([[x, y] for _track_id, x, y in candidates], dtype=np.float32)
    return points, [track_id for track_id, _x, _y in candidates]


def select_conditioning_frames(
    rows_by_frame: dict[int, list[dict[str, str]]],
    *,
    stride: int,
    visibility_threshold: float,
    minimum_visible_points: int,
) -> list[int]:
    selected = {1}
    previous_visible: int | None = None
    for frame in sorted(rows_by_frame):
        visible = sum(float(row["visible"]) >= visibility_threshold for row in rows_by_frame[frame])
        regular = (frame - 1) % stride == 0
        reliability_transition = previous_visible is not None and (
            (previous_visible >= minimum_visible_points) != (visible >= minimum_visible_points)
        )
        if visible >= minimum_visible_points and (regular or reliability_transition):
            selected.add(frame)
        previous_visible = visible
    return sorted(selected)


def run_refinement(
    *,
    sample_dir: Path,
    tracks_path: Path,
    source_masks_dir: Path,
    output_dir: Path,
    model_id: str,
    prompt_stride: int,
    visibility_threshold: float,
    minimum_visible_points: int,
    maximum_prompt_points: int,
) -> dict[str, object]:
    import torch
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    rows_by_frame = read_tracks(tracks_path)
    segmentation_dir = sample_dir / "results" / "segmentation"
    jpg_paths = prepare_jpg_frames(sample_dir / "frames", segmentation_dir / "sam2_jpg_frames")
    conditioning_frames = select_conditioning_frames(
        rows_by_frame,
        stride=prompt_stride,
        visibility_threshold=visibility_threshold,
        minimum_visible_points=minimum_visible_points,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*_mask.png"):
        stale.unlink()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = SAM2VideoPredictor.from_pretrained(model_id).to(device)
    prompt_records: list[dict[str, object]] = []
    written: list[int] = []
    with torch.inference_mode():
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            state = predictor.init_state(str(segmentation_dir / "sam2_jpg_frames"))
            for frame in conditioning_frames:
                mask_path = source_masks_dir / f"{frame:05d}_mask.png"
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                points, track_ids = visible_object_points(
                    rows_by_frame[frame],
                    mask,
                    visibility_threshold=visibility_threshold,
                    maximum_points=maximum_prompt_points,
                )
                if frame != 1 and len(points) < minimum_visible_points:
                    continue
                labels = np.ones(len(points), dtype=np.int32)
                predictor.add_new_points_or_box(
                    state,
                    frame_idx=frame - 1,
                    obj_id=1,
                    points=points,
                    labels=labels,
                    box=mask_bbox(mask) if frame == 1 else None,
                    clear_old_points=True,
                )
                prompt_records.append(
                    {
                        "frame": frame,
                        "positive_track_ids": track_ids,
                        "source_mask": str(mask_path),
                        "source_mask_sha256": file_sha256(mask_path),
                    }
                )
            for out_frame_idx, _object_ids, out_mask_logits in predictor.propagate_in_video(state):
                mask = (out_mask_logits[0] > 0.0).detach().cpu().numpy().squeeze()
                frame = int(out_frame_idx) + 1
                cv2.imwrite(str(output_dir / f"{frame:05d}_mask.png"), mask.astype(np.uint8) * 255)
                written.append(frame)

    manifest = {
        "schema_version": 1,
        "method": "sam_pt_compatible_persistent_point_prompting",
        "sam_model": model_id,
        "tracks": str(tracks_path),
        "tracks_sha256": file_sha256(tracks_path),
        "source_masks": str(source_masks_dir),
        "output_masks": str(output_dir),
        "frame_count": len(jpg_paths),
        "written_frame_count": len(written),
        "prompt_stride": prompt_stride,
        "visibility_threshold": visibility_threshold,
        "minimum_visible_points": minimum_visible_points,
        "maximum_prompt_points": maximum_prompt_points,
        "conditioning_prompts": prompt_records,
        "overwrites_canonical_masks": False,
    }
    manifest_path = output_dir.parent / "sam_pt_candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {**manifest, "manifest": str(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, default=None)
    parser.add_argument("--source-masks", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sam2-model", default=DEFAULT_SAM2_MODEL)
    parser.add_argument("--prompt-stride", type=int, default=24)
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-visible-points", type=int, default=16)
    parser.add_argument("--maximum-prompt-points", type=int, default=16)
    args = parser.parse_args()

    tracks = args.tracks or args.sample_dir / "results/tracking/rigid_point_tracks.csv"
    source_masks = args.source_masks or args.sample_dir / "results/segmentation/masks"
    output_dir = args.output_dir or args.sample_dir / "results/segmentation/sam_pt_candidate_masks"
    summary = run_refinement(
        sample_dir=args.sample_dir,
        tracks_path=tracks,
        source_masks_dir=source_masks,
        output_dir=output_dir,
        model_id=args.sam2_model,
        prompt_stride=args.prompt_stride,
        visibility_threshold=args.visibility_threshold,
        minimum_visible_points=args.minimum_visible_points,
        maximum_prompt_points=args.maximum_prompt_points,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
