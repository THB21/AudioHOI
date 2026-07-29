from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..measurements import Line2DMeasurement
from ..state.geometry_provider import FeaturePointGeometryProvider, PinholeCamera


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_line_reprojection_evidence(
    *,
    frame_paths: Mapping[int, Path],
    object_states: Mapping[int, Sequence[float]],
    geometry_provider: FeaturePointGeometryProvider,
    measurements: Sequence[Line2DMeasurement],
    cameras_by_frame: Mapping[int, PinholeCamera],
    output_dir: Path,
    selected_frames: Sequence[int],
) -> dict[str, object]:
    """Render object-only typed line observations against predicted geometry."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on declared audiohoi runtime
        raise RuntimeError("line render evidence requires OpenCV in the audiohoi runtime") from exc

    frames = tuple(dict.fromkeys(int(frame) for frame in selected_frames))
    if not frames:
        raise ValueError("line render evidence requires selected frames")
    measurements_by_frame: dict[int, list[Line2DMeasurement]] = {}
    for measurement in measurements:
        measurements_by_frame.setdefault(int(measurement.meta.frame), []).append(measurement)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for frame in frames:
        frame_path = frame_paths.get(frame)
        state = object_states.get(frame)
        camera = cameras_by_frame.get(frame)
        if frame_path is None or state is None or camera is None:
            raise ValueError(f"line render evidence missing frame/state/camera for frame {frame}")
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot read render evidence frame: {frame_path}")
        squared_errors: list[float] = []
        rendered = 0
        for measurement in measurements_by_frame.get(frame, []):
            points = geometry_provider.feature_points_world(state, measurement.meta.feature.geometry_feature_id)
            if points.shape != (2, 3):
                raise ValueError("line render geometry must resolve to two endpoints")
            predicted = camera.project(points)
            observed = np.asarray((measurement.start_uv, measurement.end_uv), dtype=float)
            squared_errors.extend((predicted - observed).reshape(-1) ** 2)
            cv2.line(image, tuple(np.rint(observed[0]).astype(int)), tuple(np.rint(observed[1]).astype(int)), (30, 220, 30), 3, cv2.LINE_AA)
            cv2.line(image, tuple(np.rint(predicted[0]).astype(int)), tuple(np.rint(predicted[1]).astype(int)), (220, 40, 220), 2, cv2.LINE_AA)
            rendered += 1
        output_path = output_dir / f"frame_{frame:05d}.png"
        cv2.putText(image, "observed=green predicted=magenta", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2, cv2.LINE_AA)
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"failed to write render evidence: {output_path}")
        records.append(
            {
                "frame": frame,
                "measurement_count": rendered,
                "line_rmse_px": float(np.sqrt(np.mean(squared_errors))) if squared_errors else None,
                "source_frame": str(frame_path),
                "source_frame_sha256": _sha256(frame_path),
                "overlay": output_path.name,
                "overlay_sha256": _sha256(output_path),
            }
        )
    manifest = {
        "schema_version": 1,
        "mode": "generic_line_reprojection_object_evidence",
        "scope": "object_only",
        "observed_color": "green",
        "predicted_color": "magenta",
        "records": records,
        "accepted_outputs_written": False,
    }
    manifest_path = output_dir / "render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return manifest
