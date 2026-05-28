#!/usr/bin/env python3
"""Scene-level support geometry utilities for shared-camera pipelines.

Current minimal implementation exports a single support line parameter
`floor_v` estimated from contact events in image space.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SupportGeometry:
    support_type: str
    floor_v: float
    source: str
    confidence: float
    num_support_events: int


def read_contact_frames(path: Path) -> list[int]:
    frames: list[int] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["visual_frame"]))
    if not frames:
        raise RuntimeError(f"No contact frames in {path}")
    return frames


def estimate_support_geometry_from_contacts(
    frames: np.ndarray,
    obs_bottoms: np.ndarray,
    contact_frames: list[int],
) -> SupportGeometry:
    frame_to_idx = {int(frame): idx for idx, frame in enumerate(frames.tolist())}
    contact_indices = [frame_to_idx[f] for f in contact_frames if f in frame_to_idx]
    if not contact_indices:
        raise RuntimeError("No overlapping contact frames for support geometry estimation")

    support_values = np.asarray([obs_bottoms[idx] for idx in contact_indices], dtype=np.float64)
    floor_v = float(np.median(support_values))
    if len(support_values) > 1:
        mad = float(np.median(np.abs(support_values - floor_v)))
        confidence = float(1.0 / (1.0 + mad / 8.0))
    else:
        confidence = 0.5

    return SupportGeometry(
        support_type="floor",
        floor_v=floor_v,
        source="ball_contact_events",
        confidence=confidence,
        num_support_events=len(contact_indices),
    )


def write_support_geometry_json(path: Path, support: SupportGeometry) -> None:
    payload = {
        "support_type": support.support_type,
        "floor_v": support.floor_v,
        "source": support.source,
        "confidence": support.confidence,
        "num_support_events": support.num_support_events,
    }
    path.write_text(json.dumps(payload, indent=2))
