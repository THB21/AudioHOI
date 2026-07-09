"""SMPL-X per-part VERTEX index sets for vertex-accurate penetration/contact.

Stage C and the HOI metrics originally used 21 skeleton points per hand (wrist +
finger joints + tips). The flesh BETWEEN those points can still penetrate an object
even when every skeleton point is clear — the chair backrest case showed 42% of
frames with hand-mesh vertices inside the chair while the 21 skeleton points read 0.

This builds dense vertex index sets per part by proximity to that part's joints on
the SMPL-X template (zero pose), so both the refiner and the evaluator measure/clear
the actual mesh surface. Cached to json keyed by body-model path so the ~1 s template
forward runs once.

Parts and their SMPL-X joint anchors (index into the 55-joint SMPL-X `.joints`):
  left_hand  = L_wrist(20) + left finger joints(25-39) + left fingertips(66-70)
  right_hand = R_wrist(21) + right finger joints(40-54) + right fingertips(71-75)
  left_foot  = L_ankle(7) + L_foot(10) + left toes/heel(60-62)
  right_foot = R_ankle(8) + R_foot(11) + right toes/heel(63-65)
  pelvis     = pelvis(0) + L/R hip(1,2) + spine1(3)          (chair sit / body contact)
  back       = spine1-3(3,6,9) + neck(12)                    (chair backrest lean)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_PART_JOINTS = {
    "left_hand": [20, *range(25, 40), 66, 67, 68, 69, 70],
    "right_hand": [21, *range(40, 55), 71, 72, 73, 74, 75],
    "left_foot": [7, 10, 60, 61, 62],
    "right_foot": [8, 11, 63, 64, 65],
    "pelvis": [0, 1, 2, 3],
    "back": [3, 6, 9, 12],
}
# radius (m) around each part's joints within which template verts are claimed
_PART_RADIUS = {
    "left_hand": 0.035, "right_hand": 0.035,
    "left_foot": 0.06, "right_foot": 0.06,
    "pelvis": 0.14, "back": 0.16,
}


def part_vertex_indices(body_model_root: str | Path, parts: list[str] | None = None,
                        cache_dir: str | Path | None = None) -> dict[str, np.ndarray]:
    """Return {part: vertex_index_array} for the given parts (default all)."""
    parts = parts or list(_PART_JOINTS)
    root = Path(body_model_root)
    cache = Path(cache_dir) if cache_dir else root
    cache_file = cache / "part_vertex_indices.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        if all(p in data for p in parts):
            return {p: np.asarray(data[p], dtype=np.int64) for p in parts}
    else:
        data = {}

    import torch
    import smplx
    model = smplx.create(str(root), model_type="smplx", gender="neutral", ext="npz",
                         use_pca=False, flat_hand_mean=True, num_betas=10, batch_size=1)
    with torch.no_grad():
        out = model(return_verts=True)
    V = out.vertices[0].numpy()          # [10475, 3]
    J = out.joints[0].numpy()            # [55+, 3]
    for p in _PART_JOINTS:
        anchors = J[_PART_JOINTS[p]]
        d = np.linalg.norm(V[:, None, :] - anchors[None, :, :], axis=2).min(axis=1)
        data[p] = np.nonzero(d < _PART_RADIUS[p])[0].tolist()
    try:
        cache_file.write_text(json.dumps({k: v for k, v in data.items()}))
    except OSError:
        pass
    return {p: np.asarray(data[p], dtype=np.int64) for p in parts}
