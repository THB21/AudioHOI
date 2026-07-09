"""Stage 4 — ground each audio event in the surrounding video frames.

For every onset we read the visual context at that time so the audio label can be
confirmed and, crucially, *attributed to an entity*:
  - object 2D position / speed / acceleration + vertical velocity reversal (bounce/impact)
  - nearest body part (COCO wrists/ankles from cached ViTPose) and its pixel distance
  - frame-difference motion magnitude (a generic "something moved here" spike)

All inputs are optional; missing ones degrade gracefully (generated clips may lack object
tracking). Uses GVHMR's cached ``vitpose.pt`` (COCO-17) — no SMPL-X forward needed.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# COCO-17 keypoint indices for the plausible contactors
_PARTS = {"left_hand": 9, "right_hand": 10, "left_foot": 15, "right_foot": 16, "nose": 0}


@dataclass
class VisualContext:
    obj_u: float
    obj_v: float
    obj_speed: float          # px/frame
    obj_accel: float          # px/frame^2
    vel_reversal: bool        # vertical object velocity flips sign at the event
    nearest_part: str
    part_dist_px: float       # object↔nearest body part (NaN if no object)
    part_speed: float         # speed of that part (px/frame)
    flow_mag: float           # frame-difference motion magnitude
    proximity_min: bool       # object↔part distance is a local minimum
    flow_spike: bool          # motion magnitude is a local maximum
    visual_cue: str           # vel_reversal | proximity_min | flow_spike | none

    def as_dict(self) -> dict:
        d = asdict(self)
        d["vel_reversal"] = int(d["vel_reversal"])
        d["proximity_min"] = int(d["proximity_min"])
        d["flow_spike"] = int(d["flow_spike"])
        return d


def _load_object_uv(sample_dir: Path, n_frames: int) -> np.ndarray | None:
    csv_path = sample_dir / "results" / "tracking" / "ball_trajectory.csv"
    if not csv_path.exists():
        return None
    fr, u, v = [], [], []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row.get("ball_center_x") in (None, ""):
                continue
            fr.append(int(row["frame"]))
            u.append(float(row["ball_center_x"]))
            v.append(float(row["ball_center_y"]))
    if len(fr) < 2:
        return None
    grid = np.arange(1, n_frames + 1)
    uu = np.interp(grid, fr, u)
    vv = np.interp(grid, fr, v)
    return np.stack([uu, vv], axis=1)  # [N,2], frame index = row+1


def _load_vitpose(sample_dir: Path) -> np.ndarray | None:
    p = sample_dir / "results" / "gvhmr" / "preprocess" / "vitpose.pt"
    if not p.exists():
        return None
    import torch

    v = torch.load(str(p), map_location="cpu")
    return np.asarray(v, dtype=np.float64)  # [N,17,3] (x,y,conf)


def _frame_motion(sample_dir: Path, frames_needed: set[int]) -> dict[int, float]:
    """Mean abs frame-difference for the requested frames (cheap motion proxy)."""
    import cv2

    fdir = sample_dir / "frames"
    files = sorted(fdir.glob("*.png")) or sorted(fdir.glob("*.jpg"))
    if not files:
        return {}
    out: dict[int, float] = {}
    for fr in sorted(frames_needed):
        i = fr - 1
        if i < 1 or i >= len(files):
            continue
        a = cv2.imread(str(files[i - 1]), cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(str(files[i]), cv2.IMREAD_GRAYSCALE)
        if a is None or b is None or a.shape != b.shape:
            continue
        scale = 240.0 / a.shape[0]
        if scale < 1.0:
            a = cv2.resize(a, None, fx=scale, fy=scale)
            b = cv2.resize(b, None, fx=scale, fy=scale)
        out[fr] = float(np.mean(np.abs(b.astype(np.float32) - a.astype(np.float32))))
    return out


def build_visual_contexts(sample_dir: Path, onset_frames: list[int], n_frames: int,
                          fps: float = 24.0) -> list[VisualContext]:
    sample_dir = Path(sample_dir)
    obj = _load_object_uv(sample_dir, n_frames)
    kpts = _load_vitpose(sample_dir)

    # object kinematics over the full clip
    if obj is not None:
        vel = np.gradient(obj, axis=0)               # px/frame
        speed = np.linalg.norm(vel, axis=1)
        accel = np.gradient(speed)
    # part speeds
    if kpts is not None:
        part_xy = {name: kpts[:, idx, :2] for name, idx in _PARTS.items()}
        part_conf = {name: kpts[:, idx, 2] for name, idx in _PARTS.items()}
        part_spd = {name: np.concatenate([[0.0], np.linalg.norm(np.diff(xy, axis=0), axis=1)])
                    for name, xy in part_xy.items()}

    motion = _frame_motion(sample_dir, {f for fr in onset_frames for f in (fr - 1, fr, fr + 1)})
    motion_vals = np.array([motion.get(f, 0.0) for f in range(1, n_frames + 1)])
    flow_thr = float(np.percentile(motion_vals[motion_vals > 0], 75)) if np.any(motion_vals > 0) else 0.0

    out: list[VisualContext] = []
    for fr in onset_frames:
        i = int(np.clip(fr - 1, 0, n_frames - 1))

        obj_u = obj_v = obj_speed = obj_accel = 0.0
        vel_reversal = False
        if obj is not None:
            obj_u, obj_v = float(obj[i, 0]), float(obj[i, 1])
            obj_speed, obj_accel = float(speed[i]), float(accel[i])
            lo, hi = max(0, i - 2), min(n_frames - 1, i + 2)
            vy = vel[lo:hi + 1, 1]
            vel_reversal = bool(np.any(vy[:-1] * vy[1:] < 0)) if vy.size >= 2 else False

        nearest_part, part_dist, part_speed = "none", float("nan"), 0.0
        proximity_min = False
        if kpts is not None:
            cand = [(n, part_xy[n][i], part_conf[n][i]) for n in _PARTS if n != "nose"]
            cand = [(n, xy) for n, xy, cf in cand if cf > 0.3]
            if obj is not None and cand:
                dists = {n: float(np.hypot(*(xy - obj[i]))) for n, xy in cand}
                nearest_part = min(dists, key=dists.get)
                part_dist = dists[nearest_part]
                part_speed = float(part_spd[nearest_part][i])
                # proximity local minimum over +-3 frames
                idxr = range(max(0, i - 3), min(n_frames, i + 4))
                seq = [float(np.hypot(*(part_xy[nearest_part][j] - obj[j]))) for j in idxr]
                proximity_min = bool(part_dist <= min(seq) + 1e-6)
            elif cand:
                # no object: the fastest-moving plausible contactor at this frame
                nearest_part = max(cand, key=lambda c: part_spd[c[0]][i])[0]
                part_speed = float(part_spd[nearest_part][i])

        flow_mag = float(motion.get(fr, 0.0))
        nb = motion_vals[max(0, i - 3):min(n_frames, i + 4)]
        flow_spike = bool(flow_mag >= flow_thr and (nb.size == 0 or flow_mag >= nb.max() - 1e-6))

        if vel_reversal:
            cue = "vel_reversal"
        elif proximity_min and not np.isnan(part_dist) and part_dist < 120.0:
            cue = "proximity_min"
        elif flow_spike:
            cue = "flow_spike"
        else:
            cue = "none"

        out.append(VisualContext(
            obj_u=obj_u, obj_v=obj_v, obj_speed=obj_speed, obj_accel=obj_accel,
            vel_reversal=vel_reversal, nearest_part=nearest_part, part_dist_px=part_dist,
            part_speed=part_speed, flow_mag=flow_mag, proximity_min=proximity_min,
            flow_spike=flow_spike, visual_cue=cue,
        ))
    return out
