# Radius-Free Generic Object Pipeline from Mesh Tracks

## 0. Goal

The current mainline is effective for basketball because the object is modeled as a sphere. The next generic version should remove the physical radius prior from the generic branch while preserving the two-stage structure:

```text
Stage 0: video preprocessing
    frames -> SAM2 mask -> CoTracker object mesh tracks -> generic object proxy observations

Stage 1: sharedcam baseline
    object proxy observations -> radius-free object reference trajectory

Stage 2: anchor interpolation
    human contact depth anchors object contact-proxy depth -> refined object reference trajectory
```

The key replacement is:

```text
basketball-specific:
    ball center + radius + projected circle + ball bottom

generic:
    object mesh points + reference proxy + contact proxy + support proxy + depth prior
```

This does not require object-specific parts such as `mug handle`, `chair leg`, `blanket corner`. Those parts are represented implicitly as tracked mesh/proxy points selected by geometry and interaction context.

---

## 1. Current Radius Dependencies to Remove from the Generic Path

### 1.1 Current sharedcam path

Current file:

```text
scripts/shared/sharedcam/run_basketball_pose6d_sharedcam.py
```

Current ball-specific components:

```python
BALL_RADIUS_M = 0.12

class SphereShape:
    def __init__(self, radius_m: float) -> None:
        self.radius_m = float(radius_m)

    def project(self, translation, camera):
        r_px = camera.fx * self.radius_m / z
        bottom_v = v + r_px
        diameter_px = 2.0 * r_px
        area_px = pi * r_px * r_px
```

Current initialization:

```python
z = camera.fx * shape.radius_m / max(row["r"], 1e-6)
x = (row["u"] - camera.cx) * z / camera.fx
r_px = camera.fx * shape.radius_m / max(z, 1e-6)
v_center_from_floor = camera.floor_v - r_px
y = (v_center_from_floor - camera.cy) * z / camera.fy
```

Current residuals depend on sphere geometry:

```python
circle contour residual
projected diameter vs mask width / height / size
projected area vs mask area
bottom_v = v + r_px vs floor_v
```

Current support line estimation also uses:

```python
obs_bottoms = v + r
```

These are all radius-derived constraints.

### 1.2 Current anchorinterp path

Current file:

```text
scripts/shared/human_ball/contact/run_human_ball_contact_phase_calibration_anchorinterp_generic.py
```

Current anchor logic:

```python
deltas = part_z[human_event_mask] - z_init[human_event_mask]
global_z_shift = median(deltas)
z_ref = z_init + global_z_shift

anchor_values = part_z
```

This means:

```text
z_object_ref(t_contact) = z_human_part(t_contact)
```

For generic objects, this should become:

```text
z_object_contact_proxy(t_contact) = z_human_part(t_contact)
```

Since the solver still optimizes object reference depth, the anchor value must be converted:

```text
z_object_contact_proxy = z_object_ref + delta_contact

z_object_ref_anchor = z_human_part - delta_contact
```

---

## 2. New Generic Representation

### 2.1 Object mesh tracks

Replace five basketball points:

```text
center, left, right, top, bottom
```

with a category-agnostic point set sampled from the SAM2 object mask:

```text
boundary points
interior points
extremal points
```

Output file:

```text
results/tracking/object_mesh_tracks.csv
```

Long-format schema:

```csv
frame,time,point_id,point_name,point_type,x,y,visible,source_chunk
1,0.000000,0,boundary_000,boundary,421.2,301.5,0.98,0
1,0.000000,1,boundary_001,boundary,424.0,299.8,0.97,0
1,0.000000,64,interior_000,interior,448.3,333.0,0.92,0
```

This format avoids hard-coded object parts.

### 2.2 Object proxy observations

Create a generic observation table from mask, mesh tracks, human contact proxy, and depth prior.

Output file:

```text
results/object_observations/object_proxy_observations.csv
```

Minimum schema:

```csv
frame,time,
ref_u,ref_v,
support_u,support_v,support_dv,
contact_u,contact_v,
object_ref_depth_m,contact_proxy_depth_m,contact_depth_offset_m,
ref_conf,support_conf,contact_conf,depth_conf,observation_conf
```

Definitions:

```text
ref_u, ref_v:
    object reference proxy, usually robust center of visible mesh/mask.

support_u, support_v:
    support-facing proxy, usually lowest visible object mesh points or mask bottom.

support_dv:
    support_v - ref_v.
    This replaces pixel radius in support projection.

contact_u, contact_v:
    contact proxy selected from object mesh points near active human contact part.

object_ref_depth_m:
    depth prior at reference proxy from DA3 / depth map / aligned monocular depth.

contact_proxy_depth_m:
    depth prior at contact proxy.

contact_depth_offset_m:
    contact_proxy_depth_m - object_ref_depth_m.
    This replaces the implicit assumption that object center depth equals hand depth.
```

---

## 3. Stage 0A: Generic Object Mesh Tracking

### 3.1 New file

Create:

```text
scripts/shared/tracking/run_cotracker_object_mesh.py
```

Keep the old basketball file unchanged:

```text
scripts/shared/tracking/run_cotracker_basketball.py
```

### 3.2 Generic mask point sampling

Key function:

```python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch


def sample_contour_uniform(contour: np.ndarray, n: int) -> np.ndarray:
    """Uniformly sample approximately n points along a contour."""
    contour = contour.reshape(-1, 2).astype(np.float32)
    if len(contour) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if len(contour) <= n:
        return contour
    idx = np.linspace(0, len(contour) - 1, n, dtype=np.int32)
    return contour[idx]


def sample_interior_grid(binary: np.ndarray, n: int) -> np.ndarray:
    """Sample n interior mask pixels with deterministic spacing."""
    ys, xs = np.where(binary)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    if len(coords) <= n:
        return coords

    # Deterministic uniform subsampling. Replace with farthest-point sampling later if needed.
    idx = np.linspace(0, len(coords) - 1, n, dtype=np.int32)
    return coords[idx]


def initial_points_from_mask_generic(
    mask_path: Path,
    n_boundary: int = 64,
    n_interior: int = 64,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Return generic object query points from a SAM2 mask.

    No object category is used. The output points are only geometric proxies.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask {mask_path}")

    binary = mask > 0
    ys, xs = np.where(binary)
    if len(xs) == 0:
        raise RuntimeError(f"Empty mask {mask_path}")

    points: list[list[float]] = []
    names: list[str] = []
    types: list[str] = []

    contours, _ = cv2.findContours(
        binary.astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if contours:
        contour = max(contours, key=cv2.contourArea)
        boundary = sample_contour_uniform(contour, n_boundary)
        for k, (x, y) in enumerate(boundary):
            points.append([float(x), float(y)])
            names.append(f"boundary_{k:03d}")
            types.append("boundary")

    interior = sample_interior_grid(binary, n_interior)
    for k, (x, y) in enumerate(interior):
        points.append([float(x), float(y)])
        names.append(f"interior_{k:03d}")
        types.append("interior")

    center_x = float(np.mean(xs))
    center_y = float(np.mean(ys))
    extremal = {
        "center": [center_x, center_y],
        "left": [float(xs.min()), center_y],
        "right": [float(xs.max()), center_y],
        "top": [center_x, float(ys.min())],
        "bottom": [center_x, float(ys.max())],
    }
    for name, point in extremal.items():
        points.append(point)
        names.append(name)
        types.append("extreme")

    return np.asarray(points, dtype=np.float32), names, types
```

### 3.3 CoTracker output should be long-format

Instead of writing one wide row with `center_x`, `left_x`, etc., write one row per tracked point:

```python
def append_track_rows(
    rows: list[dict[str, object]],
    tracks_chunk: np.ndarray,
    visibility_chunk: np.ndarray,
    point_names: list[str],
    point_types: list[str],
    start_frame: int,
    fps: float,
    source_chunk: int,
) -> None:
    """Append CoTracker results in long format.

    tracks_chunk: (T, P, 2)
    visibility_chunk: (T, P)
    start_frame: zero-based global frame index of the chunk start
    """
    for local_t in range(tracks_chunk.shape[0]):
        frame_0 = start_frame + local_t
        frame_1 = frame_0 + 1
        time = frame_0 / fps
        for point_id, name in enumerate(point_names):
            x, y = tracks_chunk[local_t, point_id]
            rows.append({
                "frame": frame_1,
                "time": f"{time:.6f}",
                "point_id": point_id,
                "point_name": name,
                "point_type": point_types[point_id],
                "x": f"{float(x):.3f}",
                "y": f"{float(y):.3f}",
                "visible": f"{float(visibility_chunk[local_t, point_id]):.6f}",
                "source_chunk": source_chunk,
            })
```

### 3.4 Main tracking loop

The logic remains close to the current CoTracker chunk loop, but the query points are now generic mesh samples:

```python
for chunk_idx, start in enumerate(range(0, frames.shape[0], args.chunk_len)):
    end = min(start + args.chunk_len, frames.shape[0])

    mask_path = masks_dir / f"{start + 1:05d}_mask.png"
    points, point_names, point_types = initial_points_from_mask_generic(
        mask_path,
        n_boundary=args.n_boundary,
        n_interior=args.n_interior,
    )

    scaled_points = points.copy()
    scaled_points[:, 0] *= sx
    scaled_points[:, 1] *= sy

    video = (
        torch.from_numpy(frames[start:end])
        .permute(0, 3, 1, 2)[None]
        .float()
        .to(args.device)
    )

    queries = torch.zeros((1, len(scaled_points), 3), dtype=torch.float32, device=args.device)
    queries[0, :, 1:] = torch.from_numpy(scaled_points).to(args.device)

    pred_tracks, pred_visibility = cotracker(video, queries=queries)
    tracks_chunk = pred_tracks[0].detach().cpu().numpy()
    visibility_chunk = pred_visibility[0].detach().cpu().numpy()

    tracks_chunk[:, :, 0] /= sx
    tracks_chunk[:, :, 1] /= sy

    append_track_rows(
        rows=mesh_rows,
        tracks_chunk=tracks_chunk,
        visibility_chunk=visibility_chunk,
        point_names=point_names,
        point_types=point_types,
        start_frame=start,
        fps=args.fps,
        source_chunk=chunk_idx,
    )
```

---

## 4. Stage 0B: Build Generic Object Proxy Observations

### 4.1 New file

Create:

```text
scripts/shared/object_observation/build_object_proxy_observations.py
```

This should be the generic path. Keep `build_object_observations.py` compatible with old outputs, but the generic path should not read `mug_body_features()` or `enclosing_radius_px` as a depth cue.

### 4.2 Reading mesh tracks

```python
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def read_mesh_tracks(path: Path) -> dict[int, list[dict[str, object]]]:
    by_frame: dict[int, list[dict[str, object]]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            by_frame.setdefault(frame, []).append({
                "point_id": int(row["point_id"]),
                "point_name": row["point_name"],
                "point_type": row["point_type"],
                "x": float(row["x"]),
                "y": float(row["y"]),
                "visible": float(row["visible"]),
            })
    return by_frame


def visible_points(
    rows: list[dict[str, object]],
    min_visible: float = 0.5,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    kept = [r for r in rows if float(r["visible"]) >= min_visible]
    if not kept:
        return np.zeros((0, 2), dtype=np.float64), []
    pts = np.asarray([[float(r["x"]), float(r["y"])] for r in kept], dtype=np.float64)
    return pts, kept
```

### 4.3 Reference proxy

The reference proxy should be stable. Use robust mesh center if enough visible points exist; otherwise use mask centroid.

```python
def select_ref_proxy(
    mesh_rows: list[dict[str, object]],
    mask_center: tuple[float, float] | None,
) -> tuple[float, float, float, str]:
    pts, kept = visible_points(mesh_rows)
    if len(pts) >= 8:
        # Median is more robust than mean under partial occlusion / drift.
        u = float(np.median(pts[:, 0]))
        v = float(np.median(pts[:, 1]))
        conf = min(1.0, len(pts) / 64.0)
        return u, v, conf, "mesh_visible_median"

    if mask_center is not None:
        u, v = mask_center
        return float(u), float(v), 0.5, "mask_center_fallback"

    return math.nan, math.nan, 0.0, "missing"
```

### 4.4 Support proxy

Support proxy is the visible object point facing the support surface. In image coordinates for floor/table contact, the simplest first version is the lowest visible object region.

```python
def select_support_proxy(
    mesh_rows: list[dict[str, object]],
    mask_bbox: tuple[float, float, float, float] | None,
    top_fraction: float = 0.10,
) -> tuple[float, float, float, str]:
    pts, kept = visible_points(mesh_rows)
    if len(pts) >= 8:
        # Image y increases downward. Select bottom top_fraction of visible points.
        q = float(np.quantile(pts[:, 1], 1.0 - top_fraction))
        bottom_pts = pts[pts[:, 1] >= q]
        if len(bottom_pts) > 0:
            u = float(np.median(bottom_pts[:, 0]))
            v = float(np.median(bottom_pts[:, 1]))
            conf = min(1.0, len(bottom_pts) / 8.0)
            return u, v, conf, "lowest_visible_mesh_points"

    if mask_bbox is not None:
        x1, y1, x2, y2 = mask_bbox
        return float(0.5 * (x1 + x2)), float(y2), 0.4, "mask_bbox_bottom_fallback"

    return math.nan, math.nan, 0.0, "missing"
```

### 4.5 Contact proxy

Contact proxy is selected from visible object mesh points near the active human contact proxy. It does not require semantic object parts.

Inputs can be one of:

```text
human hand mask
projected SMPL-X hand/wrist/finger joints
active body part 2D proxy from contact labels
```

First implementation using a single active human 2D point:

```python
def select_contact_proxy_from_human_point(
    mesh_rows: list[dict[str, object]],
    human_uv: tuple[float, float] | None,
    distance_sigma_px: float = 40.0,
    top_k: int = 8,
) -> tuple[float, float, float, str]:
    if human_uv is None:
        return math.nan, math.nan, 0.0, "missing_human_proxy"

    pts, kept = visible_points(mesh_rows)
    if len(pts) == 0:
        return math.nan, math.nan, 0.0, "missing_object_mesh"

    h = np.asarray(human_uv, dtype=np.float64)
    d = np.linalg.norm(pts - h[None, :], axis=1)

    # Prefer boundary points if available, because contact often occurs on visible surface/boundary.
    type_bonus = np.asarray([
        1.25 if str(r["point_type"]) == "boundary" else 1.0
        for r in kept
    ], dtype=np.float64)

    scores = np.exp(-d / max(distance_sigma_px, 1e-6)) * type_bonus
    order = np.argsort(-scores)
    selected = order[: min(top_k, len(order))]

    selected_pts = pts[selected]
    selected_scores = scores[selected]
    weight_sum = float(np.sum(selected_scores))
    if weight_sum <= 1e-8:
        # Fallback to nearest point.
        j = int(np.argmin(d))
        return float(pts[j, 0]), float(pts[j, 1]), 0.2, "nearest_mesh_point_fallback"

    uv = np.sum(selected_pts * selected_scores[:, None], axis=0) / weight_sum
    min_dist = float(np.min(d))
    conf = float(np.clip(np.exp(-min_dist / distance_sigma_px), 0.0, 1.0))
    return float(uv[0]), float(uv[1]), conf, "human_nearest_mesh_topk"
```

If a hand mask is available, use a distance transform instead:

```python
def select_contact_proxy_from_human_mask(
    mesh_rows: list[dict[str, object]],
    hand_mask: np.ndarray | None,
    top_k: int = 8,
) -> tuple[float, float, float, str]:
    if hand_mask is None:
        return math.nan, math.nan, 0.0, "missing_hand_mask"

    pts, kept = visible_points(mesh_rows)
    if len(pts) == 0:
        return math.nan, math.nan, 0.0, "missing_object_mesh"

    binary_hand = hand_mask > 0
    inv = (~binary_hand).astype(np.uint8)
    dist_map = cv2.distanceTransform(inv, cv2.DIST_L2, 3)

    h, w = dist_map.shape[:2]
    xi = np.clip(np.round(pts[:, 0]).astype(np.int32), 0, w - 1)
    yi = np.clip(np.round(pts[:, 1]).astype(np.int32), 0, h - 1)
    d = dist_map[yi, xi]

    order = np.argsort(d)
    selected = order[: min(top_k, len(order))]
    selected_pts = pts[selected]

    # Small distance gets larger weight.
    weights = 1.0 / (1.0 + d[selected])
    uv = np.sum(selected_pts * weights[:, None], axis=0) / max(float(np.sum(weights)), 1e-8)
    conf = float(np.clip(1.0 / (1.0 + float(np.min(d)) / 20.0), 0.0, 1.0))
    return float(uv[0]), float(uv[1]), conf, "hand_mask_distance_transform"
```

### 4.6 Depth sampling

Depth source can be DA3 or another monocular depth model. The generic path should treat it as a depth prior, not as an object size prior.

```python
def sample_depth_at_uv(
    depth: np.ndarray,
    u: float,
    v: float,
    window: int = 5,
) -> tuple[float, float]:
    """Robustly sample depth at image coordinate.

    Returns:
        depth_value, confidence
    """
    if not np.isfinite(u) or not np.isfinite(v):
        return math.nan, 0.0

    h, w = depth.shape[:2]
    x = int(round(u))
    y = int(round(v))
    if x < 0 or x >= w or y < 0 or y >= h:
        return math.nan, 0.0

    r = max(0, int(window))
    x1 = max(0, x - r)
    x2 = min(w, x + r + 1)
    y1 = max(0, y - r)
    y2 = min(h, y + r + 1)

    patch = np.asarray(depth[y1:y2, x1:x2], dtype=np.float64)
    vals = patch[np.isfinite(patch) & (patch > 0)]
    if vals.size == 0:
        return math.nan, 0.0

    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    conf = float(1.0 / (1.0 + mad / max(abs(med), 1e-6)))
    return med, conf
```

Build offsets:

```python
ref_depth, ref_depth_conf = sample_depth_at_uv(depth_map, ref_u, ref_v)
contact_depth, contact_depth_conf = sample_depth_at_uv(depth_map, contact_u, contact_v)

if np.isfinite(ref_depth) and np.isfinite(contact_depth):
    contact_depth_offset = contact_depth - ref_depth
    depth_conf = min(ref_depth_conf, contact_depth_conf)
else:
    contact_depth_offset = 0.0
    depth_conf = 0.0
```

### 4.7 Output row

```python
support_dv = support_v - ref_v if np.isfinite(support_v) and np.isfinite(ref_v) else math.nan

row = {
    "frame": frame,
    "time": f"{time:.6f}",
    "ref_u": f"{ref_u:.3f}",
    "ref_v": f"{ref_v:.3f}",
    "ref_source": ref_source,
    "support_u": f"{support_u:.3f}",
    "support_v": f"{support_v:.3f}",
    "support_dv": f"{support_dv:.3f}",
    "support_source": support_source,
    "contact_u": f"{contact_u:.3f}" if np.isfinite(contact_u) else "",
    "contact_v": f"{contact_v:.3f}" if np.isfinite(contact_v) else "",
    "contact_source": contact_source,
    "object_ref_depth_m": f"{ref_depth:.6f}" if np.isfinite(ref_depth) else "",
    "contact_proxy_depth_m": f"{contact_depth:.6f}" if np.isfinite(contact_depth) else "",
    "contact_depth_offset_m": f"{contact_depth_offset:.6f}",
    "ref_conf": f"{ref_conf:.6f}",
    "support_conf": f"{support_conf:.6f}",
    "contact_conf": f"{contact_conf:.6f}",
    "depth_conf": f"{depth_conf:.6f}",
    "observation_conf": f"{min(ref_conf, max(depth_conf, 0.1)):.6f}",
}
```

---

## 5. Stage 1: Radius-Free Sharedcam Baseline

### 5.1 New file

Create:

```text
scripts/shared/sharedcam/run_object_proxy_pose6d_sharedcam.py
```

Keep old sphere baseline unchanged:

```text
scripts/shared/sharedcam/run_basketball_pose6d_sharedcam.py
```

### 5.2 Replace SphereShape with ObjectProxyShape

```python
from dataclasses import dataclass
import numpy as np


@dataclass
class CameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    floor_v: float


class ObjectProxyShape:
    """Radius-free projection model for a single object reference proxy."""

    def project_ref(self, translation: np.ndarray, camera: CameraModel) -> dict[str, float]:
        x, y, z = translation.tolist()
        z = max(float(z), 1e-6)
        u = camera.cx + camera.fx * x / z
        v = camera.cy + camera.fy * y / z
        return {"u": float(u), "v": float(v)}
```

Do not include:

```text
radius_m
r_px
diameter_px
area_px
bottom_v = v + r_px
```

### 5.3 Read object proxy observations

```python
import csv
import math
from pathlib import Path


def parse_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    return float(value)


def read_object_proxy_observations(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref_u = parse_float(row, "ref_u")
            ref_v = parse_float(row, "ref_v")
            depth = parse_float(row, "object_ref_depth_m")
            if not np.isfinite(ref_u) or not np.isfinite(ref_v) or not np.isfinite(depth):
                continue

            support_v = parse_float(row, "support_v", default=ref_v)
            support_dv = parse_float(row, "support_dv", default=support_v - ref_v)

            rows.append({
                "frame": int(row["frame"]),
                "time": float(row["time"]),
                "u": ref_u,
                "v": ref_v,
                "support_v": support_v,
                "support_dv": support_dv,
                "object_ref_depth_m": depth,
                "depth_conf": parse_float(row, "depth_conf", 0.0),
                "observation_conf": parse_float(row, "observation_conf", 1.0),
            })

    if not rows:
        raise RuntimeError(f"No valid object proxy observations in {path}")
    return rows
```

### 5.4 Radius-free initialization

```python
def build_init_translations_from_depth(
    obs_rows: list[dict[str, float]],
    camera: CameraModel,
) -> np.ndarray:
    translations = []
    for row in obs_rows:
        z = max(float(row["object_ref_depth_m"]), 0.20)
        x = (row["u"] - camera.cx) * z / camera.fx
        y = (row["v"] - camera.cy) * z / camera.fy
        translations.append(np.array([x, y, z], dtype=np.float64))
    return np.stack(translations, axis=0)
```

This replaces:

```python
z = fx * radius_m / observed_radius_px
```

### 5.5 Support geometry from support proxy

Current support estimation can stay structurally the same, but its input should be `support_v`, not `v + r`.

```python
obs_supports = np.asarray([row["support_v"] for row in obs_rows], dtype=np.float64)
support = estimate_support_geometry_from_contacts(frames, obs_supports, contact_frames)
camera.floor_v = support.floor_v
```

The function `estimate_support_geometry_from_contacts()` can remain unchanged because it only computes the median image-space support line from event frames.

### 5.6 Radius-free sharedcam residuals

Use center reprojection, depth prior, temporal smoothness, and optional support image-line residual.

```python
def pose_residuals_object_proxy(
    flat_state: np.ndarray,
    obs_rows: list[dict[str, float]],
    camera: CameraModel,
    shape: ObjectProxyShape,
    segments: list[dict[str, np.ndarray | int]],
    contact_indices: set[int],
    weak_contact_indices: set[int],
    center_weight: float,
    depth_weight: float,
    support_weight: float,
    temp_weight: float,
    z_temp_weight: float,
    z_boundary_weight: float,
    z_slope_weight: float,
) -> np.ndarray:
    t, ab = unpack_state(flat_state, len(obs_rows), segments)
    residuals: list[float] = []

    for idx, row in enumerate(obs_rows):
        pred = shape.project_ref(t[idx], camera)

        obs_conf = float(row.get("observation_conf", 1.0))
        depth_conf = float(row.get("depth_conf", 1.0))

        # Reference proxy reprojection.
        residuals.append(obs_conf * center_weight * (pred["u"] - row["u"]))
        residuals.append(obs_conf * center_weight * (pred["v"] - row["v"]))

        # Generic depth prior.
        residuals.append(depth_conf * depth_weight * (t[idx, 2] - row["object_ref_depth_m"]))

        # Support proxy image-line residual.
        # support_dv replaces projected radius.
        if idx in contact_indices or idx in weak_contact_indices:
            weight = support_weight if idx in contact_indices else 0.5 * support_weight
            pred_support_v = pred["v"] + row.get("support_dv", 0.0)
            residuals.append(weight * ((pred_support_v - camera.floor_v) / 20.0))

        # Keep object in front of camera.
        residuals.append(0.30 * max(0.0, 0.20 - t[idx, 2]))

    # XY temporal smoothness.
    for idx in range(1, len(t) - 1):
        accel_xy = t[idx + 1, :2] - 2.0 * t[idx, :2] + t[idx - 1, :2]
        residuals.extend((temp_weight * accel_xy).tolist())

    # Z temporal smoothness inside each segment.
    for seg in segments:
        indices = seg["indices"]
        if len(indices) >= 3:
            zvals = t[indices, 2]
            accel_z = zvals[2:] - 2.0 * zvals[1:-1] + zvals[:-2]
            residuals.extend((z_temp_weight * accel_z).tolist())

    # Segment boundary continuity.
    for seg_idx in range(len(segments) - 1):
        cur_seg = segments[seg_idx]
        next_seg = segments[seg_idx + 1]
        end_idx = int(cur_seg["end"])
        next_start_idx = int(next_seg["start"])
        residuals.append(z_boundary_weight * (t[next_start_idx, 2] - t[end_idx, 2]))
        residuals.append(z_slope_weight * (ab[seg_idx + 1, 0] - ab[seg_idx, 0]))

    return np.asarray(residuals, dtype=np.float64)
```

Remove from generic sharedcam:

```text
circle contour residual
projected diameter residual
projected area residual
physical radius depth initialization
bottom_v = v + radius_px
```

### 5.7 Output schema

Output:

```text
results/pose6d_object_proxy/object_pose6d_sharedcam_trajectory.csv
```

Recommended fields:

```csv
frame,time,
tx,ty,tz,
qw,qx,qy,qz,
coord_frame,
u_ref_obs,v_ref_obs,
u_ref_proj,v_ref_proj,
support_v_obs,support_proj_v,floor_v,
residual_px,
contact_frame,audio_contact_frame,
depth_prior_m,depth_prior_gap_m
```

Do not require these fields in the generic output:

```text
radius_m
radius_obs_px
radius_proj_px
bottom_proj_v
```

If backward compatibility is needed, write them as empty strings, not as active variables.

---

## 6. Stage 2: Radius-Free Anchor Interpolation

### 6.1 New file

Create:

```text
scripts/shared/human_ball/contact/run_object_contact_anchorinterp.py
```

Keep old ball-specific anchorinterp file unchanged until the generic path is stable.

### 6.2 Read object proxy pose

```python
def read_object_proxy_pose(path: Path) -> list[dict[str, float | int | str]]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "frame": int(row["frame"]),
                "time": float(row["time"]),
                "tx": float(row["tx"]),
                "ty": float(row["ty"]),
                "tz": float(row["tz"]),
                "qw": float(row.get("qw", 1.0) or 1.0),
                "qx": float(row.get("qx", 0.0) or 0.0),
                "qy": float(row.get("qy", 0.0) or 0.0),
                "qz": float(row.get("qz", 0.0) or 0.0),
                "coord_frame": row.get("coord_frame", "gvhmr_incam"),
                "u_obs": float(row["u_ref_obs"]),
                "v_obs": float(row["v_ref_obs"]),
                "floor_v": float(row.get("floor_v", 0.0) or 0.0),
                "contact_frame": int(row.get("contact_frame", 0) or 0),
                "audio_contact_frame": int(row.get("audio_contact_frame", 0) or 0),
            })
    if not rows:
        raise RuntimeError(f"No object proxy pose rows found in {path}")
    return rows
```

### 6.3 Read contact depth offsets

```python
def read_contact_offsets(path: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            offset_str = row.get("contact_depth_offset_m", "")
            if offset_str == "":
                offset = 0.0
                conf = 0.0
            else:
                offset = float(offset_str)
                conf = float(row.get("contact_conf", 0.0) or 0.0)
            out[frame] = {
                "contact_depth_offset_m": offset,
                "contact_conf": conf,
                "contact_u": float(row.get("contact_u", "nan") or "nan"),
                "contact_v": float(row.get("contact_v", "nan") or "nan"),
            }
    return out
```

### 6.4 Core anchor replacement

Current behavior:

```python
anchor_values = part_z
```

Generic behavior:

```python
contact_offset = np.asarray([
    contact_offsets.get(frame, {}).get("contact_depth_offset_m", 0.0)
    for frame in object_frames
], dtype=np.float64)

anchor_values = part_z - contact_offset
```

Meaning:

```text
z_contact_proxy = z_ref + contact_offset
z_contact_proxy = z_human_part
therefore:
z_ref = z_human_part - contact_offset
```

Then keep the current interpolation solver almost unchanged:

```python
deltas = anchor_values[human_event_mask] - z_init[human_event_mask]
global_z_shift = float(np.median(deltas))
z_ref = np.maximum(z_init + global_z_shift, 0.20)

z_final = solve_anchor_interpolation(
    z_ref=z_ref,
    anchor_mask=human_event_mask,
    anchor_values=anchor_values,
    u_obs=u_obs,
    v_obs=v_obs,
    K=K,
    times=times,
    flight_mask=flight_mask,
    w_ref=args.w_ref,
    w_temp=args.w_temp,
    w_phys_xz=args.w_phys_xz,
    w_phys_y=args.w_phys_y,
    gravity_mps2=args.gravity_mps2,
)
```

### 6.5 Contact gap reporting

The final contact depth gap should be computed on the contact proxy, not the object reference proxy:

```python
z_contact_final = z_final + contact_offset
contact_depth_gap = z_contact_final - part_z
```

Output fields:

```csv
frame,time,
tx,ty,tz,
qw,qx,qy,qz,
coord_frame,
u_ref_obs,v_ref_obs,
contact_u,contact_v,
contact_depth_offset_m,
active_part,active_part_z,
z_contact_final,
contact_depth_gap,
global_z_ref,
human_contact_event,floor_contact_event,human_contact_state,floor_contact_state
```

Do not output `radius_proj_px` or `bottom_proj_v` as active generic quantities.

---

## 7. Recommended Command Flow

### 7.1 Existing preprocessing

Run SAM2 as currently done. The output masks remain:

```text
results/segmentation/masks/*.png
```

### 7.2 Generic mesh tracking

```bash
python -m scripts.shared.tracking.run_cotracker_object_mesh \
  --sample-dir samples/<sample_name> \
  --fps 24 \
  --resize-width 256 \
  --chunk-len 32 \
  --n-boundary 64 \
  --n-interior 64
```

Expected output:

```text
results/tracking/object_mesh_tracks.csv
```

### 7.3 Build object proxy observations

```bash
python -m scripts.shared.object_observation.build_object_proxy_observations \
  --sample-dir samples/<sample_name> \
  --mesh-tracks-csv results/tracking/object_mesh_tracks.csv \
  --mask-dir results/segmentation/masks \
  --depth-dir results/depth \
  --human-proxy-csv results/contact_candidates/human_contact_proxy_2d.csv
```

Expected output:

```text
results/object_observations/object_proxy_observations.csv
```

### 7.4 Radius-free sharedcam baseline

```bash
python -m scripts.shared.sharedcam.run_object_proxy_pose6d_sharedcam \
  --sample-dir samples/<sample_name> \
  --object-proxy-observation-csv results/object_observations/object_proxy_observations.csv \
  --center-weight 0.04 \
  --depth-weight 1.0 \
  --support-weight 10.0 \
  --temp-weight 0.08 \
  --z-temp-weight 0.22
```

Expected output:

```text
results/pose6d_object_proxy/object_pose6d_sharedcam_trajectory.csv
```

### 7.5 Contact anchor interpolation

```bash
python -m scripts.shared.human_ball.contact.run_object_contact_anchorinterp \
  --sample-dir samples/<sample_name> \
  --object-pose-csv results/pose6d_object_proxy/object_pose6d_sharedcam_trajectory.csv \
  --object-proxy-observation-csv results/object_observations/object_proxy_observations.csv \
  --contact-state-csv results/contact_candidates/contact_state_frames.csv \
  --contact-event-csv results/contact_candidates/contact_candidates_labeled.csv
```

Expected output:

```text
results/pose6d_object_proxy_contactphase/object_pose6d_sharedcam_contactphase_trajectory.csv
```

---

## 8. Minimal Implementation Order

### Step 1: Add generic mesh tracking

Implement:

```text
scripts/shared/tracking/run_cotracker_object_mesh.py
```

Output:

```text
object_mesh_tracks.csv
```

Do not modify the old basketball tracker yet.

### Step 2: Add generic proxy observation builder

Implement:

```text
scripts/shared/object_observation/build_object_proxy_observations.py
```

First version can use:

```text
ref proxy:
    median visible mesh point

support proxy:
    lowest visible mesh points

contact proxy:
    nearest visible mesh points to projected active human part

depth:
    DA3 / depth map at ref and contact proxy
```

### Step 3: Add radius-free sharedcam baseline

Implement:

```text
scripts/shared/sharedcam/run_object_proxy_pose6d_sharedcam.py
```

Delete these from generic residuals:

```text
SphereShape
ball_radius_m
circle_contour_points
projected diameter residual
projected area residual
bottom_v = v + r_px
z = fx * radius / r_px
```

Replace with:

```text
ObjectProxyShape
z_init = object_ref_depth_m
support_v = ref_v + support_dv
support residual at contact frames
```

### Step 4: Add contact-proxy anchorinterp

Implement:

```text
scripts/shared/human_ball/contact/run_object_contact_anchorinterp.py
```

Core line:

```python
anchor_values = part_z - contact_depth_offset
```

Fallback:

```python
contact_depth_offset = 0.0
```

This fallback reproduces the old center-depth anchoring assumption, but in the new file it is explicit and measurable.

---

## 9. Mapping from Old Variables to New Variables

| Old basketball variable | Generic replacement | Meaning |
|---|---|---|
| `ball_center_x`, `ball_center_y` | `ref_u`, `ref_v` | object reference proxy |
| `radius_obs_px` | removed | no projected radius cue |
| `ball_radius_m` | removed from generic path | no physical size prior |
| `bottom_v = v + r_px` | `support_v = ref_v + support_dv` | support-facing proxy |
| `z = fx * R / r_px` | `z = object_ref_depth_m` | depth prior from DA3/depth model |
| `z_ball == z_human_part` | `z_ref + delta_contact == z_human_part` | human anchors contact proxy depth |
| `radius_proj_px` | removed / empty | no sphere projection |
| `circle_contour_points()` | optional mask/mesh reprojection later | not needed in first radius-free baseline |

---

## 10. Sanity Checks

### 10.1 Mesh track overlay

Visualize `object_mesh_tracks.csv` over frames:

```text
boundary points: should remain on object contour
interior points: should remain inside object region
visible scores: should drop under occlusion
```

Failure mode:

```text
CoTracker points drift to hand/background.
```

Response:

```text
increase chunk reinitialization frequency
filter tracked points by current SAM2 mask containment
discard points far outside object mask
```

### 10.2 Proxy overlay

For every frame, overlay:

```text
ref proxy: center marker
support proxy: bottom/support marker
contact proxy: marker near active hand/body part
```

Failure mode:

```text
contact proxy jumps to wrong side of object.
```

Response:

```text
use temporal smoothing on contact proxy
use top-k weighted mean instead of nearest single point
increase boundary preference only during contact frames
```

### 10.3 Sharedcam baseline check

Expected after Stage 1:

```text
u_ref_proj and v_ref_proj should match ref_u/ref_v.
tz should roughly follow depth prior but be smoother.
support_proj_v should be near floor_v on floor contact frames.
```

Failure mode:

```text
tz collapses or explodes.
```

Response:

```text
check object_ref_depth_m scale
reduce depth_weight if depth prior is noisy
increase temporal weight
add z lower/upper bounds
```

### 10.4 Anchorinterp check

Expected after Stage 2:

```text
z_contact_final = z_final + contact_depth_offset_m
z_contact_final should be close to active_part_z at human contact frames.
```

Main diagnostic:

```text
contact_depth_gap = z_contact_final - active_part_z
```

Failure mode:

```text
object center moves too much to satisfy hand depth.
```

Response:

```text
inspect contact_depth_offset_m
if contact proxy/depth is unreliable, set offset confidence low or fallback to 0
increase w_ref
reduce anchor influence for low contact_conf frames
```

---

## 11. Ablation Plan

Run these variants:

```text
A. old sphere sharedcam + old anchorinterp
B. object proxy sharedcam with contact_depth_offset = 0
C. object proxy sharedcam with DA3 contact_depth_offset
D. object proxy sharedcam + contact proxy selected from mesh + hand proximity
E. object proxy sharedcam + support proxy floor contact
```

Important metrics:

```text
2D ref reprojection error
support proxy image-line error at floor events
contact_depth_gap at human events
temporal smoothness / trajectory jumps
qualitative overlay
```

This separates:

```text
radius removal effect
DA3 depth prior effect
contact proxy effect
support proxy effect
```

---

## 12. Main Design Decision

The generic path should not solve object-specific semantic parts.

Do not define:

```text
mug handle
chair leg
blanket corner
```

Define only:

```text
tracked object mesh points
reference proxy
contact proxy
support proxy
```

Then the same code handles:

```text
mug: contact proxy lands near handle/side surface
chair: contact proxy lands near back/arm/seat/leg depending on hand
blanket: contact proxy lands near grasped cloth boundary
ball: contact proxy lands on the visible ball side, without requiring radius
```

The final generic anchor equation is:

```text
z_ref_anchor = z_human_part - (z_contact_proxy - z_ref_proxy)
```

In code:

```python
anchor_values = part_z - contact_depth_offset_m
```

This is the minimal change that preserves the original two-stage anchor interpolation structure while removing the physical radius prior from the generic object path.
