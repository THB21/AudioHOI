# Radius-free contact candidate design

## 0. Context

The current pipeline has three separable stages:

```text
Stage 0: SAM2 / CoTracker preprocessing
Stage 1: sharedcam baseline optimization
Stage 2: anchorinterp depth refinement
```

This note focuses only on the **contact candidate** stage.

The main question is:

```text
If we remove the ball radius prior from sharedcam / anchorinterp,
what should happen to contact candidate detection?
```

The answer is:

```text
Short term:
    Contact candidate code can stay as-is if we only need event frames
    for the current basketball branch.

Long term / generic object branch:
    It must be changed from center-radius contact detection
    to mesh / mask-boundary contact detection.
```

---

## 1. Current contact candidate logic

The current file is:

```text
scripts/shared/contact_candidates/run_contact_candidate_detection.py
```

It is already useful because it produces the two files needed by downstream contact-phase logic:

```text
results/contact_candidates/contact_state_frames.csv
results/contact_candidates/contact_candidates_labeled.csv
```

The downstream `anchorinterp` stage mostly needs:

```text
human_contact_event frames
floor_contact_event frames
human_contact_state
floor_contact_state
contact_part / contact_side / contact_label
```

So as an event proposal layer, it is still usable.

However, the current implementation is still **ball-specific**.

---

## 2. Where the radius prior appears in contact candidate detection

### 2.1 Input is ball trajectory

Current code reads:

```python
ball_rows = read_ball_track(results_dir / "tracking" / "ball_trajectory.csv")
```

and expects these fields:

```text
ball_center_x
ball_center_y
radius
mask_area
source
```

This means the detector assumes the object state is:

```text
object = 2D center + pixel radius
```

---

### 2.2 Human-object contact uses center-radius geometry

Current logic is effectively:

```python
ball_uv = np.stack([ball_u, ball_v], axis=1)
min_contact_dist = min(distance(left_anchor, ball_center), distance(right_anchor, ball_center))

min_contact_gap = max(
    min_contact_dist - ball_r - anchor_region_radius_px,
    0.0,
)
```

This has the geometric meaning:

```text
contact gap = distance(human anchor, ball center) - ball radius
```

For a ball this approximates the distance from the hand/foot to the ball boundary.

For a mug, chair, blanket, bag, box, etc., this is not a valid contact boundary model.

---

### 2.3 Floor contact also uses center-radius geometry

Current floor contact logic is effectively:

```python
ball_bottom_v = ball_v + ball_r
floor_gap = abs(ball_bottom_v - support_v)
```

This has the geometric meaning:

```text
object support point = ball center y + ball radius
```

For general objects, the support point should instead come from:

```text
lowest visible mask point
lowest visible tracked mesh point
support-facing boundary point
```

---

## 3. What should replace radius in contact candidate detection

The generic replacement is:

```text
ball center + radius
→ object mesh / object mask boundary
```

Specifically:

```text
human-object contact:
    old: distance(human_anchor, ball_center) - ball_radius
    new: distance(human_anchor, object_boundary_or_mesh_points)

object-support contact:
    old: ball_center_y + ball_radius
    new: support_proxy_v
```

This keeps the detector category-agnostic. It does not require:

```text
mug handle
chair leg
blanket corner
bottle cap
```

It only requires generic object proxies:

```text
object_ref_proxy
object_contact_boundary_or_mesh_points
object_support_proxy
```

---

## 4. Proposed new file

Do not overwrite the basketball detector immediately.

Create a new file:

```text
scripts/shared/contact_candidates/run_object_contact_candidate_detection.py
```

Keep the old file for basketball:

```text
scripts/shared/contact_candidates/run_contact_candidate_detection.py
```

The new generic file should preserve the output contract:

```text
anchor_contact_candidates.csv
floor_contact_candidates.csv
contact_state_frames.csv
contact_intervals.csv
contact_candidates_labeled.csv
```

This allows `anchorinterp` to keep reading the same event/state files.

---

## 5. Required inputs for the generic detector

### 5.1 Object observation table

From `object_observations.csv`, use:

```text
frame
time
ref_u
ref_v
support_proxy_u
support_proxy_v
observation_conf
```

For backward compatibility, these can initially map to existing fields:

```text
ref_u = center_x
ref_v = center_y
support_proxy_u = center_x
support_proxy_v = bbox_y2
```

The key is that `support_proxy_v` replaces:

```text
ball_center_y + radius
```

---

### 5.2 Object mesh / boundary tracks

Preferred generic input:

```text
results/tracking/object_mesh_tracks.csv
```

Suggested long-format schema:

```csv
frame,time,point_id,point_type,x,y,visible
1,0.000,boundary_000,boundary,421.2,301.5,0.98
1,0.000,boundary_001,boundary,424.0,299.8,0.97
1,0.000,interior_000,interior,430.5,315.0,0.95
...
```

Use primarily:

```text
point_type == boundary
```

for contact detection. Interior points can be used as fallback when boundary tracks are sparse or unreliable.

---

## 6. Key code replacement 1: read generic object track

Replace the ball-specific reader:

```python
def read_ball_track(path: Path) -> list[dict[str, float | int | str]]:
    ...
```

with a generic reader:

```python
def read_object_proxy_track(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref_u = row.get("ref_u") or row.get("center_x")
            ref_v = row.get("ref_v") or row.get("center_y")
            support_u = row.get("support_proxy_u") or ref_u
            support_v = row.get("support_proxy_v") or row.get("bbox_y2")

            if not ref_u or not ref_v:
                continue

            rows.append(
                {
                    "frame": int(row["frame"]),
                    "time": float(row.get("time", 0.0) or 0.0),
                    "ref_u": float(ref_u),
                    "ref_v": float(ref_v),
                    "support_u": float(support_u) if support_u else float(ref_u),
                    "support_v": float(support_v) if support_v else float(ref_v),
                    "observation_conf": float(row.get("observation_conf", 1.0) or 1.0),
                }
            )

    if not rows:
        raise RuntimeError(f"No object proxy rows found in {path}")
    return rows
```

This removes the dependency on:

```text
ball_center_x
ball_center_y
radius
```

---

## 7. Key code replacement 2: read object mesh tracks

Add a reader for long-format mesh tracks:

```python
def read_object_mesh_tracks(path: Path) -> dict[int, dict[str, np.ndarray]]:
    by_frame: dict[int, list[dict[str, object]]] = {}

    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            by_frame.setdefault(frame, []).append(row)

    out: dict[int, dict[str, np.ndarray]] = {}
    for frame, rows in by_frame.items():
        xy = []
        visible = []
        is_boundary = []
        point_ids = []

        for r in rows:
            x = r.get("x", "")
            y = r.get("y", "")
            if x == "" or y == "":
                continue
            xy.append([float(x), float(y)])
            visible.append(float(r.get("visible", 1.0) or 1.0) > 0.5)
            is_boundary.append(str(r.get("point_type", "")) == "boundary")
            point_ids.append(str(r.get("point_id", "")))

        if xy:
            out[frame] = {
                "xy": np.asarray(xy, dtype=np.float64),
                "visible": np.asarray(visible, dtype=bool),
                "is_boundary": np.asarray(is_boundary, dtype=bool),
                "point_ids": np.asarray(point_ids, dtype=object),
            }

    if not out:
        raise RuntimeError(f"No object mesh tracks found in {path}")
    return out
```

---

## 8. Key code replacement 3: distance to object boundary / mesh

Old concept:

```python
min_contact_gap = min_contact_dist - ball_r - anchor_region_radius_px
```

New concept:

```python
min_contact_gap = distance(human_anchor, nearest visible object boundary point)
```

Implementation:

```python
def nearest_object_points(
    anchor_uv: np.ndarray,
    anchor_valid: np.ndarray,
    frames: np.ndarray,
    object_mesh_by_frame: dict[int, dict[str, np.ndarray]],
    prefer_boundary: bool = True,
) -> dict[str, np.ndarray | list[str]]:
    n = len(frames)
    nearest_dist = np.full(n, np.inf, dtype=np.float64)
    nearest_uv = np.full((n, 2), np.nan, dtype=np.float64)
    nearest_point_id: list[str] = [""] * n
    nearest_valid = np.zeros(n, dtype=bool)

    for i, frame in enumerate(frames.tolist()):
        if not anchor_valid[i]:
            continue
        mesh = object_mesh_by_frame.get(int(frame))
        if mesh is None:
            continue

        pts = mesh["xy"]
        valid = mesh["visible"].copy()
        if prefer_boundary:
            boundary_valid = valid & mesh["is_boundary"]
            if np.any(boundary_valid):
                valid = boundary_valid

        if not np.any(valid):
            continue

        candidates = pts[valid]
        candidate_ids = mesh["point_ids"][valid]
        dists = np.linalg.norm(candidates - anchor_uv[i][None, :], axis=1)
        j = int(np.argmin(dists))

        nearest_dist[i] = float(dists[j])
        nearest_uv[i] = candidates[j]
        nearest_point_id[i] = str(candidate_ids[j])
        nearest_valid[i] = True

    return {
        "dist": nearest_dist,
        "uv": nearest_uv,
        "point_id": nearest_point_id,
        "valid": nearest_valid,
    }
```

This is the core generic replacement for radius-based contact geometry.

---

## 9. Key code replacement 4: generic anchor contact detection

Old detector signature:

```python
def detect_anchor_contact(
    left_contact_uv,
    right_contact_uv,
    left_valid,
    right_valid,
    ball_u,
    ball_v,
    ball_r,
    ...
):
    ...
```

New detector signature:

```python
def detect_object_anchor_contact(
    *,
    frames: np.ndarray,
    left_contact_uv: np.ndarray,
    right_contact_uv: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    object_mesh_by_frame: dict[int, dict[str, np.ndarray]],
    dist_thresh_px: float,
    score_sigma_px: float,
    local_radius: int,
    state_dist_thresh_px: float,
    state_score_thresh: float,
    gap_bridge: int,
    audio_support: np.ndarray | None = None,
) -> dict[str, np.ndarray | list[str]]:
    left_nn = nearest_object_points(
        left_contact_uv,
        left_valid,
        frames,
        object_mesh_by_frame,
        prefer_boundary=True,
    )
    right_nn = nearest_object_points(
        right_contact_uv,
        right_valid,
        frames,
        object_mesh_by_frame,
        prefer_boundary=True,
    )

    left_dist = np.asarray(left_nn["dist"], dtype=np.float64)
    right_dist = np.asarray(right_nn["dist"], dtype=np.float64)

    active_is_left = left_dist <= right_dist
    active_contact = np.where(active_is_left, "left", "right")
    min_contact_gap = np.minimum(left_dist, right_dist)

    if audio_support is None:
        audio_support = np.zeros_like(min_contact_gap)
    else:
        audio_support = np.asarray(audio_support, dtype=np.float64)

    proximity_score = gaussian_score(min_contact_gap, score_sigma_px)

    # Object response can still use the reference trajectory if available,
    # but it no longer depends on radius.
    object_response_score = np.zeros_like(min_contact_gap)

    contact_score = (
        0.55 * proximity_score
        + 0.15 * object_response_score
        + 0.30 * audio_support
    )

    contact_local_min = local_min_mask(min_contact_gap, local_radius)

    is_candidate = (
        (min_contact_gap <= dist_thresh_px)
        & contact_local_min
    )

    state = (
        (min_contact_gap <= state_dist_thresh_px)
        & (contact_score >= state_score_thresh)
    )
    state = bridge_short_gaps(state, gap_bridge)

    active_point_id = []
    active_object_u = np.full(len(frames), np.nan, dtype=np.float64)
    active_object_v = np.full(len(frames), np.nan, dtype=np.float64)

    left_ids = list(left_nn["point_id"])
    right_ids = list(right_nn["point_id"])
    left_uv = np.asarray(left_nn["uv"], dtype=np.float64)
    right_uv = np.asarray(right_nn["uv"], dtype=np.float64)

    for i in range(len(frames)):
        if active_is_left[i]:
            active_point_id.append(left_ids[i])
            active_object_u[i] = left_uv[i, 0]
            active_object_v[i] = left_uv[i, 1]
        else:
            active_point_id.append(right_ids[i])
            active_object_u[i] = right_uv[i, 0]
            active_object_v[i] = right_uv[i, 1]

    return {
        "left_dist": left_dist,
        "right_dist": right_dist,
        "active_contact": active_contact.tolist(),
        "min_contact_gap": min_contact_gap,
        "proximity_score": proximity_score,
        "object_response_score": object_response_score,
        "audio_support": audio_support,
        "score": contact_score,
        "candidate": is_candidate,
        "state": state,
        "active_object_point_id": active_point_id,
        "active_object_u": active_object_u,
        "active_object_v": active_object_v,
    }
```

Important change:

```text
No ball_r.
No min_contact_dist - radius.
No center-radius boundary approximation.
```

---

## 10. Key code replacement 5: generic support / floor contact detection

Old floor contact:

```python
ball_bottom_v = ball_v + ball_r
floor_gap = abs(ball_bottom_v - support_v)
```

New support contact:

```python
def detect_object_support_contact(
    *,
    object_support_v: np.ndarray,
    support_v: np.ndarray,
    gap_thresh_px: float,
    score_sigma_px: float,
    local_radius: int,
    state_gap_thresh_px: float,
    state_score_thresh: float,
    gap_bridge: int,
) -> dict[str, np.ndarray]:
    support_gap = np.abs(object_support_v - support_v)
    support_gap = np.where(np.isnan(support_gap), np.inf, support_gap)
    support_score = gaussian_score(support_gap, score_sigma_px)

    support_local_peak = local_max_mask(object_support_v, local_radius)

    is_candidate = support_local_peak & (support_gap <= gap_thresh_px)

    state = (support_gap <= state_gap_thresh_px) & (support_score >= state_score_thresh)
    state = bridge_short_gaps(state, gap_bridge)

    return {
        "object_support_v": object_support_v,
        "gap": support_gap,
        "score": support_score,
        "local_peak": support_local_peak,
        "candidate": is_candidate,
        "state": state,
    }
```

Here:

```text
object_support_v = support_proxy_v
```

where `support_proxy_v` comes from:

```text
lowest visible tracked mesh point
or mask bottom boundary
or bbox_y2 fallback
```

---

## 11. Main function structure for the generic detector

Pseudo-main:

```python
def main() -> None:
    args = build_parser().parse_args()

    sample_dir = args.sample_dir
    results_dir = sample_dir / "results"
    out_dir = args.out_dir or (results_dir / "contact_candidates")
    out_dir.mkdir(parents=True, exist_ok=True)

    object_rows = read_object_proxy_track(
        results_dir / "object_observations" / "object_observations.csv"
    )
    object_mesh_by_frame = read_object_mesh_tracks(
        results_dir / "tracking" / "object_mesh_tracks.csv"
    )

    human = read_human_result(results_dir / "gvhmr" / "result.pkl")
    joints = build_body_joints(args.body_model_root, human)
    K = np.asarray(human["K_fullimg"], dtype=np.float64)

    frames = np.asarray([int(r["frame"]) for r in object_rows], dtype=np.int32)
    times = np.asarray([float(r["time"]) for r in object_rows], dtype=np.float64)
    ref_u = np.asarray([float(r["ref_u"]) for r in object_rows], dtype=np.float64)
    ref_v = np.asarray([float(r["ref_v"]) for r in object_rows], dtype=np.float64)
    object_support_v = np.asarray([float(r["support_v"]) for r in object_rows], dtype=np.float64)

    audio_rows = read_audio_events(results_dir / "events" / "audio_events.csv")
    audio_support = build_audio_support(frames, audio_rows, radius=2)

    if args.contact_anchor == "hand":
        left_contact_cam, right_contact_cam = build_palm_centers(joints)
        contact_side_name = "hand"
    else:
        left_contact_cam, right_contact_cam = build_foot_points(joints)
        contact_side_name = "foot"

    left_contact_uv, left_valid = project_points(left_contact_cam, K)
    right_contact_uv, right_valid = project_points(right_contact_cam, K)

    anchor_det = detect_object_anchor_contact(
        frames=frames,
        left_contact_uv=left_contact_uv,
        right_contact_uv=right_contact_uv,
        left_valid=left_valid,
        right_valid=right_valid,
        object_mesh_by_frame=object_mesh_by_frame,
        dist_thresh_px=args.contact_dist_thresh_px,
        score_sigma_px=args.contact_score_sigma_px,
        local_radius=args.contact_local_radius,
        state_dist_thresh_px=args.contact_state_dist_thresh_px,
        state_score_thresh=args.contact_state_score_thresh,
        gap_bridge=args.contact_gap_bridge,
        audio_support=audio_support,
    )

    support_v = estimate_support_v_from_proxy(object_support_v)
    support_v_arr = np.full(len(frames), support_v, dtype=np.float64)

    floor_det = detect_object_support_contact(
        object_support_v=object_support_v,
        support_v=support_v_arr,
        gap_thresh_px=args.floor_gap_thresh_px,
        score_sigma_px=args.floor_score_sigma_px,
        local_radius=args.floor_local_radius,
        state_gap_thresh_px=args.floor_state_gap_thresh_px,
        state_score_thresh=args.floor_state_score_thresh,
        gap_bridge=args.floor_gap_bridge,
    )

    # Then reuse the same fusion / interval writing logic:
    # - fuse_contact_types
    # - enforce_audio_contact_coverage, possibly simplified
    # - mask_to_intervals
    # - contact_candidates_labeled.csv
    # - contact_state_frames.csv
```

---

## 12. Output schema changes

To preserve compatibility with existing downstream code, keep these columns in:

```text
contact_state_frames.csv
```

```text
frame
time
anchor_type
contact_part
contact_side
contact_label
anchor_score
floor_score
anchor_contact_state
floor_contact_state
transition_contact_state
multi_contact_state
```

Add generic object-contact columns:

```text
active_object_point_id
active_object_u
active_object_v
min_object_boundary_gap_px
```

Old ball-specific columns can either be removed in the generic file or retained as empty compatibility fields:

```text
ball_center_x
ball_center_y
ball_radius_px
ball_bottom_v
```

Recommended generic naming:

```text
object_ref_u
object_ref_v
object_support_v
object_contact_u
object_contact_v
```

---

## 13. Minimal compatibility layer for anchorinterp

The current `anchorinterp` does not need all geometric details from contact candidate. It mainly needs event/state frames.

Therefore, the generic detector must at least write:

```text
contact_candidates_labeled.csv
```

with:

```text
frame
time
contact_type
anchor_type
contact_part
contact_side
contact_label
target
score
confidence
source
```

and:

```text
contact_state_frames.csv
```

with:

```text
frame
time
anchor_contact_state
floor_contact_state
contact_label
contact_side
```

Then `anchorinterp` can continue to identify:

```text
human contact frames
floor contact frames
active human side / part
```

The extra object contact point columns are needed for the improved depth-anchor version:

```text
contact_depth_offset_m = depth(object_contact_proxy) - depth(object_ref_proxy)
```

---

## 14. Relation to anchorinterp depth anchoring

The radius-free anchorinterp should use:

```text
z_object_contact = z_human_part
```

with:

```text
z_object_contact = z_object_ref + contact_depth_offset_m
```

Therefore:

```text
z_object_ref = z_human_part - contact_depth_offset_m
```

Code-level replacement in anchorinterp:

```python
anchor_values = part_z - contact_depth_offset_m
```

instead of:

```python
anchor_values = part_z
```

The contact candidate detector provides the 2D object contact proxy:

```text
object_contact_u, object_contact_v
```

A depth module or observation builder then converts it into:

```text
contact_depth_offset_m
```

For the first radius-free baseline, this offset can be set to zero:

```python
contact_depth_offset_m = 0.0
```

Then the method reduces to:

```text
object ref depth ≈ human contact depth
```

This is acceptable as a simple baseline, but it should be documented as an approximation.

---

## 15. What can remain unchanged

These parts of the current contact candidate code can mostly remain:

```text
project_points()
build_palm_centers()
build_foot_points()
gaussian_score()
local_min_mask()
local_max_mask()
bridge_short_gaps()
build_transition_state()
fuse_contact_types()
mask_to_intervals()
split_contact_label()
write_csv()
```

These parts need replacement or generic variants:

```text
read_ball_track()
estimate_floor_v_from_ball()
detect_anchor_contact()
detect_floor_contact()
main() input loading and output field names
```

---

## 16. Direct old-to-new mapping

```text
ball_center_x / ball_center_y
→ object_ref_u / object_ref_v

ball_r
→ removed

min_contact_dist - ball_r
→ distance(human_anchor, nearest visible object boundary / mesh point)

object_contact_score based on radius overlap
→ proximity score based on boundary / mesh distance

ball_bottom_v = ball_v + ball_r
→ object_support_v = support_proxy_v

estimate_floor_v_from_ball(ball_v, ball_r)
→ estimate_support_v_from_proxy(object_support_v)

anchor_contact_candidates.csv ball fields
→ object_contact_u / object_contact_v / active_object_point_id
```

---

## 17. Practical implementation order

Recommended order:

```text
1. Keep current run_contact_candidate_detection.py unchanged.

2. Add run_object_contact_candidate_detection.py.

3. Make it read object_observations.csv and object_mesh_tracks.csv.

4. Replace center-radius distance with nearest mesh/boundary distance.

5. Replace ball_bottom_v with support_proxy_v.

6. Keep output CSV names and core state/event columns unchanged.

7. Add object_contact_u/v and active_object_point_id to outputs.

8. Let anchorinterp use the same contact_state_frames.csv and contact_candidates_labeled.csv.

9. Later, use object_contact_u/v to compute contact_depth_offset_m for better depth anchoring.
```

---

## 18. One-sentence summary

The issue in contact candidate detection is not the event logic itself. The issue is the object geometry used for contact distance:

```text
current:
    contact = distance(human anchor, ball center) - radius

radius-free generic:
    contact = distance(human anchor, object mesh / mask boundary)
```

The rest of the module can mostly be preserved as an event scoring and interval extraction layer.
