# Mug Handle-Phase & Depth-Anchor Pipeline

**Entry point:** `python src/mug/pipeline.py --sample-dir samples_known_object/02_mug`

---

## Overview

The pipeline estimates the **3D pose of a mug** (position + handle orientation) from a monocular RGB video in which a person picks up, holds, and sets down a mug.  It has four sequential stages; only the final stage produces video renders — all intermediate stages output CSV files only.

```
Stage 1  phase.py       VLM filter + joint phase optimizer   → handle_phase.csv
Stage 2  correction.py  Hidden-segment arc interpolation     → corrected_phase.csv
Stage 3  anchor.py      Z + XY depth anchor refinement       → anchored_pose.csv
Stage 4  render         Final scene render (6 videos)        → results/renders/pipe/
```

---

## VLM Calls (Pre-pipeline)

Two separate Qwen3-VL-8B-Instruct inference passes are run **before** `pipeline.py`.  Their outputs are consumed as inputs by the main pipeline stages.

---

### VLM Call 1 — Handle Visibility Labeling

**Script:** `scripts/shared/radius_free_proxy/stage1_observation/run_qwen_vlm_handle_visibility.py`  
**Runs on:** Every frame in `object_observations.csv` (or a user-specified stride/range)  
**Output:** `annotations/vlm_handle_visibility_full/qwen_handle_visibility.csv` + `.json`

#### Image crop

For each frame, the script crops a padded region around the mug body bounding box:
- Left/Right: +65% of bbox width on each side
- Top: +35% of bbox height
- Bottom: +45% of bbox height

This gives the VLM enough context to see the handle (which often extends past the body bbox) and any occluding hand.

#### Prompt (compact JSON output)

The model is asked to inspect the mug crop and return structured JSON describing:

| Output field | Values | Used by pipeline |
|---|---|---|
| `visibility` | `"visible"` / `"hidden"` / `"uncertain"` | Stage 1 VLM filter; Stage 2 segment detection |
| `handle_visible` | bool | Stage 1 VLM filter |
| `handle_contact` | bool — true only if fingers are on the C-shaped loop, not just the body | Stage 1 attachment residual gating |
| `hand_contact_part` | `"handle"` / `"body"` / `"rim"` / `"unknown"` / `"none"` | Contact event classification |
| `recommended_visibility_constraint` | `"force_visible"` / `"force_hidden_far_side"` / `"force_hidden_by_hand"` / `"weak_unknown"` | Stage 1 VLM filter overrides |
| `confidence` | 0.0–1.0 | Stage 1 attachment weight scaling |
| `occlusion_reason` | `"none"` / `"hand"` / `"mug_body"` / `"far_side"` / `"out_of_crop"` | Diagnostic only |
| `visible_side` | `"left"` / `"right"` / `"front"` / `"back"` / `"unknown"` | Diagnostic only |
| `handle_shape_visible` | `"clear_c_loop"` / `"partial_arc"` / `"small_attachment"` / `"not_visible"` | Diagnostic only |
| `yaw_anchor_quality` | `"high"` / `"medium"` / `"low"` / `"none"` | Diagnostic only |
| `body_contact`, `rim_contact` | bool | Diagnostic / contact event classification |
| `hand_occludes_handle`, `body_self_occludes_handle` | bool | Diagnostic only |
| `short_reason` | string | Diagnostic only |

#### How the pipeline uses this output

- **Stage 1 `phase.py`** reads `visibility`, `handle_contact`, `hand_contact_part`, `recommended_visibility_constraint`, and `confidence` to:
  - Temporally filter short noisy islands (≤3 frames)
  - Gate the hand-handle attachment residual (only active when `handle_contact==True` and `confidence ≥ 0.15`)
  - Determine which frames are "confirmed handle grasp" vs. "rim contact" vs. "keep previous"
- **Stage 2 `correction.py`** reads `visibility` to locate hidden/visible segment boundaries for arc-path interpolation.

```bash
python scripts/shared/radius_free_proxy/stage1_observation/run_qwen_vlm_handle_visibility.py \
    --sample-dir samples_known_object/02_mug \
    --model-id Qwen/Qwen3-VL-8B-Instruct \
    --stride 1 \
    --save-crops
```

---

### VLM Call 2 — Contact Keyframe Labeling

**Script:** `scripts/shared/radius_free_proxy/stage2_contact_candidates/run_qwen_mug_contact_keyframes.py`  
**Runs on:** Representative frames from confirmed direct-grasp-anchor runs (start / middle / end of each run — typically ~3–10 frames total)  
**Output:** `annotations/vlm_mug_contact_keyframes/mug_contact_keyframe_annotations.csv` + `.json`

#### Frame selection

The script automatically selects **representative frames** from `mug_grasp_anchor_state.csv`:
- Finds all frames where `frame_mode == "direct_grasp_anchor"`
- Groups them into continuous runs
- Picks 3 frames per run: first, middle, last

#### Image crop

A larger, context-rich square crop (minimum 420 px side) centered on the union of:
- Mug body bbox
- Known contact UV coordinates (`active_part_u/v`, `hand_contact_u/v`)
- 150 px margin on all sides

This captures finger-level detail while keeping wrist, palm, and mug handle in frame.

A **context hint** is injected into the prompt: tracker hand side (`left_hand`/`right_hand`), frame mode, stable grasp source frame.

#### Prompt (one-line JSON output)

| Output field | Values | Used by pipeline |
|---|---|---|
| `contact_visible` | bool | Validation / filtering |
| `hand_side` | `"left_hand"` / `"right_hand"` / `"unknown"` | Confirms or corrects tracker label |
| `object_part` | `"handle"` / `"body"` / `"rim"` / `"bottom"` / `"unknown"` | Semantic annotation |
| `object_region` | `"upper_handle"` / `"middle_handle"` / `"lower_handle"` / `"handle_inner"` / `"handle_outer"` / `"body_side"` / ... | Localizes which part of handle is gripped |
| `handle_grasp_type` | `"pinch_handle"` / `"hook_handle"` / `"palm_support"` / `"body_grasp"` / `"not_handle_grasp"` | Grasp style classification |
| `contact_fingers` | list of `"thumb"` / `"index"` / ... / `"palm"` | Which fingers are in contact |
| `primary_contact_finger` | single finger name | Dominant contact finger |
| `use_as_stable_grasp_keyframe` | bool — VLM confirmation that this frame is a good anchor | Anchor quality gate |
| `confidence` | 0.0–1.0 | Weight or filter threshold |
| `reason` | string | Diagnostic only |

#### How the pipeline uses this output

This call is **not** consumed by the four main pipeline stages directly.  Its outputs serve as:
- **Semantic annotations** enriching `mug_grasp_anchor_state.csv` with VLM-verified handle region labels (`object_region`, `handle_grasp_type`)
- **Quality confirmation**: frames where VLM says `use_as_stable_grasp_keyframe=False` or `confidence < 0.4` can be downweighted when building anchor targets in Stage 3
- **Debugging aid**: crops and JSON are saved for manual inspection of grasp quality

```bash
python scripts/shared/radius_free_proxy/stage2_contact_candidates/run_qwen_mug_contact_keyframes.py \
    --sample-dir samples_known_object/02_mug \
    --model-id Qwen/Qwen3-VL-8B-Instruct
```

---

## Pre-pipeline inputs

These must exist before running `pipeline.py` (produced by earlier preparation scripts):

| File | Description |
|------|-------------|
| `proxy/mug_body_only_cylinder_pose_table_static_sequence.csv` | Per-frame mug body pose (x, y, z, yaw, pitch, roll, scale in camera space). Fitted by `fit_mug_body_only_cylinder_pose.py`. |
| `results/renders/M12_articraft_rigid_mesh_vlm/handle_phase_all.csv` | Per-frame handle phase prior from VLM-overlay fitting (M12). Used as a weak initialization. |
| `results/mug_articraft_contact_points/mug_articraft_contact_points.csv` | Per-frame contact event labels: which frames are confirmed handle grasps, rim contacts, keep-previous frames, etc. |
| `results/mug_grasp_anchor_state/mug_grasp_anchor_state.csv` | Stable grasp anchor: the contact point in mug-local coordinates (`stable_grasp_local_{x,y,z}`) and hand-side label. |
| `results/contact_candidates_object_proxy/contact_state_frames.csv` | Per-frame contact/floor state flags and depth offset (`contact_depth_offset_m`). |
| `results/gvhmr/result.pkl` | GVHMR human pose estimation output (3D joints + camera intrinsics). |
| `annotations/vlm_handle_visibility_full/qwen_handle_visibility.csv` | Per-frame VLM classification: is the handle visible or hidden? |
| `articraft/materialized_mug_mesh/` | Articraft-parameterized mug mesh parts (`.obj` files for body_shell, rim_ring, bottom_disk, handle_loop). |

---

## Stage 1 — Handle Phase Optimization (`src/mug/phase.py`)

**Input →** body pose CSV, M12 phase prior CSV, contact points CSV, VLM visibility CSV  
**Output →** `results/pipe/handle_phase.csv`  (columns: `frame, time, phase_rad, phase_deg`)

### 1a. VLM temporal filter

Raw per-frame VLM labels are noisy — a frame may be labeled "handle visible" in the middle of a clearly-hidden run.  Short **islands** (runs of ≤3 frames surrounded by the opposite state) are flipped to match their neighbors.  This is iterated until stable.

### 1b. Visibility alpha

A per-frame continuous weight α ∈ [0, 1] is computed:
- Fully visible frames: α = 1.0
- Fully hidden frames: α = 0.0
- Boundary ramp: linear ramp over ±5 frames at each visible/hidden transition
- Short visible islands (≤10 frames): capped at α ≤ 0.85
- Hidden frames adjacent to a visible run: small bleed-in of up to 0.35

The alpha is **not** used in the current optimizer directly — it is available for future residual weighting.

### 1c. Phase optimizer

**Variables:** θ[0..N-1] — unwrapped handle phase for each frame (in radians).

**Initialization:** `theta0 = unwrap(M12_prior)` — the M12 VLM-overlay phase, unwrapped continuously.

**Residuals** (least-squares, `loss='soft_l1'`, `f_scale=1.0`, `max_nfev=120`):

| Term | Formula | Purpose |
|------|---------|---------|
| M12 prior | `0.35 · wrap(θ[i] - θ0[i]) / 25°` | Weak pull toward M12 initialization |
| Hand-handle attachment | `(24·w·max(0.35, conf)) · ‖proj(handle_contact, θ[i]) - hand_uv‖ / 7px` | Project the stable grasp point through the current phase; minimize reprojection to observed hand pixel |
| Table-static freeze | `3.5·wt · wrap(θ[i] - θ[table_start]) / 4°` | Once the mug rests on the table (sustained floor support ≥12 frames), lock all subsequent phases to the table-start phase |
| Velocity smoothness | `0.8 · Δθ / 10°` + excess penalty `2.0 · max(0, |Δθ|−8°) / 3°` | Penalize fast rotation; hard-penalize jumps > 8°/frame |
| Acceleration smoothness | `1.2 · Δ²θ / 5°` | Penalize jerk |

**Attachment weight `w`** per frame:
- `use_this_point_for_hand_attachment == 1` (confirmed new grasp): w = 1.0
- `use_previous_grasp_for_hand_attachment == 1` (carry previous): w = 0.60
- `rim_drinking_contact == 1` (rim contact, no handle): w = 0.30

**Stable grasp local point** (`stable_grasp_local_{x,y,z}`) is updated only on `use_this_point = 1` frames; rim frames never update it.  It is carried forward from any confirmed frame before the current segment.

**Table detection:** Scans for the first run of ≥12 consecutive frames with `support_conf ≥ 0.65` and `object_motion_score ≤ 0.15`.

---

## Stage 2 — Far-side Phase Correction (`src/mug/correction.py`)

**Input →** `handle_phase.csv`, body pose CSV, VLM visibility CSV  
**Output →** `results/pipe/corrected_phase.csv`  (columns: `frame, time, m17_phase_rad, m17_phase_deg, m14_phase_rad, m14_phase_deg, is_error_frame, phase_correction_rad, vlm_visibility`)

### Problem

Within **hidden segments** (frames where VLM labels the handle as not visible), Stage 1 has no image evidence.  The optimizer may choose a branch that puts the handle on the **near side** (facing the camera), which is physically wrong for a hidden handle.

### Algorithm

For each hidden segment [i, j):

1. Read the last visible phase before (`left = phase[i-1]`) and first visible phase after (`right = phase[j]`).
2. Compute two candidate interpolation arcs:
   - **Short arc** (`delta_short`): the ≤180° arc from `left` to `right` — less total rotation.
   - **Long arc** (`delta_long`): the complementary >180° arc going the other way.
3. For each candidate arc, count how many interpolated hidden frames would have `sin(phase + yaw) > 0` — meaning the handle is on the **far side** (away from camera, as expected when hidden).
4. **Choose the arc with more far-side frames.**  Tie-break: prefer the shorter arc.
5. Linearly interpolate hidden frames along the chosen arc:  `phase[i+k] = left + (k+1)/(n+1) · arc_delta`

Visible frames are **never modified**.

### Post-smoothing

A light Gaussian smooth (`sigma/2 = 1.5` frames, `mode='nearest'`) is applied to the full unwrapped sequence to remove sub-degree roughness in visible frames from Stage 1.

---

## Stage 3 — Depth + XY Anchor Refinement (`src/mug/anchor.py`)

**Input →** body pose CSV, `corrected_phase.csv`, grasp-state CSV, contact-state CSV, `gvhmr/result.pkl`  
**Output →** `results/pipe/anchored_pose.csv`  (same schema as body pose CSV, with refined x, y, z)

### Motivation

The mug body pose CSV has accurate **orientation** (yaw, pitch, roll) and **image-plane position**, but its camera-space depth (z) comes from proxy tracking which can drift significantly during hand contact.  GVHMR provides accurate 3D hand joint positions; we use the hand-mug contact constraint to anchor the mug depth.

### 3a. Anchor target construction

For each frame that is in a **continuous contact** state (`human_contact_state == 1` or `anchor_contact_state == 1` or `use_this_point` or `use_previous_grasp`):

1. Rotate the stable grasp local point `stable_grasp_local` by the corrected handle phase to get the **world-space position of the contact point on the mug** (`anchor_cam`).
2. Compute the **offset from mug center** to this contact point in camera space: `delta_{x,y,z} = anchor_cam - mug_center`.
3. Read the GVHMR hand joint position (`hand_xyz`) for the appropriate hand side.
4. Clip the raw `contact_depth_offset_m` to **±0.08 m** (physical constraint: hand-mug gap is always < 8 cm; larger values indicate proxy tracking errors).
5. Compute anchor targets:
   - `target_z = hand_z - clipped_offset - delta_z`
   - `target_x = hand_x - delta_x`
   - `target_y = hand_y - delta_y`

**Anchor weight** per frame:
- `use_this_point = 1` (confirmed update): weight = `3.0 · conf`, clipped to [0.25, 4.0]
- `use_previous_grasp = 1` (carry previous): weight = `1.4 · conf`
- Other continuous contact: weight = `1.0 · conf`

### 3b. Z optimization

**Reference trajectory:** Linear interpolation between all anchor `target_z` values.  Pre-anchor frames are set to the first anchor; post-anchor frames to the last anchor.  All values clipped to ≥ 0.30 m.

**Optimizer** (`loss='soft_l1'`, `f_scale=0.03`, `max_nfev=300`):

| Residual | Formula | Weight |
|----------|---------|--------|
| Reference pull | `z[i] - z_ref[i]` | 0.45 |
| Anchor constraint | `z[i] - target_z[i]` | `anchor_weight` (per-frame) |
| Velocity smoothness | `Δz[i]` | 4.0 |
| Acceleration smoothness | `Δ²z[i]` | 12.0 |
| Table-static freeze | `z[i] - z[table_start]` for i ≥ table_start | 20.0 |

### 3c. XY optimization

After Z is finalized, re-project the mug center UV coordinates at the new depth to get a new XY reference:
`x_ref, y_ref = (u - cx) · z_final / fx,  (v - cy) · z_final / fy`

**Optimizer** (same settings, `f_scale=0.04`):

| Residual | Weight |
|----------|--------|
| Reference pull `xy[i] - xy_ref[i]` | 0.30 |
| Anchor constraint `xy[i] - target_xy[i]` | `anchor_weight · 0.75` |
| Velocity smoothness `Δxy` | 3.0 |
| Acceleration smoothness `Δ²xy` | 8.0 |
| Table-static freeze | 15.0 |

---

## Stage 4 — Final Render

**Script:** `scripts/shared/radius_free_proxy/stage5_render/render_mug_articraft_camera3d_scene.py`  
**Input →** `anchored_pose.csv`, `corrected_phase.csv`, Articraft mesh  
**Output →** `results/renders/pipe/`  (6 videos)

| Video | Description |
|-------|-------------|
| `camera3d_object_only.mp4` | 3D camera-space view, mug only |
| `camera3d_with_human.mp4` | 3D camera-space view, mug + SMPL-X body |
| `overlay_object_only.mp4` | Mug mesh projected onto original video frames |
| `overlay_with_human.mp4` | Mug + body overlay on video |
| `side_yz_object_only.mp4` | Side view (depth vs height), mug only |
| `side_yz_with_human.mp4` | Side view, mug + hand joints |

---

## Running the pipeline

```bash
# Full pipeline (from repo root, audiohoi env)
python src/mug/pipeline.py --sample-dir samples_known_object/02_mug

# Re-run from a specific stage onwards
python src/mug/pipeline.py --sample-dir samples_known_object/02_mug --from anchor

# Run only one stage
python src/mug/pipeline.py --sample-dir samples_known_object/02_mug --only render

# Force re-run everything
python src/mug/pipeline.py --sample-dir samples_known_object/02_mug --force

# Quick wireframe preview render (fast, no solid mug)
python src/mug/pipeline.py --sample-dir samples_known_object/02_mug --only render --wireframe
```

Each stage can also be run standalone:

```bash
python src/mug/phase.py --sample-dir samples_known_object/02_mug
python src/mug/correction.py --sample-dir samples_known_object/02_mug
python src/mug/anchor.py --sample-dir samples_known_object/02_mug
```

---

## Key design decisions

- **No intermediate renders.** Stages 1–3 output only CSV files.  Rendering happens once at the end.
- **±0.08 m depth offset clip.** Proxy tracking errors during contact can produce offsets of ±1 m or more.  Physical hand-mug gap is always < 8 cm, so values outside this range are tracking noise.
- **Stable grasp local point.** The contact point is stored in mug-local coordinates and rotated by the current phase estimate.  This means the attachment residual is differentiable with respect to phase — changing the phase rotates where the handle contact point projects onto the image.
- **Table-static lock.** Once the mug is stably supported on the table (detected by sustained floor contact + low motion score), all subsequent frames are frozen in phase and depth.  This prevents the optimizer from drifting on frames where the mug is stationary and there is no informative signal.
- **Soft-L1 loss.** All optimizers use Huber-like robust loss to suppress outlier anchor frames (e.g., a single frame with an erroneous VLM label or a bad proxy track).
