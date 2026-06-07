# Current Ball Pipeline Status

This document summarizes the current basketball / football ball-interaction pipeline in this repo, including:

- the current mainline pipeline
- the sharedcam baseline
- contact-candidate logic
- contactphase / anchor refinement
- the current DA3 experimental branch
- current conclusions and known issues


## 1. High-Level Status

There are currently two classes of pipelines in the repo:

1. **Mainline pipeline**
   - object observations
   - contact candidate detection
   - sharedcam baseline
   - contactphase anchor refinement
   - rendering

2. **Experimental DA3-based branch**
   - DA3 scene depth / priors
   - DA3-based sharedcam initialization / optimization
   - optional downstream anchor refinement

At the moment:

- the **mainline anchorinterp result** is still the best practical result
- the **unified DA3 + anchor single-stage solver** is experimental and not yet better than the current mainline
- the **DA3-init -> anchorinterp** branch is also experimental and has not clearly beaten the current mainline


## 2. Current Mainline Pipeline

### 2.1 Overall flow

Current mainline logic is:

```text
video / masks / tracking
-> object_observations
-> contact_candidates
-> sharedcam baseline
-> contactphase_anchorinterp
-> render
```

For basketball, current mainline outputs are under:

- `samples/basketball_01/results/pose6d_sharedcam`
- `samples/basketball_01/results/pose6d_sharedcam_contactphase_anchorinterp`

For football, current mainline outputs are under:

- `samples/football_10/results/pose6d_sharedcam`
- `samples/football_10/results/pose6d_sharedcam_contactphase_anchorinterp`


## 3. Object Observations Layer

### Main file

- `scripts/shared/object_observation/build_object_observations.py`

### Role

This layer standardizes object-side 2D observations from tracking + masks.

It currently writes:

- center position
  - `center_x`
  - `center_y`
- mask / bbox geometry
- enclosing radius in pixels
  - `enclosing_radius_px`
- newly added lowest visible point fields
  - `lowest_visible_x`
  - `lowest_visible_y`
  - `lowest_visible_x1`
  - `lowest_visible_x2`

### Important current status

The original fields are preserved.

Current consumers are gradually moving to:

- prefer `lowest_visible_y`
- fallback to `center + radius` only when needed


## 4. Contact Candidate Logic

### Main file

- `scripts/shared/contact_candidates/run_contact_candidate_detection.py`

### Role

This stage produces first-pass 2D contact proposals and contact states from:

- object 2D observations
- GVHMR human joints
- optional audio support

### Current main logic

It builds:

- hand palm proxies
- foot proxies
- floor support reference from projected body support points

It then scores:

- human anchor candidates
- floor contact candidates

and writes:

- `contact_candidates_labeled.csv`
- `contact_state_frames.csv`
- `anchor_contact_candidates.csv`
- `contact_intervals.csv`

### Current data sources

Ball-side inputs now prefer `object_observations.csv`:

- `center_x / center_y`
- `enclosing_radius_px`
- `lowest_visible_y`

For floor contact detection:

- current code prefers `lowest_visible_y`
- fallback is still `ball_center_y + radius`

### Output semantics

The current generic semantics are:

- `anchor_contact_event`
- `anchor_contact_state`
- `floor_contact_event`
- `floor_contact_state`
- `active_anchor_side`

For basketball this usually maps to:

- `anchor_type = hand`
- `target = right_hand` or `left_hand`

For football this usually maps to:

- `anchor_type = foot`
- `target = right_foot` or `left_foot`


## 5. Sharedcam Baseline

### Main file

- `scripts/shared/sharedcam/run_basketball_pose6d_sharedcam.py`

### Current logic

This is still the main baseline solver.

It currently does:

```text
radius-based z initialization
-> segment construction from contact frames
-> segment-wise line fit
-> least-squares optimization
```

### Important detail

The baseline is **not** just “radius gives z once and done”.

It uses:

1. **radius-based initialization**
   - `build_init_translations(...)`

2. **segment-wise z parameterization**
   - `build_segments(...)`
   - `fit_segment_lines(...)`

3. **least-squares optimization**
   - `pose_residuals(...)`

### Does baseline optimization still use known ball radius?

Yes.

The baseline optimization still uses:

- `SphereShape(args.ball_radius_m)`

That means the optimization stage uses known ball radius in:

- projected radius residual
- bottom projection
- sphere geometry
- floor/contact geometry through the sphere model

### Support geometry

Support geometry is estimated from contact frames using:

- `scripts/shared/sharedcam/support_geometry.py`

Current support geometry input prefers:

- `lowest_visible_y`

with fallback to:

- `v + r`


## 6. Current Mainline Contactphase Anchor Refinement

### Main file

- `scripts/shared/human_ball/contact/run_human_ball_contact_phase_calibration_anchorinterp_generic.py`

### Role

This is the current mainline “final refinement” layer.

It takes:

- `pose6d_sharedcam` baseline trajectory
- contact candidates / states
- GVHMR body joints

and refines the ball depth using human contact anchors.

### Core logic

Current logic is:

```text
sharedcam baseline z_init
-> compute human-contact z deltas
-> global z shift from human event frames
-> anchor interpolation with hard human contact anchors
-> flight physics prior
```

### Hard constraints

Human contact event frames are treated as hard depth anchors:

```text
z_ball == z_part
```

This is true for the current anchorinterp mainline.

### Outside-window behavior

The generic anchorinterp script supports:

- `global_ref`
- `boundary_constant`

and we have used these for different experiments.

### Current render target

This script writes the mainline final ball trajectory used for current comparison and rendering:

- `ball_pose6d_sharedcam_contactphase_trajectory.csv`


## 7. Current Rendering

### Main file

- `scripts/shared/sharedcam/render_pose6d_sharedcam_direct.py`

### Role

This renders:

- ball-only
- human + ball
- overlay
- worldlike 3D / camera3d / side_yz

### Current status

The renderer now prefers generic fields from the trajectory CSV:

- `contact_part`
- `contact_side`
- `contact_label`
- `active_part`

It no longer relies primarily on old `active_hand` / `active_foot` guessing.

### Active ring logic

Current rule:

- if explicit `contact_side` exists, draw active ring
- if no side exists, **do not fallback**

This was added to stop left/right active-marker jumping.

### Floor rendering rule

Current rendering behavior for floor frames is:

- do not use fallback active-proxy selection
- for basketball, both hands can remain visible as candidate proxies
- no active ring unless explicit side is present


## 8. Current DA3 Experimental Branch

### Main files

- `scripts/shared/sharedcam/da3/run_da3_scene_depth.py`
- `scripts/shared/sharedcam/da3/extract_da3_priors.py`
- `scripts/shared/sharedcam/da3/run_basketball_pose6d_sharedcam_da3_init.py`

### Intended role

The DA3 branch is trying to answer:

```text
Can DA3 provide z_init directly, instead of using known ball radius as the main driver?
```

### Current DA3-init logic

Current DA3-init does:

```text
DA3 depth prior
-> interpolate missing z
-> build init translation from depth prior
-> same segment-wise optimization structure
-> optimize
```

So:

- it is **not** “just output DA3 z with no optimization”
- it already performs a full optimization step

### Does current DA3-init still use known ball radius?

Current DA3-init has been modified so that the solver path no longer uses known physical ball radius as the main prior.

Specifically:

- no radius-based z init
- no DA3-to-radius affine calibration
- no SphereShape-based residual in the DA3-init solver path

However:

- the object observation layer still carries pixel radius observations
- support geometry still historically came from center/radius fallback when lowest-point information was missing

### Current floor logic in DA3-init

This area is still experimental.

We tried several variants:

1. support based on human support `z` only
2. support based on human support `y` + weak `z`

Current lesson:

- floor should **not** be treated as a true depth anchor
- floor should **not** directly constrain ball center `y`
- floor should eventually be expressed through **lowest visible object point**


## 9. DA3 -> Anchor Experiment

### Current experimental output

We also tried:

```text
DA3-init trajectory
-> anchorinterp
```

Current output is stored separately under:

- `samples/basketball_01/results/pose6d_sharedcam_da3init_contactphase_anchorinterp`

This branch does **not** overwrite the mainline.

### Current result

This branch is still experimental and has not clearly beaten the current mainline anchorinterp.


## 10. Unified Single-Stage DA3 + Anchor Solver

### Main file

- `scripts/shared/human_ball/contact/run_human_ball_contact_phase_unified_generic.py`

### Intended idea

The unified solver was introduced to test:

```text
DA3 prior + human anchors + floor cues + physics
-> one final solve
```

instead of:

```text
DA3 init
-> baseline / sharedcam
-> anchor refine
```

### Current status

This solver is **experimental**.

It was useful for clarifying current failure modes, but it is **not yet better than the current mainline anchorinterp**.

### Key lessons from unified experiments

1. **Human contact hard anchors are still reliable**
   - these should remain hard constraints

2. **Floor is not a true depth anchor**
   - floor does not currently provide a real plane depth
   - calling floor an “anchor” is misleading

3. **Constraining ball center `y` to support `y` is wrong**
   - in the current parameterization, `x/y` are reconstructed from `u/v/z`
   - so constraining center `y` indirectly pushes `z`
   - this produced wrong deep pushes in floor frames

4. **Lowest visible point is the right abstraction**
   - floor support should eventually be written on the object’s lowest visible point
   - not on the ball center


## 11. Current Best Practical Result

For basketball, the current best practical result is still:

- `samples/basketball_01/results/pose6d_sharedcam_contactphase_anchorinterp`

For football, the current mainline practical result is:

- `samples/football_10/results/pose6d_sharedcam_contactphase_anchorinterp`

At the moment:

- **mainline anchorinterp** is still better than
  - unified DA3 + anchor
  - DA3-init -> anchor


## 12. Known Issues / Open Problems

### 12.1 Floor support is not fully correct yet

Current floor logic still needs a proper formulation based on:

- object lowest visible point
- support plane / support level

instead of:

- ball center y
- or weak proxy-only z logic

### 12.2 Lowest-point logic is not fully propagated into final optimization

We already added:

- `lowest_visible_x`
- `lowest_visible_y`
- `lowest_visible_x1`
- `lowest_visible_x2`

to object observations.

This now needs to be used more consistently in:

- floor contact detection
- sharedcam support geometry
- DA3-init solver
- final optimization / unified solver

### 12.3 DA3 is promising but not yet strong enough to replace the mainline

Current DA3 branch is useful for experimentation, but:

- it has not clearly beaten the current mainline
- it still needs better floor treatment
- it still needs stronger lowest-point-aware geometry


## 13. Short Summary

### Current mainline

```text
object observations
-> contact candidates
-> sharedcam baseline (radius-based init + optimization)
-> contactphase anchorinterp (hard human z anchors)
-> render
```

### Current DA3 branch

```text
DA3 depth prior
-> DA3-init optimization
-> optional anchorinterp
```

### Current verdict

- keep DA3 as an experimental branch
- keep mainline anchorinterp as the current best result
- next meaningful improvement should focus on:
  - lowest visible object point
  - better floor support formulation
  - not on forcing DA3 to replace the mainline too early
