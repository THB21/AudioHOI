# Mug Local M18 Plan: HaMeR Hand Proxy + Palm-Handle Stabilization

> Status correction, 2026-06-17:
> The `M12/M13/M14/M14.5/M14.6/M15` labels in this document came from the
> previous cloud push and are historical stage names only. The real local
> baseline is now `M18`, specifically:
>
> - pose baseline: `samples_known_object/02_mug/results/pipe/anchored_pose_obs.csv`
> - handle phase baseline: `samples_known_object/02_mug/results/renders/M17_phase_corrected/corrected_handle_phase.csv`
> - render baseline: `samples_known_object/02_mug/results/renders/M18_obs_scene/`
>
> New work must not overwrite any old `M15` outputs. Use explicit local M18
> experiment names such as `mug_m18_hamer_hand_proxy/`,
> `mug_m18_grasp_anchor_state/`, and `mug_m18_palm_attachment_experiment/`.
>
> Current local experiment, 2026-06-17:
>
> - 2D-aligned M18 pose:
>   `samples_known_object/02_mug/results/mug_m18_opening_2d_video_correction/mug_m18_opening_2d_video_pose.csv`
> - current M45 pose with automatically detected rotation-jump smoothing and
>   automatically detected table-static release:
>   `samples_known_object/02_mug/results/mug_m18_pose_M45_table_static_release/mug_m18_pose_m45_table_static_release.csv`
> - current no-hide physical handle phase with smoother drinking-entry schedule:
>   `samples_known_object/02_mug/results/mug_m18_handle_phase_M43_smooth_entry_no_hide/corrected_handle_phase_m43_smooth_entry_no_hide.csv`
> - single clean reproducible optimization entrypoint:
>   `scripts/known_object/mug/run_mug_m18_physical_nohide_pipeline.py`
> - render output:
>   `samples_known_object/02_mug/results/renders/final_result/`
>
> M45 must not hide `handle_loop` based on `vlm_visibility`. The handle remains
> real geometry; visibility labels are diagnostics only. If the handle is wrong,
> fix pose/phase/contact constraints rather than masking the handle in render.
> Current clean logic:
> - relax frame-57/frame-62 phase anchors so the drinking-entry handle turn is
>   continuous. This is still a M45 physical phase prior, not a full automatic
>   phase optimizer;
> - automatically detect the mug rotation branch jump from neighboring-frame
>   rotation geodesic outliers, then Slerp only that local rotation branch while
>   preserving translation/depth/scale;
> - automatically detect table-static release from sustained support/contact
>   evidence, then freeze all mug pose parameters after that frame.

> HaMeR environment correction, 2026-06-17:
> HaMeR must run from the separate conda env `hamer`, not from `audiohoi`,
> because its detector/render/MANO dependencies can conflict with the main
> AudioHOI pipeline. The source checkout is `third-party/hamer`, and the
> checkpoint download is present at
> `third-party/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt`.
> Current blocker: `third-party/hamer/_DATA/data/mano/MANO_RIGHT.pkl` is still
> missing; HaMeR imports and starts checkpoint loading, but MANO mesh loading
> cannot complete until that licensed file is placed there.

## 0. Problem Definition

The local M18 mug pipeline already has a reasonably developed object-side representation.
The old labels below are historical references from the previous cloud push,
not the names of the current local outputs:

```text
cloud M12:
    Articraft mug mesh
    mug body / rim / bottom / handle decomposition
    rigid mug transform
    handle phase / visibility diagnostic

cloud M13:
    object-side contact point export
    contact usage split
    handle / rim / hidden / occluded contact labels

cloud M14:
    M12 + M13 joint optimization
    handle phase + VLM visibility + contact evidence
```

The current missing part is:

```text
hand-side concrete contact proxy
```

At the moment, the system can reason about:

```text
where the mug handle/contact candidate is
```

but it does not yet have a stable representation of:

```text
where the hand is actually supposed to contact the mug
```

The new goal is therefore:

```text
Build a stable palm-level hand contact proxy first,
then use it to validate and stabilize mug grasp anchors.
```

---

## 1. Main Meeting Takeaway for Mug

The supervisor's new suggestion is not to recover an exact hand pose.

The suggestion is:

```text
Do not require exact finger pose.

Use a coarse hand proxy,
preferably palm-only,
to define a stable hand-object contact constraint.
```

This means the mug contact should be formulated as:

```text
palm ↔ mug handle
```

instead of:

```text
whole hand ↔ mug
finger tip ↔ mug
nearest hand point ↔ nearest object point
```

The reason is that a nearest-point or whole-hand Chamfer loss can produce unstable or misleading contact:

```text
- one fingertip can accidentally satisfy the loss
- the mug body or rim can steal the contact
- hidden handle frames can cause hallucinated contact updates
- finger pose noise can rotate or drag the mug incorrectly
```

Therefore the next stage must explicitly introduce:

```text
P_palm(t)
```

where `P_palm(t)` is a palm-level proxy point or palm patch in 3D.

---

## 2. Revised Pipeline

The updated local M18 pipeline should be:

```text
cloud M12 / local pre-M18:
    Articraft mug mesh and rigid mug pose

cloud M13 / local pre-M18:
    object-side mug contact candidates

cloud M14 / local pre-M18:
    handle phase / visibility / object-side contact reasoning

local M18.HaMeR:
    HaMeR/MANO hand keypoints and grasp-capable palm proxy

local M18.palm:
    hand-side palm proxy construction

local M18.verify:
    palm-handle contact verification and gating

local M18.attach experiment:
    stable grasp anchor state machine
    + translation-only mug attachment optimization
```

The critical new stage is:

```text
local M18.HaMeR + local M18.palm = hand-side palm proxy construction
```

---

# Part A: Local M18.palm Hand-Side Palm Proxy

## 3. Objective of Local M18.palm

For every frame `t`, construct:

```text
palm_proxy_world(t)
palm_proxy_image(t)
palm_conf(t)
palm_state(t)
```

The proxy does not need to represent the full hand.

It only needs to be stable enough to answer:

```text
Is the mug handle close to the palm region?
```

---

## 4. Input Sources

Possible sources, ordered by priority:

```text
1. hand keypoints
2. SMPL-X / GVHMR wrist and hand joints
3. hand mask
4. VLM/manual palm keyframe annotation
5. temporal propagation from previous reliable palm proxy
```

The first implementation should use whichever source is already available in the current mug data.

---

## 5. Palm Proxy Construction Rules

### 5.1 If hand keypoints are available

Use palm-related keypoints:

```text
wrist
index_mcp
middle_mcp
ring_mcp
pinky_mcp
```

Define:

```python
palm_center =
mean([
    wrist,
    index_mcp,
    middle_mcp,
    ring_mcp,
    pinky_mcp,
])
```

Confidence:

```python
palm_conf =
mean([
    wrist_conf,
    index_mcp_conf,
    middle_mcp_conf,
    ring_mcp_conf,
    pinky_mcp_conf,
])
```

This is the preferred representation because it approximates the palm rather than the fingertips.

---

### 5.2 If only wrist and hand joint are available

Use a coarse interpolation:

```python
palm_center =
alpha * wrist
+
(1.0 - alpha) * hand_joint
```

Recommended first value:

```python
alpha = 0.4
```

So:

```python
palm_center =
0.4 * wrist
+
0.6 * hand_joint
```

Confidence:

```python
palm_conf =
min(wrist_conf, hand_conf)
```

This is less accurate than MCP-based palm estimation but enough for a first contact proxy.

---

### 5.3 If only 2D hand information is available

Construct:

```text
palm_proxy_u
palm_proxy_v
```

from one of:

```text
hand mask centroid
hand keypoint center
manual/VLM palm point
```

Then lift to 3D using:

```text
available hand depth
SMPL/GVHMR depth
local depth map
previous reliable palm depth
```

If depth is unreliable, mark:

```text
palm_conf = low
```

and do not allow the frame to update the grasp anchor.

---

### 5.4 If hand is occluded

Do not create a new palm observation from image evidence.

Instead:

```text
keep previous palm proxy
decay confidence
mark palm_state = propagated
```

Example:

```python
palm_proxy_world[t] = palm_proxy_world[t - 1]
palm_conf[t] = 0.8 * palm_conf[t - 1]
```

This propagated proxy can be used for temporal continuity diagnostics, but it should not create a new direct grasp anchor.

---

## 6. Palm Patch Instead of Single Palm Point

A single palm point may still be too brittle.

Therefore the internal representation should allow a small palm patch:

```text
P_palm(t) = {palm points around palm center}
```

First version:

```text
palm_center only
```

Second version:

```text
small local patch with radius 3–5 cm
```

If a palm normal is available:

```text
sample a local disk on the palm plane
```

If no palm normal is available:

```text
use palm_center only for Local M18.palm evaluation
```

Do not block the implementation on normal estimation.

---

## 7. Output: Palm Proxy CSV

Create:

```text
samples_known_object/02_mug/results/hand_contact_proxy/mug_palm_proxy.csv
```

Schema:

```csv
frame,
time,
active_hand,

palm_proxy_x,
palm_proxy_y,
palm_proxy_z,

palm_proxy_u,
palm_proxy_v,

palm_conf,
palm_state,
source,

is_hand_visible,
is_palm_observed,
is_palm_propagated
```

Optional fields:

```csv
palm_normal_x,
palm_normal_y,
palm_normal_z,

palm_patch_radius_m,

wrist_x,
wrist_y,
wrist_z,

hand_joint_x,
hand_joint_y,
hand_joint_z
```

Recommended `palm_state` values:

```text
observed_keypoints
observed_wrist_hand
observed_mask
manual_keyframe
propagated
missing
```

---

## 8. New Script: Build Palm Proxy

Create:

```text
scripts/known_object/mug/build_mug_palm_proxy.py
```

Inputs:

```text
--sample-dir samples_known_object/02_mug
--human-pose-csv <path>
--hand-keypoints-csv <optional path>
--hand-mask-dir <optional path>
--mug-contact-csv results/mug_articraft_contact_points/mug_articraft_contact_points.csv
--out-dir results/hand_contact_proxy
```

Outputs:

```text
results/hand_contact_proxy/mug_palm_proxy.csv
results/hand_contact_proxy/mug_palm_proxy_summary.txt
```

The script should not modify local M18 pose or object contact files.

---

# Part B: Palm Debug Rendering

## 9. Purpose

Before using the palm proxy in optimization, visually check whether it is placed on the correct hand region.

This stage answers:

```text
Is the generated palm proxy actually near the palm,
or is it on the wrist, fingertips, arm, or background?
```

---

## 10. Render Output

Create:

```text
samples_known_object/02_mug/results/renders/M18_hamer_palm_proxy_debug/overlay.mp4
```

Render per frame:

```text
1. original video frame
2. mug mesh overlay
3. mug handle highlight
4. palm proxy point
5. palm confidence text
6. palm state text
7. active hand label
```

Color rules:

```text
green:
    observed_keypoints

yellow:
    observed_wrist_hand

blue:
    propagated

red:
    missing or invalid

purple:
    manual_keyframe
```

---

## 11. Debug Questions

The debug render should make these failures visible:

```text
1. palm point is on fingertip
2. palm point is on wrist/forearm
3. palm point jumps between hands
4. palm point disappears during visible grasp
5. propagated palm point drifts too far
6. palm proxy is behind/inside the mug incorrectly
```

If these happen, fix local M18.HaMeR / palm proxy before continuing.

---

# Part C: Palm-Handle Contact Verification

## 12. Why This Stage Is Needed

M13 object-side contact candidates are not sufficient.

A mug handle candidate should only become a direct grasp anchor when there is also hand-side evidence.

Therefore:

```text
object-side handle candidate
```

must be gated by:

```text
palm-side proximity
```

---

## 13. Contact Candidate Acceptance Rule

For each frame:

```text
candidate_world(t) = T_world_mug(t) * object_contact_local(t)
palm_world(t) = palm_proxy_world(t)
```

Compute:

```python
dist_palm_handle =
norm(candidate_world - palm_world)
```

Accept direct hand attachment only if:

```text
candidate is on handle
AND palm_conf is high enough
AND dist_palm_handle is below threshold
AND frame is not rim drinking contact
AND frame is not hidden/occluded hallucination
```

Recommended first thresholds:

```text
palm_conf >= 0.4
dist_palm_handle <= 0.08 m
```

A stricter threshold can be tested later:

```text
dist_palm_handle <= 0.05 m
```

---

## 14. Output: Palm-Handle Verification CSV

Create:

```text
samples_known_object/02_mug/results/hand_contact_proxy/palm_handle_verification.csv
```

Schema:

```csv
frame,
time,

object_contact_event,
nearest_articraft_part,

object_contact_local_x,
object_contact_local_y,
object_contact_local_z,

object_contact_world_x,
object_contact_world_y,
object_contact_world_z,

palm_proxy_x,
palm_proxy_y,
palm_proxy_z,

palm_conf,
palm_state,

dist_palm_handle_m,

passes_palm_gate,
passes_handle_gate,
passes_visibility_gate,
passes_direct_anchor_gate,

reject_reason
```

Recommended `reject_reason` values:

```text
missing_palm_proxy
low_palm_conf
not_handle_part
rim_drinking_contact
hidden_or_occluded
distance_too_large
unconfirmed_visible_handle
accepted
```

---

## 15. New Script: Verify Palm-Handle Contact

Create:

```text
scripts/known_object/mug/verify_mug_palm_handle_contact.py
```

Inputs:

```text
--m18-trajectory-csv samples_known_object/02_mug/results/pipe/anchored_pose_obs.csv
--mug-contact-csv results/mug_articraft_contact_points/mug_articraft_contact_points.csv
--palm-proxy-csv results/hand_contact_proxy/mug_palm_proxy.csv
--out-dir results/hand_contact_proxy
```

Outputs:

```text
palm_handle_verification.csv
palm_handle_verification_summary.txt
```

---

# Part D: VLM Verification

## 16. Role of VLM

VLM should not be used as the optimizer.

VLM should only be used as a conservative verifier for fragile steps.

For mug, VLM should verify:

```text
1. whether the highlighted object region is the handle
2. whether the highlighted hand point is on the palm region
3. whether hand and handle are plausibly close in the rendered/cropped view
```

---

## 17. Do Not Ask Broad Questions

Avoid:

```text
Is the contact reasonable?
```

This question is too vague and may produce biased yes answers.

Use specific forced-choice questions.

---

## 18. VLM Question 1: Object Side

Input:

```text
object-only render
mug handle/contact candidate highlighted in red
```

Question:

```text
Which part of the mug is highlighted in red?

Choose one:
A. handle
B. cup body
C. rim
D. bottom
E. unclear
```

Accept only:

```text
A. handle
```

---

## 19. VLM Question 2: Hand Side

Input:

```text
cropped hand-mug interaction image
palm proxy highlighted
```

Question:

```text
Where is the highlighted point on the hand?

Choose one:
A. palm
B. fingertip
C. wrist
D. forearm
E. background
F. unclear
```

Accept only:

```text
A. palm
```

---

## 20. VLM Question 3: Palm-Handle Proximity

Input:

```text
rendered mug + hand crop
mug handle highlighted
palm proxy highlighted
```

Question:

```text
Is the highlighted palm proxy close to the highlighted mug handle?

Choose one:
A. yes
B. no
C. unclear
```

Accept only:

```text
A. yes
```

---

## 21. VLM Usage Rule

Use VLM output as a gate:

```text
accept / reject / unclear
```

Do not use free-form VLM text directly as a continuous loss value.

If VLM says:

```text
unclear
```

then the frame should not update `stable_grasp_local`.

---

# Part E: Stable Grasp Anchor State

## 22. State Machine Principle

Only confirmed visible palm-handle contact frames can update the stable mug grasp anchor.

The stable anchor is stored in mug-local coordinates:

```text
stable_grasp_local
```

This point should not be overwritten by:

```text
rim contact
body contact
nearest visible mesh candidate
occluded handle candidate
unconfirmed VLM output
```

---

## 23. Frame Modes

Use these frame modes:

```text
direct_grasp_anchor:
    confirmed visible palm-handle contact
    allowed to update stable_grasp_local

keep_previous_grasp_anchor:
    hand is probably still holding mug
    but no reliable new direct contact observation
    stable_grasp_local is reused

rim_contact_no_anchor_update:
    mug rim contacts mouth
    this must not update hand grasp anchor

no_attachment:
    no reliable hand-mug attachment evidence

invalid:
    contradictory evidence or geometry failure
```

Important distinction:

```text
rim contact is not hand grasp contact.
```

If the person is drinking while still holding the mug, the previous hand grasp anchor may be kept, but the rim point must never become the hand anchor.

---

## 24. Anchor Update Rule

Pseudo-code:

```python
stable_grasp_local = None
stable_grasp_conf = 0.0

for frame in frames:
    palm_row = palm_proxy[frame]
    verify_row = palm_handle_verification[frame]
    contact_row = mug_contact_points[frame]

    if verify_row["passes_direct_anchor_gate"] == 1:
        stable_grasp_local = [
            contact_row["object_contact_local_x"],
            contact_row["object_contact_local_y"],
            contact_row["object_contact_local_z"],
        ]
        stable_grasp_conf = min(
            palm_row["palm_conf"],
            verify_row["verification_conf"],
        )
        frame_mode = "direct_grasp_anchor"

    elif should_keep_previous_grasp(contact_row, palm_row) and stable_grasp_local is not None:
        frame_mode = "keep_previous_grasp_anchor"

    elif contact_row["rim_drinking_contact"] == 1:
        frame_mode = "rim_contact_no_anchor_update"

    else:
        frame_mode = "no_attachment"
```

---

## 25. Output: Grasp Anchor State

Create:

```text
samples_known_object/02_mug/results/mug_m18_grasp_anchor_state/mug_grasp_anchor_state.csv
```

Schema:

```csv
frame,
time,

object_contact_event,
hand_mug_contact_state,

frame_mode,

stable_grasp_local_x,
stable_grasp_local_y,
stable_grasp_local_z,

stable_grasp_conf,

palm_proxy_x,
palm_proxy_y,
palm_proxy_z,
palm_conf,

dist_palm_handle_m,

update_stable_grasp,
keep_previous_grasp,

reject_reason
```

---

## 26. New Script: Build Grasp Anchor State

Create:

```text
scripts/known_object/mug/build_mug_grasp_anchor_state.py
```

Inputs:

```text
--mug-contact-csv results/mug_articraft_contact_points/mug_articraft_contact_points.csv
--palm-proxy-csv results/hand_contact_proxy/mug_palm_proxy.csv
--palm-handle-verification-csv results/hand_contact_proxy/palm_handle_verification.csv
--out-dir results/mug_m18_grasp_anchor_state
```

Outputs:

```text
mug_grasp_anchor_state.csv
mug_grasp_anchor_state_summary.txt
```

---

# Part F: Evaluation Before Optimization

## 27. Why Evaluation Comes Before Optimization

Do not run the local M18 attachment experiment immediately.

First check whether the palm proxy and stable anchor logic are geometrically reasonable.

If the evaluation fails, optimization will only hide the actual problem.

---

## 28. Hand-to-Grasp Distance

For each frame where `frame_mode` is:

```text
direct_grasp_anchor
keep_previous_grasp_anchor
```

compute:

```python
grasp_world =
T_world_mug(t) * stable_grasp_local

dist =
norm(grasp_world - palm_proxy_world(t))
```

Output:

```text
samples_known_object/02_mug/results/mug_m18_grasp_anchor_state/hand_to_grasp_distance.csv
```

Schema:

```csv
frame,
time,

frame_mode,

grasp_world_x,
grasp_world_y,
grasp_world_z,

palm_proxy_x,
palm_proxy_y,
palm_proxy_z,

palm_conf,

hand_to_grasp_dist_m,

is_large_gap
```

Recommended large-gap threshold:

```text
0.10 m
```

---

## 29. Evaluation Questions

Check:

```text
1. Are direct_grasp_anchor frames actually palm-handle contact frames?

2. Is the palm proxy on the palm, not fingertip/wrist?

3. Does stable_grasp_local remain constant during hidden/drinking segments?

4. Does rim contact avoid overwriting the hand anchor?

5. Does hand_to_grasp_dist_m remain reasonable?

6. Are large distance gaps caused by palm proxy failure or object pose failure?
```

If these fail:

```text
do not run the local M18 attachment experiment
```

Fix:

```text
palm proxy
contact gate
state machine
```

first.

---

## 30. Evaluation Summary

Create:

```text
samples_known_object/02_mug/results/mug_m18_grasp_anchor_state/evaluation_summary.txt
```

Include:

```text
num_frames
num_direct_grasp_anchor
num_keep_previous_grasp_anchor
num_rim_contact_no_anchor_update
num_no_attachment

mean_hand_to_grasp_dist_direct
max_hand_to_grasp_dist_direct

mean_hand_to_grasp_dist_keep_previous
max_hand_to_grasp_dist_keep_previous

num_large_gaps
frames_large_gaps

num_anchor_updates
num_anchor_rejections
```

---

# Part G: Local M18 Translation-Only Attachment Experiment

## 31. Local M18 Attachment Should Be Conservative

The local M18 attachment experiment should not optimize full 6D mug pose at first.

Optimize only translation corrections:

```text
delta_x(t)
delta_y(t)
delta_z(t)
```

Do not optimize:

```text
yaw
pitch
roll
scale
handle phase
mug mesh geometry
hand pose
```

Reason:

```text
the palm-handle attachment residual may otherwise rotate or distort the mug pose
```

The purpose of the local M18 attachment experiment is only to reduce attachment gaps while staying close to local M18.

---

## 32. Local M18 Attachment Variables

For `N` frames:

```python
params =
[
    delta_x_0, delta_y_0, delta_z_0,
    delta_x_1, delta_y_1, delta_z_1,
    ...
    delta_x_N, delta_y_N, delta_z_N,
]
```

Apply correction:

```python
T_corr = T_M18.copy()
T_corr[:3, 3] += delta_xyz[t]
```

---

## 33. Local M18 Attachment Objective

Use:

```text
E =
    E_keep_M18
  + E_palm_handle_attachment
  + E_temporal_smooth_delta
```

---

### 33.1 Keep-M18 Residual

```python
E_keep_M18(t) =
||delta_xyz(t)||²
```

Purpose:

```text
prevent contact residual from pulling the mug too far away from visual alignment
```

---

### 33.2 Palm-Handle Attachment Residual

For each active attachment frame:

```python
grasp_world =
T_corr(t) * stable_grasp_local

E_attach(t) =
rho(
    ||grasp_world - palm_proxy_world(t)||²
)
```

Use only frames with:

```text
frame_mode == direct_grasp_anchor
or
frame_mode == keep_previous_grasp_anchor
```

Weight by confidence:

```python
w_attach_t =
base_w_attach
*
stable_grasp_conf
*
palm_conf
```

---

### 33.3 Temporal Smoothness

Use second-order smoothness on translation correction:

```python
acc_delta(t) =
delta(t + 1)
-
2 * delta(t)
+
delta(t - 1)

E_smooth =
||acc_delta(t)||²
```

Purpose:

```text
avoid frame-to-frame correction jitter
```

---

## 34. Local M18 Attachment Initial Weights

First run:

```text
w_keep = high
w_attach = medium
w_smooth = medium-high
sigma_attach = 0.05–0.08 m
```

Concrete starting values:

```text
w_keep = 10.0
w_attach = 2.0
w_smooth = 5.0
sigma_attach = 0.07
```

These are not final hyperparameters.

They are a conservative starting point.

---

## 35. Local M18 Attachment Output

Create:

```text
samples_known_object/02_mug/results/mug_m18_palm_attachment_experiment/
```

Files:

```text
mug_m18_attachment_trajectory.csv
mug_m18_attachment_residual_report.csv
mug_m18_attachment_summary.txt
```

`mug_m18_attachment_trajectory.csv` schema:

```csv
frame,
time,

tx_m18,
ty_m18,
tz_m18,

delta_x,
delta_y,
delta_z,

tx_attach,
ty_attach,
tz_attach,

qw,
qx,
qy,
qz,

frame_mode,
palm_conf,
stable_grasp_conf,

hand_to_grasp_dist_before_m,
hand_to_grasp_dist_after_m
```

---

# Part H: Diagnostic Rendering After Local M18 Attachment

## 36. Render Local M18 Attachment Debug Video

Create:

```text
samples_known_object/02_mug/results/renders/M18_palm_handle_attachment_debug/overlay.mp4
```

Render:

```text
1. local M18 mug pose
2. local M18 attachment-corrected mug pose
3. palm proxy
4. stable grasp anchor
5. attachment line before correction
6. attachment line after correction
7. frame mode
8. hand-to-grasp distance before/after
```

Color rules:

```text
green:
    improved after attachment correction

red:
    worse after attachment correction

blue:
    keep previous grasp

purple:
    rim contact no anchor update

gray:
    no attachment
```

---

# Part I: Ablation Plan

## 37. Experiments

Run the following experiments:

```text
A0:
    local M18 current baseline

A1:
    local M18 + palm proxy only
    no grasp state
    no optimization

A2:
    local M18 + palm proxy + direct anchor state
    no optimization

A3:
    local M18 attachment, direct anchors only

A4:
    local M18 attachment, direct + previous grasp anchors

A5:
    local M18 attachment, direct + previous grasp anchors
    + strong keep-M18 prior
```

---

## 38. Metrics

For each experiment, report:

```csv
experiment,
num_direct_anchor_frames,
num_keep_previous_frames,

mean_hand_to_grasp_dist_before_m,
mean_hand_to_grasp_dist_after_m,

max_hand_to_grasp_dist_before_m,
max_hand_to_grasp_dist_after_m,

mean_pose_delta_from_M18_m,
max_pose_delta_from_M18_m,

num_large_gaps_before,
num_large_gaps_after,

visible_overlay_quality,
hidden_segment_stability,
rim_contact_anchor_leakage
```

---

## 39. Success Criteria

Local M18 palm proxy is successful if:

```text
1. palm proxy is visually located on the palm in most visible grasp frames

2. direct_grasp_anchor frames are only created when palm and handle are close

3. rim contact does not overwrite the hand grasp anchor

4. hidden/drinking frames preserve the previous object-local grasp anchor

5. hand-to-grasp distance does not explode in hidden segments
```

Local M18 attachment experiment is successful if:

```text
1. hand-to-grasp distance decreases

2. pose delta from local M18 remains small

3. visual overlay quality does not degrade

4. mug handle does not flip or rotate unnaturally

5. hidden segment stability improves
```

---

# Part J: Minimal Implementation Order

## Step 1

Implement:

```text
scripts/known_object/mug/build_mug_palm_proxy.py
```

Output:

```text
results/hand_contact_proxy/mug_palm_proxy.csv
```

---

## Step 2

Render:

```text
results/renders/M18_hamer_palm_proxy_debug/overlay.mp4
```

Check whether palm proxy is actually on the palm.

---

## Step 3

Implement:

```text
scripts/known_object/mug/verify_mug_palm_handle_contact.py
```

Output:

```text
results/hand_contact_proxy/palm_handle_verification.csv
```

---

## Step 4

Implement:

```text
scripts/known_object/mug/build_mug_grasp_anchor_state.py
```

Output:

```text
results/mug_m18_grasp_anchor_state/mug_grasp_anchor_state.csv
```

---

## Step 5

Evaluate:

```text
results/mug_m18_grasp_anchor_state/hand_to_grasp_distance.csv
results/mug_m18_grasp_anchor_state/evaluation_summary.txt
```

If this fails, stop and fix local M18.HaMeR / palm proxy.

---

## Step 6

Run conservative local M18 attachment experiment:

```text
scripts/known_object/mug/run_mug_m18_translation_attachment.py
```

Only optimize translation correction.

---

## Step 7

Render local M18 attachment debug video and run ablation A0–A5.

---

# Part K: Out of Scope for This Stage

Do not do the following yet:

```text
full hand pose optimization
finger-level grasp reconstruction
full 6D mug pose correction
new object reconstruction
per-frame handle redetection
generic radius-free proxy replacement for mug
physics simulation
force estimation
```

These may be useful later, but they are not the current bottleneck.

The current bottleneck is:

```text
missing hand-side palm contact proxy
```

---

# Final Summary

The corrected mug plan is:

```text
1. Build palm proxy.
2. Verify palm proxy visually.
3. Gate object-side handle candidates using palm proximity.
4. Update stable grasp anchor only from confirmed palm-handle frames.
5. Preserve previous grasp anchor during hidden/drinking frames.
6. Evaluate hand-to-grasp distance before optimization.
7. Run translation-only local M18 attachment only after evaluation passes.
```

The central change is:

```text
The local M18 attachment experiment should not attach the mug to an undefined hand point.

It should attach the mug handle anchor to a stable palm proxy.
```
