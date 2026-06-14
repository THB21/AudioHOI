# Radius-Free Object Proxy Pipeline (English)

## 0. Goal and Constraints

This experimental line generalizes the known-radius ball pipeline into a generic object-proxy pipeline.

Hard constraints:

- Do not use a known ball radius.
- Do not use `radius_px -> depth`.
- Do not use the constraint “ball center to floor distance equals radius”.
- Object depth comes from DA3 depth priors and mask/proxy sampling.
- Contact detection provides candidate frames, body-part labels, support/floor states, and anchor frames for later optimization.

Current layout:

```text
scripts/shared/radius_free_proxy/
  stage0_preprocess/
  stage1_observation/
  stage2_contact_candidates/
  stage3_da3_init_optimization/
  stage4_anchor_refinement/
  stage5_render/
```

Full order:

```text
Stage0 preprocess
  video/audio/frames + SAM2 + CoTracker + GVHMR + DA3 + audio events
↓
Stage1 observation
  object_observations + object_proxy_observations
↓
Stage2 contact candidates
  object proxy + body part proxy + audio -> contact states/events
↓
Stage3 DA3 contact-aware init optimization
  first optimization: 2D ref + DA3 depth + support/contact soft terms
↓
Stage4 anchor refinement
  second optimization: human contact event frames as z anchors
↓
Stage5 render
  overlay / camera3d / side_yz, h264
```

## 1. Stage0 Preprocess

Code:

```text
scripts/shared/radius_free_proxy/stage0_preprocess/prepare_sample_inputs.py
scripts/shared/radius_free_proxy/stage0_preprocess/prepare_known_object_samples.py
scripts/shared/radius_free_proxy/stage0_preprocess/run_sam2_segmentation.py
scripts/shared/radius_free_proxy/stage0_preprocess/run_cotracker_object_mesh.py
scripts/shared/radius_free_proxy/stage0_preprocess/prepare_articraft_mug_proxy.py
scripts/shared/radius_free_proxy/stage0_preprocess/register_da3_scene_depth.py
scripts/shared/radius_free_proxy/stage0_preprocess/extract_da3_depth_priors.py
scripts/shared/radius_free_proxy/stage0_preprocess/align_audio_events.py
```

Responsibilities:

- `prepare_sample_inputs.py`: copy/normalize video, extract frames, extract `audio.wav`, optionally build audio events.
- `run_sam2_segmentation.py`: unified SAM2 video helper. It supports automatic GroundingDINO first-frame boxes and manual box/point prompts.
- `run_cotracker_object_mesh.py`: creates object mesh/boundary tracks from masks and tracked points.
- `prepare_articraft_mug_proxy.py`: optional mug-only Articraft handoff. It exports keyframes, Articraft prompts, and the pipeline-owned `proxy/mug_proxy.json`.
- GVHMR is consumed as `results/gvhmr/result.pkl` for body-part proxies.
- `register_da3_scene_depth.py`: registers DA3 scene depth maps.
- `extract_da3_depth_priors.py`: samples DA3 depth from object proxy locations.
- `align_audio_events.py`: audio-only event proposal generation; `audio_visual_alignment.csv` is no longer used.

Audio event generation uses `librosa` onset strength, RMS rise, local sharpness, and `scipy.signal.find_peaks`.

Outputs:

```text
<sample>/results/events/audio_events.csv
<sample>/results/segmentation/masks/
<sample>/results/tracking/object_mesh_tracks_test.csv
<sample>/results/gvhmr/result.pkl
<sample>/results/da3/scene_depth/
<sample>/results/da3/priors/
```


### Optional Stage0.5: Articraft Mug Proxy

Articraft is used only as a canonical geometry proxy generator for mug-like objects. It is not used for frame-by-frame tracking.

The intended flow is:

```text
key mug frame / clean mug image
↓
Articraft text-only or image-conditioned mug generation
↓
semantic parts: body / handle / rim / bottom
↓
pipeline-owned proxy/mug_proxy.json
↓
Stage1 tracks body center, Stage2 anchors hand to handle/side region
```

The radius-free pipeline must not depend directly on Articraft's internal asset format. Stage2 reads only:

```text
<sample>/proxy/mug_proxy.json
```

For `samples_known_object/02_mug`, prepare the handoff files with:

```bash
python scripts/shared/radius_free_proxy/stage0_preprocess/prepare_articraft_mug_proxy.py \
  --sample-dir samples_known_object/02_mug \
  --copy-keyframes
```

This creates:

```text
<sample>/keyframes/
<sample>/articraft/prompt_mug_proxy_text_only.txt
<sample>/articraft/prompt_mug_proxy_image_conditioned.txt
<sample>/proxy/mug_proxy.json
<sample>/annotations/
```

The clean mug image can be created by inpainting/removing the hand from a selected keyframe. The generated Articraft asset should be abstracted back into `mug_proxy.json`; the first version can be manually filled as a normalized cylinder-with-handle proxy.

Current mug handoff outputs also include:

```text
<sample>/annotations/001_contact_region_mask.png
<sample>/annotations/001_contact_region_preview.png
<sample>/annotations/001_contact_region_mask.json
<sample>/articraft/generated_record/mug_proxy_record.json
<sample>/results/stage1_mug_body_trajectory.csv
<sample>/results/stage2_mug_contact_test.csv
```

The first contact-region mask is intentionally derived from the human hand/contact side stored in `object_observations.csv`. It does not try to visually recognize the mug handle. For the current mug sample, this means the proxy marks the left-side handle/contact region. Once Articraft produces a real mug asset with a separate `handle` part, that generated handle can replace the approximate side box, while the downstream Stage2 interface remains the same.

Current usable Articraft/Codex record is the final embedded C-loop version:

```text
record_id: rec_edit-the-current-mug-model-to-fix-the-handle-bod_20260609_140343_925699_db515086
model: gpt-5.5 via --provider codex-cli
source images:
  samples_known_object/02_mug/keyframes/zoom/mug_zoom_reference_montage.png
  samples_known_object/02_mug/keyframes/zoom/001_mug_clean_zoom_handle.png
```

Intermediate pad/connector versions are not used as the final proxy. The current record keeps the handle as a single C-shaped tube whose endpoints embed directly into the body wall.

The generated model has semantic parts `body`, `handle`, `rim`, and `bottom`. Its copied canonical outputs live under:

```text
samples_known_object/02_mug/articraft/generated_record/rec_create-a-simple-3d-mug-proxy-for-contact-reasoni_20260609_124656_628392_db105e6a/
```

`mug_proxy.json` is now Articraft-backed: it stores metric dimensions from the generated model, marks the Articraft `handle` as the 3D contact part, and mirrors the canonical right-side handle to the observed left-side contact region in the video.

Stage2 copies the audio table it uses to:

```text
<sample>/results/contact_candidates_object_proxy/audio_events.csv
```

## 2. Stage1 Observation

Code:

```text
scripts/shared/radius_free_proxy/stage1_observation/build_object_observations.py
scripts/shared/radius_free_proxy/stage1_observation/build_object_proxy_observations.py
scripts/shared/radius_free_proxy/stage1_observation/object_proxy_observation_utils.py
```

Outputs:

```text
<sample>/results/object_observations/object_observations.csv
<sample>/results/object_proxy_observations/object_proxy_observations.csv
```

Important object proxy fields:

- `ref_u/ref_v`: 2D object reference proxy.
- `support_u/support_v`: smoothed support/bottom proxy.
- `support_v_raw`: raw bottom proxy; Stage2 floor peak uses this field.
- `contact_u/contact_v`: object-side contact region proxy. For the mug/Articraft path, this comes from `mug_proxy.json` `contact_region` plus the stable body bbox; it no longer follows per-frame visible handle pixels or nearest mesh points to the human part.
- `object_ref_depth_m`: DA3 depth at the reference proxy.
- `contact_proxy_depth_m`: DA3 depth at the contact proxy.
- `contact_depth_offset_m = contact_proxy_depth_m - object_ref_depth_m`.
- `active_label/active_part_u/active_part_v/active_part_z`: current likely human contact part.
- `object_motion_score`: 2D object proxy acceleration response.
- `audio_score`: audio support copied from audio events.

Radius-free means there is no radius estimation and no radius-based z reconstruction.

### Mug / Articraft Contact Point

For mug-like continuous grasp, Stage1 uses `ref_u/ref_v` only as the stable cup body center observation. `contact_u/contact_v` is a fixed object-side region:

```text
keyframe contact mask / observed side
+ mug_proxy.json contact_region(handle, left/right)
+ current frame body bbox
=> handle:left/right:canonical_contact_region
```

This no longer computes the contact point directly from per-frame noisy proxies or bbox side, and the hand no longer decides where the handle is. `mug_proxy.json` fixes the semantic region as the handle. The object-side contact region comes from the painted contact mask on a selected hand-holding keyframe (currently `annotations/001_contact_region_mask.json`). Stage1 converts that mask contact point into a body-bbox-normalized offset and propagates it through the video using the mug body center/bbox. Per-frame `handle_visible` can be used for diagnostics/visual checks, but not as contact ground truth. The hand is compared against this object-side contact trajectory in Stage2.


## 3. Stage2 Contact Candidates

Code:

```text
scripts/shared/radius_free_proxy/stage2_contact_candidates/run_contact_candidate_detection.py
scripts/shared/radius_free_proxy/stage2_contact_candidates/object_contact_candidate_utils.py
```

Outputs:

```text
<sample>/results/contact_candidates_object_proxy/audio_events.csv
<sample>/results/contact_candidates_object_proxy/anchor_contact_candidates.csv
<sample>/results/contact_candidates_object_proxy/floor_contact_candidates.csv
<sample>/results/contact_candidates_object_proxy/contact_state_frames.csv
<sample>/results/contact_candidates_object_proxy/contact_candidates_labeled.csv
<sample>/results/contact_candidates_object_proxy/contact_intervals.csv
```


### Mug Continuous Contact Mode

When `anchor_event_mode=continuous_state` and Stage1 emits `contact_proxy_name=handle:*:canonical_contact_region`, Stage2 skips the hand-to-object-boundary detector. It directly computes:

```text
d_left  = distance(left_hand_uv,  contact_u/contact_v)
d_right = distance(right_hand_uv, contact_u/contact_v)
active hand = smaller distance
anchor_score = Gaussian(distance, sigma=30px)
anchor_state = distance <= 48px and score >= 0.25
```

So mug grasp produces continuous `anchor_contact_event` frames rather than one peak per interval.

### Contact Part Policy

The human candidate part is selected from sample name/metadata/prompt:

```text
football / soccer / kick / feet -> feet-only
basketball / dribble / bounce / catch / hand -> hands-only
other objects -> active body proxy
```

This prevents basketball from briefly switching to foot due to a nearest-body-part glitch, while football only considers feet.

### Human Anchor Candidate

For hands/feet, the detector computes distances from left/right body proxies to the object boundary/contact proxy:

```text
min_gap = min(distance(left_part, object_boundary), distance(right_part, object_boundary))
proximity_score = Gaussian(min_gap, sigma=18px)
anchor_score = 0.55 * proximity_score + 0.30 * audio_support
candidate = local_min(min_gap, radius=2) and min_gap <= 28px
base_state = min_gap <= 38px and anchor_score >= 0.35
```

For impulse contacts, the final state uses a mainline-like interval gate:

```text
motion_gate = object_motion_score >= 0.80 or audio_support >= 0.20
anchor_state = bridge((min_gap <= 52px) and motion_gate, gap=2)
```

This prevents a stationary object near a body part from becoming an anchor after the action ends.


### How Audio Is Used in Contact Candidates

Stage2 no longer reads `audio_visual_alignment.csv`. It only reads:

```text
<sample>/results/events/audio_events.csv
```

It also copies the exact audio table used by contact detection to:

```text
<sample>/results/contact_candidates_object_proxy/audio_events.csv
```

Each `audio_events.csv` row contains:

```text
event, audio_time, audio_frame, peak, prominence, rms_rise, sharpness, audio_score
```

Stage2 converts sparse audio events into per-frame `audio_support`:

```text
for each audio event:
    audio_support[frame +/- 2] = max(audio_support, audio_score)
```

So each audio event is dilated by two frames on both sides and used as soft contact evidence.

Audio contributes to the human anchor score:

```text
anchor_score = 0.55 * proximity_score
             + 0.15 * object_response_score
             + 0.30 * audio_support
```

The current object response term is effectively zero, so in practice:

```text
anchor_score ≈ 0.55 * proximity_score + 0.30 * audio_support
```

Audio also helps open impulse-contact intervals:

```text
motion_gate = object_motion_score >= 0.80 or audio_support >= 0.20
anchor_state = (min_gap <= 52px) and motion_gate
```

Therefore audio does not create contacts by itself. It:

- raises anchor scores for frames already near a human part,
- opens a contact interval when object motion is weak but an impact sound exists,
- helps interval-peak selection choose a better contact frame.

The final `contact_candidates_labeled.csv` keeps:

```text
source_audio
```

This is the per-frame `audio_support` used for that candidate.

### Floor/Support Candidate

There is no radius-based floor constraint. For impulse scenes, floor events are detected from the object bottom proxy itself:

```text
object_support_peak_v = support_v_raw
support_enter = percentile(object_support_peak_v, 85)
support_soft = percentile(object_support_peak_v, 65)
proxy_floor_score = clip((support_v_raw - support_soft) / (support_enter - support_soft), 0, 1)
floor_state = local_max(support_v_raw, radius=2) and support_v_raw >= support_enter
```

Absolute support-plane gap is kept for diagnostics but is not the sole floor event criterion because the GVHMR floor proxy can be globally biased.

### Event Selection

`interval_peak` mode:

- Split frame-level `anchor_state` into intervals.
- For football/feet, select the frame with maximum `anchor_score` in each interval.
- For basketball/hand impulse contacts, select the frame with minimum `object_ref_v`, i.e. the highest image point.
- For floor intervals, select maximum `support_v_raw`.

Event output:

```text
anchor_event = anchor_state and frame == interval_peak and anchor_score >= 0.15
floor_event = floor_state and frame == floor_interval_peak
```

`continuous_state` is used for mug/hold/grasp scenes where contact persists instead of peaking.

## 4. Stage3 DA3 Contact-Aware Init Optimization

Code:

```text
scripts/shared/radius_free_proxy/stage3_da3_init_optimization/run_da3_init_optimization.py
```

Inputs:

```text
object_proxy_observations.csv
contact_state_frames.csv
gvhmr/result.pkl
```

Optimization variable per frame:

```text
X_t = (tx_t, ty_t, tz_t)
```

Initialization:

```text
X_init = backproject(ref_u, ref_v, object_ref_depth_m, K)
```

The optimizer uses `scipy.optimize.least_squares(..., loss='soft_l1', max_nfev=400)`.

Default weights:

```text
center_weight = 0.05
depth_weight = 0.3
support_weight = 10.0
contact_weight = 3.0
vel_weight = 0.0
z_vel_weight = 2.0
z_acc_weight = 0.50
xy_acc_weight = 0.08
```

Residual terms:

```text
E_center  : projected X_t matches ref_u/ref_v
E_depth   : tz_t follows object_ref_depth_m / object_depth_smooth
E_support : projected support point matches support_v
E_contact : contact state/score softly pulls object contact proxy to human part z
E_z_vel   : first-order z velocity smoothing
E_z_acc   : second-order z acceleration smoothing
E_xy_acc  : light x/y acceleration smoothing
```

Outputs:

```text
<sample>/results/pose6d_object_proxy_da3_init/object_pose6d_sharedcam_trajectory.csv
<sample>/results/pose6d_object_proxy_da3_init/object_pose6d_sharedcam_reprojection_comparison.csv
<sample>/results/pose6d_object_proxy_da3_init/object_pose6d_sharedcam_summary.txt
<sample>/results/pose6d_object_proxy_da3_init/support_geometry.json
```

## 5. Stage4 Anchor Refinement

Code:

```text
scripts/shared/radius_free_proxy/stage4_anchor_refinement/run_anchor_refinement.py
```

Inputs:

```text
pose6d_object_proxy_da3_init/object_pose6d_sharedcam_trajectory.csv
object_proxy_observations/object_proxy_observations.csv
contact_candidates_object_proxy/contact_state_frames.csv
contact_candidates_object_proxy/contact_candidates_labeled.csv
gvhmr/result.pkl
```

Anchor z construction:

```text
raw_offset = contact_depth_offset_m
if abs(raw_offset) <= max_contact_depth_offset_m (default 1.0m):
    offset_used = raw_offset
else:
    offset_used = 0

anchor_value = human_contact_part_z - offset_used
```

Both raw and used offsets are written:

```text
contact_depth_offset_m
contact_depth_offset_used_m
```

This guards against DA3/contact-proxy outliers such as 7m or 20m offsets pulling the trajectory to an invalid depth layer.

Default reference behavior:

```text
z_ref_mode = anchor_segment
outside_window_mode = boundary_constant
```

Meaning:

- Interpolate reference z between human anchor events.
- Hold the first anchor value before the first anchor.
- Hold the last anchor value after the last anchor.
- Avoid letting DA3 pull the contact window exterior back to a wrong depth layer.

Stage4 optimizes only z. Anchor frames are fixed to `anchor_value`; non-anchor z values are optimized.

Default weights:

```text
w_ref = 0.7
w_temp = 5.0
w_phys_xz = 1.25
w_phys_y = 1.5
gravity_mps2 = 9.81
```

Residuals:

```text
E_ref     : z_t follows z_ref_t
E_temp    : second-order z smoothness
E_phys_xz : in flight triplets, x/z second difference approaches 0
E_phys_y  : in flight triplets, y second difference approaches g * dt^2
```

Uses `least_squares(..., loss='soft_l1', max_nfev=400)`.

Outputs:

```text
<sample>/results/pose6d_object_proxy_anchor_refined/object_pose6d_sharedcam_contactphase_trajectory.csv
<sample>/results/pose6d_object_proxy_anchor_refined/object_pose6d_sharedcam_contactphase_reprojection_comparison.csv
<sample>/results/pose6d_object_proxy_anchor_refined/object_pose6d_sharedcam_contactphase_summary.txt
```

## 6. Stage5 Render

Code:

```text
scripts/shared/radius_free_proxy/stage5_render/render_pose6d_scene.py
```

Default input:

```text
<sample>/results/pose6d_object_proxy_anchor_refined/object_pose6d_sharedcam_contactphase_trajectory.csv
```

Outputs:

```text
<sample>/results/renders/pose6d_object_proxy_anchor_refined/ball/overlay.mp4
<sample>/results/renders/pose6d_object_proxy_anchor_refined/ball/camera3d.mp4
<sample>/results/renders/pose6d_object_proxy_anchor_refined/ball/side_yz.mp4
<sample>/results/renders/pose6d_object_proxy_anchor_refined/with_human/overlay.mp4
<sample>/results/renders/pose6d_object_proxy_anchor_refined/with_human/camera3d.mp4
<sample>/results/renders/pose6d_object_proxy_anchor_refined/with_human/side_yz.mp4
```

Render rules:

- h264 is the default video codec.
- Candidate human parts are shown throughout the sequence.
- The active candidate human part is circled in red only on contact frames.
- Floor/support frames still show the current candidate human part; floor itself is not treated as a human part.
- Mug-like objects are rendered as object proxies instead of always as spheres.

## 7. Recommended Run Order

```bash
python scripts/shared/radius_free_proxy/stage0_preprocess/prepare_known_object_samples.py
python scripts/shared/radius_free_proxy/stage0_preprocess/run_sam2_segmentation.py --sample-dir <sample>
python scripts/shared/radius_free_proxy/stage0_preprocess/run_cotracker_object_mesh.py --sample-dir <sample>
python scripts/shared/radius_free_proxy/stage0_preprocess/register_da3_scene_depth.py --sample-dir <sample> --source-depth-dir <da3_depth_export>
python scripts/shared/radius_free_proxy/stage0_preprocess/extract_da3_depth_priors.py --sample-dir <sample>
python scripts/shared/radius_free_proxy/stage0_preprocess/align_audio_events.py --sample-dir <sample> --fps <fps>
python scripts/shared/radius_free_proxy/stage1_observation/build_object_observations.py --sample-dir <sample>
python scripts/shared/radius_free_proxy/stage1_observation/build_object_proxy_observations.py --sample-dir <sample>
python scripts/shared/radius_free_proxy/stage2_contact_candidates/run_contact_candidate_detection.py --sample-dir <sample>
python scripts/shared/radius_free_proxy/stage3_da3_init_optimization/run_da3_init_optimization.py --sample-dir <sample>
python scripts/shared/radius_free_proxy/stage4_anchor_refinement/run_anchor_refinement.py --sample-dir <sample>
python scripts/shared/radius_free_proxy/stage5_render/render_pose6d_scene.py --sample-dir <sample> --with-human
```
