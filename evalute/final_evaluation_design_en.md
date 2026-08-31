# Final Evaluation Design EN

This document defines the next final-product evaluation protocol for AudioHOI. It is based on the current `benchmark_vlm_qwen` outputs, the human/audio/HOI results, the meeting transcript, and the HOI-PAGE-style part-level evaluation perspective.

The current complete final-result evaluation scope contains two ball cases only:

- `basketball`
- `football`

These two cases have aligned final videos, source videos, object 6DoF trajectories,
human/contact artifacts, temporal/audio artifacts, and paired gate-trace evidence.
Mug is intentionally excluded for now; chair and stick are not yet published as
complete deliverable videos under `final_result/videos/`. This document still keeps
the mug/chair/stick metric design because they are the next expansion targets, not
because they currently have complete final-result evaluation outputs.

The core rule is:

```text
Compute what can be computed. Use VLM only for perceptual ambiguity. Use LLM for artifact auditing and failure localization.
```

## 1. Why the evaluation needs to change

The current object-level final summary can already check SE3 pose, proxy contact, jump count, and static drift. However, it is not sufficient as the final evaluation:

1. Object-level penetration/floating proxies are not the same as true human-object geometry penetration.
2. Overlay should not primarily rely on VLM. If masks, rendered silhouettes, and reprojections are available, we should compute overlay metrics directly.
3. The benchmark must prove component contribution. If `with VLM`, `without VLM`, `with audio`, and `without audio` point to the same result directory or produce no optimizer difference, the ablation is not meaningful.

The new protocol has three layers:

```text
A. Final result hard metrics
B. VLM/LLM judge and audit
C. Ablation benchmark
```

Hard metrics are the primary evidence. VLM/LLM provide interpretation and edge-case review.

## 2. Relation to HOI-PAGE

HOI-PAGE emphasizes part-level affordance reasoning: object parts are decomposed, human-object part relations are represented, and object motion is optimized with part-level contact constraints. Its project page describes a pipeline that decomposes objects into geometric parts, extracts reference video constraints, and optimizes motion while enforcing part-level contacts.

We adopt the following ideas:

- part-level human-object relations instead of only object centers;
- fitting, contact, penetration, and smoothness families of constraints;
- LLM/VLM as semantic/audit support, not continuous pose solvers;
- evaluation of whether the correct human part contacts the correct object part.

We do not copy the setting directly:

- HOI-PAGE is text + known 3D objects generation; AudioHOI is video-conditioned reconstruction;
- AudioHOI must evaluate video overlay and reprojection;
- AudioHOI has audio timing, so we need audio-windowed contact and event-vs-flight acceleration metrics;
- AudioHOI must evaluate both object 6DoF and human-object interaction geometry.

## 3. Evaluation targets

Object result directory:

```text
samples_known_object/<case>/results/benchmark_vlm_qwen/
```

Human / HOI layer outputs:

```text
samples_known_object/<case>/results/gvhmr or human_gvhmr
samples_known_object/<case>/results/hands or human_hands
samples_known_object/<case>/results/contact_refine or human_contact_refine
samples_known_object/<case>/results/hoi_eval/hoi_interaction_metrics.json
samples_known_object/<case>/results/renders/*human_full_scene_3d*
```

Cross-case summaries:

```text
final_result/evaluation/
```

## 4. Layer 1: Object 6DoF and video fit

### 4.1 SE3 completeness

Input: `object_pose.csv`

Required fields:

```text
frame,time,tx,ty,tz,qw,qx,qy,qz
```

Metrics:

```text
se3_valid = required columns exist
translation_valid_rate = finite tx/ty/tz frames / total frames
rotation_valid_rate = finite normalized quaternion frames / total frames
```

Ball rotations are weakly observable, but the SE3 schema still exists. Mug, chair, and stick rotations must affect rendering and semantic part orientation.

### 4.2 Overlay / reprojection metrics

Overlay must be primarily a hard metric, not a VLM score.

Inputs:

- SAM2/object mask;
- rendered object mask or silhouette;
- object observations, correspondences, line correspondences;
- rendered overlay frames.

Metric priority:

1. **Silhouette IoU**

```text
silhouette_iou_t = |M_render_t ∩ M_sam2_t| / |M_render_t ∪ M_sam2_t|
overlay_iou = mean_t(silhouette_iou_t)
```

Implementation rule:

```text
If rendered object masks already exist, use them directly.
If rendered object masks are missing, the final evaluator must generate evaluation-only render masks.
Output directory: samples_known_object/<case>/results/<result>/evaluation/render_masks/
These masks are evaluation artifacts only. They do not replace the human-readable overlay video and do not write back to pose.
```

The current implementation reports two source types:

```text
generated_eval_proxy_render_mask_iou:
  lightweight proxy masks generated from pose plus observation radius/line/bbox.

generated_eval_full_geometry_mask_iou:
  full geometry masks rasterized from URDF/Articraft geometry and object_pose.csv.
```

Basketball and football still use proxy circle masks. Mug, chair, and stick can generate full geometry masks when a parseable URDF exists. The protocol treats rendered masks as required evaluation artifacts instead of relying on VLM judgment.

2. **Reprojection Chamfer distance**

```text
D_chamfer = 0.5 mean_{p in rendered edge} min_q ||p-q||
          + 0.5 mean_{q in observed edge} min_p ||q-p||
```

3. **Line alignment** for stick-like objects

```text
angle_error_t = acos(|dot(axis_render_2d, axis_observed_2d)|)
center_error_px_t = ||center_render_2d - center_observed_2d||
endpoint_error_px_t = endpoint reprojection error when visible
```

4. **Coverage / false coverage**

```text
coverage = |M_render ∩ M_obs| / |M_obs|
false_coverage = |M_render - M_obs| / |M_render|
```

VLM is only used when masks are unreliable, occlusion makes IoU unfair, or the question is perceptual acceptability.

The final table must separate:

```text
Overlay Hard Score
Overlay VLM Judge
Overlay Conflict Flag
```

## 5. Layer 2: Human and object parts

Following the part-level idea, the evaluator must know which human part and which object part are involved.

### 5.1 Human part coverage

Standard parts:

```text
left_hand,right_hand,left_foot,right_foot,torso,hip,back,head,mouth
```

Artifacts:

```text
evaluation/human_parts.csv
evaluation/human_part_points.csv
evaluation/human_part_metrics.csv
```

Implemented now:

```text
evaluation/human_parts.csv
evaluation/object_parts.csv
evaluation/object_part_vocab_map.csv
evaluation/part_metrics.csv
```

`human_parts.csv` records whether each standard human part has GVHMR, hand, or contact-pair evidence. `object_parts.csv` uses canonical object-part names for metrics and preserves the original evidence names in `raw_parts`. `object_part_vocab_map.csv` records how each raw part maps to a canonical part and whether it came from case config, object surface points, or HOI contact pairs.

This separates metric vocabulary from raw evidence vocabulary:

| Raw part examples | Canonical metric part |
| --- | --- |
| `handle`, `handle_loop` | `handle` |
| `cup_body`, `body_shell`, `body_shell_or_occluded_handle_region` | `cup_body` |
| `rim`, `rim_ring` | `rim` |
| `front_leg`, `rear_leg` | `legs` |
| `backrest`, `backrest_board` | `back` |
| `main_body` | `shaft` |
| `ball_boundary` | `surface` |
| `ball_bottom`, `floor_support` | `support_region` |

Therefore, `object_part_contact_coverage` is not a raw string-match score. It measures how many normalized semantic object-part slots have surface or contact evidence.

Metrics:

```text
part_available_rate(part) = valid frames / total frames
part_smoothness(part) = mean ||x_t - 0.5(x_{t-1}+x_{t+1})||
part_confidence(part) = detector/model confidence if available
```

### 5.2 Object part coverage

Object parts:

| Case | Object parts |
| --- | --- |
| basketball | center, surface, support_region |
| football | center, surface, support_region |
| mug | cup_body, handle, rim, bottom |
| chair | legs, seat, back, top_rail, stretcher, hole, feet |
| stick | shaft, grip_region, support_region |

Artifacts:

```text
evaluation/object_parts.csv
evaluation/object_part_surface_points.csv
evaluation/object_part_metrics.csv
```

Metrics:

```text
object_part_available_rate
object_part_overlay_score
object_part_semantic_consistency
asymmetric_rotation_plausibility
```

## 6. Layer 3: HOI contact metrics

Contact is evaluated as part-pair contact, not as a single scalar proxy.

Artifacts:

```text
evaluation/hoi_contact_pairs.csv
evaluation/hoi_contact_intervals.csv
evaluation/hoi_contact_metrics.csv
```

Implemented artifact sources:

```text
object_contact_points.csv / contact_candidates.csv:
  frame, human_part, human_side, object_part, contact_active, contact_confidence, contact_u/v.

anchor_state.csv:
  contact_persistent, anchor_update_allowed, pose_anchor_allowed, anchor_action.

hoi_interaction_metrics.json:
  Aggregate human/audio/HOI metrics, such as contact_frame_ratio, contact_gap_mm, and part_correct_ratio.
```

`hoi_contact_pairs.csv` is now the standard part-pair audit table. If an object family does not expose stable/observed local coordinates, `contact_anchor_drift_mean` remains blank. That means the evidence is missing, not that the contact is proven stable.

Schema:

```text
frame,human_part,object_part,expected,observed,persistent,rel_static,
min_distance_m,surface_gap_m,penetration_depth_m,contact_confidence,
contact_state,source
```

Metrics:

```text
contact_frame_ratio = observed-contact frames / valid frames
contact_interval_recall = observed frames inside expected intervals / expected interval frames
contact_gap = |distance(human_part, object_surface) - target_gap|
contact_proxy = exp(-contact_gap_mm / 50)
part_correct_ratio = correct closest human/object part over expected-contact frames
contact_drift = std of object-local human contact point during persistent contact
switch_accuracy = correct contact-state transitions / expected transitions
```

`contact_proxy` is a distance-derived proxy score. Higher is better. It does not replace ground-truth Contact F1: when manual contact labels exist, Contact F1 should be reported separately; without labels, `contact_proxy` and the raw `contact_gap_mm` are the auditable proxy.

## 7. Layer 4: penetration, floating, and support tradeoff

Penetration and floating must be evaluated together. A hand far away from the object has zero penetration but failed contact; a hand deep inside the object has good contact distance but failed physics.

### 7.1 Geometry penetration

Using SDF or sphere/capsule distance:

```text
sdf(v) > 0 outside
sdf(v) = 0 surface
sdf(v) < 0 inside
penetration_depth(v) = max(0, -sdf(v))
penetration_frame_ratio = frames with any depth > eps / valid frames
penetration_vertex_ratio = penetrating checked points / checked points
penetration_depth_mean = mean positive penetration depth
penetration_depth_max = max positive penetration depth
```

### 7.2 Floating / contact gap

```text
floating_gap_t = max(0, distance_to_surface_t - allowed_contact_band)
floating_rate = expected-contact frames with floating gap / expected-contact frames
```

### 7.3 Contact-aware physical score

The meeting discussion explicitly mentioned the tradeoff between penetration and floating. This must be part of the final interpretation. Reducing penetration alone is not sufficient because moving the object or hand away can make penetration zero. Reducing floating alone is also not sufficient because pushing the human body into the object can reduce surface distance while creating illegal penetration. The final physical score must combine expected contact, surface gap, penetration depth, and support state.

```text
contact_success_t = expected_contact_t and surface_gap_t within allowed band
illegal_penetration_t = penetration_depth_t > allowed_shallow_contact_depth
floating_failure_t = expected_contact_t and surface_gap_t > allowed_contact_band

tradeoff_score = contact_success_rate
               * (1 - penetration_frame_ratio)
               * (1 - floating_rate)
```

Interpretation rule:

```text
low penetration + high floating  = likely separated object/body, not a good result
low floating + high penetration  = likely contact obtained by interpenetration, not a good result
low penetration + low floating + correct contact interval = physically plausible interaction
```

```text
physical_contact_score = exp(- contact_gap_mean / sigma_gap)
non_penetration_score = exp(- penetration_depth_mean / sigma_pen)
tradeoff_score = sqrt(physical_contact_score * non_penetration_score)
```

This exposes both cheating modes: floating away to avoid penetration, and penetrating deeply to satisfy contact.

Severity must be part-aware: shallow chair back contact is not the same as a hand going through a ball.

## 8. Layer 5: temporal and audio-aware metrics

Temporal metrics are numerical first.

Object translation:

```text
v_t = ||T_t - T_{t-1}||
a_t = ||T_t - 2T_{t-1} + T_{t-2}||
j_t = ||T_t - 3T_{t-1} + 3T_{t-2} - T_{t-3}||
```

Rotation:

```text
omega_t = angle(q_{t-1}^{-1} q_t)
alpha_t = |omega_t - omega_{t-1}|
```

Audio-aware split:

```text
event_window = frames within ±k of audio/contact event
accel_at_events = mean acceleration inside event windows
accel_in_flight = mean acceleration outside event windows
```

A bounce/kick spike near an audio/contact event is plausible. A spike away from events is likely an artifact.

Current implementation now materializes the motion-regime-aware layer instead of relying on one coarse `jump_count`:

- `temporal_plausibility_metrics.csv/json`
- `translation_spike_count`
- `rotation_spike_count`
- `event_aligned_spike_count`
- `non_event_spike_count`
- `high_speed_recall`
- `oversmooth_rate`
- `static_tail_drift_m`
- `temporal_failure_intervals`

Spike thresholds are sequence-adaptive:

```text
threshold = max(floor, median(values) + 3 * MAD(values), percentile_95(values))
```

Event windows are read from `audio_contact_csv` / `contact_records.csv`. Spikes inside those windows are treated as possible impact motion; spikes outside them are reported as likely temporal artifacts.

Implementation:

- `scripts/shared/generic_contact_pipeline/core/evaluation/final_hoi/temporal_plausibility_metrics.py`
- `final_result/evaluation/<case>/evaluation/temporal_plausibility_metrics.csv`

## 9. VLM / LLM final judge

### 9.1 VLM

VLM only checks selected frames/windows:

- hard overlay and visual perception conflict;
- occlusion makes IoU unfair;
- small penetration may be visually acceptable;
- semantic part contact is ambiguous;
- final render is visually implausible.

VLM output schema:

```json
{
  "query_type": "perceptual_contact_or_overlay_check",
  "question": "Does the rendered object visually overlap the real target object, ignoring occluded regions?",
  "allowed_answers": ["pass", "unclear", "fail"],
  "score_1_to_5": 4,
  "short_reason": "...",
  "failure_stage_hint": "observation|contact|pose|optimizer|render|human_refine|unclear"
}
```

### 9.2 LLM

LLM reads CSV/JSON artifacts and explains consistency:

- gate timeline;
- optimizer decisions;
- metric failures;
- ablation deltas;
- failure stage.

LLM does not compute pose.

### 9.3 Implemented QA artifacts

`run_final_hoi_evaluator.py` now refreshes both hard metrics and QA audit artifacts. Standard location:

```text
samples_known_object/<case>/results/<result_name>/vlm_trace/06_evaluation/
```

Implemented files:

```text
pipeline_qa_summary.csv
pipeline_qa_summary.json
pipeline_qa_summary.md
vlm_eval_queries.csv
vlm_eval_raw_responses.jsonl
vlm_eval_parsed_scores.csv
vlm_eval_summary.json
llm_eval_summary.md
qa_audit_report.html
```

Important fields:

```text
stage
frame
evidence_path
question
raw_answer
visibility
contact_correctness
support_consistency
penetration_absence
temporal_plausibility
overall_plausibility
failure_stage_hint
affected_constraint
changed_optimizer_behavior
```

The current default source is a `metric-grounded dry-run final judge`: it uses hard metrics and existing trace artifacts to emit the fixed QA schema, with `changed_optimizer_behavior=0`. This does not pretend that Qwen has re-judged the result. It fixes the evaluation interface so every conclusion is traceable to evidence, question, raw answer, and parsed score. A real Qwen final judge can later replace the raw answer source while preserving the same schema.

`pipeline_qa_summary.*` and `vlm_eval_*` have different roles:

- `pipeline_qa_summary.*` aggregates VLM/LLM questions, answers, gates, affected constraints, and optimizer-effect flags that happened inside the pipeline stages;
- `vlm_eval_*` contains final-evaluator representative-interval questions over the final result;
- neither layer directly generates pose.

## 10. Final tables

Human-readable table:

The human-readable table intentionally keeps only six columns for quick review. Audio, VLM, LLM, and anchor ablation differences are reported in `final_result/evaluation/ablation/ablation_table.csv` and `ablation_report.md`, not in the final human-readable table.

| Case | Object 6DoF | Visual Overlay | Contact/Anchor | Physical | Temporal |
| --- | --- | --- | --- | --- | --- |

Current artifact:

```text
final_result/evaluation/final_evaluation_human_readable.md
```

Detailed CSV columns:

```text
case,result_name,n_frames,se3_valid,translation_valid_rate,rotation_valid_rate,
overlay_iou,overlay_chamfer_px,line_angle_error_deg,
contact_frame_ratio,contact_gap_mm,part_correct_ratio,contact_drift_mm,
penetration_frame_ratio,penetration_depth_mean_mm,penetration_depth_max_mm,
floating_rate,tradeoff_score,object_jerk,rotation_jerk,static_tail_drift_m,
high_speed_recall,contact_ratio_audio_windows,accel_at_events,accel_in_flight,
vlm_overlay_judge,vlm_contact_judge,llm_failure_stage,final_pass
```

## 11. Current human/audio/HOI outputs

The current `hoi_summary` already provides:

- penetration frame ratio and max depth;
- contact frame ratio and contact gap;
- part correctness;
- audio-window contact ratio;
- event-vs-flight acceleration;
- object jerk;
- grasp stability / MDev* when available.

The next implementation should merge these into the unified final evaluator rather than treat them as a separate table.

### 11.1 Current handoff artifact gap and TODO

The current results are connected to the upstream object/contact/audio pipeline. The checked artifacts show:

- `pose6d_object_proxy_anchor_refined/object_pose6d_sharedcam_contactphase_trajectory.csv`
  keeps the final object SE3 trajectory, support proxy, contact frame, and audio-contact frame fields;
- `human_audio_semantics/contact_records.csv` keeps audio/visual contact events, contact target, contact state, and manipulation weights;
- `human_audio_semantics/body_surface_contacts.csv` keeps body-surface contact points, contact part/region, and surface distance;
- `benchmark_vlm_qwen` keeps gate timeline, optimizer decisions, pre-smooth pose, and residuals.

However, these artifacts are still not enough for the final evaluator to compute three hard-metric families:

| Missing metric | Why it cannot be computed as a hard metric now | Required artifact |
| --- | --- | --- |
| `anchor_drift` | Basketball/football `anchor_state.csv` contains anchor gates, but `stable_local_x/y/z/s` and `observed_local_x/y/z/s` are empty. Ball cases currently use surface-gap / center-depth anchors rather than mug/stick-style object-local stable grasp points. | `final_anchor_state.csv` |
| `static_tail_drift` | The final result does not mark which interval should be static. Basketball/football also may not have a semantically valid static tail, so the evaluator must not automatically treat the last frames as static. | `final_motion_intervals.csv` |
| `floating_rate` / support gap | Final HOI metrics include penetration and contact surface distance, but not final 3D object-to-floor/support-surface gap. Pipeline traces have 2D/proxy fields such as `support_gap_px/floor_v`, but those must not be presented as final 3D floating hard metrics. | `final_support_state.csv` |

Therefore the human-readable table moves these fields into Evidence Notes instead of displaying `n/a` in the main cells. This is not simply an unfinished run; it is a final-result handoff contract gap.

Next TODO:

```text
final_anchor_state.csv
  frame,time,human_part,object_part,anchor_type,
  stable_local_x,stable_local_y,stable_local_z,stable_local_s,
  observed_local_x,observed_local_y,observed_local_z,observed_local_s,
  surface_gap_m,anchor_drift_m,anchor_update_allowed,pose_anchor_allowed,source

final_support_state.csv
  frame,time,support_type,support_part,
  object_bottom_x,object_bottom_y,object_bottom_z,
  support_surface_x,support_surface_y,support_surface_z,
  support_gap_m,floating_flag,penetration_flag,support_confidence,source

final_motion_intervals.csv
  interval_id,start_frame,end_frame,
  motion_regime,expected_static,expected_high_speed,
  contact_required,audio_event_aligned,reason,source
```

Once these artifacts exist, the evaluator can upgrade the currently unavailable fields into:

- `contact_anchor_drift_mean/max`
- `floating_rate`
- `support_consistency`
- `static_tail_drift_m`
- `static_rotation_drift_rad`

## 12. Acceptance criteria

1. All five cases produce detailed final evaluation.
2. Overlay has hard metrics and is not VLM-only.
3. Contact, penetration, and floating are co-reported to show the tradeoff.
4. VLM/LLM conclusions are traceable to evidence, question, raw answer, and parsed score.
5. Ablation methods map to real different result directories.
6. The report can answer whether audio, VLM, contact anchors, and smoothness actually help.
