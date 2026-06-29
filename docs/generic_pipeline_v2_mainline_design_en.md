# Generic Pipeline v2 Mainline Design

This document describes the `generic_pipeline_v2_llm_vlm_gate` mainline: inputs and outputs, stage design, losses/energy terms, LLM/VLM roles, case-specific logic for the four solved objects, and validation.

## 1. Goal

The new mainline unifies the four solved cases, basketball, football, mug, and chair, into one modular pipeline:

```text
case profile / config
+ reusable capability components
+ stage runner
```

Principles:

- A new object should first add a config, not a full object-specific runner.
- A new component is added only when a genuinely new capability is needed.
- LLM/VLM never solve continuous pose directly.
- The optimizer is the only continuous solver for 2D-to-3D/6D, depth, contact, temporal smoothing, and static/freeze constraints.

Mainline result directories:

```text
samples_known_object/<case>/results/generic_pipeline_v2_llm_vlm_gate/
samples_known_object/<case>/results/renders/generic_pipeline_v2_llm_vlm_gate/
```

Main command:

```bash
python scripts/shared/generic_contact_pipeline/run_pipeline.py \
  --case all \
  --result-name generic_pipeline_v2_llm_vlm_gate \
  --from-stage stage-1 \
  --to-stage stage7 \
  --llm-mode mistral \
  --vlm-mode dry-run
```

Real VLM smoke example:

```bash
/home/yang/miniconda3/envs/qwen-vl/bin/python \
  scripts/shared/generic_contact_pipeline/stages/stage_vlm_qwen.py \
  --case chair \
  --stage stage2 \
  --result-name generic_pipeline_v2_llm_vlm_gate \
  --limit 2 \
  --no-refresh-queries
```

## 2. Code Layout

```text
scripts/shared/generic_contact_pipeline/
  run_pipeline.py

  configs/
    cases/*.yaml
    llm_provider.yaml
    vlm_provider.yaml

  core/
    config.py
    schema.py
    camera.py
    hoi_profile.py
    llm_provider.py
    vlm.py
    vlm_gates.py
    loss_analysis.py
    compare.py

  components/
    contact/
    geometry/
    observation/
    pose/
    refinement/
    render/

  stages/
    stage_minus1_llm_prior.py
    stage0_preprocess.py
    stage1_observation.py
    stage2_contact.py
    stage3_initial_pose.py
    stage4_contact_refine.py
    stage5_render.py
    stage6_compare.py
    stage_loss_analysis.py
    stage_ablation.py
    stage_vlm_qwen.py
    stage_vlm_verify.py
```

## 3. Stage Design

### Stage -1: LLM Semantic HOI Prior

Inputs:

- case yaml
- `samples_known_object/<case>/metadata.json`
- original video-generation prompt
- Articraft prompt / record preview
- seed profile: `samples_known_object/<case>/annotations/hoi_profile_seed.json`

Outputs:

```text
hoi_profile.json
hoi_profile_resolved.yaml
prompt_context.json
llm_prior_trace.json
```

Purpose:

- Mistral generates only a discrete HOI semantic prior.
- The output contains object parts, human parts, interaction edges, support priors, motion priors, and VLM query policy.
- It must not output pose, coordinates, loss weights, or SE(3) corrections.

Fallback:

- If Mistral returns invalid JSON or fails schema validation, the pipeline falls back to the seed profile and records that in the trace.

### Stage 0: Preprocess Manifest

Inputs:

- SAM2 masks
- CoTracker tracks
- DA3 depth
- audio events
- GVHMR human pose
- camera intrinsics / `K_fullimg`
- mesh/proxy/URDF availability

Outputs:

```text
stage0_inputs_manifest.json
stage0_metrics.json
```

Stage0 registers and validates inputs. It does not perform object-specific pose optimization.

### Stage 1: 2D Observation

Unified outputs:

```text
object_observations.csv
object_local_points.csv
object_local_segments.csv
stage1_metrics.json
```

Observation component by profile:

| case | observation model | observations |
|---|---|---|
| basketball | `mask_track_center` | center, boundary, bottom support, bbox |
| football | `mask_track_center` | center, boundary, bottom support, bbox |
| mug | `rigid_body_plus_parts` | cup body, rim, bottom, handle visibility/contact mesh points |
| chair | `semantic_graph_tracks` | top rail, hole, seat, front/rear legs, stretchers, feet |

### Stage 2: Contact Candidates

Unified outputs:

```text
contact_candidates.csv
contact_state_frames.csv
stage2_metrics.json
```

Contact policy by profile:

| case | contact policy | logic |
|---|---|---|
| basketball | `hand_floor` | hand-ball and ball-floor |
| football | `foot_floor` | foot-ball and ball-floor |
| mug | `palm_handle_rim_body` | palm-handle/body, mouth-rim, table release |
| chair | `two_hand_toprail_endpoint` | left palm -> right endpoint, right palm -> left endpoint |

### Stage 3: Initial Pose

Unified outputs:

```text
object_pose_init.csv
stage3_metrics.json
```

Pose model:

| case | pose model | variables |
|---|---|---|
| basketball | `translation3` | `tx, ty, tz` |
| football | `translation3` | `tx, ty, tz` |
| mug | `rigid6_plus_phase` | `rx, ry, rz, tx, ty, tz, handle_phase` |
| chair | `semantic_graph_6d` | `rx, ry, rz, tx, ty, tz` |

### Stage 4: Contact / Depth / SE(3) Refinement

Unified outputs:

```text
object_pose.csv
object_contact_points.csv
object_phase.csv
stage4_metrics.json
```

Purpose:

- Contact candidates and VLM gates decide whether contact residuals are active.
- The stage applies depth anchors, XY backprojection, stable grasp, small SE(3), anchor propagation, and freeze.
- It keeps 2D overlay as a strong constraint while allowing necessary small 3D corrections.

### Stage 5: Render

Six standard videos:

```text
object_only/overlay.mp4
object_only/camera3d.mp4
object_only/side_yz.mp4
with_human/overlay.mp4
with_human/camera3d.mp4
with_human/side_yz.mp4
```

Rendering policy:

- basketball / football: proxy sphere render, automatic H264.
- mug: overlay and camera3d use the real Articraft mesh; side_yz uses an Articraft-ratio compact diagnostic model.
- chair: camera3d and overlay use the original URDF solid mesh; diagnostic views are standardized by the renderer.

### Stage 6: Baseline Comparison

Outputs:

```text
stage6_compare_report.json
migration_audit.csv
migration_audit.json
```

Checks:

- required CSVs exist and are non-empty
- frame count matches the solved baseline
- pose delta gate passes
- phase/event gate passes
- six render videos exist and pass codec/frame checks
- chair also passes semantic/contact/freeze quality gates

### Stage 6.5: LLM CSV / Data Audit

Stage 6.5 is a symbolic audit over tables and logs. It does not inspect images and does not perform continuous optimization. It reads the CSV/JSON/metrics produced by the pipeline and checks whether the stages are internally consistent.

Inputs:

```text
object_observations.csv
contact_candidates.csv
contact_state_frames.csv
object_pose_init.csv
object_pose.csv
object_contact_points.csv
object_phase.csv
stage3_metrics.json
stage4_metrics.json
stage6_compare_report.json
vlm_gates.csv
loss_analysis/loss_summary.json
```

Outputs:

```text
llm_csv_audit_queries.json
llm_csv_audit_results.json
llm_csv_audit_summary.md
llm_csv_audit_failures.csv
```

Audit checks:

- Schema completeness: for example, chair should contain left/right endpoint contacts, mug should contain handle phase, and ball cases should contain contact/floor fields.
- Stage consistency: for example, Stage2 may contain contact while Stage4 contains no object contact points, or a VLM-rejected frame may still activate `E_contact`.
- Object-specific rule violations:
  - basketball / football: unstable depth after contact, drifting floor support.
  - mug: handle jumps during drinking, pose drift after table release.
  - chair: left/right palm endpoint swap, missing freeze/interp before and after contact interval.
- Failure range summaries: largest contact gap, rotation jumps, static drift, and missing-contact frame ranges.

The LLM CSV audit may only output discrete labels and explanations:

```text
pass
schema_missing
stage_inconsistent
contact_empty
contact_side_swapped
static_drift
rotation_jump
depth_outlier
vlm_gate_ignored
unclear
```

Forbidden:

- no `tx/ty/tz/rx/ry/rz` corrections
- no loss weights
- no direct rewrite of `object_pose.csv`
- no conversion from natural-language explanation into continuous residuals

### Stage 7: Loss / Residual Logging

Outputs:

```text
loss_analysis/per_frame_residuals.csv
loss_analysis/loss_summary.json
```

Per-frame fields:

```text
E_total
E_2d
E_depth
E_visual / E_mask
E_contact
E_audio / E_support
E_smooth / E_temporal
E_static
E_penetration
E_prior / E_reg
vlm_contact_gate
vlm_anchor_gate
contact_active
static_active
failure_label
```

## 4. Energy / Loss Design

Unified form:

```text
E_total =
  w_2d      * E_2d_projection
+ w_depth   * E_depth_order_or_metric
+ w_contact * E_contact
+ w_smooth  * E_temporal_smooth
+ w_static  * E_static_freeze
+ w_pen     * E_penetration_or_floor_violation
+ w_prior   * E_pose_prior
```

Stage3 initialization:

```text
E_stage3 =
  w_2d    * E_2d_projection
+ w_depth * E_depth_prior
+ w_mask  * E_mask_bbox
+ w_geom  * E_geometry_consistency
+ w_smooth* E_temporal
```

Key residuals:

- `E_2d_projection`: projected centers, keypoints, or semantic lines match 2D observations.
- `E_depth_order_or_metric`: DA3 or anchor depth provides metric/depth-order constraints.
- `E_contact`: 3D gap between human palm/foot/mouth/table and the object contact point.
- `E_temporal_smooth`: suppresses frame-to-frame jitter and rotation jumps.
- `E_static_freeze`: freezes pose after audio/support indicates a static state.
- `E_pose_prior`: preserves object geometry, handle hidden-state continuity, chair URDF relations, and other physical priors.

Object-specific residuals:

- Balls: center reprojection, radius/mask consistency, floor/contact depth.
- Mug: body/rim/bottom projection, handle phase continuity, stable grasp anchor, table freeze.
- Chair: top-rail endpoints, seat edge, front/rear leg lines, stretcher orientation, weak backrest-hole anchor, fixed URDF geometry, two-hand endpoint SE(3) propagation.

## 5. LLM / VLM Roles

### LLM: Mistral

The LLM has two allowed roles:

```text
Stage -1: semantic prior generation
Stage 6.5: CSV/data consistency audit
```

Stage -1 generates the HOI semantic prior. Stage 6.5 reads CSV/JSON/metrics and produces discrete consistency checks and failure explanations.

Inputs:

- case config
- original video prompt
- Articraft/model prompt context
- seed profile

Outputs:

- object parts
- human parts
- interaction edges
- support/motion priors
- VLM query policy
- CSV audit labels / failure summary

Forbidden:

- no continuous pose
- no coordinates
- no loss weights
- no direct optimizer updates
- no direct rewrite of result CSVs

### VLM: Qwen-VL

The VLM is a forced-choice visual gate.

Typical queries:

- `target_mask_check`
- `keypart_identity_check`
- `track_stability_check`
- `contact_relation_check`
- `overlay_alignment_check`
- `anchor_update_check`
- `post_render_sanity_check`

Rules:

- ask one question per call
- use forced-choice labels
- always allow `unclear`
- VLM output gates predefined actions/residuals only
- free-form VLM text is never converted into a continuous loss weight

Gate behavior:

```text
pass    -> enable anchor/contact/update
reject  -> disable residual / block if non-report-only
unclear -> no update / no hard contact
```

## 6. Four Solved Cases

| case | geometry | observation | contact | pose/refine |
|---|---|---|---|---|
| basketball | sphere proxy | mask center | hand/floor | translation3 + anchor depth + backproject xy |
| football | sphere proxy | mask center | foot/floor | translation3 + foot contact depth + smoothing |
| mug | Articraft mesh | rigid body + parts | palm-handle/rim/table | rigid6+phase + stable grasp + table freeze |
| chair | Articraft/URDF | semantic graph tracks | two-hand top rail endpoint | semantic 6D + small SE(3) + anchor propagation/freeze |

## 7. Tests and Acceptance

Full mainline:

```bash
python scripts/shared/generic_contact_pipeline/run_pipeline.py \
  --case all \
  --result-name generic_pipeline_v2_llm_vlm_gate \
  --from-stage stage-1 \
  --to-stage stage7 \
  --llm-mode mistral \
  --vlm-mode dry-run
```

Qwen-VL key-stage smoke:

```bash
/home/yang/miniconda3/envs/qwen-vl/bin/python \
  scripts/shared/generic_contact_pipeline/stages/stage_vlm_qwen.py \
  --case mug \
  --stage stage5 \
  --result-name generic_pipeline_v2_llm_vlm_gate \
  --limit 2 \
  --no-refresh-queries
```

Ablation:

```bash
python scripts/shared/generic_contact_pipeline/stages/stage_ablation.py \
  --case all \
  --to-stage stage4 \
  --run-variant A2_v2_no_llm_prior \
  --run-variant A3_v2_llm_prior_only \
  --run-variant A4_v2_vlm_gate_only \
  --run-variant A5_v2_llm_prior_plus_vlm_gate \
  --run-variant A6_v2_no_contact_gate \
  --run-variant A7_v2_no_depth_gate \
  --run-variant A8_v2_no_anchor_propagation
```

Current audit:

- all four cases run from Stage -1 to Stage7
- Stage6 compare passes for all cases
- all six render videos exist for each case
- real Qwen-VL key-stage smoke returns code 0
- synthetic gate test proves reject/unclear disables residuals
- Mistral ablation is materialized independently and does not overwrite the mainline

Audit document:

```text
docs/generic_pipeline_v2_mainline_audit_zh.md
```
