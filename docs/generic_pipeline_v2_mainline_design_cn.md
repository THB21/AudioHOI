# Generic Pipeline v2 主线设计说明

本文档说明 `generic_pipeline_v2_llm_vlm_gate` 的完整主线：输入输出、stage 设计、loss/energy、LLM/VLM 分工、四个 solved cases 的复用与特化、测试和验收。

## 1. 总体目标

新主线把 basketball、football、mug、chair 四个已解 case 统一到同一套模块化 pipeline：

```text
case profile / config
+ reusable capability components
+ stage runner
```

原则：

- 新 object 优先新增 config，而不是新增完整 object-specific runner。
- 只有出现新能力时才新增 component。
- LLM/VLM 不直接求连续位姿。
- Optimizer 是唯一连续求解器，负责 2D-to-3D/6D、depth、contact、smooth、static/freeze。

主线结果目录：

```text
samples_known_object/<case>/results/generic_pipeline_v2_llm_vlm_gate/
samples_known_object/<case>/results/renders/generic_pipeline_v2_llm_vlm_gate/
```

运行入口：

```bash
python scripts/shared/generic_contact_pipeline/run_pipeline.py \
  --case all \
  --result-name generic_pipeline_v2_llm_vlm_gate \
  --from-stage stage-1 \
  --to-stage stage7 \
  --llm-mode mistral \
  --vlm-mode dry-run
```

真实 VLM smoke：

```bash
/home/yang/miniconda3/envs/qwen-vl/bin/python \
  scripts/shared/generic_contact_pipeline/stages/stage_vlm_qwen.py \
  --case chair \
  --stage stage2 \
  --result-name generic_pipeline_v2_llm_vlm_gate \
  --limit 2 \
  --no-refresh-queries
```

## 2. 代码结构

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

## 3. Stage 设计

### Stage -1：LLM Semantic HOI Prior

输入：

- case yaml
- `samples_known_object/<case>/metadata.json`
- original video prompt
- Articraft prompt / record preview
- seed profile：`samples_known_object/<case>/annotations/hoi_profile_seed.json`

输出：

```text
hoi_profile.json
hoi_profile_resolved.yaml
prompt_context.json
llm_prior_trace.json
```

作用：

- Mistral 只生成离散 HOI semantic prior。
- 输出 object parts、human parts、interaction edges、support priors、motion priors、VLM query policy。
- 不输出 pose、坐标、loss weight 或 SE(3) 修正。

失败策略：

- 如果 Mistral 返回非法 JSON 或 schema 不通过，回退 seed profile，并在 trace 里记录。

### Stage 0：Preprocess Manifest

输入：

- SAM2 mask
- CoTracker tracks
- DA3 depth
- audio events
- GVHMR human pose
- camera intrinsics / `K_fullimg`
- mesh/proxy/URDF availability

输出：

```text
stage0_inputs_manifest.json
stage0_metrics.json
```

Stage0 只检查和登记输入，不做 object-specific pose 优化。

### Stage 1：2D Observation

统一输出：

```text
object_observations.csv
object_local_points.csv
object_local_segments.csv
stage1_metrics.json
```

按 profile 选择 observation component：

| case | observation model | 主要观测 |
|---|---|---|
| basketball | `mask_track_center` | center、boundary、bottom support、bbox |
| football | `mask_track_center` | center、boundary、bottom support、bbox |
| mug | `rigid_body_plus_parts` | cup body、rim、bottom、handle visibility/contact mesh points |
| chair | `semantic_graph_tracks` | top rail、hole、seat、front/rear legs、stretchers、feet |

### Stage 2：Contact Candidates

统一输出：

```text
contact_candidates.csv
contact_state_frames.csv
stage2_metrics.json
```

按 profile 选择 contact component：

| case | contact policy | 接触逻辑 |
|---|---|---|
| basketball | `hand_floor` | hand-ball、ball-floor |
| football | `foot_floor` | foot-ball、ball-floor |
| mug | `palm_handle_rim_body` | palm-handle/body、mouth-rim、table release |
| chair | `two_hand_toprail_endpoint` | left palm -> right endpoint，right palm -> left endpoint |

### Stage 3：Initial Pose

统一输出：

```text
object_pose_init.csv
stage3_metrics.json
```

pose model：

| case | pose model | variables |
|---|---|---|
| basketball | `translation3` | `tx, ty, tz` |
| football | `translation3` | `tx, ty, tz` |
| mug | `rigid6_plus_phase` | `rx, ry, rz, tx, ty, tz, handle_phase` |
| chair | `semantic_graph_6d` | `rx, ry, rz, tx, ty, tz` |

### Stage 4：Contact / Depth / SE(3) Refinement

统一输出：

```text
object_pose.csv
object_contact_points.csv
object_phase.csv
stage4_metrics.json
```

作用：

- 使用 contact candidates 和 VLM gates 控制 contact residual 是否启用。
- 做 depth anchor、backproject xy、stable grasp、small SE(3)、anchor propagation/freeze。
- 保持 2D overlay 强约束，同时允许必要的小范围 3D 修正。

### Stage 5：Render

标准输出六个视频：

```text
object_only/overlay.mp4
object_only/camera3d.mp4
object_only/side_yz.mp4
with_human/overlay.mp4
with_human/camera3d.mp4
with_human/side_yz.mp4
```

策略：

- basketball / football：proxy sphere render，自动 H264。
- mug：overlay/camera3d 用真实 Articraft mesh，side_yz 用 Articraft-ratio compact diagnostic。
- chair：camera3d/overlay 用原始 URDF solid mesh；YZ/diagnostic 由 renderer 标准化输出。

### Stage 6：Baseline Comparison

输出：

```text
stage6_compare_report.json
migration_audit.csv
migration_audit.json
```

检查：

- required CSVs 存在且非空
- frame count 和 solved baseline 一致
- pose delta pass
- phase/event pass
- 六个 render videos 存在且 codec/frame pass
- chair 额外 semantic/contact/freeze quality gate

### Stage 7：Loss / Residual Logging

输出：

```text
loss_analysis/per_frame_residuals.csv
loss_analysis/loss_summary.json
```

每帧记录：

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

## 4. Energy / Loss 设计

统一形式：

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

Stage3 初始位姿：

```text
E_stage3 =
  w_2d    * E_2d_projection
+ w_depth * E_depth_prior
+ w_mask  * E_mask_bbox
+ w_geom  * E_geometry_consistency
+ w_smooth* E_temporal
```

关键 residual：

- `E_2d_projection`：投影后的 center / keypoints / semantic lines 对齐 2D observations。
- `E_depth_order_or_metric`：DA3 或 anchor depth 提供 metric/depth-order 约束。
- `E_contact`：人体 palm/foot/mouth/table 与 object contact point 的 3D gap。
- `E_temporal_smooth`：抑制 frame-to-frame 抖动和 rotation jump。
- `E_static_freeze`：audio + support 判定 static 后冻结。
- `E_pose_prior`：保持物体几何、handle hidden state、chair URDF relation 等物理先验。

object-specific：

- 球类：center reprojection、radius/mask consistency、floor/contact depth。
- mug：body/rim/bottom projection、handle phase continuity、stable grasp anchor、table freeze。
- chair：top rail endpoints、seat edge、front/rear leg lines、stretcher orientation、backrest hole weak anchor、URDF fixed geometry、two-hand endpoint SE(3) propagation。

## 5. LLM / VLM 分工

### LLM：Mistral

LLM 只在 Stage -1 使用。

输入：

- case config
- original video prompt
- Articraft/model prompt context
- seed profile

输出：

- object parts
- human parts
- interaction edges
- support/motion priors
- VLM query policy

禁止：

- 不输出 continuous pose。
- 不输出 coordinates。
- 不输出 loss weights。
- 不直接修改 optimization。

### VLM：Qwen-VL

VLM 是 forced-choice gate。

典型 query：

- `target_mask_check`
- `keypart_identity_check`
- `track_stability_check`
- `contact_relation_check`
- `overlay_alignment_check`
- `anchor_update_check`
- `post_render_sanity_check`

规则：

- 一次只问一个问题。
- 必须 forced-choice。
- 必须允许 `unclear`。
- VLM 输出只 gate predefined residual/action。
- 不把 VLM free text 转成连续 loss weight。

gate 行为：

```text
pass    -> enable anchor/contact/update
reject  -> disable residual / block if non-report-only
unclear -> no update / no hard contact
```

## 6. 四个 solved cases

| case | geometry | observation | contact | pose/refine |
|---|---|---|---|---|
| basketball | sphere proxy | mask center | hand/floor | translation3 + anchor depth + backproject xy |
| football | sphere proxy | mask center | foot/floor | translation3 + foot contact depth + smoothing |
| mug | Articraft mesh | rigid body + parts | palm-handle/rim/table | rigid6+phase + stable grasp + table freeze |
| chair | Articraft/URDF | semantic graph tracks | two-hand top rail endpoint | semantic 6D + small SE(3) + anchor propagate/freeze |

## 7. 测试和验收

主线全量：

```bash
python scripts/shared/generic_contact_pipeline/run_pipeline.py \
  --case all \
  --result-name generic_pipeline_v2_llm_vlm_gate \
  --from-stage stage-1 \
  --to-stage stage7 \
  --llm-mode mistral \
  --vlm-mode dry-run
```

Qwen-VL 关键 stage smoke：

```bash
/home/yang/miniconda3/envs/qwen-vl/bin/python \
  scripts/shared/generic_contact_pipeline/stages/stage_vlm_qwen.py \
  --case mug \
  --stage stage5 \
  --result-name generic_pipeline_v2_llm_vlm_gate \
  --limit 2 \
  --no-refresh-queries
```

Ablation：

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

当前审计：

- 四个 case 均从 Stage -1 到 Stage7 跑通。
- Stage6 compare 全 pass。
- 六个 render videos 均存在。
- 真实 Qwen-VL 关键 stages 串行 smoke 全 returncode 0。
- synthetic gate test 证明 reject/unclear 会禁用 residual。
- Mistral ablation 独立产出，不覆盖主线。

审计文档：

```text
docs/generic_pipeline_v2_mainline_audit_zh.md
```
