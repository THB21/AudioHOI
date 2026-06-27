# Generic Pipeline v2 代码框架说明

本文只说明当前主线代码如何组织，以及哪些旧目录已迁移/删除。主线入口是：

```text
scripts/shared/generic_contact_pipeline/run_pipeline.py
```

主线输出默认写到：

```text
samples_known_object/<case>/results/generic_pipeline_v2_llm_vlm_gate/
samples_known_object/<case>/results/renders/generic_pipeline_v2_llm_vlm_gate/
```

## 目录职责

```text
scripts/shared/generic_contact_pipeline/
  run_pipeline.py
  configs/
  core/
  components/
  stages/
```

### `configs/`

只放配置，不放算法逻辑。

```text
configs/cases/
  basketball.yaml
  football.yaml
  mug.yaml
  chair.yaml
configs/llm_provider.yaml
configs/vlm_provider.yaml
configs/vlm_query_templates.yaml
configs/runtime_envs.yaml
```

case yaml 决定每个 object 组合哪些能力，例如 observation、contact、pose、refinement、render backend。新增 object 时，优先新增 case config；只有出现新能力时才新增 component。

### `core/`

只放跨 object、跨 stage 的基础设施，不放某个 object 的专门优化逻辑。

```text
core/base/
  camera.py
  config.py
  io.py
  runtime.py
  schema.py

core/semantics/
  hoi_profile.py
  provenance.py

core/gates/
  llm_provider.py
  vlm.py
  vlm_gates.py
  vlm_provider.py

core/evaluation/
  audit.py
  ball_residuals.py
  compare.py
  loss_analysis.py
```

- `base`: 路径、schema、相机、环境、CSV/JSON IO。
- `semantics`: LLM/seed 语义 profile、旧结果 provenance 重建。
- `gates`: VLM/LLM provider、forced-choice query、gate 写入。
- `evaluation`: baseline compare、loss/residual 记录、审计。

### `components/`

放可复用能力模块。这里允许 object-specific 能力，但不能写成完整 object runner。

```text
components/geometry/
components/observation/
components/contact/
components/pose/
components/refinement/
components/render/
```

例子：

- 球类复用 `mask_track_center + hand_floor/foot_floor + translation3 + anchor_depth + proxy_sphere`。
- mug 复用 `rigid_body_plus_parts + palm_handle_rim_body + rigid6_plus_phase + stable_grasp_anchor + articraft_mesh`。
- chair 复用 `semantic_graph_tracks + two_hand_toprail_endpoint + semantic_graph_6d + small_se3/anchor_propagate_freeze + urdf_solid`。

### `stages/`

只放 stage runner，runner 负责调 component，不直接堆 object 逻辑。

```text
stages/main/
  stage_minus1_llm_prior.py
  stage0_preprocess.py
  stage1_observation.py
  stage2_contact.py
  stage3_initial_pose.py
  stage4_contact_refine.py
  stage5_render.py
  stage6_compare.py

stages/gates/
  stage_llm_mistral_profile.py
  stage_llm_qwen_profile.py
  stage_vlm_verify.py
  stage_vlm_qwen.py

stages/analysis/
  stage_loss_analysis.py
  stage_ablation.py
  stage_related_work_*.py
```

- `main`: 真正 pipeline 主线阶段。
- `gates`: LLM/VLM 的可执行阶段。VLM 只做 gate，不直接输出连续 pose。
- `analysis`: loss、ablation、related-work proxy，对主线结果做分析，不是主线求解器。

## 主线阶段

```text
stage-1  LLM/seed HOI profile
stage0   preprocess manifest: SAM2, CoTracker, DA3, audio, GVHMR, K_fullimg
stage1   2D observation / semantic tracks
stage2   contact candidates
stage3   initial 2D-to-6D pose
stage4   contact/depth/small SE(3)/freeze refine
stage5   six-video render
stage6   baseline compare
stage7   loss / related-work proxy analysis
```

## VLM 在代码里的位置

VLM query 由 `core/gates/vlm.py` 根据 `hoi_profile.json` 和 case config 生成。

每个 stage 的 VLM 结果统一落到：

```text
vlm_queries.csv
vlm_results.csv
vlm_gates.csv
vlm_summary.md
```

VLM 只能做三件事：

```text
pass/reject/unclear gate
failure label
reporting
```

VLM 不直接写连续坐标、不直接给 loss weight、不直接修改 pose。优化器只读取 gate 后决定是否启用预定义 residual。

## 旧目录处理

以下目录不是 v2 主线运行时依赖，必要能力已迁移到 `generic_contact_pipeline/components` 后从本分支清理：

```text
scripts/shared/human_ball/
scripts/shared/radius_free_proxy/
scripts/shared/sam3d_radius_estimation/
scripts/shared/tracking/   # 空目录/缓存
```

保留：

```text
scripts/shared/human/
```

原因：这里是人体/手部相关工具，包含 HaMeR/GVHMR 辅助脚本；它不是当前 v2 主线的直接 runtime import，但后续手部抓握或人体侧重建仍可能复用，所以暂不删除。

旧 mug runner：

```text
scripts/known_object/mug/run_mug_m18_physical_nohide_pipeline.py
```

是历史 M18/M17 单 object runner。其稳定逻辑已经迁移到：

```text
components/pose/mug_handle_phase_correction.py
components/pose/mug_opening_2d_pose_correction.py
components/refinement/stable_grasp_anchor.py
components/render/render_mug_articraft_camera3d_scene.py
```

因此主线不再依赖该旧 runner。

## 新 object 应该怎么加

优先只新增：

```text
configs/cases/<new_object>.yaml
```

如果已有能力不够，再新增一个小 component，例如：

```text
components/observation/<new_observation>.py
components/contact/<new_contact_policy>.py
components/refinement/<new_refinement>.py
```

不要新增整套 `run_<object>_xxx.py`。stage runner 应该继续根据 config 选择 component。
