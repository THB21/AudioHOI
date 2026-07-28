# Interaction-State-Conditioned Generic Solver Plan

维护分支：`refactor/interaction-state-production`

最后更新：2026-07-29

## 当前维护状态

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| `refactor/migrate-chair-case` | frozen | 作为 chair extraction / candidate evidence 保留，不继续堆新功能。 |
| `refactor/interaction-state-production` | in_progress | 已引入生产级 `InteractionStateIR`，并接入 sequence problem / diagnostics / candidate sandbox shadow 输入链；已新增通用 factor activation ledger、`CompiledFactor` shadow contract 和 generic `SequenceProblemContract`；暂不改变 accepted output、loss、阈值或求解路径。 |
| 下一步 | pending | 继续把 `SequenceProblemContract + CompiledFactor` 从 shadow 边界推进到 runtime factor executor 输入；保持 residual parity，不引入 case dispatcher。 |

当前主线目标收束为一句话：

> Vision 决定物体大致在哪里，audio 决定关键交互何时发生，VLM 决定可见性、接触部位和语义关系，统一的 factor-graph solver 决定连续三维轨迹如何同时满足这些约束。

核心方向不是“五个 case 的超级 dispatcher”，而是一个 interaction-state-conditioned generic solver。`case_name` 必须逐步从生产 solver 中消失，替换为：

- `StateSpec`
- `GeometryProvider`
- `Measurements`
- `InteractionState`
- `CompiledFactors`

## 0. 当前代码判断

当前代码已有一部分正确底座：

- `MeasurementIR` 已支持 `point`、`line`、`mask`、`depth`、`track`、`visibility`。
- `ContactConstraintIR` 已支持 `grasp`、`support`、`impact`、`sliding`、`release`。
- contact state 已包含 `active`、`occluded_hold`、`inactive`。
- `StateSpec` 已能表达 `sphere`、`line/capsule`、`rigid mesh`、`articulated URDF`。
- `FactorKind` 已列出 `reprojection`、`contact`、`penetration`、`temporal`、`audio`、`joint_limit`、`gauge` 等。

主要问题：

- 这些能力大多还在 shadow、diagnostics、candidate sandbox。
- `FactorSpec` 仍显式禁止 `consumed_by_solver=True`，说明 Factor IR 尚不是生产求解器真实输入。
- `candidate.py` 仍通过 `is_sphere` / `is_mug` / `is_chair` / `is_line_contact` 和 `profile.case_name` 选择不同 executor。
- 当前所谓 generic mainline 主要统一调度和文件名，真实求解仍由 case adapter、历史结果和专用 solver 驱动。

因此统一计划的核心不是继续增加 zero-shot case，而是先让 solver 的输入统一，solver 的选择不再由 case 决定。

## 1. Zero-shot 定义

本项目当前是 known-object / known-asset reconstruction。近期合理声称的 zero-shot 应限定为以下三类。

### ZS-1：新视频 zero-shot

同一物体或同一几何族的新视频，不重新调阈值、不写新代码。

例子：

- 新角度背身篮球；
- 新动作速度足球；
- 新人物拿杯子。

### ZS-2：新物体 zero-shot

物体属于已有几何族，只增加：

- mesh、URDF 或几何 proxy；
- asset manifest；
- semantic feature mapping；
- detector prompt。

不得增加：

- `pingpong_solver.py`
- `suitcase_solver.py`
- `run_new_object.py`
- 对象专用 least-squares optimizer。

### ZS-3：新交互组合 zero-shot

已有通用 interaction modes 的重新组合：

- occlusion + impact + flight
- persistent grasp + sliding support
- small object + high speed + intermittent impact

背身遮挡拍球主要测试 ZS-1 和 ZS-3；高速乒乓测试 ZS-2、ZS-3 和小目标退化；suitcase 拖动测试新 rigid object、新 contact part 和新的 sliding / rolling 模式。

暂时不要表述为“完全未知物体、没有资产、自动生成精确几何”的 open-world reconstruction。

## 2. 最终一条龙架构

保留现有 Stage 0–7 编号，但固定内部职责，并在 object 完成后增加 human stages。

```text
video + audio + object label/asset
               |
               v
Stage -1  Asset / HOI semantic profile
               |
               v
Stage 0   Multimodal preprocessing
          frames / camera / mask / tracking / depth /
          body / hands / audio events
               |
               v
Stage 1   Measurement IR
          point / line / mask / depth / track / visibility
               |
               v
Stage 2   Interaction State IR
          contact candidates / visibility / audio events /
          VLM semantic relations / support state
               |
               v
Stage 3   StateSpec + GeometryProvider + initial hypotheses
               |
               v
Stage 4   One Generic Sequence Solver
          factor compilation -> solve -> state update -> re-solve
               |
               v
Stage 5   Atomic object result publication
          object_pose.csv + object_handoff.json
               |
               v
Stage 6   object render + hard metrics
Stage 6.5 VLM selected-window audit
Stage 7   residual / failure analysis
               |
               v
Stage 8   GVHMR + HaMeR human reconstruction
               |
               v
Stage 9   body-side contact refinement
               |
               v
Stage 10  HOI render + final evaluation
```

目标命令形式：

```bash
python -m audiohoi.run_full \
  --video input.mp4 \
  --object-label suitcase \
  --asset assets/suitcase.glb \
  --geometry rigid_mesh \
  --result-name heldout_suitcase_full \
  --audio auto \
  --vlm qwen
```

## 3. 不再混合“遮挡、接触、运动状态”

旧的 `motion_regime` 会混合 contact 和速度，产生：

- `sustained_grasp`
- `short_impact`
- `high_speed`
- `static_hold`
- `free_motion`

它目前主要用于 pose seed 后面的 smoother reweighting。统一后必须拆成四个正交状态轴。

### 3.1 VisibilityState

- `visible`
- `partially_visible`
- `occluded`
- `absent`
- `unknown`

### 3.2 ContactState

- `candidate`
- `active`
- `persistent`
- `occluded_hold`
- `release`
- `inactive`

### 3.3 ContactMode

- `grasp`
- `impact`
- `support`
- `sliding`
- `rolling`
- `release`
- `unknown`

### 3.4 MotionMode

- `free`
- `ballistic`
- `attached`
- `supported_static`
- `supported_moving`
- `high_speed`
- `unknown`

例子：

背身拍球被身体挡住，但球在空中：

```text
visibility = occluded
contact_state = inactive
motion_mode = ballistic
```

不能误写成 `occluded_hold`。

手抓 suitcase handle，同时 suitcase 被身体挡住：

```text
visibility = occluded
contact_state = occluded_hold
contact_mode = grasp
motion_mode = attached + supported_moving
```

乒乓球撞击球拍：

```text
visibility = partially_visible
contact_state = active
contact_mode = impact
motion_mode = high_speed
```

## 4. 生产级 InteractionStateIR

新增位置：

```text
core/interaction/
  types.py
  estimator.py
  state_machine.py
  timeline.py
  validation.py
```

核心 schema：

```text
FrameInteractionState:
    frame
    time
    target_entity_id
    visibility_state
    motion_mode
    active_contact_ids
    support_contact_ids
    audio_event_ids
    semantic_relation_ids
    confidence
    provenance
```

Contact 仍由现有 `ContactConstraintIR` 表示，不复制新 contact schema。

新增 interval-level 输出：

```text
interaction_timeline.jsonl
interaction_intervals.csv
interaction_state_metrics.json
```

状态不是硬编码 case，而是离散推断：

```text
S* =
argmin_S
sum_t E_emission(S_t | M_t, A_t, R_t)
+ lambda_transition C(S_{t-1}, S_t)
```

其中：

- `M_t`：视觉、depth、tracking、human site；
- `A_t`：audio event；
- `R_t`：VLM semantic relation；
- `S_t`：离散 interaction state。

判定逻辑：

| 证据 | 倾向状态 |
| --- | --- |
| 短时间 proximity + audio impulse + direction change | impact |
| 长时间 proximity + local anchor stable | persistent grasp |
| 原来处于 grasp，随后视觉被遮挡 | occluded_hold |
| 地面 proximity + tangential motion | sliding / rolling |
| 无 contact + 下落 / 抛物运动 | ballistic / free |
| 长时间低速 + support | supported_static |

需要 hysteresis 和最短持续长度，避免逐帧在 impact / free / contact 之间闪烁。

## 5. 统一连续 solver 数学结构

连续 object state：

```text
X_t = [T_t, R_t, s_t, q_t^joint, phi_t]
```

具体包含哪些项由 `StateSpec` 决定：

- ball：translation；rotation 可保持 weakly observable。
- stick：root SE(3)。
- mug：root SE(3) + scale + periodic axial phase。
- chair：root SE(3) + articulated joints。
- suitcase：root SE(3)，可选 prismatic handle joint。

统一目标函数：

```text
X* =
argmin_X
sum_{t,k} g_k(S_t) lambda_k E_k(X; M, C, A, R, G)
```

其中：

- `g_k(S_t)` 是 interaction-state-dependent gate；
- `M` 是 Measurement IR；
- `C` 是 Contact Constraint IR；
- `A` 是 Audio Event IR；
- `R` 是 VLM Relation IR；
- `G` 是 GeometryProvider。

推荐两轮交替优化：

1. observation-derived initial hypotheses
2. infer preliminary interaction state
3. compile factors and solve trajectory
4. recompute 3D gap / relative velocity / support
5. update interaction state
6. recompile factors and solve once more
7. hard validation
8. publish best candidate

形式：

```text
S^{k+1} = infer(M, A, R, X^k)
X^{k+1} = argmin_X E(X | S^{k+1})
```

通常两轮足够；不能变成无边界自我修复循环。

## 6. FactorCompiler：状态选择约束，而不是选择 solver

新增：

```text
core/factors/
  base.py
  compiler.py
  runtime.py
  activation.py
  visual.py
  depth.py
  contact.py
  support.py
  temporal.py
  audio.py
  relation.py
```

保留当前 `FactorSpec` 作为审计声明，再增加真正运行的：

```text
CompiledFactor:
    factor_id
    frame_interval
    residual_fn
    robust_loss
    base_weight
    active_mask
    input_ids
    gate_provenance
```

### 6.1 通用 factor 集

| Factor | 用途 |
| --- | --- |
| `PointReprojectionFactor` | feature / center / endpoint 投影 |
| `LineReprojectionFactor` | stick、handle、chair rail |
| `MaskSilhouetteFactor` | object mask overlay |
| `MetricDepthFactor` | metric depth |
| `DepthOrderFactor` | 前后关系 |
| `ContactDistanceFactor` | human / tool site 到 object feature |
| `ContactRelativeVelocityFactor` | grasp 时相对速度接近零 |
| `LocalAnchorConstancyFactor` | 持续抓握时 local contact 不漂移 |
| `SupportPenetrationFactor` | ground / table support 与 penetration |
| `NormalVelocityFactor` | sliding 时不离开支撑面 |
| `RollingOrSlidingFactor` | 允许 tangential motion |
| `VelocityFactor` | 普通时序平滑 |
| `AccelerationFactor` | 普通加速度约束 |
| `BallisticPriorFactor` | 无接触自由飞行的弱 gravity prior |
| `StaticFreezeFactor` | 静止区间 |
| `JointLimitFactor` | articulated joint limit |
| `GaugeConstraintFactor` | mug phase、chair chord twist |
| `AudioEventAlignmentFactor` | audio 与 impact / contact transition 对齐 |
| `AudioMotionEnvelopeFactor` | continuous dragging sound 与速度区间 |
| `SemanticRelationFactor` | above / below / front / behind / support 等不等式 |

### 6.2 Factor activation matrix

| 状态 | 激活 | 降低或关闭 |
| --- | --- | --- |
| visible / free | reprojection、mask、depth、velocity | contact |
| occluded / free | ballistic、temporal、depth-order | point / mask visual |
| active impact | contact、audio alignment | acceleration smoothing |
| persistent grasp | contact、relative velocity、anchor constancy | free-motion prior |
| occluded hold | anchor constancy、relative pose、temporal | visual update |
| sliding | support、normal velocity、tangential motion | static freeze |
| rolling | support、rolling relation | generic strong smoothing |
| static support | support、freeze | high-speed model |

关键原则：

- solver 永远相同；
- 变化的是 `CompiledFactor.active_mask` 和有限的一组离散权重档位；
- 不通过 `case_name` 选择 solver。

## 7. GeometryProvider 与 SceneEntity

统一接口：

```python
class GeometryProvider:
    def feature_world(self, state, feature_ref, coordinate): ...
    def project_feature(self, state, feature_ref, camera): ...
    def signed_distance(self, state, query_points): ...
    def support_points(self, state): ...
    def feature_jacobian(self, state, feature_ref): ...
```

实现：

- `SphereProvider`
- `CapsuleProvider`
- `RigidMeshProvider`
- `ArticulatedURDFProvider`

引入 `SceneEntity`，不再只允许 human-object：

```text
SceneEntity:
    entity_id
    entity_type: human | object | tool | environment
    state_role: optimized | observed | static
    geometry_provider
```

接触边统一为：

```text
InteractionEdge:
    source_entity
    source_site
    target_entity
    target_feature
    mode
    interval
```

例子：

Basketball：

```text
human:right_hand -> ball:surface
floor:plane -> ball:surface
```

Ping-pong：

```text
paddle:face -> ball:surface
table:top -> ball:surface
```

Suitcase：

```text
human:right_hand -> suitcase:handle
floor:plane -> suitcase:wheels
```

第一版 ping-pong 不必同时优化 paddle：

```text
ball   = optimized entity
paddle = observed entity
table  = static entity
```

## 8. AudioEventIR

Audio 不只是峰值检测。需要生产级：

```text
AudioEvent:
    event_id
    start_time
    peak_time
    end_time
    event_type:
        impact
        contact_onset
        contact_offset
        sustained_motion
        silence
        unknown
    confidence
    snr
    energy
    band_profile
    source
```

### 8.1 Impact audio

适合：

- basketball bounce；
- football kick；
- ping-pong paddle / table hit。

作用：

- 给出 contact transition 的亚帧级 timing；
- impact window 内降低 acceleration smoothing；
- 鼓励合理速度方向变化；
- 非 event window 增强平滑，抑制伪 jitter。

不能做：

- 直接告诉 solver 球的 3D 坐标；
- 假设每个 spike 都是 object contact。

联合门控至少满足两类证据：

- audio impulse；
- visual / predicted proximity；
- motion direction change or VLM relation。

### 8.2 Sustained audio

Suitcase 拖动不是单个 spike，而可能是：

- drag sound begins
- continuous rolling / scraping texture
- pause
- drag resumes

需要 `AudioMotionEnvelopeFactor`。它只约束：

- 有持续拖动声时 tangential speed 不应被压到零；
- 声音停止附近允许速度快速降到零；
- 无声且视觉静止时加强 freeze。

它不要求音频振幅与速度严格线性，只使用弱 rank / correlation constraint。

研究主张：

> Audio 主要改善 contact transition timing、fast-motion preservation 和 motion segmentation，而不是直接提高每一帧三维位置精度。

## 9. VLM SemanticRelationIR

VLM 的角色是生成离散语义关系，不直接修连续姿态。

当前问题：

- `object_vlm_profile()` 仍保留 basketball、football、mug、chair 的 case-specific vocabulary。
- `anchor_update_check` 和 `temporal_motion_check` 的问题写死 “brown predicted stick / green tracked stick”。

新增：

```text
SemanticRelation:
    relation_id
    frame_interval
    subject_entity
    subject_feature
    predicate:
        visible
        occluded_by
        contacting
        held_by
        supported_by
        above
        below
        in_front_of
        behind
        inside
        separated
        moving_with
    object_entity
    object_feature
    confidence
    source
```

VLM forced-choice 问题由 asset profile 自动生成：

- part identity：`handle | wheel | body | unclear`
- visibility：`visible | partial | occluded_by_human | absent | unclear`
- contact relation：`right_hand-handle | left_hand-handle | object-ground | no_contact | unclear`
- spatial relation：`above | below | in_front_of | behind | overlapping | unclear`
- interaction mode：`grasp | impact | support | sliding | release | unclear`

允许 VLM：

- 禁止错误 anchor update；
- 把 `visible` 改为 `occluded`；
- 在预定义 contact pair 中选择一个；
- 激活 support / grasp / impact 的离散状态；
- reject 一个违反语义关系的 candidate；
- 在 hard metrics 相近的候选之间 tie-break。

不允许 VLM：

- 输出 xyz；
- 输出 quaternion；
- 自由生成 loss weight；
- 单独覆盖 geometry hard metric；
- 把一张图判断扩展成整段视频真值。

Evidence package：

- original crop
- mask crop
- human / object keypoint overlay
- contact candidate overlay
- 3–5 frame temporal strip
- candidate render
- hard metric summary

单目深度歧义可以提供 original-view render + side-view candidate render + contact gap visualization，但问题必须问“是否存在明显 floating、penetration 或错误 contact”，不能让 VLM 假装知道 side-view ground truth。

## 10. Stage 0 一条龙

现在 Stage 0 只能自动运行 SAM2 和 CoTracker；DA3、GVHMR、audio event 缺失时只记录：

```text
required external preprocessing artifact is missing;
no generic in-pipeline runner is registered yet
```

需要增加 Preprocess Runner Registry：

```text
PreprocessTask:
    task_id
    environment
    input_artifacts
    output_artifacts
    command_builder
    cache_policy
    required
```

注册：

- `extract_frames`
- `extract_audio`
- `sam2_segment`
- `cotracker_track`
- `da3_depth`
- `gvhmr_body`
- `hamer_hands`
- `audio_event_extract`
- `asset_geometry_prepare`

Stage 0 流程：

```text
resolve DAG
-> check cache hashes
-> run missing required tasks
-> verify output contracts
-> write one stage0 manifest
```

关键要求：

- 从空 result directory 能跑；
- 失败时明确是哪一个 task；
- 不静默复用其他 case artifact；
- 所有输入和模型版本进入 manifest；
- audio、body、depth 不再要求用户提前准备。

## 11. Object → Human handoff contract

当前 human refiner 已支持 sphere、capsule、mesh / SDF，并且只调整对应人体 kinematic chain，但仍需要手动传入：

- `--trajectory-csv`
- `--object-type`
- `--object-length-m`
- `--object-radius-m`
- `--object-mesh-path`

并从 trajectory 中读取 `contact_frame`、`audio_anchor`、`active_part` 等 legacy 字段。

object solver 应发布：

```text
object_handoff/
  handoff_manifest.json
  object_trajectory.csv
  state_spec.json
  geometry_descriptor.json
  geometry_resource.*
  contact_constraints.jsonl
  interaction_timeline.jsonl
  human_sites.csv
  audio_events.jsonl
  coordinate_frames.json
  uncertainty.csv
```

`handoff_manifest.json`：

```json
{
  "schema_version": 2,
  "object_entity_id": "target_object",
  "trajectory": "object_trajectory.csv",
  "state_spec": "state_spec.json",
  "geometry": "geometry_descriptor.json",
  "contacts": "contact_constraints.jsonl",
  "interaction_timeline": "interaction_timeline.jsonl",
  "human_sites": "human_sites.csv",
  "coordinate_frame": "camera_meters",
  "frame_count": 240,
  "fps": 30.0
}
```

Human side 只执行：

```python
handoff = load_object_handoff(...)
geometry = GeometryProvider.from_descriptor(handoff.geometry)
refine_human_against_fixed_object(handoff, geometry)
```

object solver 和 human refiner 最终应共享同一个 `GeometryProvider`。

## 12. 五个现有 case 迁移顺序

### 12.1 Basketball / Football

当前最接近完成：

- `translation3:sphere`
- `generic_sphere_sequence`

它们应作为第一个 production reference。

需要补：

- 正式 `InteractionStateIR`；
- audio event production factors；
- hidden / free-flight state；
- case_name 无关的 hand / foot contact site；
- single accepted-output publisher。

### 12.2 Mug

迁移：

```text
rigid6_plus_phase
```

进入统一 `StateSpec`：

- root SE3
- scale
- assembly axial phase
- coupled gauge

要删除 accepted path 中对以下内容作为独立连续 solver 的依赖：

- `stable_grasp_anchor`
- `anchor_depth`
- `table_freeze`

改写为：

- `ContactDistanceFactor`
- `MetricDepthFactor`
- `SupportFactor`
- `PeriodicPhaseFactor`
- `StaticFreezeFactor`

### 12.3 Stick

保留：

- `LineS`
- endpoint identity
- capsule geometry
- 当前 line contact 可以暂时保留为 nonblocking gap

删除：

- `line_contact_lock` special final refinement

替换为：

- `LineReprojectionFactor`
- `ContactDistanceFactor`
- `LocalAnchorConstancyFactor`
- `GaugeConstraintFactor`
- `TemporalFactor`

### 12.4 Chair

当前已经提取：

- contact chord initializer；
- articulated kinematic provider；
- point / contact / joint / gauge residual；
- candidate sandbox。

canonical accepted output 尚未切换。

最终需要：

```text
StateSpec:
  root SE3
  joint.front_to_rear
  joint.front_to_seat

Factors:
  point reprojection
  line reprojection
  two-hand contact distance
  joint limits
  chord twist gauge
  temporal
  support/freeze
```

然后由同一个 `GenericSequenceExecutor` 写 candidate，再通过 publisher 写 accepted result。

## 13. 分支与 PR 顺序

不要继续把所有工作堆进 `refactor/migrate-chair-case`。该分支冻结为 chair migration evidence。

| 分支 | 核心工作 | 退出条件 |
| --- | --- | --- |
| `refactor/interaction-state-production` | Visibility、Contact、Motion、Audio、Relation IR | 五 case IR 完整，数值输出不变 |
| `refactor/scene-geometry-provider` | SceneEntity、EntitySiteRef、统一 GeometryProvider | sphere / capsule / mesh / URDF parity |
| `refactor/generic-factor-compiler` | `FactorSpec` → `CompiledFactor`、activation matrix | 五 case residual parity |
| `refactor/unified-sequence-executor` | 一个 sequence problem、一个 executor、一个 publisher | core solver 无 `case_name` 分支 |
| `refactor/promote-five-cases` | ball、mug、stick、chair 全部迁移 | 五 case accepted outputs 都来自 unified executor |
| `refactor/audio-vlm-production-gates` | AudioEventFactor、RelationFactor、profile-driven VLM | full / no-audio / no-VLM 真正产生不同轨迹或 gate trace |
| `refactor/full-hoi-orchestrator` | Stage 0 runners、object handoff、human stages | raw video + asset 一条命令完成 |
| `benchmark/heldout-zero-shot` | 三个 held-out scenarios | 添加 case 后不允许改 solver |

## 14. 详细实现和验收

### Phase A：Architecture invariants

CI：

- `test_no_case_name_branching_in_core_solver.py`
- `test_single_object_pose_writer.py`
- `test_no_baseline_pose_reads_stage1_4.py`
- `test_factor_provenance_complete.py`
- `test_fresh_result_directory.py`
- `test_zero_shot_config_only.py`

硬规则：

- `core/solver/` 不允许 `profile.case_name`
- `core/factors/` 不允许 `profile.case_name`
- `core/state/` 不允许 `profile.case_name`
- `core/geometry/` 不允许 `profile.case_name`

允许 case-specific 的地方：

- `configs/`
- asset manifests
- legacy adapters
- evaluation annotations

`object_pose.csv` 只能由一个 publisher 写出。

### Phase B：IR 生产化

把当前 shadow builders 变成正式 stage outputs：

- `measurements.jsonl`
- `contact_constraints.jsonl`
- `semantic_relations.jsonl`
- `audio_events.jsonl`
- `interaction_timeline.jsonl`

legacy CSV 保留兼容期，但 solver 不再读取 legacy CSV。

验收：

- 五 case typed row coverage 不下降；
- source / provenance 完整；
- occluded / missing 不填零；
- 不读取 final / baseline pose；
- canonical 输出仍不变。

### Phase C：Factor parity

先只复刻旧数学，不改变权重、loss、bounds。

对每个 case 比较：

```text
old residual block
vs
compiled factor residual block
```

要求：

- row-by-row residual 一致或在冻结 tolerance 内；
- active frame mask 一致；
- weight source 一致；
- gate source 一致；
- factor ledger 可追踪到原始 measurement / constraint。

### Phase D：Generic executor

接口：

```python
problem = SequenceProblem(
    state_spec=...,
    geometry_provider=...,
    measurements=...,
    interactions=...,
    factors=...,
)
attempts = GenericSequenceExecutor.solve(problem)
best = CandidateSelector.select(attempts)
AcceptedOutputPublisher.publish(best)
```

初始化器按 geometry capability 选择：

- `sphere_center_depth_initializer`
- `line_endpoint_initializer`
- `rigid_correspondence_initializer`
- `articulated_correspondence_initializer`

这是 geometry-family initializer，不是 object-specific solver。

### Phase E：多 hypothesis

单目歧义时生成有限候选：

- depth near / median / far
- orientation alternatives
- left / right contact alternatives
- visible / occluded hypotheses

每个 attempt 写：

```text
attempts/<attempt_id>/
  state.csv
  residuals.csv
  factor_ledger.json
  hard_metrics.json
  vlm_gates.json
  status.json
```

CandidateSelector 顺序：

1. NaN、schema、joint limit 等硬失败直接 reject；
2. 严重 penetration、support violation reject；
3. contact / overlay / temporal hard metrics 排序；
4. VLM semantic reject；
5. 只有硬指标相近时，VLM preference 作 tie-break；
6. publisher 原子写入 canonical output。

## 15. 三个 held-out zero-shot case

### Case A：背身遮挡篮球 dribble

主要退化：

- 球被人体遮挡；
- 正面 mask / tracker 丢失；
- 球可能从身体前后两个深度假设中产生歧义；
- contact 只持续极短时间；
- bounce acceleration 容易被 smoothing 抹掉。

实体：

- `human hands`：observed
- `human torso`：observed occluder
- `basketball`：optimized sphere
- `floor`：static plane

状态序列：

```text
visible free-flight
-> hand impact
-> occluded free-flight
-> floor impact
-> visible rebound
```

中间不是 `occluded_hold`。

Audio 证明：

- hand / floor impact timing；
- 遮挡区间 bounce phase；
- acceleration spike 不被压平；
- reappearance 前后的 motion segmentation。

VLM 证明：

- 球被人体遮挡，而不是消失；
- contact 更可能是 hand 还是 floor；
- 哪只手参与；
- tracker 是否跟错到衣服或身体；
- ball 应位于手下方、地面上方等粗关系。

必须 ablation：

- `full`
- `no_audio`
- `no_vlm`
- `no_occlusion_state`
- `vision_only`
- `oracle_contact`

关键指标：

- `impact_timing_error_frames`
- `contact_pair_accuracy`
- `occlusion_state_F1`
- `hidden_interval_reappearance_error_px`
- `high_speed_preservation`
- `non_event_acceleration_spikes`
- `contact_gap`
- `penetration`

预期主张：

- full - no_audio 主要改善 timing 和 over-smoothing；
- full - no_vlm 主要改善 visibility / contact part 和错误 tracker gate；
- raw overlay 不一定大幅提升。

### Case B：高速小乒乓球

主要退化：

- 球尺寸可能只有数个像素；
- motion blur；
- frame-to-frame displacement 很大；
- 球拍和桌面是额外 contact entity；
- impact 可能发生在两个视频帧之间。

第一版作用域：

- ping-pong ball = optimized sphere
- paddle = observed tool entity
- table = static plane
- human hand = observed

球拍轨迹可先由 tool tracker 或 hand + paddle rigid attachment 产生，不要求第一版同时优化球拍。

Observation 改造：

- full-frame detector 只负责发现大致区域；
- 以 motion proposal 生成高分辨率 crop；
- crop 内重新 segment / track；
- Measurement covariance 随 blur、大小和 visibility 调整；
- 失踪帧写 `occluded` / `unknown`，不能补零。

Audio 证明：

- paddle hit / table hit 亚帧 timing；
- impact 时速度方向变化；
- 防止高速轨迹被普通 smoothness 压平；
- 区分真实 impact spike 与 tracker jitter。

VLM 证明：

- contact 对象是 paddle 还是 table；
- 当前小亮点是否确实是球；
- 球是否被 paddle / hand 遮挡；
- candidate 是否出现明显错误物体跟踪。

VLM 输入必须是放大的 temporal crop，而不是整张 1080p 图。

必须 factors：

- small-sphere visual / depth
- weak ballistic prior
- paddle / table contact distance
- audio event alignment
- impact direction-change preservation
- non-event smoothness

关键指标：

- tiny-object detection / track recall
- impact timing error
- direction-change preservation
- event-vs-non-event acceleration ratio
- reappearance error
- wrong-object tracking count

这是三个 case 中最难的，应放在背身篮球和 suitcase 之后。

### Case C：无规则拖动 suitcase

第一版物体模型优先使用：

- rigid mesh + handle feature + wheel / support features

先把拉杆固定视为 rigid feature。后续再扩展：

- prismatic handle joint

实体：

- human right / left hand = observed
- suitcase = optimized rigid mesh
- floor = static plane

状态序列：

```text
grasp handle
-> static support
-> irregular sliding/rolling
-> pause
-> occluded hold + rolling
-> release/static
```

Audio 应使用持续音频：

- drag onset
- rolling / scraping interval
- pause
- restart
- stop

Audio 主要控制：

- motion interval segmentation；
- start / stop timing；
- pause 不被过度平滑；
- 拖动区间不被 freeze；
- 静音区间不出现持续滑动。

VLM 证明：

- 手接触的是 handle，不是 suitcase body；
- suitcase 在地面上；
- suitcase 是 behind / in-front-of human；
- 被人体遮挡时延续 grasp；
- part identity 和 upright orientation 是否合理。

Factors：

- hand-handle contact distance
- contact relative velocity
- local anchor constancy
- wheel / floor support
- normal velocity approximately zero
- tangential motion allowed
- penetration
- audio motion envelope
- piecewise temporal smoothing

关键指标：

- handle contact part accuracy
- grasp anchor drift
- ground penetration
- support gap
- start / stop timing
- pause preservation
- occluded grasp continuity
- orientation drift

## 16. 数据采集和评估

Pipeline 输入不能使用人工 contact GT。

人工标注只用于 held-out evaluation：

- contact event frames
- contact human / object part
- visibility intervals
- object 2D center / mask
- support state

如果可以重新录制，建议额外同步一台侧面相机或 depth camera：

- pipeline 仍只输入单目主视角；
- 第二视角只作为评估 GT；
- 用于判断前后深度、真实 contact 和 occlusion recovery。

这样比让同一个 VLM 同时定义 contact 和评价 contact 更可信。

## 17. Ablation 设计

最小方法组：

- `full_audio_vlm`
- `vision_only`
- `no_audio`
- `no_vlm`
- `no_contact`
- `no_interaction_state`
- `no_sequence_solver`
- `oracle_contact`

Audio 表：

| Case | Timing error ↓ | Event accel preserved ↑ | Non-event spikes ↓ | Interval F1 ↑ |
| --- | --- | --- | --- | --- |

VLM 表：

| Case | Visibility F1 ↑ | Contact-part accuracy ↑ | Wrong anchor blocked ↑ | Wrong track blocked ↑ |
| --- | --- | --- | --- | --- |

最终 HOI 表：

| Case | Overlay ↑ | Contact gap ↓ | Penetration ↓ | Support error ↓ | Jerk ↓ | High-speed preservation ↑ |
| --- | --- | --- | --- | --- | --- | --- |

每个主要 constraint 都必须有对应指标和 ablation，不能只展示视频。

需要保留一个 audio 不明显有用的 negative-control case，例如清晰、缓慢、视觉完全可见的操作，避免只选 audio-helpful clips。

## 18. Definition of Done

### 软件条件

1. `core/solver` 中不存在 `case_name == basketball/mug/chair/...`。
2. 五个 canonical case 的 accepted output 都由一个 `GenericSequenceExecutor` 产生。
3. `object_pose.csv` 只有一个 publisher。
4. Stage 1–4 不读取 baseline / final solved pose。
5. 新对象只增加 config、asset 和 feature mapping。
6. 遮挡、grasp、impact、sliding 等通过 `InteractionState` 激活 factors。
7. Audio 和 VLM 都有真实 factor / gate provenance。
8. 从 raw video + asset 可以 fresh run。
9. Object handoff 自动触发 human pipeline。
10. 最终结果目录同时含 object、human、HOI render 和 evaluation。

### Zero-shot 条件

添加背身篮球、乒乓、suitcase 后：

| 操作 | 是否允许 |
| --- | --- |
| 新增 case config / asset | 允许 |
| 新增 detector prompt | 允许 |
| 新增 semantic feature map | 允许 |
| 修改 core solver | 禁止 |
| 新增对象专用 optimizer | 禁止 |
| 修改通用 loss 阈值 | 禁止 |
| 使用人工 contact 作为输入 | 禁止 |
| 复制旧 final pose | 禁止 |

### 科研条件

- Audio 在 timing / high-speed / segmentation 指标上有可解释增益；
- VLM 在 visibility / contact-part / wrong-track gate 上有可解释增益；
- hard geometry metrics 不依赖 VLM judge；
- full 方法在多个 held-out clips 上稳定改善，而不是只改善一个展示视频；
- 失败 case 也完整报告。

## 19. 当前分支实际执行顺序

当前最合理落地顺序：

1. 冻结 `refactor/migrate-chair-case`。
2. 建 `refactor/interaction-state-production`。
3. 建 `refactor/scene-geometry-provider`。
4. 建 `refactor/generic-factor-compiler`。
5. 建唯一 `GenericSequenceExecutor + Publisher`。
6. 依次切 mug -> stick -> chair。
7. 接入 production `AudioEventIR` 和 `VLM RelationIR`。
8. 补齐 Stage 0 runners 与 object handoff。
9. 先跑背身遮挡篮球，再跑 suitcase，最后跑高速乒乓。
10. 冻结 held-out benchmark 后禁止再改 solver。

最终研究主线：

> 一个通用的离散 interaction-state estimator，加一个通用的连续 geometry-aware sequence solver；audio 负责事件时序，VLM 负责语义与可见性，vision / depth 负责连续空间观测，最后通过正式 handoff 进入 human-side refinement。

## 20. 本文件维护规则

后续每次修改 solver 主线时，必须同步更新本文件顶部“当前维护状态”，并在对应章节记录：

- 新增或删除了哪些 core abstraction；
- 是否改变 accepted output；
- 哪些测试 / gate 已通过；
- 哪些 gap 仍是 blocking / nonblocking；
- 是否仍符合“solver 不靠 case_name 分派”的原则。

维护记录格式：

```text
YYYY-MM-DD:
- branch:
- commit:
- change:
- verification:
- remaining gap:
```

## 21. 维护记录

2026-07-29:

- branch: `refactor/interaction-state-production`
- commit: `20d6a257 Introduce production interaction state IR`
- change: 新增生产级 `core/interaction`、五 case timeline builder、timeline / intervals / metrics 导出 CLI、InteractionStateIR 测试。
- verification: `tests/test_interaction_state_ir.py`、`tests/test_factors.py`、`tests/test_sequence_solver_shadow.py` 指定五 case shadow/candidate 测试共 22 个通过；`verify_sequence_problem_shadow.py`、`verify_sequence_solver_diagnostics.py`、`verify_candidate_sandbox.py` 通过。
- remaining gap: 新 worktree 缺少 gitignored DA3 / GVHMR / SAM2 / tracking / frames 等输入，因此 Phase 0 full materialized gate 未在该 worktree 声称通过；`stick` 仍保留 `line_contact_lock_special_refinement` 为 nonblocking gap。

2026-07-29:

- branch: `refactor/interaction-state-production`
- commit: pending local commit after this update
- change: `InteractionStateIR` 接入 `build_sequence_problem_shadow()`，新增 `interaction_state_shadow` 输入摘要、canonical hash、状态分布统计和 diagnostics assemble reads；更新 sequence problem / diagnostics / candidate sandbox golden，使五 case problem manifest 开始显式携带 interaction state provenance。
- verification: `python -m pytest -q ...` 轻量范围 34 个测试通过；`verify_sequence_problem_shadow.py`、`verify_sequence_solver_diagnostics.py`、`verify_candidate_sandbox.py` 通过。
- remaining gap: `interaction_state_shadow` 仍是 `consumed_by_solver=False`，尚未进入真正 factor activation；mug/chair 当前 timeline 仍暴露为 mostly `inactive/free`，说明其正式 contact/semantic interaction source 尚需从 adapter/legacy trace 迁移；materialized candidate / Phase 0 full gate 在该 worktree 仍受 gitignored 外部输入缺失限制。

2026-07-29:

- branch: `refactor/interaction-state-production`
- commit: pending local commit after this update
- change: 新增 case-name-free `core/factors/activation.py`，用 `VisibilityState`、`ContactState`、`ContactMode`、`MotionMode`、`audio_event_ids` 为 `FactorKind` 生成 activation ledger；`build_sequence_problem_shadow()` 开始记录 `factor_activation_shadow`、policy 分布、active/downweighted/inactive frame totals；diagnostics assemble reads 增加 `factor_activation_shadow:canonical`。
- verification: 轻量范围 35 个 pytest 通过；`verify_sequence_problem_shadow.py`、`verify_sequence_solver_diagnostics.py`、`verify_candidate_sandbox.py` 通过。
- remaining gap: activation ledger 仍是 shadow-only、`consumed_by_solver=False`；下一步应把该 ledger 下沉到 `CompiledFactor` / runtime factor contract，而不是继续增加 object-specific executor。

2026-07-29:

- branch: `refactor/interaction-state-production`
- commit: pending local commit after this update
- change: 新增 case-name-free `core/factors/compiler.py`，把 `FactorSpec + FactorActivationLedger` 编译为 shadow-only `CompiledFactor`，记录 residual function ref、robust loss、base weight source、input ids、active/downweighted/inactive mask summary 和 gate provenance；`build_sequence_problem_shadow()` 开始携带 `compiled_factor_shadow`，diagnostics assemble reads 增加 `compiled_factor_shadow:canonical`。
- verification: 轻量范围 36 个 pytest 通过；`verify_sequence_problem_shadow.py`、`verify_sequence_solver_diagnostics.py`、`verify_candidate_sandbox.py` 通过。
- remaining gap: `CompiledFactor` 仍是 shadow-only、`consumed_by_solver=False`，尚未成为实际 executor 输入；mug/chair 的 interaction timeline 仍需补正式 contact/semantic state source，避免泛化状态估计依赖 legacy 空洞。

2026-07-29:

- branch: `refactor/interaction-state-production`
- commit: pending local commit after this update
- change: 新增 `core/solver/problem_contract.py`，把 `StateSpec`、geometry kind、Measurement shadow、ContactConstraint shadow、InteractionState shadow 和 CompiledFactor shadow 固化为 generic `SequenceProblemContract`；`build_sequence_problem_shadow()`、diagnostics 和 golden summary 开始携带该 contract 的 canonical hash 和 count 对齐检查。
- verification: 轻量范围 37 个 pytest 通过；`verify_sequence_problem_shadow.py`、`verify_sequence_solver_diagnostics.py`、`verify_candidate_sandbox.py` 通过。
- remaining gap: `SequenceProblemContract` 仍是 shadow-only、`consumed_by_solver=False`，下一步需要把 runtime executor 的入口改成消费该 contract / compiled factors，而不是从 case adapter 直接选择 executor。
