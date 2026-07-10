# 下一步计划：最终成品 Evaluation 与 Ablation

本文档参考 `视频重建项目评估方案与最终汇报安排-文稿-转写结果.docx`，只聚焦下一步实验、评估和 ablation。暂时不展开最终 report 和 presentation。

## 1. 核心目标

现在最重要的不是继续堆展示视频，而是把“系统到底哪些部分有用”用评估证明出来。

需要回答的问题：

1. 最终重建是否在视觉上贴合输入视频？
2. 物体 6DoF translation / rotation 是否对五类 object 都成立？
3. human-object contact 是否发生在正确的人体部位和物体部位？
4. penetration / floating / support 是否物理合理？
5. temporal motion 是否稳定，同时不抹掉高速真实运动？
6. audio、VLM、LLM、contact anchor、sequence optimizer 各自贡献是什么？
7. 和相关工作相比，我们的输入条件、输出能力和指标优势在哪里？

一句话目标：

```text
用 hard metrics + VLM perceptual judge + LLM artifact auditor 证明：
我们的 audio/vision/contact/optimizer constraints 对最终 HOI reconstruction 有实际贡献。
```

## 2. 从会议转写提取出的评估原则

### 2.1 先评估自己声称有用的 loss / constraint

如果方法里用了 audio loss、vision/overlay loss、contact anchor、surface-gap anchor、smoothness、penetration loss，就必须有对应评估。

也就是说，方法部分每一个主要 constraint 都要在结果表里有一列或一个 ablation：

| 方法组件 | 必须评估什么 |
| --- | --- |
| audio timing / intensity | 有无 audio 时 contact timing、impact/kink、audio-window contact 是否改善 |
| visual overlay / mask | object render 是否贴合视频，mask/overlay proxy 是否提高 |
| contact anchor | hand/foot/body 到 object surface 是否更接近，anchor drift 是否下降 |
| surface-gap anchor | hand 不穿进 object，同时仍保持接触 |
| penetration loss | penetration rate/depth 是否下降 |
| smoothness / acceleration | jump 是否下降，motion 是否更稳定 |
| high-speed preservation | 快速转动/击打/弹跳是否没有被过度平滑 |
| VLM/LLM gate | gate 是否真实改变 residual weight、anchor update、freeze/interpolation |

### 2.2 能直接计算的不要交给 VLM

会议里明确说：如果可以用 geometry、mask、pose、SDF、distance 直接计算，就应该先算 hard metric。

VLM 适合做：

- hard metric 可能误判的 edge case
- 视觉可接受性判断
- 小穿透是否 visually acceptable
- overlay 是否“看起来合理”
- contact region 是否语义正确
- final render 是否明显错向、错物体、错接触

VLM 不适合做：

- 多帧精确 temporal smoothness
- 精确 penetration depth
- 6DoF numerical error
- object acceleration / jerk
- anchor drift 数值

因此最终评估必须分层：

```text
Hard metrics first
VLM checks only perceptual / ambiguous cases
LLM audits CSV / gate / failure localization
```

### 2.3 Temporal 不要只靠 VLM

会议里提到 VLM 对单帧 contact/region check 更合适，对多帧 temporal 不是很可靠。

Temporal 应该主要用数值指标：

- translation velocity / acceleration / jerk
- rotation angular velocity / angular acceleration
- object bbox / surface point temporal consistency
- human joint / hand trajectory smoothness
- contact anchor drift over time
- static-tail drift
- high-speed preservation

VLM 只看 selected windows：

- 是否有明显跳变
- 是否高速段方向错了
- 是否静止段还在抖
- 是否 render 和视频观感不一致

### 2.4 Penetration 需要区分“视觉可接受”和“物理严格”

会议里指出：轻微坐下/靠背 penetration 和手穿进篮球不是同一类问题。

所以 penetration 不能只报一个数字，要分：

- human part：hand / foot / torso / hip / mouth / head
- object part：surface / handle / rim / shaft / seat / back / support
- contact type：allowed shallow contact / illegal penetration / support / grasp
- depth：mean / max / frame ratio

最终判断要同时有：

- hard SDF/distance penetration
- VLM perceptual acceptability
- LLM failure localization

### 2.5 Comparison 要先做 method input table

会议里建议先列相关方法的输入输出条件，判断怎么公平比较。

比如有的方法需要：

- text prompt
- generated video
- known object mesh
- object category
- multi-view
- ground-truth human/object model
- manual contact labels

如果对方方法依赖更多输入，给它这些输入是 fair 的；只要比较表里写清楚即可。

因此下一步要先做：

```text
related_work_comparison_matrix.csv
```

列：

- Method
- Input video
- Text prompt
- Known object mesh
- Known object category
- Human model required
- Audio used
- Output human mesh
- Output object 6DoF
- Output contact
- Output penetration check
- Code/checkpoint available
- Feasible to run
- Fair comparison setting

## 3. 最终成品 Evaluation 设计

最终评估拆成四层：

```text
Layer A: Object-level 6DoF evaluation
Layer B: Human-part and object-part evaluation
Layer C: HOI contact / penetration / support evaluation
Layer D: Temporal + audio-aware plausibility evaluation
```

外加两类 judge：

```text
VLM visual judge: selected visual/perceptual checks
LLM CSV auditor: artifact consistency and failure stage localization
```

### 3.1 Layer A：Object-level 6DoF Metrics

目标：证明每个 object 都有统一 SE3，并且视频 overlay 合理。

输入 artifact：

- `object_pose.csv`
- `object_pose_pre_smooth.csv`
- `object_observations.csv`
- `object_correspondence.csv`
- `object_surface_points.csv`
- `object_semantic_points.csv`
- `physical_smooth_residuals.csv`
- `pose_jump_audit.csv`
- render overlay / camera3d / side_yz videos

指标：

| Metric | 含义 |
| --- | --- |
| SE3 schema completeness | 是否包含 `tx,ty,tz,qw,qx,qy,qz` |
| translation valid rate | 平移是否有限、连续、不 NaN |
| rotation valid rate | quaternion 是否归一化、连续、不 NaN |
| overlay proxy | mask / line / render 贴合度 |
| mask IoU / silhouette IoU | 有 mask 时计算 object silhouette overlap |
| line alignment | stick 等 line object 的轴线和 visible line / tracked endpoints 贴合 |
| geometry consistency | 物体长度、半径、scale 是否稳定 |
| pose jump count | visual/contact/smoothness spike 个数 |
| static-tail drift | 静止尾段最大漂移 |
| high-speed preservation | 高速段是否保留真实速度/角速度而不是被压平 |

注意：

- basketball/football 的 rotation 是 weakly observable，保留 SE3 schema，但 rotation 不作为强成败指标。
- mug/chair/stick 的 rotation 必须参与 render / part orientation / contact plausibility。

### 3.2 Layer B：Human-Part Metrics

目标：不只看 object，还要知道人体部位是否存在、是否稳定、是否能支撑 HOI 判断。

需要补齐或标准化 artifact：

- `human_parts.csv`
- `human_part_points.csv`
- `human_part_mesh_vertices.npz` 或 sampled lightweight points
- `human_part_visibility.csv`
- `human_part_confidence.csv`

人体部位：

- `left_hand`
- `right_hand`
- `left_foot`
- `right_foot`
- `torso`
- `hip`
- `back`
- `head`
- `mouth`
- optional fingers / palm / toe / heel

指标：

| Metric | 含义 |
| --- | --- |
| part availability | 需要的部位是否存在 |
| part confidence | GVHMR / HaMeR / VLM 检查置信度 |
| hand trajectory smoothness | 手部轨迹是否抖动 |
| foot trajectory smoothness | 脚部轨迹是否抖动 |
| mouth/head availability | mug drinking 等是否有 mouth/head 语义点 |
| part temporal continuity | 部位是否突然跳变或消失 |

不同 object 需要的重点人体部位：

| Case | Required human parts |
| --- | --- |
| basketball | hands, body, floor/support, optional feet |
| football | feet, legs, floor/support |
| mug | hand/palm/fingers, mouth/head |
| chair | hands, hip/back/torso, feet |
| stick | left/right palm, fingers, torso/arms |

### 3.3 Layer C：Object-Part Metrics

目标：非对称物体不能只看中心和 SE3，要看语义部位是否正确。

需要补齐或标准化 artifact：

- `object_parts.csv`
- `object_part_surface_points.csv`
- `object_part_pose.csv`
- `object_part_visibility.csv`
- `object_part_semantic_confidence.csv`

物体部位：

| Case | Object parts |
| --- | --- |
| basketball | surface, contact_patch |
| football | surface, contact_patch |
| mug | body, handle, rim, bottom |
| chair | seat, back, top_rail, legs, feet |
| stick | shaft, endpoints, grip_regions |

指标：

| Metric | 含义 |
| --- | --- |
| object-part coverage | 关键 object parts 是否存在 |
| part-level overlay | part projection 是否贴合视频区域 |
| semantic consistency | handle/rim/top rail/shaft 等语义是否稳定 |
| asymmetric rotation plausibility | mug handle/rim、chair back、stick shaft 朝向是否合理 |
| part visibility consistency | 遮挡时不应误更新真实 part pose |

### 3.4 Layer D：HOI Contact Metrics

目标：把当前 single contact proxy 升级成 part-pair contact。

标准 artifact：

- `hoi_contact_pairs.csv`
- `hoi_contact_intervals.csv`
- `hoi_contact_metrics.csv`

推荐 schema：

```text
frame,
human_part,
object_part,
observed,
expected,
persistent,
rel_static,
min_distance_m,
surface_gap_m,
penetration_depth_m,
contact_confidence,
contact_state,
source
```

指标：

| Metric | 含义 |
| --- | --- |
| Contact GT F1 | 有人工 label 才算 |
| Contact Proxy | 无 GT 时按 part-to-surface distance / confidence |
| Contact Interval Recall | 持续接触段是否覆盖 |
| Contact Drift | rel_static contact 中人体部位在 object local 坐标漂移 |
| Contact Switch Accuracy | stick 单手/双手切换、mug grasp/release 是否合理 |
| Part Correct Ratio | 最近人体部位是否等于 expected human part |
| Contact Gap | expected contact 帧中离 surface 的距离 |

### 3.5 Layer E：Penetration / Non-Collision Metrics

目标：替代当前 object-level penetration proxy，做真实 human-object geometry 检查。

输入：

- human mesh vertices / sampled part points
- object sphere / capsule / mesh SDF
- object SE3 trajectory

标准 artifact：

- `penetration_metrics.csv`
- `non_collision_metrics.csv`
- `part_penetration_metrics.csv`

公式：

```text
penetration_rate = #penetrating_vertices / #checked_vertices
penetration_frame_ratio = #frames_with_penetration / #valid_frames
penetration_depth_mean = mean(max(0, -sdf(v)))
penetration_depth_max = max(max(0, -sdf(v)))
non_collision_score = 1 - penetration_frame_ratio
```

必须分部位：

```text
human_part, object_part, contact_type, allowed_or_illegal
```

否则 contact 和 penetration 会混在一起。

解释规则：

- grasp/support 的浅层接触可接受，但要报告 depth。
- hand 深穿 ball、foot 深穿 ball、mug 穿手，属于严重失败。
- chair torso/back 轻微 leaning penetration 和 hand-object penetration 不应同权重。

### 3.6 Layer F：Temporal / Motion Plausibility Metrics

目标：判断 motion 是否稳定、物理可信，同时保留高速真实运动。

指标：

| Metric | 含义 |
| --- | --- |
| object translation velocity | `||T_t - T_{t-1}||` |
| object translation acceleration | `||T_t - 2T_{t-1} + T_{t-2}||` |
| object jerk | `||Δ^3 T||` |
| rotation angular velocity | quaternion relative angle |
| rotation angular acceleration | angular velocity difference |
| human joint smoothness | human joints second difference |
| contact anchor smoothness | contact anchor local/world trajectory second difference |
| static-tail drift | static interval object/human drift |
| high-speed preservation | audio/visual impact windows 中应保留高速变化 |
| over-smoothing score | impact/kick/spin 被压平的程度 |

关键点：

- stick 高速转动不能用普通 smoothness 强行压平。
- football kick / basketball bounce 的 acceleration spike 如果发生在 audio/contact event 附近，是物理合理的。
- acceleration spike 如果发生在非 contact / 非 high-speed interval，才是 artifact。

新增 audio-aware temporal 指标：

```text
accel_at_audio_events
accel_in_flight
contact_ratio_audio_windows
impact_timing_error_frames
```

## 4. VLM / LLM Final Judge 设计

### 4.1 VLM 只看 selected evidence

不要让 VLM 看整段视频然后泛泛打分。应该把 hard metrics 找出的关键帧/窗口喂给 VLM：

- overlay worst frames
- max penetration frames
- max contact gap frames
- anchor drift spike frames
- high-speed motion windows
- static-tail drift windows
- render failure windows

每个 query 都要给：

- original frame
- mask/overlay frame
- contact overlay frame
- final render frame
- optional before/after render
- hard metric values
- one specific question

### 4.2 VLM checklist

每个 representative frame/window 问固定问题：

```json
{
  "visibility": 1-5,
  "object_overlay": 1-5,
  "contact_correctness": 1-5,
  "part_correctness": 1-5,
  "support_consistency": 1-5,
  "penetration_absence": 1-5,
  "temporal_plausibility": 1-5,
  "overall_plausibility": 1-5,
  "short_reason": "...",
  "failure_stage_hint": "observation|contact|pose|optimizer|render|human_refine|unclear"
}
```

VLM 不直接改 pose，只输出 visual judgment。

### 4.3 LLM CSV auditor

LLM 读：

- `pipeline_manifest.json`
- `stage_audit_gates.csv`
- `vlm_trace/04_gating/gate_timeline.csv`
- `anchor_state.csv`
- `physical_smooth_residuals.csv`
- `pose_jump_audit.csv`
- `optimizer_decisions.csv`
- `hoi_contact_metrics.csv`
- `penetration_metrics.csv`
- `temporal_plausibility_metrics.csv`

LLM 要回答：

- gate 是否真的影响 optimizer？
- anchor 是否 stale？
- contact 是否误判？
- penetration 主要来自哪个 human/object part？
- temporal spike 是否发生在合理的 audio/contact window？
- failure stage 应该定位到哪里？

输出：

- `llm_eval_summary.md`
- `llm_failure_stage_report.json`
- `qa_audit_report.html`

## 5. Ablation Benchmark 设计

### 5.1 必须比较真实 result directory

之前的问题是不同 method label 可能指向同一结果目录，导致数值一样。这次必须强制：

```text
每个 method = 一个真实 run variant
不能拿同一个 result directory 冒充不同 method
```

### 5.2 Required variants

每个 case 至少需要：

| Method | 目的 |
| --- | --- |
| `full_audio_vlm_llm` | 完整方法 |
| `no_audio` | 去掉 audio timing / audio loss |
| `no_vlm` | 去掉 VLM gate |
| `no_llm` | 去掉 LLM semantic prior / CSV audit |
| `no_contact_anchor` | 去掉 contact anchor residual |
| `no_surface_gap_anchor` | 去掉 surface-gap / body-side contact refine |
| `no_sequence_smooth` | 去掉 velocity/acceleration optimizer |
| `no_static_tail` | 去掉 static-tail freeze |
| `object_only` | 不做人侧 refinement，只看 object pipeline |
| `human_refine_only_no_audio` | 有 human refine，但不用 audio records |
| `oracle_contact` | 只有人工 contact label 存在时可跑 |

如果时间不够，最低优先级：

```text
full_audio_vlm_llm
no_audio
no_vlm
no_contact_anchor
no_sequence_smooth
object_only
```

### 5.3 Variant 命名规范

统一 result name：

```text
eval_full_audio_vlm_llm
eval_no_audio
eval_no_vlm
eval_no_llm
eval_no_contact_anchor
eval_no_surface_gap_anchor
eval_no_sequence_smooth
eval_no_static_tail
eval_object_only
eval_human_refine_no_audio
eval_oracle_contact
```

每个 result 必须写：

- `run_config.json`
- `ablation_flags.json`
- `pipeline_manifest.json`
- `evaluation_summary.json`
- `hoi_evaluation_summary.json`

### 5.4 Ablation 表

主表：

| Case | Method | Object Overlay ↑ | Contact Proxy ↑ | Contact Gap ↓ | Part Correct ↑ | Pen Rate ↓ | Pen Max ↓ | Non-Collision ↑ | Object Jerk ↓ | High-Speed Preservation ↑ | Static Drift ↓ | VLM Judge ↑ | Final Pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

Delta 表：

```text
delta(method) = metric(method) - metric(full)
```

重点展示：

- `full - no_audio`
- `full - no_vlm`
- `full - no_contact_anchor`
- `full - no_sequence_smooth`
- `full - object_only`

### 5.5 Audio-specific ablation

会议里特别强调有无 audio 的区别需要体现。

需要的指标：

| Metric | 解释 |
| --- | --- |
| audio event recall proxy | audio event 附近是否有真实 geometric contact |
| impact timing error | audio peak 到 contact frame 的偏差 |
| accel_at_events | event 附近是否保留合理 acceleration spike |
| accel_in_flight | 非 event 区间是否没有不合理 spike |
| contact interval correction | audio 是否帮助持续/短时 contact 分段 |
| with-vs-without audio VLM preference | VLM 是否更偏好 full audio render |

需要重点 case：

- basketball bounce / dribble
- football kick
- mug set-down / grasp / drinking
- chair drag/support if audio exists
- stick 可能 audio 弱，作为 audio 不一定有效的 honest failure case

### 5.6 VLM/LLM ablation

需要证明 VLM/LLM 不是摆设。

指标：

- gate count
- effective gate count
- gate changed residual weight count
- gate blocked anchor update count
- gate triggered freeze/interpolation count
- LLM audit detected failure count
- rerun/reweight count

输出：

- `gate_effectiveness_summary.csv`
- `vlm_llm_ablation_summary.csv`

如果 full 和 no_vlm 数值几乎相同，必须解释：

- 是否 VLM gates 太弱？
- 是否 hard constraints 已经足够？
- 是否 VLM 只做 audit 而没有影响 optimizer？
- 是否 query 太少或不在关键帧？

## 6. 相关工作比较计划

### 6.1 先做输入/输出条件表

不要直接说“比某 paper 好”。先做 comparison matrix。

候选：

- HOI-PAGE
- CHORE / PHOSA 类 HOI reconstruction
- InterCap / BEHAVE / ARCTIC metric conventions
- VideoPhy / PhyGenBench / VBench 类 video physical evaluation
- 其他可运行或可定性比较的方法

表：

| Method | Input | Known Object Mesh | Audio | Human Mesh | Object 6DoF | Contact | Penetration | Code Available | Feasible Run | Fair Setting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

### 6.2 两种比较层级

#### Quantitative runnable comparison

只有在代码/模型能跑、输入公平时做。

输出：

- same or similar video input
- same object if method requires known mesh
- same metrics if possible

#### Qualitative / protocol comparison

如果不能公平跑，就比较：

- input assumptions
- output capability
- failure examples
- visual reconstruction side-by-side
- whether method handles audio/contact/6DoF/human mesh

### 6.3 Fairness 规则

如果对方方法需要额外 object mesh，可以给它，因为这是它的方法条件。只要表里写清楚。

如果我们使用 generated video + known object asset，也要写清楚。

不能做的事：

- 用完全不同输入还直接比较数值。
- 不说明对方需要/不需要 object mesh。
- 只挑一个最好看的 visual case 当 quantitative result。

## 7. 数据规模计划

会议里提到五个例子对于 proof of concept 可以，但要更有说服力最好扩展。

分三档：

### Tier 0：当前五个主 case

必须完整：

- basketball
- football
- mug
- chair
- stick

每个都要 full + key ablations。

### Tier 1：最小可信扩展

目标：10-15 clips。

新增类型建议：

- ball bouncing / dribbling variants
- foot kick variants
- mug/cup grasp variants
- chair sit / drag variants
- stick / tool manipulation variants

### Tier 2：更强 publication-like evidence

目标：20+ clips。

包含：

- working cases
- failure cases
- occlusion-heavy cases
- fast-motion cases
- audio-helpful cases
- audio-not-helpful cases

本阶段先把 Tier 0 做完整，再决定是否扩到 Tier 1。

## 8. 需要实现或整理的代码接口

### 8.1 Evaluation module 整理

建议目录：

```text
scripts/shared/generic_contact_pipeline/core/evaluation/
  final_evaluator.py
  final_summary.py
  ablation_benchmark.py
  method_variant_registry.py
  object_6d_metrics.py
  human_part_metrics.py
  object_part_metrics.py
  hoi_contact_metrics.py
  penetration_metrics.py
  temporal_plausibility_metrics.py
  audio_ablation_metrics.py
  vlm_final_judge.py
  llm_eval_auditor.py
```

已有：

- `final_evaluator.py`
- `final_summary.py`
- `benchmark.py`
- `vlm_trace.py`

Tom layer 已有：

- `scripts/shared/evaluation/compute_hoi_interaction_metrics.py`
- `scripts/shared/human_ball/contact/object_geometry.py`
- `scripts/shared/human_ball/contact/refine_body_pose_contact.py`

下一步应统一入口，不要分散手跑。

### 8.2 标准输出目录

每个 result 下：

```text
evaluation/
  object_6d_metrics.csv
  human_part_metrics.csv
  object_part_metrics.csv
  hoi_contact_pairs.csv
  hoi_contact_intervals.csv
  hoi_contact_metrics.csv
  penetration_metrics.csv
  non_collision_metrics.csv
  temporal_plausibility_metrics.csv
  audio_ablation_metrics.csv
  vlm_final_judge.csv
  llm_eval_audit.json
  final_evaluation_summary.json
```

跨 case：

```text
samples_known_object/final_result_evaluation/
  final_evaluation_human_readable.md
  final_evaluation_detailed.csv
  final_evaluation_summary_manifest.json

samples_known_object/ablation_evaluation/
  ablation_table.csv
  ablation_delta_table.csv
  ablation_method_registry.csv
  ablation_method_registry_manifest.json
  ablation_report.md
```

### 8.3 CLI

最终建议：

```bash
# 五个 final result 汇总
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/tools/run_final_hoi_evaluator.py \
  --result-name benchmark_vlm_qwen \
  --output-dir samples_known_object/final_result_evaluation

# ablation 跑表：当前默认只包含已经有真实目录的六个 materialized variants
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/tools/run_ablation_evaluation.py \
  --cases basketball football mug chair stick \
  --methods full_audio_vlm_llm audio_enabled no_audio no_vlm no_llm no_contact_anchor \
  --require-existing
```

## 9. 执行顺序

### Phase 1：评估定义收敛

1. 冻结 final result / ablation result 命名。
2. 写 `method_variant_registry.yaml`。
3. 明确每个 metric 的输入 artifact、公式、输出列。
4. 修改文档：
   - `docs/final_result_evaluation_method_cn.md`
   - `docs/final_result_evaluation_method_en.md`
   - `docs/hoi_interaction_evaluation_method_en.md`

验收：

- 不同 metric 不再混名。
- proxy 和 GT 明确分开。
- object-level 和 HOI-level 明确分开。

### Phase 2：补齐 artifact 和 hard metrics

1. 标准化 human/object part artifact。
2. 输出 part-pair contact。
3. 输出 penetration / non-collision 分部位指标。
4. 输出 temporal / audio-aware 指标。
5. 重新生成五个 case final evaluation。

验收：

- 每个 case 都有 `evaluation/final_evaluation_summary.json`。
- 每个 case 都有 human/object/contact/penetration/temporal 五类指标。
- chair/stick/mug 这种 mesh/capsule 物体不能只用 sphere proxy 冒充最终 HOI metric。

### Phase 3：VLM/LLM final judge

1. 从 hard metrics 自动选关键帧/窗口。
2. 生成 VLM evidence panels。
3. 跑 VLM checklist。
4. 汇总 QA：
   - question
   - evidence path
   - raw response
   - parsed score
   - affected stage/constraint
5. LLM 读 CSV 做 failure localization。

验收：

- `qa_audit_report.html` 能直接看每个问题、证据和回答。
- VLM score 不覆盖 hard metric。
- LLM 能定位 observation/contact/pose/optimizer/render/human_refine。

### Phase 4：Ablation variants 生成

1. 跑/整理 `full_audio_vlm_llm`。
2. 跑 `no_audio`。
3. 跑 `no_vlm`。
4. 跑 `no_contact_anchor`。
5. 跑 `no_sequence_smooth`。
6. 跑 `object_only`。
7. 必要时补 `no_llm`, `no_static_tail`, `oracle_contact`。

验收：

- 每个 method 都指向真实不同 result directory。
- benchmark 禁止 same-result 冒充。
- 每个 result 都有 `run_config.json` 和 `ablation_flags.json`。

### Phase 5：Ablation 分析

1. 生成主 ablation table。
2. 生成 delta table。
3. 单独生成 audio ablation table。
4. 单独生成 VLM/LLM gate effectiveness table。
5. 写 short analysis：
   - 哪个 loss 有用？
   - 哪些 case audio 有用？
   - 哪些 case VLM/LLM 有用？
   - 哪些 failure 仍然存在？

验收：

- 能回答“有无 audio 是否改善？”
- 能回答“有无 VLM/LLM 是否改善？”
- 能回答“contact anchor / smooth / static-tail 各自贡献？”
- 能指出失败来自 object pose、human refine、contact 还是 temporal。

### Phase 6：相关工作比较准备

1. 做 method input/output matrix。
2. 选 1-2 个最可运行/最公平的 baseline。
3. 如果可跑，准备 same/similar input comparison。
4. 如果不可跑，准备 protocol comparison + visual side-by-side。

验收：

- 不需要现在完成所有 SOTA comparison，但必须有清晰可解释的 fairness table。

## 10. 最小可交付清单

如果时间紧，下一步至少交付：

1. 五个 case 的 final evaluation detailed table。
2. 五个 case 的 human-readable final table。
3. 五个 case 的 HOI interaction table。
4. full vs no_audio ablation。
5. full vs no_vlm ablation。
6. full vs no_contact_anchor ablation。
7. full vs no_llm ablation。
8. VLM/LLM QA audit summary。
9. Related work input/output comparison matrix。

`no_sequence_smooth` 和 `object_only` 仍是后续扩展 ablation，不属于当前默认六个已物化方法。

## 11. 最终文件清单

建议新增：

```text
docs/next_final_evaluation_ablation_plan_cn.md
docs/related_work_comparison_matrix.md
scripts/shared/generic_contact_pipeline/configs/evaluation/method_variant_registry.yaml
scripts/shared/generic_contact_pipeline/tools/run_ablation_evaluation.py
samples_known_object/ablation_evaluation/ablation_table.csv
samples_known_object/ablation_evaluation/ablation_delta_table.csv
samples_known_object/ablation_evaluation/ablation_method_registry.csv
samples_known_object/ablation_evaluation/ablation_method_registry_manifest.json
```

后续 metric 拆分后再新增：

```text
samples_known_object/ablation_evaluation/audio_ablation_table.csv
samples_known_object/ablation_evaluation/vlm_llm_gate_effectiveness.csv
samples_known_object/ablation_evaluation/contact_penetration_tradeoff.csv
```

建议扩展：

```text
docs/final_result_evaluation_method_cn.md
docs/final_result_evaluation_method_en.md
docs/hoi_interaction_evaluation_method_en.md
README.md
```

## 12. 一句话版本

下一步不是继续调单个视频，而是把最终结果评估做成科学实验：

```text
五个对象完整 final evaluation
+ audio/VLM/contact/smooth ablation
+ hard geometry metrics
+ VLM perceptual judge
+ LLM artifact audit
+ related-work fairness table
```

这样才能说明 pipeline 不是“看起来能跑”，而是每个核心约束都能被评估和证明。
