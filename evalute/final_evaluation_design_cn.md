# Final Evaluation Design CN

本文档定义 AudioHOI 下一版最终成品评估协议。它结合三类信息：

1. 当前 `benchmark_vlm_qwen` 五个 object 的最终产物；
2. human/audio/HOI layer 的最终数据与 `hoi_summary`；
3. 会议转写里的要求：能直接算的 metric 必须直接算，VLM 只用于 perceptual / ambiguous check；评估要能证明 audio、vision、contact、penetration、smoothness 等 constraints 的实际贡献。

当前已输出完整 final-result evaluation 的范围只包括两个球类：

- `basketball`
- `football`

这两个 case 有对齐的 final video、source video、object 6DoF、human/contact、temporal/audio 和 paired gate-trace evidence。mug 当前先不纳入最终表；chair/stick 尚未作为 `final_result/videos/` 下的完整交付视频发布。本文后续仍保留 mug/chair/stick 的指标设计，因为它们是下一步扩展目标，不代表当前已经输出完整最终评估结果。

核心原则：

```text
Hard metrics first. VLM only for visual ambiguity. LLM audits evidence and failure source.
```

## 1. 为什么要重设评估

当前 object-level final summary 已经能回答：每个 object 是否有 SE3 pose、是否有 overlay/contact proxy、是否有 jump/static drift。但它仍有三个问题：

1. `Penetration Rate` / `Floating Rate` 里仍有 object-pipeline proxy，不能替代真实 human-object geometry penetration。
2. `Overlay Proxy` 不能主要依赖 VLM。老师质疑点是对的：如果有 mask、render projection、2D silhouette，就应该直接计算。
3. 当前表不能证明方法组件贡献。比如开不开 VLM/LLM/audio 如果结果一样，就说明 ablation protocol 没有真正比较不同 run，或 gate 没有实际进入 optimizer。

因此下一版评估拆为三层：

```text
A. Final result hard metrics
B. VLM/LLM judge and audit
C. Ablation benchmark
```

其中 A 是主评估，B 是解释和边界检查，C 是证明每个组件有用。

## 2. 参考 HOI-PAGE 的方式，但不照搬

HOI-PAGE 的关键思想是 part-level affordance：先把 object 分成语义部位，再建立 human body part 和 object part 的关系，最后在优化中使用 part-level contact constraints。其项目页也明确描述了三步：object part decomposition、从参考 HOI video 抽取 masks/depths/human motion、再优化 object motion 并满足 part-level contact constraints。

我们应该参考的是：

- part-level human-object relation，而不是只有 object center；
- fitting / contact / penetration / smoothness 四类约束；
- 用 LLM/VLM 做 part/contact 语义辅助，但不直接生成连续 pose；
- 评估里要看 part-level contact 是否正确。

我们不能照搬的是：

- HOI-PAGE 是 text + known 3D objects 的 generation setting；我们是 video-conditioned reconstruction；
- 我们必须加 video overlay / reprojection metrics；
- 我们有 audio timing，所以需要 audio-window contact 和 event-vs-flight acceleration 指标；
- 我们要评估 generated video 中物体 6DoF 与人体重建的一致性。

## 3. 最终评估对象

每个 case 的最终目录：

```text
samples_known_object/<case>/results/benchmark_vlm_qwen/
```

human/HOI 目录：

```text
samples_known_object/<case>/results/gvhmr or human_gvhmr
samples_known_object/<case>/results/hands or human_hands
samples_known_object/<case>/results/contact_refine or human_contact_refine
samples_known_object/<case>/results/hoi_eval/hoi_interaction_metrics.json
samples_known_object/<case>/results/renders/*human_full_scene_3d*
```

跨 case 当前结果：

```text
final_result/evaluation/
```

## 4. Layer 1: Object 6DoF and Video Fit Metrics

目标：证明 object pose 是统一 SE3，并且投影贴合输入视频。

### 4.1 SE3 Completeness

输入：`object_pose.csv`

要求字段：

```text
frame,time,tx,ty,tz,qw,qx,qy,qz
```

指标：

```text
se3_valid = all required columns exist
translation_valid_rate = finite(tx,ty,tz) frames / total frames
rotation_valid_rate = finite(qw,qx,qy,qz) and |norm(q)-1| < eps frames / total frames
```

说明：basketball/football 的 rotation 是弱可观测 gauge，但 schema 仍必须存在；mug/chair/stick 的 rotation 必须进入 render 和 part orientation 检查。

### 4.2 Overlay / Reprojection Metrics

老师质疑的点：overlay 不应该主要靠 VLM。下一版 overlay 采用 hard metrics 为主。

输入：

- SAM2/object mask；
- rendered object mask / silhouette；
- `object_observations.csv` 中的 mask bbox / line / center；
- `object_correspondence.csv` / `line_correspondence.csv`；
- render overlay frames。

指标优先级：

1. **Silhouette IoU**：有 object mask 和 render mask 时使用。

```text
silhouette_iou_t = |M_render_t ∩ M_sam2_t| / |M_render_t ∪ M_sam2_t|
overlay_iou = mean_t(silhouette_iou_t)
```

实现规则：

```text
如果 rendered object mask 已存在：直接使用。
如果 rendered object mask 缺失：final evaluator 必须生成 evaluation-only render mask。
生成目录：samples_known_object/<case>/results/<result>/evaluation/render_masks/
生成出的 mask 只用于评估，不替代人看的 overlay 视频，也不回写 pose。
```

当前实现分两类 source：

```text
generated_eval_proxy_render_mask_iou:
  由 pose + observation radius/line/bbox 生成的轻量 proxy mask。

generated_eval_full_geometry_mask_iou:
  由 URDF/Articraft geometry + object_pose.csv 直接 rasterize 得到的 full geometry mask。
```

当前 basketball/football 仍使用 proxy circle mask；mug/chair/stick 已可在存在 URDF 时生成 full geometry mask。评估协议上 render mask 已经是标准产物，不能长期用 VLM 代替。

2. **Reprojection Chamfer**：适合 chair/mug/stick 等 mesh/line edge。

```text
D_chamfer = 0.5 * mean_{p in edge_render} min_{q in edge_obs} ||p-q||
          + 0.5 * mean_{q in edge_obs} min_{p in edge_render} ||q-p||
```

3. **Line Alignment**：适合 stick。

```text
angle_error_t = acos(|dot(axis_render_2d, axis_observed_2d)|)
center_error_px_t = ||center_render_2d - center_observed_2d||
endpoint_error_px_t = mean endpoint reprojection distance when visible
```

4. **Mask Coverage / False Coverage**：避免 render 压在人身上或漏出过多。

```text
coverage = |M_render ∩ M_obs| / |M_obs|
false_coverage = |M_render - M_obs| / |M_render|
```

VLM 只用于：

- mask 本身不可靠；
- occlusion 下 hard IoU 不公平；
- 判断“视觉上是否可接受”；
- 判断 render 是否明显错 object / 错方向。

最终表里必须拆开：

```text
Overlay Hard Score
Overlay VLM Judge
Overlay Conflict Flag
```

如果 hard score 和 VLM 冲突，不能互相覆盖，要报告 conflict。

## 5. Layer 2: Human/Object Part Metrics

目标：从 HOI-PAGE 的 part-level idea 出发，不只看整体人和整体物体。

### 5.1 Human Part Coverage

标准 human parts：

```text
left_hand,right_hand,left_foot,right_foot,torso,hip,back,head,mouth
```

输入：GVHMR + HaMeR + contact_refine。

输出 artifact：

```text
evaluation/human_parts.csv
evaluation/human_part_points.csv
evaluation/human_part_metrics.csv
```

当前已实现：

```text
evaluation/human_parts.csv
evaluation/object_parts.csv
evaluation/object_part_vocab_map.csv
evaluation/part_metrics.csv
```

`human_parts.csv` 记录标准人体部位是否有 GVHMR / hands / contact-pair 证据。`object_parts.csv` 使用 canonical object part 参与指标计算，并在 `raw_parts` 列保留原始命名。`object_part_vocab_map.csv` 记录每个 raw part 如何映射到 canonical part，以及它来自 case config、object surface points 还是 HOI contact pairs。

这一步不是把语义问题藏起来，而是把“指标用词”和“证据原词”分开：

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

因此 `object_part_contact_coverage` 不是字符串匹配分数，而是：在归一化后的 object semantic part slots 中，有多少 part 获得了 surface/contact evidence。

指标：

```text
part_available_rate(part) = valid frames for part / total frames
part_smoothness(part) = mean ||x_t - 0.5(x_{t-1}+x_{t+1})||
part_confidence(part) = detector/model confidence if available
```

### 5.2 Object Part Coverage

标准 object parts：

| Case | Object parts |
| --- | --- |
| basketball | center, surface, support_region |
| football | center, surface, support_region |
| mug | cup_body, handle, rim, bottom |
| chair | legs, seat, back, top_rail, stretcher, hole, feet |
| stick | shaft, grip_region, support_region |

输出 artifact：

```text
evaluation/object_parts.csv
evaluation/object_part_surface_points.csv
evaluation/object_part_metrics.csv
```

指标：

```text
object_part_available_rate
object_part_overlay_score
object_part_semantic_consistency
asymmetric_rotation_plausibility
```

mug 的 handle/rim 朝向、chair 的 back/top rail、stick 的 shaft/endpoints 必须进入 part-level rotation plausibility。

## 6. Layer 3: HOI Contact Metrics

目标：contact 不再是一个 scalar proxy，而是 part-pair relation。

标准 artifact：

```text
evaluation/hoi_contact_pairs.csv
evaluation/hoi_contact_intervals.csv
evaluation/hoi_contact_metrics.csv
```

当前已实现的 artifact 来源：

```text
object_contact_points.csv / contact_candidates.csv:
  提供 frame、human_part、human_side、object_part、contact_active、contact_confidence、contact_u/v。

anchor_state.csv:
  提供 contact_persistent、anchor_update_allowed、pose_anchor_allowed、anchor_action。

hoi_interaction_metrics.json:
  提供 human/audio/HOI 层的 aggregate metrics，例如 contact_frame_ratio、contact_gap_mm、part_correct_ratio。
```

当前 `hoi_contact_pairs.csv` 是标准 part-pair 审计表，不再只保留一个 contact scalar。若某些 object family 没有 stable/observed local coordinate，`contact_anchor_drift_mean` 会留空；这代表当前 artifact 缺失对应证据，而不是接触一定稳定。

`hoi_contact_pairs.csv` schema：

```text
frame,human_part,object_part,expected,observed,persistent,rel_static,
min_distance_m,surface_gap_m,penetration_depth_m,contact_confidence,
contact_state,source
```

指标：

### 6.1 Contact Frame Ratio

```text
contact_frame_ratio = #frames with observed contact / #valid frames
```

### 6.2 Expected Contact Recall

有 expected contact interval 时：

```text
contact_interval_recall = #expected-contact frames with observed contact / #expected-contact frames
```

### 6.3 Contact Gap

```text
contact_gap_t = |distance(human_part, object_surface) - target_gap|
contact_gap_mean = mean over expected-contact frames
```

`target_gap` 通常是 0 或小的 surface offset。这个指标直接检查 floating hand / floating foot。

当前机器表额外输出 `contact_proxy`：

```text
contact_proxy = exp(-contact_gap_mm / 50)
```

其中 50 的单位是 mm，是当前 evaluator 的默认距离尺度。`contact_proxy` 越接近 1，说明 human part 和 object surface 越贴近；越接近 0，说明 expected contact 更像 floating。这个 proxy 不能替代 GT F1：有人工 contact label 时才计算 `Contact F1`，没有 GT 时只能读 `contact_proxy` 和 `contact_gap_mm`。

### 6.4 Part Correct Ratio

```text
part_correct = 1[closest_human_part == expected_human_part and closest_object_part == expected_object_part]
part_correct_ratio = mean(part_correct over expected-contact frames)
```

### 6.5 Contact Drift

持续抓握 / 支撑时，人体接触点在 object local coordinate 应该稳定：

```text
local_contact_t = T_object_t^{-1} * p_human_contact_t
contact_drift = std_t(local_contact_t over a persistent interval)
```

用于：mug grasp、stick palm grip、chair top rail、ball hand dribble contact。

### 6.6 Contact Switch Accuracy

用于 stick 单手/双手切换、mug grasp/release、football foot contact。

```text
switch_error = predicted_switch_frame - expected_switch_frame
switch_accuracy = #correct switches / #expected switches
```

expected switch 可以来自 manual labels、VLM event record、audio/visual contact records。

## 7. Layer 4: Penetration / Floating / Support Tradeoff

会议里强调：penetration 和 contact/floating 有 tradeoff。不能简单说所有 penetration 都同样坏，也不能为了不 penetration 让手离物体很远。

### 7.1 几何 penetration

输入：human part points / vertices + object SDF or sphere/capsule/mesh geometry。

公式：

```text
sdf(v) > 0 outside
sdf(v) = 0 surface
sdf(v) < 0 inside
penetration_depth(v) = max(0, -sdf(v))
penetration_frame_ratio = #frames with any depth > eps / #valid frames
penetration_vertex_ratio = #penetrating checked points / #checked points
penetration_depth_mean = mean positive penetration depth
penetration_depth_max = max positive penetration depth
```

### 7.2 Floating / contact gap

```text
floating_gap_t = max(0, distance_to_surface_t - allowed_contact_band)
floating_rate = #expected-contact frames with floating_gap_t > eps / #expected-contact frames
```

### 7.3 Contact-aware physical score

会议里强调的 penetration / floating tradeoff 要明确进入解释：只降低 penetration 没有意义，因为把手或物体推远也能让 penetration 变成 0；只降低 floating 也没有意义，因为把人体压进物体内部会让接触距离变小。最终 physical score 必须同时看 expected contact、surface gap、penetration depth 和 support state。

```text
contact_success_t = expected_contact_t and surface_gap_t within allowed band
illegal_penetration_t = penetration_depth_t > allowed_shallow_contact_depth
floating_failure_t = expected_contact_t and surface_gap_t > allowed_contact_band

tradeoff_score = contact_success_rate
               * (1 - penetration_frame_ratio)
               * (1 - floating_rate)
```

解释规则：

```text
low penetration + high floating  = 可能只是物体/手离得太远，不是好结果
low floating + high penetration  = 可能是穿模换来的“贴合”，不是好结果
low penetration + low floating + correct contact interval = 真正物理合理
```

为了解决 tradeoff，最终不单独优化 penetration 或 floating，而是同时报告并组合：

```text
physical_contact_score = exp(- contact_gap_mean / sigma_gap)
non_penetration_score = exp(- penetration_depth_mean / sigma_pen)
tradeoff_score = sqrt(physical_contact_score * non_penetration_score)
```

这能避免两种作弊：

- 手离物体很远，penetration=0，但 contact 失败；
- 手深插进物体，contact gap 小，但 penetration 严重。

### 7.4 Penetration severity 分级

按 human part 和 contact type 分权重：

| Case | 轻微可接受 | 严重失败 |
| --- | --- | --- |
| chair | torso/back 轻微靠背 penetration | hand/leg 深穿 chair mesh |
| ball | 手/脚浅接触 surface | hand/foot 深穿球体 |
| mug | fingers 浅接触 handle/body | mug 穿手、rim 穿脸 |
| stick | palm 浅接触 shaft | shaft 穿 torso/head |

输出：

```text
evaluation/penetration_metrics.csv
evaluation/floating_metrics.csv
evaluation/contact_physics_tradeoff.csv
```

## 8. Layer 5: Temporal and Audio-Aware Metrics

Temporal 主要用 hard metrics，不靠 VLM。

### 8.1 Object motion

```text
v_t = ||T_t - T_{t-1}||
a_t = ||T_t - 2T_{t-1} + T_{t-2}||
j_t = ||T_t - 3T_{t-1} + 3T_{t-2} - T_{t-3}||
```

Rotation：

```text
omega_t = angle(q_{t-1}^{-1} q_t)
alpha_t = |omega_t - omega_{t-1}|
```

### 8.2 Event-vs-flight split

对 audio/contact event window：

```text
event_window = frames within ±k of audio/contact event
accel_at_events = mean(a_t in event_window)
accel_in_flight = mean(a_t outside event_window)
```

解释：

- basketball bounce、football kick 在 event 附近 acceleration 高是合理的；
- 非 event 区间出现 spike 是 artifact；
- stick 高速转动不应该被过度 smooth，但方向和长度必须稳定。

### 8.3 Static-tail stability

```text
static_tail_drift = max_t ||T_t - T_static_ref|| over static interval
static_rotation_drift = max_t angle(q_static_ref^{-1} q_t)
```

### 8.4 High-speed preservation

```text
high_speed_recall = #GT/proxy high-speed windows preserved / #GT/proxy high-speed windows
oversmooth_rate = #event windows where full method suppresses expected motion / #event windows
```

GT/proxy high-speed window 来自：audio events、visual velocity peaks、line/mask motion peaks。

### 8.5 当前代码落地

当前实现已经从单一 `jump_count` 升级到 motion-regime-aware temporal hard metrics：

- `temporal_plausibility_metrics.csv/json`
- `translation_spike_count`
- `rotation_spike_count`
- `event_aligned_spike_count`
- `non_event_spike_count`
- `high_speed_recall`
- `oversmooth_rate`
- `static_tail_drift_m`
- `temporal_failure_intervals`

其中 spike threshold 使用 per-sequence robust threshold，而不是固定常数：

```text
threshold = max(floor, median(values) + 3 * MAD(values), percentile_95(values))
```

event window 来自 `audio_contact_csv` / `contact_records.csv` 的 event frames。event 内 spike
视为可能合理的 bounce/kick/impact；非 event spike 更可能是 temporal artifact。

实现位置：

- `scripts/shared/generic_contact_pipeline/core/evaluation/final_hoi/temporal_plausibility_metrics.py`
- `final_result/evaluation/<case>/evaluation/temporal_plausibility_metrics.csv`

## 9. VLM / LLM Judge 的最终位置

### 9.1 VLM 不替代 hard metrics

VLM 只审以下问题：

- hard overlay metric 和视觉观感冲突；
- mask occlusion 导致 IoU 不公平；
- penetration 很小，但是否 visually acceptable；
- contact part 是否语义正确；
- final render 是否明显违背动作语义。

VLM query 必须是简单问题，而不是泛泛问“好不好”：

```json
{
  "query_type": "perceptual_contact_or_overlay_check",
  "evidence": ["original_frame", "mask_overlay", "contact_overlay", "render_frame"],
  "question": "Does the rendered object visually overlap the real target object, ignoring occluded regions?",
  "allowed_answers": ["pass", "unclear", "fail"],
  "score_1_to_5": 4,
  "short_reason": "...",
  "failure_stage_hint": "observation|contact|pose|optimizer|render|human_refine|unclear"
}
```

### 9.2 LLM auditor

LLM 读 CSV/JSON，不看图为主：

- metric summary；
- gate timeline；
- optimizer decisions；
- ablation deltas；
- failure flags。

LLM 输出：

```text
llm_eval_summary.md
llm_failure_stage_report.json
llm_ablation_interpretation.md
```

LLM 的任务是解释，不是算 pose。

### 9.3 当前已实现的 QA artifact

`run_final_hoi_evaluator.py` 现在会同时刷新 hard metrics 和 QA audit。标准输出位置：

```text
samples_known_object/<case>/results/<result_name>/vlm_trace/06_evaluation/
```

当前已实现文件：

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

关键字段：

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

当前默认是 `metric-grounded dry-run final judge`：它根据 hard metrics 和已有 trace 生成固定 schema 的 QA 记录，`changed_optimizer_behavior=0`。这不是假装 Qwen 已经复判，而是为了固定最终评估接口，保证每个结论都能追溯到 evidence/question/raw/parsed。后续真实 Qwen final judge 应替换 raw answer 来源，但保留同一 schema。

其中 `pipeline_qa_summary.*` 和 `vlm_eval_*` 含义不同：

- `pipeline_qa_summary.*` 聚合 pipeline stage 中已经发生的 VLM/LLM 问答、gate、affected constraint，以及是否改变 optimizer/gate 行为；
- `vlm_eval_*` 是 final evaluator 对最终结果的 representative interval 审查问题；
- 两者都不直接生成 pose。

## 10. 最终汇总表设计

### 10.1 Human-readable table

只放这 6 个大类，给人快速读结论。Audio / VLM / LLM / anchor ablation 的差异放在 `final_result/evaluation/ablation/ablation_table.csv` 和 `ablation_report.md`，不塞进最终人读表。

| Case | Object 6DoF | Visual Overlay | Contact/Anchor | Physical | Temporal |
| --- | --- | --- | --- | --- | --- |

当前产物：

```text
final_result/evaluation/final_evaluation_human_readable.md
```

### 10.2 Detailed CSV

```text
case,result_name,n_frames,
se3_valid,translation_valid_rate,rotation_valid_rate,
overlay_iou,overlay_chamfer_px,line_angle_error_deg,
contact_frame_ratio,contact_gap_mm,part_correct_ratio,contact_drift_mm,
penetration_frame_ratio,penetration_depth_mean_mm,penetration_depth_max_mm,
floating_rate,tradeoff_score,
object_jerk,rotation_jerk,static_tail_drift_m,high_speed_recall,
contact_ratio_audio_windows,accel_at_events,accel_in_flight,
vlm_overlay_judge,vlm_contact_judge,llm_failure_stage,final_pass
```

## 11. 当前 human/audio/HOI 结果如何进入评估

当前已有 `samples_known_object/hoi_interaction_evaluation/hoi_summary.md`，里面有：

- `pen_frame_ratio`
- `pen_depth_max_mm`
- `contact_frame_ratio`
- `contact_gap_mm`
- `part_correct_ratio`
- `contact_ratio_audio_windows`
- `accel_at_events`
- `accel_in_flight`
- `object_jerk`
- `grasp_stability_mm`
- `mdev_star_mm`

下一步不是丢掉这些，而是把它们接进统一 final evaluator：

```text
object-level final_evaluator
+ HOI interaction metrics
+ new overlay hard metrics
+ new ablation delta metrics
= final_evaluation_detailed.csv
```

### 11.1 当前 handoff artifact 缺口与 TODO

当前结果已连接前面的 object/contact/audio pipeline。checked artifact 显示：

- `pose6d_object_proxy_anchor_refined/object_pose6d_sharedcam_contactphase_trajectory.csv`
  保留了最终 object SE3、support proxy、contact frame、audio contact frame 等字段；
- `human_audio_semantics/contact_records.csv` 保留了 audio/visual contact events、contact target、contact state、manipulation weights；
- `human_audio_semantics/body_surface_contacts.csv` 保留了 body surface contact point、contact part、contact region、surface distance；
- `benchmark_vlm_qwen` 保留了 gate timeline、optimizer decisions、pre-smooth pose、residuals。

但是这些还不足以让 final evaluator 计算下面三类 hard metrics：

| Missing metric | 为什么当前不能硬算 | 需要新增/保留的 artifact |
| --- | --- | --- |
| `anchor_drift` | basketball/football 的 `anchor_state.csv` 有 anchor gate，但 `stable_local_x/y/z/s` 和 `observed_local_x/y/z/s` 为空。球类当前使用 surface-gap / center-depth anchor，不是 mug/stick 那种 object-local stable grasp point。 | `final_anchor_state.csv` |
| `static_tail_drift` | final result 没有标注哪一段应该静止。basketball/football 也不一定有明确静止尾段，不能自动把最后几帧当 static tail。 | `final_motion_intervals.csv` |
| `floating_rate` / support gap | final HOI 有 penetration/contact surface distance，但没有 final 3D object-to-floor/support surface gap。pipeline trace 有 `support_gap_px/floor_v` 等 2D/proxy 字段，但不能冒充 final 3D floating hard metric。 | `final_support_state.csv` |

因此当前人读表把这些项放在 Evidence Notes，而不是在主表里显示 `n/a`。这不是“没跑完”，而是 final-result handoff contract 还没把 evaluator 需要的三类证据物化出来。

下一步 TODO：

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

有了这三个 artifact 后，evaluator 才能把当前 unavailable 项升级为：

- `contact_anchor_drift_mean/max`
- `floating_rate`
- `support_consistency`
- `static_tail_drift_m`
- `static_rotation_drift_rad`

## 12. Acceptance Criteria

评估系统通过的标准：

1. 五个 case 都能输出 detailed final evaluation。
2. overlay 有 hard metric，不能只靠 VLM。
3. penetration/floating/contact gap 同时报告，体现 tradeoff。
4. VLM/LLM 的每个结论都能追溯 evidence、question、raw answer、parsed score。
5. ablation 中每个 method 对应真实不同 result directory。
6. 能回答：audio 是否有用、VLM 是否有用、contact anchor 是否有用、smooth 是否有用。
7. failure stage 能定位到 observation/contact/object pose/human refine/render/evaluator，而不是只写 render bad。
