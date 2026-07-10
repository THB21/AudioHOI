# 最终结果评估方法说明

本文档说明当前 generic HOI pipeline 如何评估最终结果。这里的“最终结果”指每个 case 当前选定的最终目录，默认是 `benchmark_vlm_qwen`。它不是 benchmark 对比表，不比较 baseline、ablation 或 oracle-contact。

## 评估对象

每个 case 的最终结果必须包含统一 SE3 pose：

- 平移：`tx, ty, tz`
- 旋转四元数：`qw, qx, qy, qz`

篮球、足球、mug、chair、stick 都采用这个 schema。球类因为几何近似球体，旋转通常是弱约束或 identity gauge；这表示 schema 上是 6DoF，但球面自转不作为强可观测指标。mug、chair、stick 的 rotation 更直接参与 render 和物理一致性判断。

## 三层评估

### 1. Hard Metrics

Hard metrics 只从机器 artifact 计算，不依赖 VLM 解释：

- `Contact F1`：只有存在人工 contact label 时才计算。
- `Contact Proxy`：没有人工标签时的接触可信度代理指标，不能当作 F1。
- `Overlay Proxy`：mask/track/line/render overlay 的代理贴合度。
- `Anchor Drift`：稳定 anchor 与当前观测 anchor 的漂移。
- `Penetration Rate`：人/物体或支撑关系中的穿透代理。
- `Floating Rate`：物体悬浮或支撑 gap 代理。
- `Jump Count`：`pose_jump_audit.csv` 中的跳变帧数。
- `Static Drift`：静止尾段最大漂移。
- `Geometry Spread`：物体几何长度/尺度变化，stick 等刚体不应忽长忽短。

### Hard Metrics 公式

记第 `t` 帧的预测接触为 `p_t ∈ {0,1}`，人工标签为 `y_t ∈ {0,1}`，接触置信度为 `c_t ∈ [0,1]`，anchor 漂移为 `d_t`，穿透/悬浮代理量为 `pen_t` / `float_t`。

**Contact F1** 只在人工标签存在时计算：

```text
TP = Σ 1[p_t = 1 and y_t = 1]
FP = Σ 1[p_t = 1 and y_t = 0]
FN = Σ 1[p_t = 0 and y_t = 1]
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
Contact F1 = 2 * Precision * Recall / (Precision + Recall)
```

没有人工标签时，`Contact F1` 留空，不用 proxy 冒充。

**Contact Proxy**：

```text
Contact Proxy = mean_t( c_t if contact_observed_t = 1 else 0 )
```

**Overlay Proxy** 优先使用 mask/track/line 可信度：

```text
Overlay Proxy = mean_t(clip(mask_iou_t or observation_conf_t or track_conf_t, 0, 1))
```

如果没有直接置信度，则用 jitter 反推：

```text
Overlay Proxy = mean_t( 1 / (1 + max(0, jitter_px_t) / 20) )
```

**Anchor Drift**：

```text
d_t = || observed_local_t - stable_local_t ||_2
```

若是 line object，则使用局部杆坐标：

```text
d_t = | observed_local_s_t - stable_local_s_t |
```

报告：

```text
Anchor Drift Mean = mean_t(d_t)
Anchor Drift Max = max_t(d_t)
```

**Penetration Rate / Floating Rate**：

若有 residual：

```text
Penetration Rate = mean_t(1[penetration_depth_t > 1e-4])
Floating Rate = mean_t(1[floating_gap_t > 1e-3])
```

若没有显式 residual，则当前 evaluator 会从 `contact_depth_offset_m` 生成 proxy：

```text
penetration proxy: contact_depth_offset_m < -1e-6
floating proxy: contact_depth_offset_m > 1e-6
```

**Jump Count**：

```text
Jump Count = Σ_t 1[
  visual_spike_t = 1
  or contact_spike_t = 1
  or smoothness_spike_t = 1
]
```

**Static Drift**：

```text
Static Drift Max = max_t(static_tail_drift_m_t)
```

**Geometry Spread**：

```text
Geometry Spread = max_t(length_t) - min_t(length_t)
```

主要输入：

- `object_pose.csv`
- `object_observations.csv`
- `contact_candidates.csv`
- `anchor_state.csv`
- `physical_smooth_residuals.csv`
- `pose_jump_audit.csv`

### 2. VLM Visual Judge

VLM visual judge 是视觉检查层，问题类似：

- 目标物体是否可见并被正确跟踪？
- render overlay 是否贴合真实 object？
- contact points 是否贴到手、脚、地面、桌面等合理位置？
- 是否出现 floating、penetration、跳变、方向错误或长度变化？
- 整体交互是否物理可信？

当前报告把 pipeline 内部 VLM gate 和最终 VLM judge 分开保存。VLM 不生成 pose，也不覆盖几何事实，只给出视觉审查分数和失败原因。

### 3. LLM CSV Auditor

LLM CSV auditor 读取 CSV/JSON artifact，检查 pipeline 是否自洽：

- gate 是否真的影响 optimizer？
- anchor 是否 stale？
- static-tail freeze 是否生效？
- pose jump 是否被记录和抑制？
- 失败主要来自 observation、contact、pose init、optimizer 还是 render？

LLM 同样不输出坐标或 pose，只输出审计结论。

## 最终汇总表

最终汇总表由以下命令生成：

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/tools/run_final_hoi_evaluator.py \
  --result-name benchmark_vlm_qwen \
  --output-dir samples_known_object/final_result_evaluation
```

默认输出：

- `samples_known_object/final_result_evaluation/final_evaluation_detailed.csv`
- `samples_known_object/final_result_evaluation/final_evaluation_human_readable.md`
- `samples_known_object/final_result_evaluation/final_evaluation_summary_manifest.json`

表中每一行只有一个 case 的当前最终结果，不包含 baseline 或 ablation。若要比较方法贡献，应另看 benchmark report。

旧的 `run_final_summary.py` / `final_result_evaluation_summary.*` 是 object-only legacy summary，保留用于兼容，但不作为当前最终 HOI 汇报主表。

## 通过标准

一个最终结果至少应满足：

- `SE3 Pose = yes`
- `Jump Count = 0` 或跳变有明确审计解释
- `Static Drift` 在静止段足够小
- `Penetration Rate` 和 `Floating Rate` 低
- `Contact Proxy` 或人工 `Contact F1` 合理
- `Failure Stage` 能定位失败来源，而不是只写 render bad

## 为什么当前很多 case 显示 `stage2_contact`

当前 final summary 中，basketball、football、mug、stick 多数显示 `Failure Stage = stage2_contact`。这不是说最终 render 一定整体失败，而是 evaluator 认为失败源头首先落在 **contact/anchor evidence**：

- 这些 case 没有人工 contact label，所以 `Contact F1` 为空，只能使用 `Contact Proxy`。
- `anchor_drift_fail` 的阈值是 `Anchor Drift Max > 0.08m`。当前结果中 basketball、football、mug、stick 都超过这个阈值。
- basketball、football、mug 还因为 `contact_depth_offset_m` 被作为 floating/penetration proxy 使用，非零 depth offset 会触发较高的 `Floating Rate` 或 `Penetration Rate`。
- failure stage 的优先级是：overlay/geometry → contact/anchor → physical optimizer。因此一旦 `anchor_drift_fail` 成立，即使同时存在 penetration/floating proxy，summary 也会优先写 `stage2_contact`。

因此 `stage2_contact` 的含义更准确地说是：**最终 pose 已经有 SE3，jump 也可能为 0，但接触/anchor 证据还不足以让 evaluator 判定完全通过。** 这也是为什么 chair 通过，而其他 case 仍显示 `pass=no`。
