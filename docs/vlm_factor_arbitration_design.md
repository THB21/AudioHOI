# 通用 VLM Factor Arbitration 设计

日期：2026-07-30  
分支：`refactor/interaction-state-production`  
范围：object solver；GVHMR 只作为只读骨架 / human-site observation。

## 目标

当视觉观测、接触观测或其他 typed evidence 在某个时间窗内冲突时，让 VLM 在预定义的离散可靠性档位中选择 factor activation，而不是让对象名选择 solver，也不是让 VLM 直接生成连续姿态或任意 loss weight。

第一项验证对象是当前 stick candidate 暴露的通用冲突：`LineReprojectionFactor(endpoints)` 能恢复可见物体 extent，但与只读 GVHMR closest-line contact evidence 在部分区间不一致。实现必须同样适用于任何 line/capsule 或其他同时具有视觉与接触 evidence 的资产。

## 非目标与边界

- 不新增 stick、chair、mug 或其他 case-specific solver、query 或 dispatcher。
- 不允许 VLM 输出 xyz、quaternion、joint value、phase 或连续 loss weight。
- 不允许 VLM 完全关闭视觉或接触 factor；只允许 `active` 与 `downweighted` 两档之间转换。
- 不修改人体状态，不运行 human refinement，不发布 downstream human pipeline 输出。
- 不写 canonical `object_pose.csv`；验证 attempt 仍写隔离 candidate directory。
- 不新增测试文件；使用现有测试、artifact verifiers 与真实 stick attempt 验证。

## 当前缺口

现有 VLM gate 只能表达 anchor、contact frame 和 pose-refine 的允许 / 拒绝：

- stick Stage 1 `keypart_identity_check` 几乎全部为 `unclear`；
- Stage 2 `contact_relation_check` 接受了大多数 hand-contact；
- Stage 3 只在少数采样帧检查 overlay；
- Stage 4 `anchor_update_check` 不比较 factor evidence 的可靠性；
- production `FactorCompiler` 的 activation 目前只由 InteractionState 轴生成，无法消费 VLM 对具体 factor 冲突的判断。

因此“VLM 已运行”并不等于“VLM 已影响 generic solver”。

## 决策接口

新增 profile-driven forced-choice query：

`constraint_reliability_check`

候选标签固定为：

- `visual_observation_reliable`
- `contact_relation_reliable`
- `both_consistent`
- `unclear`

query 由 factor kinds、SceneEntity / feature metadata 和 evidence manifest 生成，模板中不得出现 case name 或对象专用词。输入 evidence package 包含：

- 原始帧或 crop；
- visual measurement overlay（例如 line endpoints）；
- GVHMR 只读骨架和 human-site overlay；
- object contact feature / closest-point overlay；
- candidate object render；
- 相邻帧 temporal strip；
- factor residual summary 与 provenance，但不向 VLM 暴露 canonical pose。

## 离散 activation 规则

VLM 决策只覆盖 InteractionState 已生成的基础 activation，且只能保持或降低可信度：

| VLM 标签 | visual factor | contact factor |
|---|---|---|
| `visual_observation_reliable` | 保持基础档位 | 降为 `downweighted` |
| `contact_relation_reliable` | 降为 `downweighted` | 保持基础档位 |
| `both_consistent` | 保持基础档位 | 保持基础档位 |
| `unclear` | 降为 `downweighted` | 降为 `downweighted` |

这里的 visual factor 包括 point / line / mask / metric-depth 等已编译视觉观测；contact factor 包括 contact distance 及后续同源 contact relation factors。某 factor 在基础 InteractionState 中已经是 `inactive` 时，VLM 不得重新激活它。

权重数值继续来自 versioned factor runtime config，例如 `active=1.0`、`downweighted=0.25`、`inactive=0.0`；VLM 只选择档位名称。

## 编译与运行路径

实现新增一个 case-independent arbitration ledger，核心记录为：

```text
FactorGateDecision
  query_id
  frame_interval
  normalized_label
  affected_factor_ids
  status_by_factor
  evidence_ids
  provider
  model
  prompt_hash
  response_hash
  provenance
```

数据流固定为：

```text
InteractionState activation
  -> VLM factor arbitration overlay
  -> merged activation intervals
  -> CompiledFactor
  -> SequenceProblemFactory
  -> GenericSequenceExecutor
```

合并操作必须在 factor compiler 边界完成。solver 只消费合并后的 activation intervals，不读取 VLM 文本、不读取 case name，也不选择 executor。

## 触发条件

不对所有帧调用 VLM。先用硬证据检测冲突时间窗，例如：

- visual residual 与 contact residual 无法同时下降；
- 两类 residual 的高分位值同时超过各自现有诊断阈值；
- observation confidence / visibility 与 contact confidence 给出相反排序。

触发器只选择需要审计的时间窗，不直接决定哪类 factor 更可信。没有有效 VLM 结果时采用 fail-closed 行为：两类 factor 均降为 `downweighted`，attempt 标记 `vlm_arbitration_unclear`，不得 canonical promotion。

## Provenance 与发布约束

每个 attempt 的 `vlm_gates.json` 必须记录：

- query template/version；
- provider/model；
- evidence artifact 路径和哈希；
- normalized label；
- 受影响 factor ids 和合并前后 activation intervals；
- fail-closed 原因（若有）。

factor ledger 的 `gate_provenance` 同时加入对应 query / decision id。hard metrics 仍优先；VLM 不能覆盖 NaN、几何穿透、非法 joint bounds 或 publication authorization gate。

## 验收

1. core solver / factors / state / geometry 中不新增 `case_name` 分支。
2. 同一个 query schema 和 activation merge 可作用于任意 factor ids。
3. stick attempt 的冲突区间产生真实、可追踪的 VLM decision，并改变 compiled activation intervals。
4. `vlm_mode=none` 时明确记录 `not_evaluated`，不伪装成已启用。
5. VLM 输出只改变离散 tier，不能改变 StateSpec、geometry、factor base weight 或 solver 类型。
6. stick 新 attempt 必须同时报告 endpoint reprojection、contact gap、termination 和 pose/render regression；如果 hard metrics 未改善，保持 blocked，不增加 case-specific 修补。

