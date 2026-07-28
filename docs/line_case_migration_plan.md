# Stick / Line Contact 迁移计划

Branch: `refactor/migrate-line-case`

Base: `2393672a Guard mug candidate sandbox gap eligibility`

## 目标与边界

stick 的 line contact 可以保留，但必须作为通用 line-object primitive，而不是
stick-only 例外。当前分支先收紧 `LineS` contact contract 和回归门禁，不改数值求解、
loss、阈值、Stage 3/4 调度或 accepted outputs。

## 当前状态

- Stage 2 contact 已适配为 `line_s_contact_v1`。
- `ContactConstraint.object_coordinate` 使用 `LineS` tagged coordinate，不转换为伪
  `LocalXYZ`。
- `line_contact_lock_special_refinement` 继续记录为 nonblocking compatibility gap。
- candidate sandbox 对 stick 保持 `sandbox_ready`，但当前只允许写 sandbox manifest；
  不允许写 `object_pose_init.csv`、`object_pose.csv` 或 contact accepted outputs。

## 本分支新增 gate

`LineS` 现在被定义为归一化局部线坐标，必须满足 `0 <= s <= 1`。这防止后续迁移时把
像素坐标、米制距离、端点 id 或其它对象专用编码误塞进 generic contact constraint。

新增测试覆盖：

- canonical stick contact CSV 的 480 条 contact 全部适配为 `LineS`；
- 所有 `LineS.s` 都在 `[0, 1]`；
- 越界 `LineS(42.0)` 被 contract 拒绝。

## 后续迁移接受标准

1. 把当前 `line_contact_lock` 的输入拆成 Measurement IR、ContactConstraint IR、
   line GeometryProvider 和 Factor IR。
2. 保留 `LineS` 与 endpoint identity disambiguation，但 residual/weight/acceptance 必须
   记录为 generic factor/candidate diagnostics。
3. candidate 输出先进入 isolated result directory；禁止覆盖 canonical
   `benchmark_vlm_qwen`。
4. 与现有 stick 结果比较 contact median/P90、line overlay、temporal spikes、static tail
   drift 和六路 render hash；不得通过放宽阈值消除差异。
5. 只有这些 gates 全过后，才能把 `line_contact_lock_special_refinement` 从 nonblocking
   compatibility gap 降级/关闭。
