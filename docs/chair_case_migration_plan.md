# Chair 迁移计划

Branch: `refactor/migrate-chair-case`

Base: `0b6da6fa Add line contact diagnostics gate`

## 目标与边界

chair 的两手 top-rail contact chord、URDF articulation 和 semantic 2D gauge 可以泛化，
但当前实现仍通过 `small_se3` / `chair_twohand_endpoint_se3.py` 路径绑定 chair-specific
pairprop solver。本分支先增加只读 diagnostics gate，不改 solver 数值、loss、阈值、
Stage 3/4 调度或 accepted outputs。

## 当前诊断结论

canonical `benchmark_vlm_qwen` 的 Stage 4 pairprop pose 被接受，但其 seed provenance
仍是 `rebuilt_from_mainline_saved2d`，并记录 `mainline_pose_csv`。因此
`semantic_graph_solver_private` 必须继续是 blocking gap；不能因为 pairprop contact quality
通过就宣称 chair 已完成算法级泛化。

## 本分支新增 gate

`generic_chair_contact_diagnostics` 只读汇总 `stage4_metrics.json` 中的 chair pairprop
证据：

- `solver_executed=false`、`accepted_outputs_written=false`、`baseline_pose_read=false`；
- 固定追踪 `semantic_graph_solver_private`；
- 记录 seed policy、historical seed reference fields、active frame count、pairprop
  contact median/P90 gap、per-frame metric rows、standard/constraint quality gate；
- validator 会拒绝把 historical/rebuilt seed 状态伪装成 nonblocking gap，也会拒绝
  solver execution、accepted output write 或负 contact gap。

这一步把 chair 专用 pairprop 的 acceptance 证据变成可机器校验的迁移边界，但还没有把
`chair_twohand_endpoint_se3.py` 替换为 generic factor executor。

## 后续接受标准

1. contact chord initializer 已提升为通用 `RigidCorrespondenceInitializer` core contract；
   legacy chair wrapper 仅保留兼容入口，不再承载 chair-specific solved state。
2. URDF joint propagation 已建立 `ArticulatedKinematicProvider` core contract；
   chair solver 现在通过 data-rule provider 调用 rear/seat/side-stretcher 传播语义。
3. twist rank deficiency、semantic 2D line factors、joint limits 和 contact distance
   进入 generic Factor IR。
4. candidate 输出先进入 isolated result directory；禁止覆盖 canonical `benchmark_vlm_qwen`。
5. semantic 2D、contact median/P90、freeze、pose lock 和六路 render 均不退化后，才可
   关闭 `semantic_graph_solver_private`。
