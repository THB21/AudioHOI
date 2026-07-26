# Basketball / Football 通用球体序列求解迁移

Branch: `refactor/migrate-ball-cases`

Base: `cc9a696f Clarify mug and chair generalization boundaries`

## 目标与边界

把 basketball、football 的 Stage 4 连续深度求解从 `anchor_depth` compatibility
adapter 迁入 typed、result-owned 的 `generic_sphere_sequence`。本分支保持原有数值算法和
参数，不调整 loss、阈值、接触事件定义、平滑器或末端 audit。

新求解器只允许读取：

- `object_observations.csv`（Measurement IR）；
- `contact_events.csv` 与 `contact_state_frames.csv`（Contact Constraint timeline）；
- `human_sites.csv`（逐帧 HumanSite camera XYZ）；
- `support_geometry.json`（显式 support observation）；
- case profile 中的 camera、`translation3:sphere` 和静态 `radius_m`。

禁止读取 `object_pose_init.csv`、`object_pose.csv`、profile baseline 或 canonical final
pose。candidate 先写入隔离目录；Stage 4 capability 明确 promotion 后才写 accepted output。

## 隐式输入审计与修复

旧实现还隐式读取了两个未在 Stage DAG 声明的路径：

- `results/contact_candidates_object_proxy/contact_candidates_labeled.csv`；
- `results/pose6d_object_proxy_da3_init/support_geometry.json`。

现已由 Stage 2 复制为 result-owned artifact，并进入 attempt input hash。Stage 2 同时从
GVHMR/SMPL-X 生成双手、双脚四个 `HumanSite`，使用九位小数保存 camera XYZ，避免六位
舍入造成最终 candidate 的 `1e-6` 漂移。

isolated worktree 初次 fresh run 还暴露 runtime hydration 漏项：basketball/football 的
`results/tracking` 只有单个已跟踪 CSV，缺少被忽略的完整 tracking 目录。runtime manifest
已补充两条目录记录：basketball `4 files / 1903059 bytes / 105f817a...`，football
`4 files / 2402209 bytes / b249101b...`；补齐后 Stage 1–5 均可 fresh run。

## 数值等价证据

`generic_sphere_migration_switched_v1` 使用切换后的配置从 Stage 1 跑到 Stage 5，
`llm_mode=none`、`vlm_mode=none`。两个 case 均退出码 0，Stage 4/5 audit 均 pass、
`blocking_count=0`、`rerun_stage=false`。

| Case | 帧数 | human/support events | typed candidate = 旧 exact seed | fresh final pose |
| --- | ---: | ---: | --- | --- |
| basketball | 192 | 15 / 15 | `9345db0b4d65e829...`（byte-identical） | `2b01296a6c11fbb1...` |
| football | 242 | 6 / 9 | `0b697481bbd5bf6f...`（byte-identical） | `9c5098505c8bb6cb...` |

basketball/football 的 candidate 均与 canonical `object_pose_pre_smooth.csv` 逐字节一致，
并在 switched run 中原样成为 `object_pose_pre_smooth.csv`。此前对相同 fresh 输入直接调用
旧 `anchor_depth` 与新 policy，也分别得到完全相同的 Stage 4 输出；football 的 direct
hash `0d48a496...` 与 full-pipeline hash `9c509850...` 的区别来自现有 stage audit 调用时序，
而不是 solver migration，且旧/新在相同调用路径中一致。

六个主要渲染视频（object overlay/camera3d/side_yz 与相应 with-human 视频）的 hash 已冻结
在 `tests/golden/sphere_sequence_migration_v1.json`；两张 with-human preview 已人工检查，
球、人和地面/support overlay 均正常。

## Provenance 与安全门禁

每次求解生成：

- `generic_sphere_sequence_candidate.csv`；
- `generic_sphere_sequence_residuals.csv`；
- `generic_sphere_sequence_attempt.json`。

attempt 固定记录全部输入 hash、算法参数、candidate/residual hash，并断言
`baseline_pose_read=false`、`accepted_outputs_written=false`。promotion provenance 单独记录
`accepted_output_written=true`，避免把 sandbox execution 与 accepted write 混为一谈。
Stage 4 metrics 也把 `mainline_implementations` 与 `compatibility_adapters` 分开记录；
`generic_sphere_sequence` 属于前者，保留的 `backproject_xy` 属于后者。

sequence shadow 的 hash 还修复了跨 worktree 可移植性：Measurement/Contact source path 在
进入 canonical hash 前转为 repository-relative。此前 mug/chair/stick 同时出现 hash gap
只是绝对 worktree 前缀污染，不是三个 case 的算法或数据变化。

## 尚未关闭的门禁

- canonical `benchmark_vlm_qwen` 含真实 Qwen/VLM gate；当前 fresh run 使用 none gate，
  所以只可比较 exact seed、相同 gate path 的 final pose 和渲染，不能宣称 canonical final
  pose 已逐字节一致。
- 仍需完成五 case 全量 pytest、Phase 0 golden、runtime hydration 和全部 shadow verifier。
- 必须审计 diff，确认旧数值参数、loss、阈值和平滑算法没有变化。
- 在上述门禁全部通过前不删除 `anchor_depth` compatibility policy；mug 仍需要该兼容入口。

stick 的 line contact 按既定决策保留，继续作为 nonblocking compatibility mechanism；本分支
不迁移或删除它。
