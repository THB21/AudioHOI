# Generic Sequence Solver 分支实施计划

Branch: `refactor/generic-sequence-solver`

Base: `64710953 Add factor shadow composition validation`

## 目标

建立通用 sequence solver 的 shadow problem/attempt contract，使后续真正执行优化器时
已有可审计的输入边界、factor 组合、attempt policy 和输出保护规则。

本分支不执行连续求解，不产生新 pose，不替换 Stage 3/4 legacy solver。

## 前置门禁

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| 新 worktree / 新分支 | done | `/mnt/hdd/AudioHOI-generic-sequence-solver` from `64710953` |
| Ignored runtime input hydration | done | dry-run `verified=41 would_copy=48 errors=0`; apply `copied=48 errors=0`; final dry-run `verified=89 would_copy=0 errors=0` |
| Factor registry base | done | `verify_factor_shadow.py` pass；factor shadow validator 已启用 |

## Scope

新增只读 sequence problem shadow：

- 从 case profile 声明 state contract，不读取 `object_pose_init.csv`。
- 汇总 Measurement IR、ContactConstraint IR 和 Factor IR 的 frozen shadow 输入。
- 生成 deterministic `shadow-<hash>` attempt id。
- 显式记录：
  - `solver_executed=false`
  - `accepted_outputs_written=false`
  - `baseline_pose_read=false`
  - `initializer_status=not_executed`
  - `writes=[]`
- 提供 export/verify CLI 与五 case golden summary。

新增 shadow execution diagnostics：

- 固定 `assemble_problem -> initialize_state -> solve_sequence -> evaluate_candidate` 四段诊断。
- `assemble_problem` 只检查 typed inputs 是否可组装。
- `initialize_state`、`solve_sequence`、`evaluate_candidate` 均不执行。
- basketball/football 标记为 `ready_for_future_shadow_solve`。
- mug/chair 因未迁移机制 gap 标记为 `blocked_by_known_gaps`。
- stick 的 `line_contact_lock_special_refinement` 可作为 compatibility refinement 保留，
  标记为 nonblocking gap。
- mug 的 `phase_snapshot_fallback` 是 canonical provenance gap。generic state 必须把它
  表达为刚性总成上的 periodic feature cue，并以 coupled yaw/phase gauge 比较组合后几何；
  handle 不是物理 revolute child，不得把 body 与 handle 当作两个独立刚体 rotation。
- chair 的 `semantic_graph_solver_private` 是实现边界 gap：状态和约束可泛化，但当前
  top-rail chord 初始化、twist/关节 2D gauge 和 pose lock 仍捆在 chair 私有 solver。
  迁移目标是通用 kinematic graph、correspondence initializer 和 rank-deficiency factor，
  不是删除两点接触这一有效几何机制。

新增 isolated candidate sandbox guard：

- 只允许 future-ready case 计划写入 sibling `generic_sequence_solver_shadow/...` 目录；
  当前包括 basketball、football，以及保留 line contact compatibility 的 stick。
- 原 shadow 分支只 materialize sandbox manifest；后续 ball migration 已允许 sphere case 在
  sibling sandbox 生成安全命名的 candidate/residual/attempt，仍禁止 accepted output 名称。
- validator 拒绝 accepted output 文件名，例如 `object_pose_init.csv`、`object_pose.csv`。
- blocked case 不计划任何 artifact。

## 非目标

- 不运行 optimizer。
- 不写 `object_pose_init.csv`、final pose、phase、contact 或 render。
- 不读取 profile `baseline` 中的 solved pose/phase。
- 不把 historical solved pose、snapshot fallback 或 compatibility seed 提升为 initializer。
- 不改 loss、阈值、求解器、Stage 3/4 调度或输出目录。

## 计划步骤

| Step | Status | Evidence |
| --- | --- | --- |
| 建立 `core/solver` shadow contract | done | `problem.py`、`validation.py`、`golden.py` |
| 增加 CLI | done | `export_sequence_problem_shadow.py`、`verify_sequence_problem_shadow.py` |
| 冻结五 case summary | done | `tests/golden/sequence_problem_shadow_v1.json` |
| 增加 shadow execution diagnostics | done | `diagnostics.py`、`diagnostics_golden.py`、export/verify CLI |
| 冻结五 case diagnostics | done | `tests/golden/sequence_solver_diagnostics_v1.json` |
| 增加 candidate sandbox guard | done | `candidate.py`、`candidate_golden.py`、export/verify CLI |
| 冻结五 case sandbox summary | done | `tests/golden/sequence_candidate_sandbox_v1.json` |
| 增加测试 | done | `tests/test_sequence_solver_shadow.py`：17 passed |
| 运行回归 | done | focused IR tests 61 passed；`python -m pytest -q`: 116 passed, 2 skipped；state/factor/sequence/candidate/golden gates pass |

## 接受标准

- sequence problem validator 必须拒绝任何 solver execution、accepted output write 或
  baseline pose read。
- sequence diagnostics 必须保持 solver/initializer/evaluator 不执行，且 `writes=[]`。
- shadow manifest 本身只写 sandbox manifest；geometry candidate solver 只能写 manifest 中
  声明的安全 artifact，且 candidate dir 不得等于 canonical result dir。
- `object_pose_init.csv` 不得出现在 sequence problem payload。
- 五 case measurement/contact/factor 数量与 gap ids 必须 frozen。
- 当前 gap 保持显式：
  - mug: `phase_snapshot_fallback`
  - chair: `semantic_graph_solver_private`、`unsupported_loss_term:E_audio`
  - stick: `line_contact_lock_special_refinement`（nonblocking compatibility）
- 受保护路径无 diff：
  - `components/`
  - `stages/`
  - `run_pipeline.py`
  - `core/base`
  - `core/provenance`
  - `src/audio`

## 当前结论

这一步让“泛化姿态求解”有了可机器校验的问题边界，但还没有真正求解。它证明的事情是：

- 通用 solver 未来应消费哪些 typed inputs 和 factor requirements；
- 当前哪些 legacy 机制仍不能安全泛化；
- shadow attempt 如何在失败时只记录诊断，不覆盖 accepted outputs。
- 哪些 case 可进入下一步 future shadow solve，哪些 case 必须先迁移 blocking legacy gap。
- line contact 可保留为 stick compatibility refinement，但必须继续记录为 nonblocking gap。
- future candidate 输出必须先进入隔离 sandbox，不能污染 canonical `benchmark_vlm_qwen`。

真正的 pose 不退化迁移仍在后续对象迁移分支中逐 case 完成。

## 后续 ball migration 进展（2026-07-26）

`refactor/migrate-ball-cases` 已真正执行 `translation3:sphere` candidate solver。它只读取
result-owned Measurement IR、contact event/timeline、HumanSite 和 support observation，
不读取 baseline/canonical pose；basketball 与 football candidate 均与旧 exact Stage 4
seed 逐字节一致。candidate sandbox summary 因而对 sphere case 计划四个安全 artifact，
stick 仍只计划 manifest，mug/chair blocking gap 不变。

同时修复 canonical shadow hash 将绝对 worktree 前缀纳入哈希的问题。更新后的五 case
hash 是路径归一化后的新 schema-v1 内容冻结；mug/chair/stick 的 hash 更新不表示其数据或
算法发生变化。
