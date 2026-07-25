# Factor Registry 分支实施计划

Branch: `refactor/factor-registry`

Base: `d6729aea Add state spec kinematics shadow gates`

## 目标

把现有 Stage 3/4 求解输入的 loss/residual 语义先映射成只读 `Factor IR`，不替换任何
连续求解器，不调整 loss、阈值或优化算法。

本分支完成后，后续 `generic-sequence-solver` 才允许消费这些 factor。

## 前置门禁

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| 新 worktree / 新分支 | done | `/mnt/hdd/AudioHOI-factor-registry` at `d6729aea` |
| Ignored runtime input hydration | done | dry-run `verified=41 would_copy=48 errors=0`; apply `copied=48 errors=0`; final dry-run `verified=89 would_copy=0 errors=0` |
| State shadow portability | done | 修复 state shadow canonical hash 的 worktree 绝对路径依赖 |
| State gates | done | `verify_state_shadow.py` pass；`verify_state_parity.py` pass with known chair warnings |
| Baseline regression | done | `python -m pytest -q`: 89 passed, 2 skipped；decoded five-case golden verify passed |

## Scope

新增只读 factor 描述与 shadow verifier：

- `PointReprojectionFactor`
- `LineReprojectionFactor`
- `MaskSilhouetteFactor`
- `MetricDepthFactor` / `DepthOrderFactor`
- `ContactDistanceFactor`
- `SupportAndPenetrationFactor`
- `TemporalVelocityFactor` / `TemporalAccelerationFactor`
- `StaticFreezeFactor`
- `JointLimitFactor`
- `GaugeConstraintFactor`
- `PosePriorFactor`

首批不要求覆盖每个 legacy solver 的所有私有调参；必须覆盖可从现有 CSV/profile 中
明确追溯到 measurement/contact/state 的 factor 等价项，并报告缺口。

## 非目标

- 不让 factor 被 Stage 3/4 solver 消费。
- 不改 `components/`、`stages/`、`run_pipeline.py` 的求解路径。
- 不改 loss weight、阈值、优化器、接触 gate 或输出目录。
- 不把 historical solved pose/phase 当作 `PosePriorFactor`。

## 计划步骤

1. 定义 `core/factors/types.py`：factor id、kind、input refs、frame interval、unit、weight source、gate source、residual source。
2. 增加五 case shadow adapter：从 measurement/contact/state shadow 和 legacy pose/debug CSV 生成 factor summary。
3. 增加 factor registry/verifier：五 case factor kind counts、source hashes、unmapped legacy fields、canonical hash。
4. 增加测试：factor schema、no case-name dependency、no solver consumption、unknown/dead factor 拒绝。
5. 运行 `pytest`、state gates、golden decoded verify、hydration dry-run。

## 接受标准

- factor shadow `consumed_by_solver=false`。
- factor 的 measurement/contact/state 输入均以 id/source hash 引用，不复制连续 pose。
- basketball/football/stick 的 translation/depth/contact/smoothness 等价项能映射为通用 factors。
- mug/chair 中不能泛化的 solved-seed 或 solver-private 项必须显式列入 gap report。
- `no_contact_anchor` 仍不得复活为有效 consumer。
- 全量回归和五 case golden 不退化。
