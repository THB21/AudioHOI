# StateSpec / GeometryProvider 分支补充计划

Branch: `refactor/state-spec-kinematics`

Base: `b05b1929 refactor: add typed contact constraint shadow IR`

## 先回答你关心的两点

### 当前整个重建算法逻辑是否已泛化？

还没有完全泛化。

- 观测与接触：我们已经把 legacy 输入变为 typed shadow（`core/measurements`、`core/contact_constraints`），并形成无 case-name 消费语义。
- 姿态/几何：本阶段只在构建 `StateSpec` 与 `GeometryProvider` 描述，不改变当前 Stage3/4 的真实连续求解入口。
- 当前可见事实：调度层仍通过 `case` 插件选择 legacy pose solver（translation3、rigid6_plus_phase、semantic_graph_6d），因此“完全泛化求解”尚未实现。

结论：泛化修复框架已搭建（观测+接触+状态声明），但“统一求解器”仍在后续 `factor-registry` / `generic-sequence-solver` 分支。

### 现有计划是否覆盖观测、接触、姿态求解泛化修复？

覆盖了，但分层次进行：

- 观测泛化：已完成 measurement IR，下一步要求 state/geometry 与其联动。
- 接触泛化：已完成 contact constraint IR，下一步要求 state/geometry 与 contact 无 case 分支对齐。
- 姿态/几何泛化：本分支目标是把姿态状态描述与几何 provider 从 solver 中解耦；未在此分支中改 Stage3-4 的最优化流程。

## Scope

五个 canonical state 的通用描述（仅用于 shadow/兼容），不引入新 loss、阈值和连续求解器。

- `translation3`（basketball, football, stick）
- `rigid6_plus_phase`（mug）
- `semantic_graph_6d`（chair）

## 当前库存结论（已完成）

| 对象状态来源 | 案例 | 当前有效状态 | 几何能力 | 观测特性 |
| --- | --- | --- | --- | --- |
| `translation3` | basketball, football | root translation + 球体半径（球体参数仍以固定行值出现） | sphere | 旋转不可观测；仅通过 `ref` 点与深度约束 | 
| `translation3` | stick | root translation + camera-plane line angle（写入 quaternion） | line/capsule | 旋转轴向不可观测（roll 可作为 gauge） |
| `rigid6_plus_phase` | mug | root 刚体位姿 + scale + 周期相位 | rigid mesh | 手柄相位可观测（含周期性约束） |
| `semantic_graph_6d` | chair | root 刚体位姿 + rear / seat 两个转动关节 | URDF（显示物理限位） | 语义边界线与支撑信息可观测 |

Chair URDF 明确限位（来自已审计源）：

- `front_to_rear`: `[-0.82, 0.12]` rad
- `front_to_seat`: `[0.0, 1.35]` rad

注意：任何优化搜索上下界均不视为物理物理限位。

## 先补齐输入缺失（你这一步必须先过）

新的 worktree 里若 DA3/GVHMR/SAM2 仍是 missing，不能继续改造。优先执行以下硬门禁：

1. 先 dry-run 同步：

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/tools/sync_golden_inputs.py \
  --source-root /mnt/hdd/AudioHOI
```

2. 关键通过条件：`errors=0` 且有预期 `would_copy`，避免把缺失输入掩盖成算法问题。

3. 再执行：

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/tools/sync_golden_inputs.py \
  --source-root /mnt/hdd/AudioHOI --apply
```

4. 同步后再跑 `python -m pytest -q` / 五 case 回归；如无 `verified` 兜底，不允许继续迁移。

## 补充计划（本分支目标）

### 1) 结构化状态/几何类型

**目标**：构建 `core/state` 与 `core/geometry_provider` 的基础类型，完全声明式、可验证。

- `StateSpec`（`spec_id`, `dof` 列表, `static_parameters`, `observability`）
- `DofSpec`（id, kind, dimension, unit, bounds, source）
- `DofKind`（`translation`, `rotation_so3`, `scalar`, `revolute`, `prismatic`, `periodic`）
- `Bound`（lower/upper/unit/source, closed/open）
- `GaugeConstraint`（target dof 集合 + 显式来源）
- Geometry descriptor（kind, feature ids, hash, capabilities）

### 2) 统一五个对象的 Adapter（schema-driven，不按 case 名）

- 依据输入列名/字段类别而非 case_name，映射到 StateSpec 与 Geometry descriptor
- 共享行为：
  - 只读 shadow；
  - `consumed_by_solver = false`；
  - 严格列映射与未映射字段上报。
- 特殊处理：
  - stick 与 ball 的 `translation3` 用同一语义，但几何通过 geometry profile 区分；
  - mug 将 yaw 固定为 gauge，不允许强制 `0.0`。

### 3) 运行时投影/变换可复核（shadow parity）

- 逐帧检查 `object_pose_init.csv` 与旧语义的一致性：
  - translation parity（root 平移）
  - 旋转观测（如果缺失则声明不可观测 + gauge）
  - 周期量 wrap 映射
- URDF 及 rigid mesh 资源 hash 与特征能力记录
- geometry feature 映射与支持点/段定义保持与现有渲染输入一致

### 4) 验收与合并门控

- 添加/复用 tests：
  - 五 case adapter shadow 生成
  - 未映射非空字段与单位一致性
  - unobservable gauge 不被零值伪造
  - chair 限位源标注与几何描述一致
- `generalized_reconstruction_kernel_plan` 中的 `refactor/state-spec-kinematics` 里程碑从 `pending` -> `in_progress`
- commit 只在本分支完成一次可复核的可交付（不允许半成品分散提交）

## 关键交付清单（可核验）

- `src/audio/...` 无改动
- 新增文件（预期）：
  - `scripts/shared/generic_contact_pipeline/core/state/types.py`
  - `scripts/shared/generic_contact_pipeline/core/state/spec.py`
  - `scripts/shared/generic_contact_pipeline/core/state/adapters.py`
  - `scripts/shared/generic_contact_pipeline/core/state/geometry.py`
  - `scripts/shared/generic_contact_pipeline/core/state/shadow.py`
  - `scripts/shared/generic_contact_pipeline/core/state/__init__.py`
  - `scripts/shared/generic_contact_pipeline/core/state/registry.py`
- 测试覆盖（至少）：
  - `tests/test_state_spec.py`
  - `tests/test_pipeline_provenance.py` 的状态/资源 hash 与五 case 输入补齐行一致性扩展
- 文档输出：
  - `docs/state_spec_kinematics_plan.md`（持续更新当前 `Step` 与 `Evidence`）
  - `docs/generalized_reconstruction_kernel_plan.md` 增加本分支完成时间戳

## 风险与冻结边界（本次不改）

- 不修改 Stage3/4 pose 求解器主逻辑
- 不引入新 loss/阈值
- 不触发任何历史 `solved_pose` 消费
- 不改写当前 canonical 输出目录格式

## 里程碑

1. 完成 `core/state` 与 `core/geometry_provider` 只读模型（含序列化 + 验证）
2. 完成五对象 shadow adapter 与 geometry 描述构建
3. 完成 projection/parity 测试与报告
4. 形成本分支可复用的 manifest（resource hash, schema, unmapped fields）
5. 运行回归（5 case 现状对比）通过后提交，进入 `refactor/factor-registry`

## 当前进展记录

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| Runtime input hydration | done | `verified=89 would_copy=0 copied=0 errors=0` |
| `StateSpec` / `GeometryDescriptor` 基础类型 | done | `core/state/types.py` |
| 五 case legacy state adapter | done | `core/state/adapters.py`；按 CSV schema 识别，不按 `case_name` 分支 |
| Read-only state shadow manifest | done | `core/state/shadow.py`；`consumed_by_solver=false` |
| State shadow export CLI | done | `tools/export_state_shadow.py`；写入审阅用 JSON，不改 Stage 3/4 |
| Frozen state shadow regression | done | `tests/golden/state_shadow_v1.json`；五 case state/geometry/hash 摘要稳定 |
| State shadow verifier | done | `tools/verify_state_shadow.py`；后续分支可作为 state/geometry 漂移门禁 |
| State parity report | done | `core/state/parity.py` + `tools/export_state_parity.py`；五 case representation parity pass，chair 当前 seed 的 URDF joint-limit mismatch 记录为 warning |
| State parity verifier | done | `tools/verify_state_parity.py`；默认 warning 不失败，`--strict-warnings` 可阻断 chair migration |
| Planned module boundary | done | `types.py/spec.py/geometry.py/registry.py/adapters.py/shadow.py/golden.py/parity.py` |
| Focused regression | done | `tests/test_state_spec.py`: 17 passed；`tests/test_measurements.py tests/test_contact_constraints.py tests/test_state_spec.py`: 34 passed |
| Full regression | done | `python -m pytest -q`: 89 passed, 2 skipped；decoded five-case golden verify passed |

### Parity observation

`export_state_parity.py` 不运行 solver，只检查 legacy `object_pose_init.csv` 与 `StateSpec`
声明的状态语义一致性。当前五 case 的表示层检查通过；chair 的两个 articulated joint
存在历史 seed 超出 URDF limit 的 warning：

- `joint.front_to_rear.within_urdf_limit`: 102 frames, max excess `0.525480457` rad
- `joint.front_to_seat.within_urdf_limit`: 82 frames, max excess `0.345642214` rad

这不是本分支引入的输出变化；它冻结为后续 generic solver / chair migration 需要解决的
已知 evidence。

## 该分支不改范围（确认）

- 不改 Stage3/4 求解器主逻辑
- 不改 loss、阈值、solver 梯度结构
- 不引入/换 DA3、GVHMR、SAM2、VLM、LLM
- 不改 canonical result 输出路径和已冻结主结果文件名
