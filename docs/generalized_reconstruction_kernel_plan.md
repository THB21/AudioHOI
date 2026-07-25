# Phase G：通用重建内核实施计划

Base：完成 mug/chair solved-seed removal 并通过五 case golden regression 后

## 目标

把当前“统一调度 + case-specific solver”推进为真正由数据描述驱动的重建内核：

```text
detector / tracker / depth / human pose
  -> typed measurements
  -> contact constraints
  -> StateSpec + GeometryProvider
  -> generic factors
  -> generic sequence solver
```

新增对象时允许增加 case/sample 配置、几何资源和观测语义映射；不允许增加新的
`run_<object>`、对象专用连续优化器或读取历史 solved pose/phase。

## 前置条件

- Phase 0 golden、attempt/provenance、artifact store、typed contracts 和
  capability registry 保持为强制回归基础。
- Mug observation-derived seed 已完成；chair seed removal 必须先达到接受标准。
- 五个 canonical case 的输入、逐阶段产物、contact/gate、pose 和 decoded render
  hashes 可重验。
- 每个新 worktree 必须先补齐 ignored runtime inputs；DA3/GVHMR/SAM2/tracking 不得
  以 missing 形式进入 golden 验证或回归结论。
- 每项迁移先 shadow-run，未通过接受标准时继续使用原实现，禁止静默替换。

## 非目标

- 不在本阶段更换 DA3、GVHMR、SAM2、VLM 或 LLM 模型。
- 不用 LLM/VLM 直接生成连续 pose、坐标或 loss weight。
- 不同时重写人体姿态 refinement；人体侧只通过 typed `HumanSite` 接口输入。
- 不为了通过新架构测试而调整现有 loss、阈值或 benchmark 定义。

## 核心数据模型

### 1. Measurement IR

所有观测共享：`sample_id/frame/time/feature_ref/coordinate_frame/confidence/source`
以及可选 covariance。首批 tagged union：

- `Point2DMeasurement`
- `Line2DMeasurement`
- `Mask2DMeasurement`
- `MetricDepthMeasurement` / `DepthOrderMeasurement`
- `TrackMeasurement`
- `VisibilityMeasurement`

`FeatureRef` 描述语义角色与几何 feature id，不包含 `case_name`。adapter 必须显式
声明坐标系、单位、缺失值和观测可识别的自由度；不得用零值填补未知量。

### 2. Contact Constraint IR

```text
ContactConstraint:
  human_site: HumanSite
  object_feature: FeatureRef
  object_coordinate: LocalXYZ | LineS | SurfaceUV | None
  mode: grasp | support | impact | sliding | release
  interval: frame range
  state: candidate | active | occluded_hold | inactive
  confidence / normal_policy / provenance
```

点、线、面接触保持 tagged union，不做伪 XYZ 转换。VLM 只能改变预定义 constraint
的 gate/state，不能产生连续坐标或权重。

### 3. StateSpec 与 GeometryProvider

`StateSpec` 以数据声明：root translation、root rotation、optional scale、revolute /
prismatic / periodic DOF、joint limits、static parameters 和 gauge constraints。

`GeometryProvider` 提供统一的 feature projection、local-to-world、surface query、
contact point 和可选 Jacobian。sphere、line/capsule、rigid mesh、articulated URDF
是 provider，不是独立优化器。

### 4. Factor IR

首批通用 factors：

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
- `PosePriorFactor`（只能引用当前 run 的观测推导结果）

每个 factor 必须记录输入 measurement/constraint ids、启用 gate、residual、权重来源
和 frame range。核心 factor 不允许读取 case profile 的 `case_name`。

## 分支与执行顺序

| 分支 | 工作内容 | 状态 | 完成条件 |
| --- | --- | --- | --- |
| `refactor/generalized-measurements` | Measurement IR、坐标系/单位/covariance、五 case read-only adapters | done | 63 tests；旧 CSV byte-stable；20 contracts、plugins、encoded/decoded golden 通过 |
| `refactor/contact-constraint-ir` | ContactConstraint、HumanSite、FeatureRef、local-coordinate union、gate adapter | done | 五 case byte-stable shadow；72 tests；contracts/plugins/golden 通过；dead flag 保持拒绝 |
| `refactor/state-spec-kinematics` | StateSpec、DOF/gauge、sphere/line/mesh/URDF GeometryProvider | done | 五 case state shadow/hash/parity verified；89 tests；decoded golden 通过；solver/loss/output 路径无改动 |
| `refactor/factor-registry` | 通用 factor registry、residual trace、组合校验 | in_progress | Factor IR shadow/verifier/validator 已建立；98 tests；decoded golden 通过；不被 solver 消费 |
| `refactor/generic-sequence-solver` | 通用初始化、单帧/序列求解、deterministic attempt provenance | in_progress | sequence problem/diagnostics/candidate sandbox 已建立；116 tests；不读取 baseline pose；不写 accepted 输出 |
| `refactor/migrate-ball-cases` | basketball/football 迁移 | pending | 两 case 全指标不退化；删除其专用连续求解分支 |
| `refactor/migrate-line-case` | stick 迁移 | pending | LineS/contact/时序指标不退化；无 line 专用 optimizer |
| `refactor/migrate-mug-case` | mug rigid mesh + periodic phase 迁移 | pending | gauge-invariant pose、handle、contact、render 接受标准通过 |
| `refactor/migrate-chair-case` | chair URDF + articulated DOF + two-hand contact 迁移 | pending | semantic 2D、contact P90、freeze、render 不退化 |
| `refactor/heldout-generalization` | 未见对象与退化条件验证 | pending | 满足下述零专用 solver 验收 |

每个分支从前一已接受提交建立新 worktree；一个分支只迁移一个 IR 或一个对象族。

## Case 迁移规则

每个现有 case 按相同流程迁移：

1. 从上一接受提交创建新 worktree，并先运行 runtime input hydration dry-run/apply。
2. 冻结现有输入、factor 等价项和逐帧指标。
3. adapter-only shadow run：只生成 IR，不参与求解。
4. factor shadow run：计算 residual，但不改 pose。
5. generic solver 写入独立 result directory。
6. 比较 observation coverage、contact/gate、pose、时序和六路 decoded render。
7. 仅在全部 mandatory gate 通过后切换 capability plugin。
8. 切换后删除对应的专用连续优化入口；保留只读 compatibility reader 的期限必须明确。

## Mug：多组件旋转的泛化模型

Mug 不能被建模为两个互不相关的刚体姿态。首批通用表达应是一个运动学图：

```text
world -> body(root SE3) -> handle(periodic/revolute joint)
```

每帧状态为 `T_world_body + scale + handle.phase`，handle 的世界变换必须由
`T_world_handle = T_world_body * T_body_handle(handle.phase)` 组合得到。body、handle、
rim 和 bottom measurement 都绑定到各自 `FeatureRef`，投影、接触和渲染统一查询
`GeometryProvider.local_to_world(feature, state)`；solver 不允许分别生成两个无约束
rotation CSV。

当前 mug body 近似轴对称，因此 root 绕局部竖直轴的 yaw 与 `handle.phase` 不是两个
独立可观测量：对任意 `delta`，给 root yaw 加 `delta`、同时给 phase 减 `delta`，
复合后的 handle 几何可以保持不变。迁移时必须声明跨 DOF 的 coupled gauge，并选择
一个确定规范（首选保持现有 fresh seed 约定：body axial yaw 固定到 gauge，所有可观测
轴向角写入 phase）。禁止对两个量各自加 identity/historical prior 来伪造可观测性。

通用初始化和优化分解为：

1. body mask/center/depth 建立 root translation、tilt 和 scale 的观测推导候选；
2. visible handle feature 约束复合轴向角，hidden span 保留 `occluded_hold` 并使用通用
   periodic temporal factor；
3. palm-handle contact 作用于组合后的 handle feature，而不是对象名为 mug 的列；
4. 以组合后 feature 的 2D/3D 位置、接触和 render 做 gauge-invariant 验收，不比较
   raw yaw/phase 数值。

为此 `StateSpec` 还需从平铺 DOF 扩展出 `KinematicNode`、`JointSpec(parent/child/axis)`、
`FeatureAttachment` 和可同时引用多个 DOF 的 coupled `GaugeConstraint`。这套表达应同样
覆盖带转动把手、盖子或铰接子件的 held-out 对象；对象配置只提供几何、关节轴和 feature
mapping，不新增 mug/cup 专用连续 solver。

当前 `phase_snapshot_fallback` gap 来自 canonical `benchmark_vlm_qwen` 的 Stage 3
provenance（其中明确记录 `snapshot_fallback_used=true`），不是“mug 的双组件旋转无法
泛化”。`mug_observation_seed_v4` 已证明 current-run 观测足以生成可接受候选；该 gap
只有在上述 phase initializer 和 coupled gauge 被 generic solver 正式消费并通过独立
sandbox 回归后才能关闭，不能仅因存在 fresh 结果而从审计中删除。

## Chair：当前专用机制及其通用替代

Chair 的物理状态本身可以用通用模型表达：root SE3、两个有 URDF limit 的 revolute
joint，以及绑定在 articulated links 上的 line/endpoint features。当前被标记为
`semantic_graph_solver_private`，原因是求解实现尚未经过这些通用边界：

- `chair_twohand_endpoint_se3.py` 直接导入 chair pose/render 模块和固定 base transform；
- articulation 根据 chair part/segment id 和局部 X 轴分支计算，未通过
  `GeometryProvider.joint_transform/Jacobian`；
- contact 字段、左右 top-rail endpoint、两个 joint 名称和 palm pairing 写在 solver 内；
- 两端点对齐、绕 contact chord 的剩余 twist、2D semantic line 拟合、关节求解和
  `pose_lock_reason` 在同一对象专用优化路径内完成。

其中纯 `align_contact_chord(local[2], target[2], init)` 已经不依赖 chair，可保留并提升
为通用 `RigidCorrespondenceInitializer`。两点对应只能确定 translation 和两个旋转自由度，
绕 chord 的 twist 是结构性 rank deficiency；通用 solver 应显式报告该 null space，再用
`LineReprojectionFactor`、`JointLimitFactor` 和 coupled `GaugeConstraintFactor` 消除，而不在
chair runner 中隐式补齐。双手和 top rail 只是两组 `HumanSite <-> FeatureRef` 对应，关节
传播由 URDF `GeometryProvider` 提供。

`pose_lock_reason` 也应迁移为 candidate attempt 的通用 constraint-acceptance invariant：
已通过硬接触/几何下界 gate 的状态不能被后续非约束 smoother 静默覆写，但不应出现
chair 专属锁。完成这些迁移并通过 semantic 2D、contact median/P90、freeze 与六路 render
回归后，才能关闭 `semantic_graph_solver_private`；它表示当前软件封装仍专用，不表示
chair 必须永远使用专用算法。

## Runtime Input Hydration Gate

所有后续分支必须在代码迁移前执行：

```bash
python scripts/shared/generic_contact_pipeline/tools/sync_golden_inputs.py \
  --source-root /mnt/hdd/AudioHOI
python scripts/shared/generic_contact_pipeline/tools/sync_golden_inputs.py \
  --source-root /mnt/hdd/AudioHOI --apply
```

接受标准：

- dry-run/apply 都必须 `errors=0`；
- DA3、GVHMR、SAM2 masks、chair tracking 等 ignored runtime inputs 必须 verified 或 copied；
- 若目标已有不同 hash，sync 必须拒绝覆盖，由人工决定是否接受新 baseline；
- golden 中的 missing 只能表示真实不存在的历史证据，不允许表示 worktree 未补齐。

当前 `refactor/state-spec-kinematics` worktree 验证结果：`verified=89 would_copy=0 copied=0 errors=0`。

## 当前泛化边界

截至本计划阶段，系统还不是完整泛化重建算法：

- 观测层已进入 `Measurement IR`，并通过五 case read-only shadow adapter 验证。
- 接触层已进入 `ContactConstraint IR`，并通过五 case read-only shadow adapter 验证。
- 姿态/几何层已进入 `StateSpec + GeometryProvider` shadow；当前 factor registry 只建立
  residual/factor trace 与组合校验，不替换 Stage 3/4 连续求解。
- 真正的泛化姿态求解必须等 `factor-registry` 和 `generic-sequence-solver` 通过后才成立。

因此，“泛化修复”计划覆盖观测、接触、姿态三层；当前已经把三层输入语义与 factor
trace 放入 shadow gate。真正的连续姿态求解泛化仍处于后续 `generic-sequence-solver`
迁移阶段。

## Mandatory regression gates

- 无 `historical_solved_*`、snapshot、canonical final pose/phase optimizer input。
- Stage 1-4 typed contract、attempt lineage、artifact-store integrity 全部通过。
- 输入 coverage 不下降；缺失/遮挡保持显式状态。
- contact part/side/mode/interval 逐帧比较，差异必须列出。
- 2D overlay、depth、contact median/P90、penetration、temporal spikes、static drift
  分项不退化；不以总分掩盖单项失败。
- articulation/periodic DOF 使用 gauge-invariant metric。
- 六路视频记录文件 hash 与 decoded RGB24 hash；变更 case 做指标和视觉审计。
- `python -m pytest -q`、五 case contracts、plugins、golden、hydration dry-run 全通过。

## 泛化验收集

| 样本/条件 | 主要能力覆盖 | 限制 |
| --- | --- | --- |
| 乒乓球 | sphere、小尺度、快速运动、impact contact | 复用 ball StateSpec/factors，不新增 solver |
| suitcase | rigid/articulated mesh、handle grasp、地面 support | 只新增 geometry/feature mapping；若新增 DOF 必须为通用 StateSpec |
| golf | line/capsule、双手/单手 grasp、快速 swing | 复用 LineS 与通用 factors，不新增 golf optimizer |
| 背视角篮球 | 遮挡、human-side ambiguity | 不使用正视角 hard-coded side mapping |
| blur/降分辨率/遮挡/丢帧 | uncertainty、missing observation、temporal propagation | 同一 sample + condition schema，不复制 case solver |

泛化通过的最低定义：至少一个未参与 factor/solver 开发的对象，只增加配置、几何
provider 数据和 detector-to-feature mapping 即完成 fresh run；不允许新增 observation /
contact / pose continuous solver 文件。

## 需要新增的代码边界

```text
core/measurements/
  types.py
  frames.py
  adapters.py
core/contact_constraints/
  types.py
  state_machine.py
core/state/
  spec.py
  geometry_provider.py
core/factors/
  registry.py
  projection.py
  depth.py
  contact.py
  temporal.py
  physical.py
core/solver/
  initializer.py
  sequence_problem.py
  diagnostics.py
```

Case plugin 最终只负责：选择现有 adapter、提供 geometry/features、声明 StateSpec 和
factor 配置。任何 plugin 若直接读取 baseline pose、直接运行对象专用 least-squares
或直接覆写最终 pose，contract audit 必须失败。

## 决策记录

- 2026-07-24：现有 typed contracts/plugin registry 被判定为必要但不充分；它们统一
  接口和调度，没有统一 measurement/contact/state/factor 语义。
- 2026-07-24：Phase G 排在 chair solved-seed removal 之后，避免把历史 seed 伪装成
  generic initializer 的训练/参考输入。
- 2026-07-24：held-out 通过条件采用“零新增专用连续 solver”，而不是仅要求 pipeline
  能运行。
- 2026-07-26：`generic-sequence-solver` 首步只建立 problem/attempt shadow contract；
  不执行 optimizer，避免把旧 pose 或 compatibility seed 偷渡为通用初始化。

## 计划维护记录

| 日期 | 状态 | 变更 | 文件 |
| --- | --- | --- | --- |
| 2026-07-24 | done | 建立 Phase G 主计划，定义三层 IR、StateSpec/factors、分支顺序与 held-out gates | 本文件 |
| 2026-07-24 | done | 将 Phase G 接入 chair 后续顺序 | `docs/chair_seed_removal_plan.md` |
| 2026-07-24 | done | 在 v2 主线设计中限定当前 generic 声明边界 | `docs/generic_pipeline_v2_mainline_design_cn.md` |
| 2026-07-24 | done | Measurement IR、三类 legacy schema adapter、显式字段 coverage 与五 case shadow hash | `docs/generalized_measurements_plan.md` |
| 2026-07-25 | done | ContactConstraint IR、LocalXYZ/LineS union、离散 gate 与五 case contact shadow hash | `docs/contact_constraint_ir_plan.md` |
| 2026-07-25 | done | 完成 StateSpec/GeometryProvider shadow、五 case frozen hash、parity/verifier 与回归门禁 | `docs/state_spec_kinematics_plan.md` |
| 2026-07-25 | in_progress | 创建 factor-registry 分支，补齐 ignored inputs，修复 state shadow canonical hash 的 worktree 路径依赖；建立首版 Factor IR shadow/verifier | `docs/factor_registry_plan.md` |
| 2026-07-26 | in_progress | 增加 Factor IR 组合校验；gap id 改为机制级命名，避免 object-name-specific 判断；更新 98-test 回归证据 | `docs/factor_registry_plan.md` |
| 2026-07-26 | in_progress | 创建 generic-sequence-solver 分支，建立只读 sequence problem shadow、deterministic shadow attempt id 与 accepted-output write 防线 | `docs/generic_sequence_solver_plan.md` |
| 2026-07-26 | in_progress | 增加 sequence solver shadow diagnostics；basketball/football 标记 future-shadow-ready，mug/chair 被机制 gap 阻断；stick line contact 改为 nonblocking compatibility | `docs/generic_sequence_solver_plan.md` |
| 2026-07-26 | in_progress | 增加 candidate sandbox guard；只允许 future-ready case 写隔离 manifest，拒绝 accepted output 文件名 | `docs/generic_sequence_solver_plan.md` |
| 2026-07-26 | in_progress | 明确 mug 根姿态/子组件 phase 的 coupled gauge，以及 chair 私有 chord/gauge 求解向通用 kinematic graph/factor 的迁移边界 | 本文件、`docs/generic_sequence_solver_plan.md` |
