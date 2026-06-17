# Mug 把手相位与深度锚定 Pipeline 文档

**入口：** `python src/mug/pipeline.py --sample-dir samples_known_object/02_mug`

---

## 当前本地清理状态（2026-06-17）

本地目前保留两类结果：

1. **主 pipeline 必需输入/基线/CSV 输出**：这些不能删，因为后续阶段会读取，或用于复现 M45。
2. **最终 M45 六视频结果**：这是当前清理后的最终可视化结果。

已清理掉的是未追踪的探索性中间渲染目录，例如早期 HaMeR/模拟抓握/palmgate/若干 M18-M39 debug render。它们不是本文主 pipeline 的输入。如果需要再次查看，可以由对应脚本重新生成。

### 必须保留的主 pipeline 输入

| 路径 | 用途 |
|---|---|
| `proxy/mug_body_only_cylinder_pose_table_static_sequence.csv` | Stage 1-3 的基础杯体 6D pose 输入。 |
| `results/renders/M12_articraft_rigid_mesh_vlm/handle_phase_all.csv` | Stage 1 的把手相位弱先验。 |
| `results/mug_articraft_contact_points/mug_articraft_contact_points.csv` | Stage 1/3 的接触事件与 mug-local 接触点。 |
| `results/mug_grasp_anchor_state/mug_grasp_anchor_state.csv` | Stage 3 的稳定抓握锚点输入。 |
| `results/contact_candidates_object_proxy/contact_state_frames.csv` | Stage 3 的接触/桌面状态与深度偏移输入。 |
| `results/gvhmr/result.pkl` | Stage 3 和最终 render 的人体 3D 关节/相机内参输入。 |
| `annotations/vlm_handle_visibility_full/qwen_handle_visibility.csv` | Stage 1/2 的把手可见性输入。 |
| `articraft/materialized_mug_mesh/` | Stage 1/4 使用的 Articraft mug mesh。 |

### 主 pipeline 标准输出

| 路径 | 来源 | 用途 |
|---|---|---|
| `results/pipe/handle_phase.csv` | Stage 1 | VLM 过滤 + 接触约束优化后的初始 handle phase。 |
| `results/pipe/corrected_phase.csv` | Stage 2 | 遮挡段远侧物理修正后的 phase。 |
| `results/pipe/anchored_pose.csv` | Stage 3 | 用手部 3D 锚定后的 mug pose。 |
| `results/renders/M17_phase_corrected/corrected_handle_phase.csv` | 历史 M17 baseline | 当前 M45 no-hide phase 的基线来源，必须保留。 |

### 当前最终 M45 结果

M45 是在主 pipeline/M17 结果基础上的局部物理修正版，目标是解决：

- Euler rotation branch jump，目前由相邻帧旋转 geodesic outlier 自动检测，再对检测窗口做 Slerp；
- drinking-entry handle 左右摆动；
- table-static release 以后 pose/scale 仍抖动的问题，目前由 `contact_state_frames.csv` 中的 support confidence / support gap / acceleration 连续段自动检测；
- 保持 `handle_loop` 为真实 mesh，不用 visibility 去隐藏 handle。

注意：drinking-entry 的 handle phase 目前仍是 M45 的物理先验轨迹，不是完整自动优化。后续应将其升级为“数据项 + 物理平滑项”的全局 phase 求解，而不是继续手动指定关键帧。

| 路径 | 用途 |
|---|---|
| `results/final_result/handle_phase.csv` | 从 M17 phase baseline 生成的 no-hide、smooth-entry handle phase。 |
| `results/final_result/object_pose.csv` | M18/M45 pose：自动检测 rotation jump 后 Slerp + 自动检测 table-static 后 freeze。 |
| `results/renders/final_result/` | 当前最终六视频结果。 |
| `scripts/known_object/mug/run_mug_m18_physical_nohide_pipeline.py` | 复现 M43 phase + M45 pose，并可选重新渲染六视频的唯一干净入口。 |

重跑 M45 的 CSV：

```bash
PYTHONPATH=. python scripts/known_object/mug/run_mug_m18_physical_nohide_pipeline.py \
    --sample-dir samples_known_object/02_mug
```

重跑 M45 六视频：

```bash
PYTHONPATH=. python scripts/known_object/mug/run_mug_m18_physical_nohide_pipeline.py \
    --sample-dir samples_known_object/02_mug \
    --render-full6
```

当前 `results/renders/` 中应保留：

```text
M12_articraft_rigid_mesh_vlm/
M14_joint_contact_handle_phase/
M15_original_phase_recovered/
M17_phase_corrected/
M18_anchor_depth_scene/
final_result/
```

其中 `M12/M14/M15/M17/M18_anchor` 是已追踪的历史基线或主 pipeline 参照；`final_result` 是当前最终结果。

---

## 总体流程

Pipeline 从单目 RGB 视频中估计马克杯的 **3D 姿态**（位置 + 把手朝向），视频内容为人物拿起、持握、放下马克杯的过程。共四个顺序执行的阶段；只有最后阶段生成视频渲染，前三个阶段仅输出 CSV 文件。

```
Stage 1  phase.py       VLM 时序过滤 + 全局相位优化器   → handle_phase.csv
Stage 2  correction.py  遮挡段弧路径插值修正            → corrected_phase.csv
Stage 3  anchor.py      Z + XY 深度锚定精化             → anchored_pose.csv
Stage 4  render         最终场景渲染（6 个视频）         → results/renders/pipe/
```

---

## VLM 调用说明（Pipeline 前置步骤）

在运行 `pipeline.py` 之前，需要执行两次独立的 **Qwen3-VL-8B-Instruct** 推理，其输出作为主 pipeline 各阶段的输入。

---

### VLM 调用 1 — 把手可见性逐帧标注

**脚本：** `scripts/shared/radius_free_proxy/stage1_observation/run_qwen_vlm_handle_visibility.py`  
**处理对象：** `object_observations.csv` 中的每一帧（可指定 stride 或帧范围）  
**输出：** `annotations/vlm_handle_visibility_full/qwen_handle_visibility.csv` + `.json`

#### 图像裁剪策略

对每帧，以杯体 bounding box 为基础向外扩展裁剪：
- 左右各扩展 bbox 宽度的 **65%**
- 上方扩展 bbox 高度的 **35%**
- 下方扩展 bbox 高度的 **45%**

这样 VLM 能看到经常超出杯体 bbox 的把手，以及可能遮挡把手的手部。

#### Prompt 设计与 JSON 输出字段

模型对杯子裁剪图进行检视，返回结构化 JSON：

| 输出字段 | 取值范围 | Pipeline 使用方式 |
|---|---|---|
| `visibility` | `"visible"` / `"hidden"` / `"uncertain"` | Stage 1 时序过滤；Stage 2 遮挡段划分 |
| `handle_visible` | bool | Stage 1 时序过滤 |
| `handle_contact` | bool — 仅当手指确实在 C 形把手环上时为 true，不包括仅接触杯身 | Stage 1 附着残差激活条件 |
| `hand_contact_part` | `"handle"` / `"body"` / `"rim"` / `"unknown"` / `"none"` | 接触事件分类 |
| `recommended_visibility_constraint` | `"force_visible"` / `"force_hidden_far_side"` / `"force_hidden_by_hand"` / `"weak_unknown"` | Stage 1 过滤覆盖规则 |
| `confidence` | 0.0–1.0 | Stage 1 附着权重缩放 |
| `occlusion_reason` | `"none"` / `"hand"` / `"mug_body"` / `"far_side"` / `"out_of_crop"` | 诊断用 |
| `visible_side` | `"left"` / `"right"` / `"front"` / `"back"` / `"unknown"` | 诊断用 |
| `handle_shape_visible` | `"clear_c_loop"` / `"partial_arc"` / `"small_attachment"` / `"not_visible"` | 诊断用 |
| `yaw_anchor_quality` | `"high"` / `"medium"` / `"low"` / `"none"` | 诊断用 |
| `body_contact`、`rim_contact` | bool | 诊断 / 接触事件分类 |
| `hand_occludes_handle`、`body_self_occludes_handle` | bool | 诊断用 |
| `short_reason` | 字符串 | 诊断用 |

#### Pipeline 使用方式

- **Stage 1 `phase.py`** 读取 `visibility`、`handle_contact`、`hand_contact_part`、`recommended_visibility_constraint`、`confidence`，用于：
  - 时序过滤：消除 ≤3 帧的短噪声岛
  - 附着残差激活：仅在 `handle_contact==True` 且 `confidence ≥ 0.15` 时触发手-把手约束
  - 区分帧类型：确认把手抓握 / 杯沿接触 / 沿用上一帧
- **Stage 2 `correction.py`** 读取 `visibility` 定位遮挡/可见段边界，用于弧路径插值。

```bash
python scripts/shared/radius_free_proxy/stage1_observation/run_qwen_vlm_handle_visibility.py \
    --sample-dir samples_known_object/02_mug \
    --model-id Qwen/Qwen3-VL-8B-Instruct \
    --stride 1 \
    --save-crops
```

---

### VLM 调用 2 — 接触关键帧精标注

**脚本：** `scripts/shared/radius_free_proxy/stage2_contact_candidates/run_qwen_mug_contact_keyframes.py`  
**处理对象：** 从确认的直接抓握锚定帧中自动选取的代表帧（每段取首/中/尾，通常约 3–10 帧）  
**输出：** `annotations/vlm_mug_contact_keyframes/mug_contact_keyframe_annotations.csv` + `.json`

#### 帧选取策略

脚本自动从 `mug_grasp_anchor_state.csv` 中选代表帧：
- 找出所有 `frame_mode == "direct_grasp_anchor"` 的帧
- 将连续帧组合成"段"
- 每段取 3 帧：首帧、中间帧、末帧

#### 图像裁剪策略

以杯体 bbox 与接触 UV 坐标的并集为中心，取正方形裁剪（最小边长 **420 px**，四周 **150 px** margin），保证 VLM 能看到手指细节、掌部与把手全貌。

同时在 prompt 中注入**上下文提示**：tracker 判定的手部标签（left/right\_hand）、frame\_mode、stable grasp 来源帧号。

#### Prompt 设计与 JSON 输出字段

| 输出字段 | 取值范围 | Pipeline 使用方式 |
|---|---|---|
| `contact_visible` | bool | 验证 / 过滤 |
| `hand_side` | `"left_hand"` / `"right_hand"` / `"unknown"` | 确认或修正 tracker 标签 |
| `object_part` | `"handle"` / `"body"` / `"rim"` / `"bottom"` / `"unknown"` | 语义标注 |
| `object_region` | `"upper_handle"` / `"middle_handle"` / `"lower_handle"` / `"handle_inner"` / `"handle_outer"` / `"body_side"` / ... | 定位把手上的具体抓握位置 |
| `handle_grasp_type` | `"pinch_handle"` / `"hook_handle"` / `"palm_support"` / `"body_grasp"` / `"not_handle_grasp"` | 抓握方式分类 |
| `contact_fingers` | 手指列表：`"thumb"` / `"index"` / ... / `"palm"` | 参与接触的手指 |
| `primary_contact_finger` | 单个手指名 | 主导接触手指 |
| `use_as_stable_grasp_keyframe` | bool — VLM 确认该帧是否适合作为锚点 | 锚点质量门控 |
| `confidence` | 0.0–1.0 | 权重或过滤阈值 |
| `reason` | 字符串 | 诊断用 |

#### Pipeline 使用方式

该调用的输出**不直接**被四个主 pipeline 阶段读取，而是作为：
- **语义标注**：为 `mug_grasp_anchor_state.csv` 中的锚点帧补充 VLM 验证过的把手区域信息（`object_region`、`handle_grasp_type`）
- **质量确认**：VLM 判定 `use_as_stable_grasp_keyframe=False` 或 `confidence < 0.4` 的帧，在 Stage 3 构建锚点目标时可降权
- **调试辅助**：保存裁剪图和 JSON 供人工检查抓握质量

```bash
python scripts/shared/radius_free_proxy/stage2_contact_candidates/run_qwen_mug_contact_keyframes.py \
    --sample-dir samples_known_object/02_mug \
    --model-id Qwen/Qwen3-VL-8B-Instruct
```

---

## Pipeline 前置输入

运行 `pipeline.py` 前需已存在以下文件（由前置准备脚本产生）：

| 文件 | 说明 |
|------|------|
| `proxy/mug_body_only_cylinder_pose_table_static_sequence.csv` | 每帧杯体姿态（相机空间下 x, y, z, yaw, pitch, roll, scale），由 `fit_mug_body_only_cylinder_pose.py` 拟合产生。 |
| `results/renders/M12_articraft_rigid_mesh_vlm/handle_phase_all.csv` | M12 阶段 VLM 覆盖拟合得到的每帧把手相位先验，作为弱初始化使用。 |
| `results/mug_articraft_contact_points/mug_articraft_contact_points.csv` | 每帧接触事件标注：确认把手抓握、杯沿接触、沿用上一帧等标记。 |
| `results/mug_grasp_anchor_state/mug_grasp_anchor_state.csv` | 稳定抓握锚点：杯体局部坐标系下的接触点 (`stable_grasp_local_{x,y,z}`) 及手部标签（左/右）。 |
| `results/contact_candidates_object_proxy/contact_state_frames.csv` | 每帧接触/地面状态标志及深度偏移量 (`contact_depth_offset_m`)。 |
| `results/gvhmr/result.pkl` | GVHMR 人体姿态估计结果（3D 关节 + 相机内参）。 |
| `annotations/vlm_handle_visibility_full/qwen_handle_visibility.csv` | VLM 逐帧分类：把手是否可见。 |
| `articraft/materialized_mug_mesh/` | Articraft 参数化马克杯网格各部件（body_shell、rim_ring、bottom_disk、handle_loop 的 .obj 文件）。 |

---

## Stage 1 — 把手相位优化（`src/mug/phase.py`）

**输入 →** 杯体姿态 CSV、M12 相位先验 CSV、接触点 CSV、VLM 可见性 CSV  
**输出 →** `results/pipe/handle_phase.csv`（列：`frame, time, phase_rad, phase_deg`）

### 1a. VLM 时序一致性过滤

原始逐帧 VLM 标注存在噪声——可能在一段明显遮挡的序列中出现几帧"把手可见"的误判。算法将**短岛**（≤3 帧、被相反状态包围的连续段）翻转为与邻居一致，迭代至收敛。

### 1b. 可见性 Alpha

计算每帧的连续权重 α ∈ [0, 1]：
- 完全可见帧：α = 1.0
- 完全遮挡帧：α = 0.0
- 边界渐变：在每次可见/遮挡切换处做 ±5 帧线性过渡
- 短可见段（≤10 帧）：α 上限 0.85
- 紧邻可见段的遮挡帧：最大渗透 0.35

Alpha 目前作为备用字段保留，未直接进入优化器残差。

### 1c. 相位优化器

**变量：** θ[0..N-1] — 每帧把手相位（弧度，展开版）

**初始化：** `theta0 = unwrap(M12_先验相位)`

**残差项**（least-squares，`loss='soft_l1'`，`f_scale=1.0`，`max_nfev=120`）：

| 残差项 | 公式 | 作用 |
|--------|------|------|
| M12 先验 | `0.35 · wrap(θ[i] - θ0[i]) / 25°` | 弱拉向 M12 初始化 |
| 手-把手附着约束 | `(24·w·max(0.35, conf)) · ‖proj(抓握点, θ[i]) - 手像素位置‖ / 7px` | 用当前相位将稳定抓握点投影到图像，最小化与观测手部像素的重投影误差 |
| 桌面静止锁定 | `3.5·wt · wrap(θ[i] - θ[table_start]) / 4°` | 一旦检测到杯子静置于桌面（持续 ≥12 帧地面支撑），将后续所有帧锁定在桌面起始相位 |
| 速度平滑 | `0.8 · Δθ / 10°` + 超限惩罚 `2.0 · max(0, |Δθ|-8°) / 3°` | 抑制快速旋转；对 >8°/帧的跳变额外惩罚 |
| 加速度平滑 | `1.2 · Δ²θ / 5°` | 抑制抖动 |

**附着权重 `w`** 按帧分类：
- `use_this_point_for_hand_attachment == 1`（确认新抓握）：w = 1.0
- `use_previous_grasp_for_hand_attachment == 1`（沿用上一帧）：w = 0.60
- `rim_drinking_contact == 1`（杯沿接触，无把手）：w = 0.30

**稳定抓握局部点** (`stable_grasp_local_{x,y,z}`) 仅在 `use_this_point = 1` 的帧更新；杯沿接触帧不更新。优化前会从当前段之前的任何确认帧继承此局部点。

**桌面检测：** 扫描第一个连续 ≥12 帧满足 `support_conf ≥ 0.65` 且 `object_motion_score ≤ 0.15` 的段，其起始帧即为 `table_start`。

---

## Stage 2 — 远侧相位修正（`src/mug/correction.py`）

**输入 →** `handle_phase.csv`、杯体姿态 CSV、VLM 可见性 CSV  
**输出 →** `results/pipe/corrected_phase.csv`（列：`frame, time, m17_phase_rad, m17_phase_deg, m14_phase_rad, m14_phase_deg, is_error_frame, phase_correction_rad, vlm_visibility`）

### 问题

在**遮挡段**（VLM 标注把手不可见的帧）内，Stage 1 没有图像证据，优化器可能选择将把手置于**近侧**（朝向摄像机），这与把手被遮挡的物理事实矛盾。

### 算法

对每个遮挡段 [i, j)：

1. 读取段前最后可见帧的相位 `left = phase[i-1]`，段后第一可见帧的相位 `right = phase[j]`。
2. 计算两条候选弧路径：
   - **短弧**（`delta_short`）：从 `left` 到 `right` 的 ≤180° 弧——旋转量更小。
   - **长弧**（`delta_long`）：互补的 >180° 弧，沿另一方向绕圈。
3. 对每条候选弧，统计插值后有多少遮挡帧满足 `sin(phase + yaw) > 0`——即把手位于**远侧**（背对摄像机，与遮挡物理一致）。
4. **选择远侧帧更多的弧。** 若两弧相等，优先选短弧。
5. 沿所选弧对遮挡帧做线性插值：`phase[i+k] = left + (k+1)/(n+1) · arc_delta`

**可见帧永远不被修改。**

### 后置平滑

对全序列展开相位施加轻度高斯平滑（`sigma/2 = 1.5` 帧，`mode='nearest'`），消除 Stage 1 可见帧的亚度级抖动。

---

## Stage 3 — 深度与图像平面 XY 锚定（`src/mug/anchor.py`）

**输入 →** 杯体姿态 CSV、`corrected_phase.csv`、抓握状态 CSV、接触状态 CSV、`gvhmr/result.pkl`  
**输出 →** `results/pipe/anchored_pose.csv`（与杯体姿态 CSV 同字段，x/y/z 被精化）

### 动机

杯体姿态 CSV 中的**朝向**（yaw/pitch/roll）和**图像平面位置**较准确，但相机空间深度（z）来自 proxy 跟踪，在手部接触期间可能有较大漂移。GVHMR 提供高精度的 3D 手部关节位置；利用手杯接触约束来锚定杯体深度。

### 3a. 锚点目标构建

对每个处于**持续接触**状态的帧（`human_contact_state == 1` 或 `anchor_contact_state == 1` 或 `use_this_point` 或 `use_previous_grasp`）：

1. 用修正后的把手相位旋转稳定抓握局部点 `stable_grasp_local`，得到接触点在相机空间的位置 `anchor_cam`。
2. 计算接触点相对杯心的偏移量：`delta_{x,y,z} = anchor_cam - mug_center`。
3. 从 GVHMR 读取对应手部的 3D 位置 `hand_xyz`。
4. 将原始 `contact_depth_offset_m` 裁剪至 **±0.08 m**（物理约束：手杯间隙始终 < 8 cm；超出此范围的值是 proxy 跟踪噪声）。
5. 计算锚点目标：
   - `target_z = hand_z - 裁剪偏移 - delta_z`
   - `target_x = hand_x - delta_x`
   - `target_y = hand_y - delta_y`

**锚点权重** 按帧分类：
- `use_this_point = 1`（确认更新）：weight = `3.0 · conf`，裁剪至 [0.25, 4.0]
- `use_previous_grasp = 1`（沿用上一帧）：weight = `1.4 · conf`
- 其他持续接触：weight = `1.0 · conf`

### 3b. Z 轴优化

**参考轨迹：** 在所有锚点 `target_z` 之间线性插值。首个锚点之前的帧设为第一个锚点值，最后一个锚点之后的帧设为最后一个锚点值，全程裁剪至 ≥ 0.30 m。

**优化器**（`loss='soft_l1'`，`f_scale=0.03`，`max_nfev=300`）：

| 残差项 | 公式 | 权重 |
|--------|------|------|
| 参考轨迹拉力 | `z[i] - z_ref[i]` | 0.45 |
| 锚点约束 | `z[i] - target_z[i]` | 逐帧 `anchor_weight` |
| 速度平滑 | `Δz[i]` | 4.0 |
| 加速度平滑 | `Δ²z[i]` | 12.0 |
| 桌面静止锁定 | `z[i] - z[table_start]`（i ≥ table_start） | 20.0 |

### 3c. XY 轴优化

Z 优化完成后，用新深度将杯心 UV 坐标反投影得到新的 XY 参考：
`x_ref = (u - cx) · z_final / fx`，`y_ref = (v - cy) · z_final / fy`

**优化器**（同上，`f_scale=0.04`）：

| 残差项 | 权重 |
|--------|------|
| 参考轨迹拉力 `xy[i] - xy_ref[i]` | 0.30 |
| 锚点约束 `xy[i] - target_xy[i]` | `anchor_weight · 0.75` |
| 速度平滑 `Δxy` | 3.0 |
| 加速度平滑 `Δ²xy` | 8.0 |
| 桌面静止锁定 | 15.0 |

---

## Stage 4 — 最终渲染

**脚本：** `scripts/shared/radius_free_proxy/stage5_render/render_mug_articraft_camera3d_scene.py`  
**输入 →** `anchored_pose.csv`、`corrected_phase.csv`、Articraft 网格  
**输出 →** `results/renders/pipe/`（6 个视频）

| 视频 | 说明 |
|------|------|
| `camera3d_object_only.mp4` | 相机空间 3D 视角，仅杯子 |
| `camera3d_with_human.mp4` | 相机空间 3D 视角，杯子 + SMPL-X 人体 |
| `overlay_object_only.mp4` | 杯子网格投影叠加到原始视频帧 |
| `overlay_with_human.mp4` | 杯子 + 人体叠加到视频 |
| `side_yz_object_only.mp4` | 侧视图（深度 vs 高度），仅杯子 |
| `side_yz_with_human.mp4` | 侧视图，杯子 + 手部关节 |

---

## 运行 Pipeline

```bash
# 完整 pipeline（从 repo 根目录，audiohoi 环境）
python src/mug/pipeline.py --sample-dir samples_known_object/02_mug

# 从指定阶段开始重跑
python src/mug/pipeline.py --sample-dir samples_known_object/02_mug --from anchor

# 只运行某一阶段
python src/mug/pipeline.py --sample-dir samples_known_object/02_mug --only render

# 强制重跑所有阶段
python src/mug/pipeline.py --sample-dir samples_known_object/02_mug --force

# 快速线框预览渲染（无实体网格，速度快）
python src/mug/pipeline.py --sample-dir samples_known_object/02_mug --only render --wireframe
```

各阶段也可单独运行：

```bash
python src/mug/phase.py --sample-dir samples_known_object/02_mug
python src/mug/correction.py --sample-dir samples_known_object/02_mug
python src/mug/anchor.py --sample-dir samples_known_object/02_mug
```

---

## 关键设计决策

- **中间步骤不渲染。** Stage 1-3 只输出 CSV 文件。渲染只在最后进行一次，避免中间视频带来的性能开销。
- **±0.08 m 深度偏移裁剪。** Proxy 跟踪错误在接触期间可产生 ±1 m 级别的偏移。物理上手杯间隙始终 < 8 cm，超出范围的值是跟踪噪声，直接裁剪。
- **稳定抓握局部点。** 接触点以杯体局部坐标存储，随当前相位旋转。这使附着残差对相位可微——改变相位会旋转把手接触点在图像上的投影位置。
- **桌面静止锁定。** 一旦检测到杯子稳定支撑在桌面（通过持续地面接触 + 低运动分数），后续所有帧的相位和深度被冻结，防止优化器在无有效信号的帧上漂移。
- **Soft-L1 损失。** 所有优化器使用类 Huber 鲁棒损失，抑制异常锚点帧（如单帧 VLM 误判或 proxy 跟踪坏点）对结果的影响。
