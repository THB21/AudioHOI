# Radius-Free Object Proxy Pipeline（中文）

## 0. 目标与核心约束

这条线的目标是把“已知半径球”的 pipeline 泛化成“任意 object proxy”的 pipeline。核心要求是：

- 不用已知球半径。
- 不用 `radius_px -> depth`。
- 不用“球心到地面距离等于半径”的约束。
- object 的 3D 深度来自 DA3 depth prior 和 object mask/proxy 采样。
- contact 只负责提供候选帧、人体部位、support/floor 状态和后续 anchor。

当前主目录：

```text
scripts/shared/radius_free_proxy/
  stage0_preprocess/
  stage1_observation/
  stage2_contact_candidates/
  stage3_da3_init_optimization/
  stage4_anchor_refinement/
  stage5_render/
```

完整顺序：

```text
Stage0 preprocess
  video/audio/frames + SAM2 + CoTracker + GVHMR + DA3 + audio events
↓
Stage1 observation
  object_observations + object_proxy_observations
↓
Stage2 contact candidates
  object proxy + body part proxy + audio -> contact states/events
↓
Stage3 DA3 contact-aware init optimization
  第一次优化：2D ref + DA3 depth + support/contact soft terms
↓
Stage4 anchor refinement
  第二次优化：human contact event frames 作为 z anchor
↓
Stage5 render
  overlay / camera3d / side_yz, h264
```

## 1. Stage0 Preprocess

代码：

```text
scripts/shared/radius_free_proxy/stage0_preprocess/prepare_sample_inputs.py
scripts/shared/radius_free_proxy/stage0_preprocess/prepare_known_object_samples.py
scripts/shared/radius_free_proxy/stage0_preprocess/run_sam2_segmentation.py
scripts/shared/radius_free_proxy/stage0_preprocess/run_cotracker_object_mesh.py
scripts/shared/radius_free_proxy/stage0_preprocess/register_da3_scene_depth.py
scripts/shared/radius_free_proxy/stage0_preprocess/extract_da3_depth_priors.py
scripts/shared/radius_free_proxy/stage0_preprocess/align_audio_events.py
```

### 做什么

- `prepare_sample_inputs.py`：复制/整理 video，抽帧，抽 `audio.wav`，可生成 audio events。
- `run_sam2_segmentation.py`：统一 SAM2 video helper。支持 GroundingDINO 自动 first-frame box，也支持手动 box/points。
- `run_cotracker_object_mesh.py`：基于 mask/track 点生成 object mesh/boundary tracks。
- GVHMR 不在此目录中实现，但结果 `results/gvhmr/result.pkl` 是后续人体 proxy 的输入。
- `register_da3_scene_depth.py`：注册 DA3 scene depth。
- `extract_da3_depth_priors.py`：从 DA3 depth map 按 object proxy 采样 depth prior。
- `align_audio_events.py`：只做 audio event detection，不再生成/使用 `audio_visual_alignment.csv`。

### Audio events 怎么生成

`align_audio_events.py` 使用 `librosa`：

- onset strength
- RMS rise
- local sharpness
- `scipy.signal.find_peaks`

输出：

```text
<sample>/results/events/audio_events.csv
```

Stage2 会把实际使用的 audio 表复制到：

```text
<sample>/results/contact_candidates_object_proxy/audio_events.csv
```

`audio_visual_alignment.csv` 已经从当前 radius-free pipeline 删除。

## 2. Stage1 Observation

代码：

```text
scripts/shared/radius_free_proxy/stage1_observation/build_object_observations.py
scripts/shared/radius_free_proxy/stage1_observation/build_object_proxy_observations.py
scripts/shared/radius_free_proxy/stage1_observation/object_proxy_observation_utils.py
```

### 输出

```text
<sample>/results/object_observations/object_observations.csv
<sample>/results/object_proxy_observations/object_proxy_observations.csv
```

### object proxy 字段

`object_proxy_observations.csv` 包含：

- `ref_u/ref_v`：object 3D ref point 的 2D proxy。
- `support_u/support_v`：object support/bottom proxy。
- `support_v_raw`：未平滑的 object bottom proxy，Stage2 floor peak 使用这个字段。
- `contact_u/contact_v`：靠近人体 active part 的 object contact proxy。
- `object_ref_depth_m`：DA3 在 ref proxy 处采样的深度。
- `contact_proxy_depth_m`：DA3 在 contact proxy 处采样的深度。
- `contact_depth_offset_m = contact_proxy_depth_m - object_ref_depth_m`。
- `active_label/active_part_u/active_part_v/active_part_z`：当前最可能接触的人体部位 proxy。
- `object_motion_score`：object 2D proxy 加速度响应。
- `audio_score`：audio event 支持，来自 `audio_events.csv`。

### Radius-free 关键点

这里不估球半径，也不用半径推 z。深度来自 DA3 + mask/proxy 上的采样点。

## 3. Stage2 Contact Candidates

代码：

```text
scripts/shared/radius_free_proxy/stage2_contact_candidates/run_contact_candidate_detection.py
scripts/shared/radius_free_proxy/stage2_contact_candidates/object_contact_candidate_utils.py
```

### 输出

```text
<sample>/results/contact_candidates_object_proxy/audio_events.csv
<sample>/results/contact_candidates_object_proxy/anchor_contact_candidates.csv
<sample>/results/contact_candidates_object_proxy/floor_contact_candidates.csv
<sample>/results/contact_candidates_object_proxy/contact_state_frames.csv
<sample>/results/contact_candidates_object_proxy/contact_candidates_labeled.csv
<sample>/results/contact_candidates_object_proxy/contact_intervals.csv
```

### Contact part policy

根据 sample metadata/name/prompt 自动选择人体候选部位：

```text
football / soccer / kick / feet -> feet-only
basketball / dribble / bounce / catch / hand -> hands-only
其他 object -> active body proxy
```

这样 basketball 不会因为单帧 nearest body part 抖动误选 foot；football 则只看双脚。

### Human anchor candidate

对 hand/foot policy，分别计算左右人体 proxy 到 object boundary/contact proxy 的距离：

```text
min_gap = min(distance(left_part, object_boundary), distance(right_part, object_boundary))
proximity_score = Gaussian(min_gap, sigma=18px)
anchor_score = 0.55 * proximity_score + 0.30 * audio_support
candidate = local_min(min_gap, radius=2) and min_gap <= 28px
base_state = min_gap <= 38px and anchor_score >= 0.35
```

对 impulse 类事件，最终 state 不直接用 proximity-only，而是：

```text
motion_gate = object_motion_score >= 0.80 or audio_support >= 0.20
anchor_state = bridge((min_gap <= 52px) and motion_gate, gap=2)
```

这样静止后靠近脚/手的 object 不会自动变成 anchor。


### Audio 在 Contact Candidates 中怎么用

Stage2 不再读 `audio_visual_alignment.csv`，只读：

```text
<sample>/results/events/audio_events.csv
```

并把实际使用的 audio 表复制到：

```text
<sample>/results/contact_candidates_object_proxy/audio_events.csv
```

`audio_events.csv` 每行包含：

```text
event, audio_time, audio_frame, peak, prominence, rms_rise, sharpness, audio_score
```

Stage2 会把稀疏的 audio event 转成逐帧 `audio_support`：

```text
for each audio event:
    audio_support[frame +/- 2] = max(audio_support, audio_score)
```

也就是 audio event 会向前后 2 帧扩散，作为 contact proposal 的软证据。

Audio 进入 human anchor score：

```text
anchor_score = 0.55 * proximity_score
             + 0.15 * object_response_score
             + 0.30 * audio_support
```

当前 object response 项基本为 0，所以实际主要是：

```text
anchor_score ≈ 0.55 * proximity_score + 0.30 * audio_support
```

Audio 也参与 impulse contact interval 的打开：

```text
motion_gate = object_motion_score >= 0.80 or audio_support >= 0.20
anchor_state = (min_gap <= 52px) and motion_gate
```

所以 audio 不会单独生成 contact；它的作用是：

- 提高接近人体部位的帧的 anchor score。
- 在 object motion 不够强但有撞击声时打开 contact interval。
- 帮助 interval peak 选出更合理的 contact frame。

最终 `contact_candidates_labeled.csv` 里保留：

```text
source_audio
```

这个字段就是该候选帧对应的逐帧 `audio_support`。

### Floor/support candidate

当前 radius-free 没有半径，floor 不能用“球心到底面 = 半径”。

对于 impulse 场景，floor event 使用 object 自己的 bottom/support proxy peak：

```text
object_support_peak_v = support_v_raw
support_enter = percentile(object_support_peak_v, 85)
support_soft = percentile(object_support_peak_v, 65)
proxy_floor_score = clip((support_v_raw - support_soft) / (support_enter - support_soft), 0, 1)
floor_state = local_max(support_v_raw, radius=2) and support_v_raw >= support_enter
```

绝对 support-plane gap 仍保留在 `floor_contact_candidates.csv` 中作为诊断字段，但 impulse floor event 不再完全依赖它，因为 GVHMR floor plane 有时整体偏移。

### Event selection

`interval_peak` 模式：

- 先把逐帧 `anchor_state` 切成 interval。
- football/feet：每个 interval 选 `anchor_score` 最大帧，避免 image height peak 把 contact frame 偏到飞行帧。
- basketball/hand 等 impulse：每个 interval 选 `object_ref_v` 最小帧，也就是图像里最高点。
- floor：每个 floor interval 选 `support_v_raw` 最大帧。

事件输出条件：

```text
anchor_event = anchor_state and frame == interval_peak and anchor_score >= 0.15
floor_event = floor_state and frame == floor_interval_peak
```

`continuous_state` 模式用于 mug/hold/grasp：持续接触不压成单峰事件。

## 4. Stage3 DA3 Contact-Aware Init Optimization

代码：

```text
scripts/shared/radius_free_proxy/stage3_da3_init_optimization/run_da3_init_optimization.py
```

### 输入

```text
object_proxy_observations.csv
contact_state_frames.csv
gvhmr/result.pkl
```

### 优化变量

每帧 object 3D ref point：

```text
X_t = (tx_t, ty_t, tz_t)
```

### 初始化

使用 2D ref point + DA3 depth 初始化：

```text
X_init = backproject(ref_u, ref_v, object_ref_depth_m, K)
```

### Energy / residuals

Stage3 使用 `scipy.optimize.least_squares(..., loss='soft_l1')`。

默认权重：

```text
center_weight = 0.05
depth_weight = 0.3
support_weight = 10.0
contact_weight = 3.0
vel_weight = 0.0
z_vel_weight = 2.0
z_acc_weight = 0.50
xy_acc_weight = 0.08
```

主要 residual：

```text
E_center  : project(X_t) 对齐 ref_u/ref_v
E_depth   : tz_t 对齐 object_ref_depth_m / object_depth_smooth
E_support : projected_support_v 对齐 support_v
E_contact : human contact state/score 下，让 object contact proxy 靠近 human part z
E_z_vel   : z 一阶速度平滑
E_z_acc   : z 二阶加速度平滑
E_xy_acc  : x/y 二阶加速度轻量平滑
```

Stage3 是 DA3 后的第一次 3D 解，负责得到一个可用但未完全 human-anchor-refined 的 trajectory。

### 输出

```text
<sample>/results/pose6d_object_proxy_da3_init/object_pose6d_sharedcam_trajectory.csv
<sample>/results/pose6d_object_proxy_da3_init/object_pose6d_sharedcam_reprojection_comparison.csv
<sample>/results/pose6d_object_proxy_da3_init/object_pose6d_sharedcam_summary.txt
<sample>/results/pose6d_object_proxy_da3_init/support_geometry.json
```

## 5. Stage4 Anchor Refinement

代码：

```text
scripts/shared/radius_free_proxy/stage4_anchor_refinement/run_anchor_refinement.py
```

### 输入

```text
pose6d_object_proxy_da3_init/object_pose6d_sharedcam_trajectory.csv
object_proxy_observations/object_proxy_observations.csv
contact_candidates_object_proxy/contact_state_frames.csv
contact_candidates_object_proxy/contact_candidates_labeled.csv
gvhmr/result.pkl
```

### Anchor z 逻辑

Stage4 使用 Stage2 的 human contact event frame 作为 z anchor。

对每个 anchor frame：

```text
raw_offset = contact_depth_offset_m
if abs(raw_offset) <= max_contact_depth_offset_m (default 1.0m):
    offset_used = raw_offset
else:
    offset_used = 0

anchor_value = human_contact_part_z - offset_used
```

保留字段：

```text
contact_depth_offset_m       # raw DA3/proxy offset
contact_depth_offset_used_m  # 实际参与 anchor 的 offset，离群值会被置 0
```

这个门控是为了防止 DA3 在 contact proxy 采样点出现 7m/20m 离群深度时，把整段 trajectory 拉爆。

### z reference

默认：

```text
z_ref_mode = anchor_segment
outside_window_mode = boundary_constant
```

含义：

- 在 human anchor event 之间做 anchor-to-anchor segment reference。
- 第一个 anchor 前保持第一个 anchor 的边界值。
- 最后一个 anchor 后保持最后一个 anchor 的边界值。
- 不让 DA3 在 contact window 外重新把 z 拉回错误深度层。

### Optimization energy

Stage4 只优化 z，自变量是非 anchor 帧的 z；anchor 帧固定为 `anchor_value`。

默认权重：

```text
w_ref = 0.7
w_temp = 5.0
w_phys_xz = 1.25
w_phys_y = 1.5
gravity_mps2 = 9.81
```

residual：

```text
E_ref  : z_t 接近 z_ref_t
E_temp : z 的二阶差分平滑
E_phys_xz : flight triplet 中 x/z 二阶差分接近 0
E_phys_y  : flight triplet 中 y 二阶差分接近 g * dt^2
```

使用：

```text
least_squares(..., loss='soft_l1', max_nfev=400)
```

### 输出

```text
<sample>/results/pose6d_object_proxy_anchor_refined/object_pose6d_sharedcam_contactphase_trajectory.csv
<sample>/results/pose6d_object_proxy_anchor_refined/object_pose6d_sharedcam_contactphase_reprojection_comparison.csv
<sample>/results/pose6d_object_proxy_anchor_refined/object_pose6d_sharedcam_contactphase_summary.txt
```

## 6. Stage5 Render

代码：

```text
scripts/shared/radius_free_proxy/stage5_render/render_pose6d_scene.py
```

默认输入：

```text
<sample>/results/pose6d_object_proxy_anchor_refined/object_pose6d_sharedcam_contactphase_trajectory.csv
```

输出：

```text
<sample>/results/renders/pose6d_object_proxy_anchor_refined/ball/overlay.mp4
<sample>/results/renders/pose6d_object_proxy_anchor_refined/ball/camera3d.mp4
<sample>/results/renders/pose6d_object_proxy_anchor_refined/ball/side_yz.mp4
<sample>/results/renders/pose6d_object_proxy_anchor_refined/with_human/overlay.mp4
<sample>/results/renders/pose6d_object_proxy_anchor_refined/with_human/camera3d.mp4
<sample>/results/renders/pose6d_object_proxy_anchor_refined/with_human/side_yz.mp4
```

Render 规则：

- h264 默认输出。
- 候选人体部位全程显示。
- active candidate human part 只在 contact frame 标红圈。
- floor/support frame 仍然显示当前 active human candidate part，但不会把 floor 当成人体部位。
- mug-like object 会画 object proxy，不强行画成球。

## 7. 推荐运行顺序

```bash
python scripts/shared/radius_free_proxy/stage0_preprocess/prepare_known_object_samples.py
python scripts/shared/radius_free_proxy/stage0_preprocess/run_sam2_segmentation.py --sample-dir <sample>
python scripts/shared/radius_free_proxy/stage0_preprocess/run_cotracker_object_mesh.py --sample-dir <sample>
python scripts/shared/radius_free_proxy/stage0_preprocess/register_da3_scene_depth.py --sample-dir <sample> --source-depth-dir <da3_depth_export>
python scripts/shared/radius_free_proxy/stage0_preprocess/extract_da3_depth_priors.py --sample-dir <sample>
python scripts/shared/radius_free_proxy/stage0_preprocess/align_audio_events.py --sample-dir <sample> --fps <fps>
python scripts/shared/radius_free_proxy/stage1_observation/build_object_observations.py --sample-dir <sample>
python scripts/shared/radius_free_proxy/stage1_observation/build_object_proxy_observations.py --sample-dir <sample>
python scripts/shared/radius_free_proxy/stage2_contact_candidates/run_contact_candidate_detection.py --sample-dir <sample>
python scripts/shared/radius_free_proxy/stage3_da3_init_optimization/run_da3_init_optimization.py --sample-dir <sample>
python scripts/shared/radius_free_proxy/stage4_anchor_refinement/run_anchor_refinement.py --sample-dir <sample>
python scripts/shared/radius_free_proxy/stage5_render/render_pose6d_scene.py --sample-dir <sample> --with-human
```
