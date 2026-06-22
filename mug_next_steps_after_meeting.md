# Mug Pipeline 下一步执行计划（基于会议前半段 mug 讨论）

## 0. 会议结论确认

本计划只覆盖 **mug 相关部分**，不包含后半段 football。

根据会议前半段，老师给出的方向可以归纳为：

```text
1. 不要把 SAM3D 当成硬依赖。
2. 对 mug 这种简单物体，可以用 3D generative prior / code-generated object / Articraft 生成 canonical object proxy。
3. 你不需要对所有 frame 做 reconstruction。
4. 你只需要一张或少数 keyframe 来得到 object reconstruction / object prior。
5. handle 被手挡住时，可以先用 image synthesis remove hand，得到 clean complete mug。
6. contact 信息不仅要知道 human side，还要知道 object side：mug 上哪一块被手接触。
7. 如果 human motion 可靠，可以用 hand-object contact/attachment 帮助 anchor object motion。
```

这说明现在 mug 线的目标不是：

```text
每帧检测 handle
每帧重建 mug
每帧重新找 contact point
```

而是：

```text
少数 keyframe 建立 mug canonical geometry + object-side contact region
整段视频中 track mug body pose
contact interval 中使用 object-local grasp/contact anchor 约束 mug motion
```

---

## 1. 当前状态重新定位

你现在已经做了：

```text
M12:
  Articraft mug rigid mesh overlay
  body / rim / bottom / handle 共用一个 mug rigid transform
  visual handle phase / visibility diagnostic

M13:
  object-side contact point export
  contact usage split:
    - use_this_point_for_hand_attachment
    - use_previous_grasp_for_hand_attachment
    - rim_drinking_contact
    - hidden / occluded grasp state

M14:
  M12 + M13 联合优化
  handle phase + VLM visibility + contact evidence 联合使用
```

所以现在不要再做：

```text
继续重新找更多 contact point
继续调每帧 handle detection
继续让 rim/body visible point 抢 hand-handle anchor
```

当前真正的问题是：

```text
handle 大多数时候不可见或被手遮挡，
所以仅靠视频当前帧无法稳定细化 hand-handle contact 点。
```

下一步应该把问题改成：

```text
在可确认的 keyframe / confirmed visible contact frame 中建立 stable object-local grasp anchor，
在 hidden / drinking / occluded frames 中沿用这个 stable grasp anchor，
而不是每帧重新估计 contact 点。
```

---

## 2. 核心设计：三类信息必须分开

### 2.1 Mug body pose

表示：

```text
杯子整体刚体位姿 T_world_mug(t)
```

来源：

```text
M5 / M12 / M14 body pose
rim-bottom observation
body mask / bbox
DA3 depth prior
```

作用：

```text
决定 body / rim / bottom / handle 这个 Articraft mug 刚体整体在哪里、怎么转。
```

---

### 2.2 Visual handle phase

表示：

```text
Articraft handle 在 mug local frame 里的固定部件，
通过 axial phase / yaw 与视频中的可见把手方向对齐。
```

来源：

```text
VLM handle visibility
visible side
hidden negative visibility
Articraft handle mesh projection
```

作用：

```text
判断 visual handle 应该在 mug 的哪一侧，
但不等于 hand contact anchor。
```

---

### 2.3 Hand-mug grasp/contact anchor

表示：

```text
手真正抓住 mug 的 object-local grasp point / grasp region。
```

来源：

```text
confirmed handle contact frame
object-side contact region annotation
VLM / image synthesis / manual keyframe paint
Articraft handle mesh projection
hand proxy / hand keypoints
```

作用：

```text
在 Stage2 / M14+ 中作为 hand-object attachment residual。
```

关键原则：

```text
visual handle ≠ hand grasp anchor
rim drinking contact ≠ hand grasp anchor
body/rim/unknown debug point ≠ hand grasp anchor
```

---

## 3. 当前最该做的事情

### 总目标

```text
把 M14 从“每帧联系可见点”的逻辑，
升级成“stable object-local grasp anchor 状态机 + hand attachment residual”。
```

这不是重新做 M12/M13，而是在你已有 M12/M13/M14 基础上增加一个稳定的 grasp anchor 层。

---

## 4. Step 1：确认哪些帧可以更新 grasp anchor

读取：

```text
samples_known_object/02_mug/results/mug_articraft_contact_points/mug_articraft_contact_points.csv
```

只允许这些帧更新 `stable_grasp_local`：

```text
use_this_point_for_hand_attachment == 1
```

这些帧必须满足：

```text
1. VLM / explicit evidence 确认 hand-handle contact
2. projected Articraft handle point 离 active hand proxy 足够近
3. 不是 rim drinking contact
4. 不是 hidden / far-side occluded
5. 不是 unconfirmed nearest-handle candidate
```

不允许这些帧更新：

```text
rim_drinking_contact == 1
use_previous_grasp_for_hand_attachment == 1
object_contact_event == occluded_hand_mug_grasp
object_contact_event == handle_contact_point_misaligned
object_contact_event == unconfirmed_handle_candidate
nearest_articraft_part == handle_loop 但没有 VLM handle evidence
```

---

## 5. Step 2：实现 stable grasp anchor 状态机

### 5.1 状态机规则

```python
stable_grasp_local = None
stable_grasp_conf = 0.0

for t in frames:
    row = m13[t]

    if row["use_this_point_for_hand_attachment"] == 1:
        # confirmed visible hand-handle anchor
        stable_grasp_local = row["object_contact_local_xyz"]
        stable_grasp_conf = row.get("confidence", 1.0)
        frame_mode[t] = "direct_grasp_anchor"

    elif row["use_previous_grasp_for_hand_attachment"] == 1 and stable_grasp_local is not None:
        # hidden / drinking / occluded but still holding mug
        frame_mode[t] = "keep_previous_grasp_anchor"

    elif row["rim_drinking_contact"] == 1:
        # mouth-rim contact, not hand-handle grasp
        frame_mode[t] = "rim_contact_no_hand_anchor"

    else:
        frame_mode[t] = "no_attachment"
```

### 5.2 输出诊断文件

保存：

```text
samples_known_object/02_mug/results/mug_grasp_anchor_state/mug_grasp_anchor_state.csv
```

字段：

```csv
frame,
object_contact_event,
hand_mug_contact_state,
use_this_point_for_hand_attachment,
use_previous_grasp_for_hand_attachment,
rim_drinking_contact,
frame_mode,
stable_grasp_local_x,
stable_grasp_local_y,
stable_grasp_local_z,
stable_grasp_conf
```

---

## 6. Step 3：先做 evaluation，不直接优化

在已有 M14 pose 上，先计算 hand-to-grasp 距离。

### 6.1 计算 object-local grasp point 的 world position

```python
import numpy as np


def transform_point(T, q):
    qh = np.array([q[0], q[1], q[2], 1.0], dtype=np.float64)
    return (T @ qh)[:3]


def hand_to_grasp_distance(T_world_mug, grasp_local, hand_points_world):
    grasp_world = transform_point(T_world_mug, grasp_local)
    dists = np.linalg.norm(hand_points_world - grasp_world[None, :], axis=1)
    return float(np.min(dists)), grasp_world
```

### 6.2 输出距离曲线

保存：

```text
samples_known_object/02_mug/results/mug_grasp_anchor_state/hand_to_grasp_distance.csv
```

字段：

```csv
frame,
frame_mode,
hand_to_grasp_dist_m,
grip_world_x,
grip_world_y,
grip_world_z,
active_hand_x,
active_hand_y,
active_hand_z,
visual_handle_phase,
visibility_alpha
```

### 6.3 你要看的问题

```text
1. direct_grasp_anchor 帧距离是否合理？
2. keep_previous_grasp_anchor 帧是否连续？
3. rim_drinking_contact 是否没有覆盖 stable_grasp_local？
4. hidden 帧是否仍保持同一个 object-local grasp point？
5. hand_to_grasp_distance 是否在 hidden/drinking 段爆炸？
```

如果这里就不合理，不要继续优化。先修 anchor 逻辑。

---

## 7. Step 4：渲染 M14-grasp diagnostic

新建渲染：

```text
samples_known_object/02_mug/results/renders/M14_grasp_anchor_debug/overlay.mp4
```

每帧画：

```text
1. Articraft mug mesh
2. stable_grasp_local 投影点
3. active hand proxy / hand point
4. hand-to-grasp 连线
5. frame_mode 文本
6. use_this / use_previous / rim_drinking 标志
```

颜色规则：

```text
yellow:
  direct_grasp_anchor

blue:
  keep_previous_grasp_anchor

gray:
  no_attachment

red:
  invalid / misaligned / geometry gap

purple:
  rim_drinking_contact, not hand anchor
```

这个渲染比继续看 M13 debug 点更有意义，因为它直接告诉你：

```text
当前系统是否真的在用稳定 object-local grasp anchor，
而不是每帧拿 rim/body/unknown 点乱更新。
```

---

## 8. Step 5：如果 evaluation 合理，再做小范围优化

先不要 full 6D optimization。

第一版只优化：

```text
mug translation correction:
  delta_x_t, delta_y_t, delta_z_t
```

不要优化：

```text
yaw / pitch / roll / scale / handle phase
```

因为现在最大风险是 contact residual 把 mug body pose 拉坏。

---

## 9. Stage2 / M15 objective

### 9.1 优化变量

```python
params = [delta_x_0, delta_y_0, delta_z_0,
          delta_x_1, delta_y_1, delta_z_1,
          ...]
```

### 9.2 corrected pose

```python
def apply_translation_delta(T, delta_xyz):
    T_new = T.copy()
    T_new[:3, 3] += delta_xyz
    return T_new
```

### 9.3 residuals

```text
E =
  E_keep_M14_pose
+ E_hand_attachment
+ E_temporal_smooth_delta
```

不要重新加：

```text
raw contact_u/contact_v residual
rim drinking as hand anchor
new hidden-frame contact point
```

---

## 10. M15 residual 伪代码

```python
def m15_residual(params, frames, T_m14, grasp_state, hand_points_world):
    deltas = params.reshape(len(frames), 3)
    res = []

    for i, t in enumerate(frames):
        delta = deltas[i]
        T_corr = apply_translation_delta(T_m14[t], delta)

        # keep close to M14 pose
        res.extend((w_keep * delta).tolist())

        mode = grasp_state[t]["frame_mode"]
        grasp_local = grasp_state[t].get("stable_grasp_local")

        if mode in {"direct_grasp_anchor", "keep_previous_grasp_anchor"} and grasp_local is not None:
            dist, _ = hand_to_grasp_distance(
                T_corr,
                grasp_local,
                hand_points_world[t],
            )
            res.append(w_attach * dist / sigma_attach)

    # smooth delta correction
    for i in range(1, len(frames) - 1):
        acc = deltas[i + 1] - 2 * deltas[i] + deltas[i - 1]
        res.extend((w_smooth * acc).tolist())

    return np.asarray(res)
```

建议初始参数：

```text
w_keep = high
w_attach = medium
w_smooth = medium-high
sigma_attach = 0.05 ~ 0.08 m
```

---

## 11. 必须做的 ablation

你已经做了 M12/M13/M14，所以现在只做这几组：

```text
A0: M14 current
  当前 joint contact + handle phase 结果

A1: M14 + grasp-state evaluation only
  不优化，只画 stable_grasp_local 和 hand distance

A2: M15 direct anchors only
  只使用 use_this_point_for_hand_attachment == 1

A3: M15 direct + previous grasp
  direct anchor 更新 stable_grasp_local
  hidden/rim/drinking 沿用 previous grasp

A4: M15 direct + previous grasp + strong keep-M14
  防止 contact residual 把 mug pose 拉坏
```

比较指标：

```csv
experiment,
mean_hand_to_grasp_dist_m,
max_hand_to_grasp_dist_m,
mean_pose_delta_from_M14_m,
max_pose_delta_from_M14_m,
rim_bottom_error_change_px,
visible_overlay_quality,
hidden_segment_stability
```

---

## 12. 判断结果

### 12.1 如果 A3 比 A2 好

说明：

```text
previous stable grasp anchor 在 hidden/drinking 段有效。
```

这是你想要的结果。

---

### 12.2 如果 A3 比 A0 手部距离更好，但 overlay 变差

说明：

```text
attachment residual 在拉对手，
但把 mug pose 拉坏。
```

解决：

```text
w_keep ↑
w_attach ↓
只优化 z 或 translation
不要动 rotation/phase
```

---

### 12.3 如果 A3 没改善

检查：

```text
1. use_this_point_for_hand_attachment 帧是否太少
2. stable_grasp_local 是否初始化成功
3. active hand proxy 是否太粗
4. GVHMR hand depth 是否和 mug depth 未对齐
5. M13 是否把 confirmed handle contact 误判成 hidden/body/rim
```

---

### 12.4 如果 direct anchor 本身错

说明：

```text
object-side handle point 或 human-side hand proxy 不够精细。
```

下一步才是：

```text
HaMeR / MANO / MediaPipe Hands
```

不要继续调 VLM label。

---

## 13. 关于 human side 是否要进一步细化

根据老师录音，fine-grained contact region 可以用 image synthesis / VLM 在 keyframe 上 paint 出来。但如果要做真正手-杯把手 attachment，GVHMR active hand proxy 可能太粗。

当前选择：

```text
短期：
  用 GVHMR active hand proxy，只做 coarse hand attachment。

中期：
  加 HaMeR / MANO / MediaPipe Hands，得到 21 hand joints 或 hand mesh。

长期：
  hand surface vertices <-> Articraft handle/body surface contact。
```

判断是否必须上 HaMeR：

```text
如果 direct_grasp_anchor 帧中，object-side handle point 已经合理，
但 hand_to_grasp_dist 仍然乱，
说明 human-side proxy 太粗，需要手部几何。
```

---

## 14. 当前不要做的事情

先不要：

```text
1. 不要继续重做 M12 body/handle phase
2. 不要继续重写 M13 contact classification
3. 不要直接 full 6D optimize mug pose
4. 不要让 rim_drinking_contact 更新 hand anchor
5. 不要在 hidden frames 新建 contact point
6. 不要用 nearest_articraft_part == handle_loop 直接当 confirmed contact
7. 不要把 VLM 当几何真值
```

---

## 15. 你现在的下一步一句话

```text
用 M13 已经分好的 contact usage 字段，
建立 stable object-local grasp anchor 状态机，
先做 hand-to-grasp distance evaluation 和可视化，
再只优化小幅 translation correction，
验证 previous grasp anchor 是否能在 hidden/drinking 段维持 hand-mug attachment。
```

这是对老师 mug 指导最贴合的下一步：

```text
object prior 已有：Articraft mug
contact region 已有：M13 / VLM / projected handle
tracking 已有：M12/M14 body + phase
现在缺的是：把 contact information 变成真正的 attachment residual
```


---

## 8. 已执行：Stable Grasp Anchor 状态机

根据上面的会议结论，已经新增一个独立状态机脚本，不直接改 M14 优化本体：

```text
scripts/shared/radius_free_proxy/stage2_contact_candidates/build_mug_grasp_anchor_state.py
```

输入：

```text
samples_known_object/02_mug/results/mug_articraft_contact_points/mug_articraft_contact_points.csv
```

输出：

```text
samples_known_object/02_mug/results/mug_grasp_anchor_state/mug_grasp_anchor_state.csv
samples_known_object/02_mug/results/mug_grasp_anchor_state/mug_grasp_anchor_state_summary.json
```

### 8.1 当前实现的状态机规则

只有 confirmed hand-handle grasp 可以更新 object-local anchor：

```text
object_contact_event == hand_handle_grasp
hand_mug_contact_state == direct_hand_grasp_point
semantic_contact_part == handle_loop
use_this_point_for_hand_attachment == 1
rim_drinking_contact == 0
```

以下情况不能更新 anchor，只能沿用已有 stable grasp：

```text
rim_drinking_contact == 1
object_contact_event == occluded_hand_mug_grasp
object_contact_event == handle_contact_point_misaligned
object_contact_event == unconfirmed_handle_candidate
object_contact_event == hand_handle_grasp_depth_misaligned
hand_mug_contact_state == held_but_handle_not_observed
hand_mug_contact_state == visual_contact_needs_depth_alignment
```

这一步解决的是：

```text
不要让 rim/body/hidden/深度错位帧覆盖 hand-handle grasp anchor。
```

### 8.2 当前输出结果

本次 mug 共 240 帧，状态机结果：

```text
direct_grasp_anchor: 20 frames
keep_previous_grasp_anchor: 215 frames
rim_contact_keep_previous_grasp_anchor: 5 frames
```

允许更新 stable grasp 的帧：

```text
1, 2, 3, 4, 5, 6, 7,
53, 54, 55, 56, 57, 58, 59, 60,
121, 122, 123, 124, 125
```

rim drinking 但不更新 hand anchor 的帧：

```text
72, 73, 77, 78, 83
```

关键抽查：

```text
48-52:
  VLM says visible/handle, but event is hand_handle_grasp_depth_misaligned.
  These frames keep previous stable grasp from frame 7 and do not update anchor.

72-83 drinking segment:
  rim_drinking_contact frames keep previous grasp from frame 60.
  rim/mouth contact does not overwrite hand-handle anchor.

89-97:
  visible/handle but projected contact is misaligned or depth-misaligned.
  These frames keep previous stable grasp from frame 60 and do not update anchor.

121-125:
  confirmed direct hand-handle grasp, so stable grasp is updated again.
```

当前最后的 stable grasp：

```text
stable_grasp_local = [-0.077386, -0.036629, 0.000000]
source_frame = 125
```

### 8.3 和会议目标的关系

这一步对应会议里说的 contact 粒度：

```text
right_hand contacts mug.handle_region
```

而不是：

```text
exact index fingertip contacts exact handle vertex
```

当前状态机先把 object-side contact 表达稳定下来：

```text
human_side = left/right hand proxy
object_side = handle / rim / body semantic region
stable_grasp_local = approximate object-local handle anchor
```

下一步再把这个状态机输出接回 M14 / Stage2 residual：

```text
minimize distance( T_mug(t) * stable_grasp_local, active_hand_proxy(t) )
```

其中 hidden / drinking / occluded frame 使用上一段 confirmed grasp，而不是从当前画面重新找 rim/body/debug 点。


---

## 9. 已执行：Keyframe Fine Contact Annotation 输入层

为了避免每帧重新猜 contact 点，现在新增了一个只针对 confirmed grasp keyframe 的细粒度标注脚本：

```text
scripts/shared/radius_free_proxy/stage2_contact_candidates/run_qwen_mug_contact_keyframes.py
```

它读取 stable grasp anchor 状态机，只选择 `direct_grasp_anchor` 的代表帧，而不是 hidden/drinking/misaligned 帧。

当前代表帧：

```text
1, 4, 7, 53, 57, 60, 121, 123, 125
```

输出目录：

```text
samples_known_object/02_mug/annotations/vlm_mug_contact_keyframes_large/
```

核心产物：

```text
crops/00001_contact_crop.png
crops/00004_contact_crop.png
crops/00007_contact_crop.png
crops/00053_contact_crop.png
crops/00057_contact_crop.png
crops/00060_contact_crop.png
crops/00121_contact_crop.png
crops/00123_contact_crop.png
crops/00125_contact_crop.png
contact_keyframe_crop_sheet.png
mug_contact_keyframe_annotations.csv
mug_contact_keyframe_annotations.json
prompt.txt
```

裁剪策略最终改回小 crop：

```text
目标：只让 VLM 看稳定抓握区域，减少全图/大 crop 里的杯沿、嘴部、身体姿态干扰。
大 crop 会让模型重新解释全局 drinking/contact，容易把 rim/body 事件混进 hand-handle grasp。
小 crop 约 220-240px，围绕 mug handle + active hand/contact point。
```

VLM prompt 要求输出：

```text
contact_visible
hand_side
contact_fingers
primary_contact_finger
object_part
object_region
handle_grasp_type
visible_evidence
use_as_stable_grasp_keyframe
confidence
reason
```

这一步的定位：

```text
不是每帧重新找 contact。
不是让 VLM 给精确 3D vertex。
而是在少数 clear confirmed keyframe 上，得到更细的 object-side region 和 finger evidence，
再转成 stable object-local grasp region。
```

### 9.1 当前 VLM 运行状态

本地 Qwen3-VL-8B 模型路径存在：

```text
/mnt/hdd/AudioHOI/models/modelscope/Qwen/Qwen3-VL-8B-Instruct
```

当前 `audiohoi` 环境连续跑多帧 8B VLM 在 10GB GPU 上会 OOM：

```text
CUDA out of memory during multi-frame generation.
```

尝试 `--load-4bit` 失败，原因是当前 transformers/bitsandbytes 与 `Qwen3VLForConditionalGeneration` 不兼容：

```text
AttributeError: Qwen3VLForConditionalGeneration has no attribute set_submodule
```

最终采用的 workaround：

```text
one frame per process
small crop
compact JSON prompt
reuse tracker/state hand side as context hint
```

已完成的小 crop VLM keyframe 标注：

```text
frames: 1, 4, 7, 53, 57, 60, 121, 123, 125
all frames: contact_visible=True
all frames: hand_side=left_hand
all frames: object_part=handle
all frames: object_region=upper_handle
all frames: handle_grasp_type=pinch_handle
primary_contact_finger=thumb
confidence=0.95-0.98
```

输出位置：

```text
samples_known_object/02_mug/annotations/vlm_mug_contact_keyframes/mug_contact_keyframe_annotations.csv
samples_known_object/02_mug/annotations/vlm_mug_contact_keyframes/mug_contact_keyframe_annotations.json
```

这批结果说明：稳定抓握的 canonical hand-object contact 应该是
`left_hand thumb/fingers -> upper_handle`，而不是逐帧从 rim/body 最近点重新找 contact。

### 9.2 已接入：VLM keyframe object-side contact part -> stable grasp state

小 crop VLM 标注已经接回状态机：

```text
scripts/shared/radius_free_proxy/stage2_contact_candidates/build_mug_grasp_anchor_state.py
```

状态机现在不仅保存 `stable_grasp_local_xyz`，还保存 VLM 确认的 object-side / hand-side 语义：

```text
stable_grasp_object_part = handle
stable_grasp_object_region = upper_handle
stable_grasp_hand_side = left_hand
stable_grasp_primary_finger = thumb
stable_grasp_type = pinch_handle
```

重新生成的输出：

```text
samples_known_object/02_mug/results/mug_grasp_anchor_state/mug_grasp_anchor_state.csv
samples_known_object/02_mug/results/mug_grasp_anchor_state/mug_grasp_anchor_state_summary.json
```

关键行为：

```text
confirmed direct grasp frames: update local xyz + VLM semantic grasp metadata
hidden / depth-misaligned frames: keep previous upper_handle grasp anchor
rim drinking frames: keep previous upper_handle grasp anchor, do not overwrite with rim contact
```

抽查结果：

```text
frame 48  -> keep previous from frame 7, handle/upper_handle/left_hand/thumb
frame 72  -> rim drinking, keep previous from frame 60
frame 83  -> rim drinking, keep previous from frame 60
frame 96  -> keep previous from frame 60
frame 121 -> confirmed grasp, update to handle/upper_handle/left_hand/thumb
```

所以目前已经得到的 object contact 具体部分是：

```text
object_part: handle
object_region: upper_handle
hand_side: left_hand
primary_contact_finger: thumb
contact_fingers: thumb + index/middle/ring/pinky/palm
handle_grasp_type: pinch_handle
```

注意：这仍然是 semantic contact region，不是 exact mesh vertex。下一步如果要进入真正优化，应把它作为稳定 object-local grasp region residual，而不是每帧重新最近点搜索。

### 9.3 已执行：M15 stable grasp anchor projection diagnostic

新增脚本：

```text
scripts/shared/radius_free_proxy/stage2_contact_candidates/render_mug_grasp_anchor_projection.py
```

输入：

```text
M14 mug body pose:
  samples_known_object/02_mug/proxy/mug_body_only_cylinder_pose_segmented_sequence.csv
M14 handle phase:
  samples_known_object/02_mug/results/renders/M14_joint_contact_handle_phase/handle_phase_joint_contact.csv
Stable grasp state:
  samples_known_object/02_mug/results/mug_grasp_anchor_state/mug_grasp_anchor_state.csv
Per-frame contact diagnostics:
  samples_known_object/02_mug/results/mug_articraft_contact_points/mug_articraft_contact_points.csv
```

输出：

```text
samples_known_object/02_mug/results/renders/M15_grasp_anchor_projection/grasp_anchor_projection.csv
samples_known_object/02_mug/results/renders/M15_grasp_anchor_projection/overlay.mp4
samples_known_object/02_mug/results/renders/M15_grasp_anchor_projection/outputs.json
```

M15 做的事情：

```text
1. 读取 stable_grasp_local_xyz + VLM semantic metadata。
2. 用 M14 mug pose + handle phase 把 stable upper_handle anchor 投影回画面。
3. 和 active hand proxy 的 2D 位置比较，输出 hand_anchor_dist_px。
4. 按 frame_mode 给出 proposed attachment residual weight。
```

当前诊断结果：

```text
num_frames: 240
num_residual_frames: 240  # diagnostic/proposed only
mean_hand_anchor_dist_px: 18.93
median_hand_anchor_dist_px: 19.47
```

最差帧集中在：

```text
72-84 drinking segment: weak keep-previous anchor, distance about 30-37 px
115-120 lowering/transition segment: weak keep-previous anchor, distance about 30 px
```

这说明 stable upper_handle anchor 比之前逐帧 nearest contact 更稳定，但它现在还不是最终优化器。正式 M15/M16 优化还需要：

```text
contact/release interval gate
weak residual for hidden/drinking frames
strong residual only for confirmed grasp keyframes
pose regularization to keep close to M14 body pose
```

