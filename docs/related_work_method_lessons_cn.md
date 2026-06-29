# Related Work 方法拆解与 AudioHOI 借鉴

本文是 `generic_pipeline_v2_llm_vlm_gate` 的论文方法补充说明。主线设计文档负责讲我们的 pipeline；本文只讲相关论文的中心思想、它们怎么做，以及 AudioHOI 如何吸收这些思想。

## 1. 总结：我们从论文中抽象出的三层结构

AudioHOI v2 的核心不是复现某一篇论文，而是把多类 HOI/HSI 方法抽象成三层，并额外加入一个表格层面的 LLM 审计：

```text
LLM semantic prior
  -> object parts / human parts / interaction edges / affordance graph

VLM visual gate
  -> mask、keypart、contact、overlay、anchor update、final render 的 forced-choice verification

optimizer
  -> 连续 2D-to-6D、depth、contact、temporal、static/freeze、small SE(3)

LLM CSV/data audit
  -> schema、stage consistency、contact emptiness、side swap、static drift、rotation jump 的离散诊断
```

对应到论文启发：

- InterCap / MOVER 说明：contact、occlusion、floor/support、penetration 是强几何约束，应该进入 optimizer。
- HOI-PAGE / InteractAnything 说明：LLM 更适合生成部件级 affordance 和 interaction graph，也适合检查结构化 CSV 是否违反语义规则，但不应该直接输出连续 pose。
- GenZI / ZeroHSI 说明：VLM/video generation 含有强视觉先验，但更适合作为视觉验证或候选生成，不应该替代数值优化。
- InterDiff / CoopDiff 说明：动态 HOI 要保持 contact-consistent motion，不能只逐帧拟合。
- Gaussian-HOI / Open3DHOI / WildHOI 类方法说明：开放词汇物体、contact region、human-conditioned object pose 是关键，但完整神经重建不是我们当前 solved cases 的目标。

## 2. 方法对照表

| 方法 | 中心思想 | 它怎么做 | AudioHOI 借鉴 |
|---|---|---|---|
| InterCap | 人体和物体必须联合估计，contact 能同时改善 body pose 和 object pose | 多视角 RGB-D + SMPL-X + 已知物体 mesh，利用接触、地面、物体位姿等约束联合优化 | 把 contact 作为真实几何 residual：mug palm-handle、chair two-hand endpoint、ball hand/foot/floor contact |
| MOVER | 人在场景中的运动能约束物体布局 | 利用遮挡深度顺序、free-space、contact surface 一致性来优化 camera、ground、object placement | 加入 `E_depth_order`、`E_penetration_or_floor_violation`、`E_contact`、floor/table static |
| InterDiff / CoopDiff | 动态 HOI 需要 human/object motion consistency 和 contact consistency | diffusion 生成人和物体运动，并用 physics/contact-aware 机制约束动态 | 不引入 diffusion solver，但保留 contact interval、temporal smooth、rotation jump penalty、static/freeze |
| HOI-PAGE | LLM 可以推理部件级 affordance graph | 从 text prompt + object parts 生成 Part Affordance Graph，指导 4D HOI 生成 | Stage -1 生成 `hoi_profile.json`；Stage 6.5 检查 CSV 是否违反这些语义规则 |
| InteractAnything | open-set object 的交互需要 LLM 解析关系、affordance、细节动作 | LLM feedback 生成/修正 interaction pose 和 object affordance，适配任意 mesh | Mistral/Qwen 只做离散语义先验、CSV 审计和检查项，不直接给坐标或 loss weight |
| GenZI | VLM 可以从 scene view 和 text prompt 想象合理人体交互 | VLM inpainting 多视角 2D human，再通过 3D optimization 还原 human-scene interaction | VLM 用来 gate mask/keypart/contact/render；optimizer 才负责 3D/6D 连续求解 |
| ZeroHSI | video generation 提供强 motion prior，可用于 zero-shot HSI | 利用视频生成模型产生运动先验，再用 differentiable rendering 重建 4D human-scene interaction | 我们的视频本身来自生成模型，因此必须反向验证：SAM2/CoTracker/VLM/audio 只给证据，最终靠 optimizer 对齐 |
| Open3DHOI / WildHOI | in-the-wild RGB 可做开放词汇 HOI 3D annotation/reconstruction | 结合 2D 图像、人体估计、物体重建/6D pose、语义标注形成 3D HOI 数据 | 输出统一的 `object_pose.csv`、`object_contact_points.csv`、part-level local points，便于扩展到新 object |
| Gaussian-HOI / HOIGS | HOI 可通过显式人/物动态表示和 contact region 学习重建 | 用 Gaussian representation 表达人和物体，优化动态场、接触区域和渲染一致性 | 我们不做 photorealistic neural rendering，但借鉴“contact region 是输出”的思想，显式写 `object_contact_points.csv` |

## 3. InterCap：contact 是几何约束，不是视觉注释

### 中心思想

InterCap 的关键观察是：人体和物体在交互时不能分开估计。手碰杯、脚踩地、嘴碰杯沿、物体接触地面等关系，本身就是强 3D 约束。

### 它怎么做

InterCap 使用多视角 RGB-D、SMPL-X whole-body 模型和已知 object mesh。它把人体 pose、物体 pose、接触和场景几何放进联合优化，得到更稳定的人体与物体重建。

### AudioHOI 如何借鉴

AudioHOI 没有多视角 RGB-D，所以不能直接复现 InterCap。但我们保留它最重要的思想：

```text
contact is an optimization constraint
```

对应实现：

- mug：`palm_handle_rim_body` + `stable_grasp_anchor`，handle 不可见时不乱跟可见点，而用 stable grasp prior。
- chair：`two_hand_toprail_endpoint` + `small_se3`，左右 palm 分别约束 top rail 两端。
- basketball / football：hand/foot/floor contact 作为 depth anchor，不只靠 2D center。

## 4. MOVER：人体轨迹能约束物体深度、遮挡和支撑

### 中心思想

MOVER 关注 monocular scene reconstruction，核心是人的运动和场景/物体布局互相约束。人在物体前后产生 depth ordering，人走过的位置形成 free space，人和物体接触时接触面应共位。

### 它怎么做

MOVER 优化 camera、ground plane、object scale 和 object position。约束包括：

- occlusion / depth ordering
- human free-space
- human-object contact surface consistency
- scene/object placement prior

### AudioHOI 如何借鉴

AudioHOI 把这些思想拆成可记录的 energy：

```text
E_total =
  w_2d      * E_2d_projection
+ w_depth   * E_depth_order_or_metric
+ w_contact * E_contact
+ w_smooth  * E_temporal_smooth
+ w_static  * E_static_freeze
+ w_pen     * E_penetration_or_floor_violation
+ w_prior   * E_pose_prior
```

对应 case：

- ball：floor support、hand/foot contact depth anchor、contact interval smooth。
- mug：table release 后 static/freeze，喝水阶段 handle phase continuity。
- chair：lift/place 由 audio + floor support + GVHMR proximity 联合判断，可信中间帧向两侧传播。

## 5. HOI-PAGE / InteractAnything：LLM 做语义先验，不做连续求解

### 中心思想

HOI-PAGE 用 LLM 推理 object parts 和 human parts 之间的 affordance graph。InteractAnything 也强调 open-set object 的交互需要关系推理、affordance parsing 和动作细节解释。

### 它们怎么做

这类方法通常从 text prompt 和 object mesh 出发，让 LLM 生成：

- 哪些 object parts 可交互
- 哪些 human parts 可能接触
- 接触关系是什么
- 动作阶段/目标是什么

然后再用 motion/pose/rendering 模型生成或优化 3D 交互。

### AudioHOI 如何借鉴

AudioHOI 的 Stage -1 采用同样的语义角色，但限定得更保守：

```json
{
  "object_parts": ["top_rail", "seat", "front_leg", "rear_leg"],
  "human_parts": ["left_palm", "right_palm"],
  "interaction_edges": [
    {"human_part": "left_palm", "object_part": "right_top_rail_endpoint", "relation": "hold"}
  ]
}
```

LLM 允许：

- 生成 object part / human part 列表。
- 生成 interaction edges。
- 生成 support/motion prior。
- 生成每个 stage 应该问 VLM 的问题类型。
- 在 Stage 6.5 读取 CSV/JSON/metrics，检查 schema、stage consistency、contact emptiness、left/right swap、static drift、rotation jump。

LLM 不允许：

- 输出 2D/3D 坐标。
- 输出 SE(3) correction。
- 输出连续 loss weight。
- 直接覆盖 optimizer 的 pose。
- 直接改写 `object_pose.csv` 或 `object_contact_points.csv`。

这就是 `hoi_profile.json` 的定位。

## 6. GenZI / ZeroHSI：VLM 和 video prior 可以帮助判断，但不能替代 optimizer

### 中心思想

GenZI 用 VLM 在 scene render 上想象/插入可能的人体交互，再通过 3D optimization 得到 human-scene interaction。ZeroHSI 则利用 video generation 的 motion prior，再通过 differentiable rendering 重建 4D interaction。

### 它们怎么做

这类方法说明两个事实：

1. 大视觉模型/视频模型确实含有丰富的人体交互先验。
2. 仅靠生成结果还不够，最终仍需要几何一致性、渲染一致性或优化约束。

### AudioHOI 如何借鉴

我们的输入视频本身就是生成视频，所以更要避免“相信生成模型”。AudioHOI 把 VLM 放在 conservative verification layer：

```text
Stage0 target_mask_check
Stage1 keypart_identity_check / track_stability_check
Stage2 contact_relation_check
Stage3 overlay_alignment_check
Stage4 anchor_update_check
Stage5 post_render_sanity_check
```

VLM 问题必须是 forced-choice，例如：

```text
Which chair part is highlighted?
labels: top_rail / front_leg / rear_leg / seat / stretcher / hole / background / unclear
```

VLM 只输出：

```text
pass / reject / unclear / failure label
```

它不直接改变 pose。若 VLM reject 某 contact candidate，该帧只是不启用 `E_contact`；若 unclear，则不更新 anchor。

## 7. InterDiff / CoopDiff：动态一致性比单帧拟合更重要

### 中心思想

InterDiff 和 CoopDiff 关注动态 HOI。它们的重点不是单帧 pose，而是 human motion、object motion 和 contact consistency 在时间上要成立。

### 它们怎么做

InterDiff 使用 physics-informed diffusion 来生成人-物运动。CoopDiff 将 human dynamics 和 object dynamics 分支建模，并通过 contact consistency 连接两者。

### AudioHOI 如何借鉴

AudioHOI 不用 diffusion 生成轨迹，而是把动态一致性写成 refinement rule 和 residual：

- `E_temporal_smooth`: 防止逐帧抖动。
- `E_static_freeze`: 放下后物体不再漂移。
- `rotation_jump_count`: 检查 mug handle / chair orientation 突变。
- `anchor_propagate_freeze`: 可信 contact interval 的头尾向前后传播。

这解释了为什么 chair 不能只看某一帧 2D overlay：中间接触帧如果 3D contact 更可信，就要允许小 SE(3) 旋转优化，再向前后传播，而不是每帧独立拟合。

## 8. Open3DHOI / Gaussian-HOI：显式输出 contact region 和 3D HOI 结构

### 中心思想

Open-vocabulary 3D HOI 和 Gaussian-HOI 类方法强调：HOI 不只是人体轨迹，也不是物体轨迹，而是人、物、接触区域、语义关系的联合结构。

### 它们怎么做

这类方法常结合：

- human pose / body mesh
- object reconstruction / object 6D pose
- semantic object category / open vocabulary labels
- contact region or interaction region
- rendering or reconstruction consistency

Gaussian-HOI/HOIGS 进一步把人和物体表示成 Gaussian/dynamic fields，用 contact-aware 或 HOI-aware optimizer 学习空间关系。

### AudioHOI 如何借鉴

AudioHOI 当前不做 photorealistic Gaussian reconstruction，但保留结构化输出：

```text
object_pose.csv
object_contact_points.csv
object_phase.csv
object_local_points.csv
object_local_segments.csv
vlm_gates.csv
stage7_loss_residuals.csv
```

尤其是 `object_contact_points.csv`：它不是 debug 可视化，而是后续要给人体精确 pkl / hand reconstruction / teacher review 使用的核心产物。

## 9. 为什么我们不是直接跑这些方法作为 baseline

这些论文和 AudioHOI 的输入假设不同：

- InterCap 需要多视角 RGB-D 和同步标定。
- MOVER 主要优化静态 scene layout，不直接输出我们需要的 object contact CSV。
- InterDiff / CoopDiff 是 motion generation，不是从生成视频恢复 object 6D/contact。
- HOI-PAGE / InteractAnything 生成 HOI，不解决我们的视频对齐恢复问题。
- GenZI / ZeroHSI 以生成/重建为主，不直接提供 per-frame object 6D/contact recovery。
- Gaussian-HOI 类方法更偏 neural reconstruction，输出形式和我们当前 CSV/diagnostic video 不同。

因此 v2 的 related-work comparison 采用 method proxy / ablation：

```text
video_only_tracking
mesh_only_alignment
human_only_contact
no_audio_event
no_vlm_gate
no_llm_prior
no_contact_refine
ours_full
```

比较目的不是证明“别人代码跑不过”，而是说明每种信息源缺失时会失败在哪：

- 无 contact：深度和 release/static 不稳定。
- 无 object semantic parts：mug handle、chair top rail/legs 会错。
- 无 audio：lift/place/static 时段不稳。
- 无 VLM gate：错误 anchor/contact 会进入 optimizer。
- 无 LLM prior：VLM 问题和 contact relation 过粗，zero-shot 新物体难扩展。

## 10. 对主线设计的直接影响

论文方法最终落到 AudioHOI v2 的设计决策：

1. LLM 在 Stage -1 生成离散 HOI profile，在 Stage 6.5 做 CSV/data audit。
2. VLM 每个 stage 都做 forced-choice gate。
3. Optimizer 是唯一连续求解器。
4. Contact candidate 不能直接信任，必须经过视觉/几何 gate。
5. object-specific 能力写成可复用 component，而不是每个 object 一个 runner。
6. LLM/VLM 都只能输出离散 gate、diagnostic label 或 summary，不能直接输出连续修正。
7. 输出必须包含 pose、contact points、phase、loss residual、LLM CSV audit 和六个 render video，便于和论文方法做 fair proxy comparison。

## References

- InterCap: Joint Markerless 3D Tracking of Humans and Objects in Interaction. Project page: https://intercap.is.tue.mpg.de/
- MOVER: Human-Aware Object Placement for Visual Environment Reconstruction. Project page: https://vlg.inf.ethz.ch/publications/Mover-Human-Aware-Object-Placement-for.html
- InterDiff: Generating 3D Human-Object Interactions with Physics-Informed Diffusion. Project page: https://sirui-xu.github.io/InterDiff/
- CoopDiff: Contact-consistent decoupled Diffusion for anticipating 3D HOI. arXiv page: https://arxiv.org/html/2508.07162v1
- HOI-PAGE: Zero-Shot Human-Object Interaction Generation with Part Affordance Graphs. Project page: https://craigleili.github.io/projects/hoipage/
- InteractAnything: Zero-shot Human Object Interaction Synthesis via LLM Feedback and Object Affordance Parsing. Project page: https://jinluzhang.site/projects/interactanything/
- GenZI: Zero-Shot 3D Human-Scene Interaction Generation. CVPR 2024 paper: https://openaccess.thecvf.com/content/CVPR2024/papers/Li_GenZI_Zero-Shot_3D_Human-Scene_Interaction_Generation_CVPR_2024_paper.pdf
- ZeroHSI: Zero-Shot 4D Human-Scene Interaction by Video Generation. Project page: https://awfuact.github.io/zerohsi/
- Open-vocabulary / Gaussian-HOI style 3D HOI annotation and reconstruction. Project page: https://wenboran2002.github.io/3dhoi/
