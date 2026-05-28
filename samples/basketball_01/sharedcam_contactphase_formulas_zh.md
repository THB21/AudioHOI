# Sharedcam 与 Contactphase 数学公式说明

这份说明总结了当前篮球 shared-camera 分支里使用的优化目标。

新链路：
SAM2 masks + CoTracker -> generic object observations -> shared-camera baseline -> contactphase anchor interpolation。

## 1. 记号定义

对第 $t$ 帧：

- 相机坐标系下的球心平移：
  $$T_t = (X_t, Y_t, Z_t)$$
- 投影后的球心坐标：
  $$(u_t, v_t)$$
- 投影后的球半径（像素）：
  $$r_t$$
- 观测到的球心 / 半径：
  $$(u_t^{obs}, v_t^{obs}, r_t^{obs})$$
- 通用观测向量（来自 SAM2 mask + CoTracker）：
  $$\mathbf{o}_t = [u_t, v_t, r_t^{obs}, a_t, w_t, h_t, \rho_t, c_t]$$
  其中 $a_t$ 是 mask 面积，$w_t, h_t$ 是 mask/bbox 宽高，$\rho_t$ 是长宽比，$c_t$ 是轮廓紧致度/圆度等形状指标。篮球场景主要使用 $u_t, v_t, r_t^{obs}$；以后在 mug / hammer 等物体上可以改吃其他 shape cues。
- 图像中的共享支撑几何：
  $$S = (support\_type, floor_v, source, confidence)$$
- 图像中的共享地面/支撑线：
  $$floor_v$$
- 投影后的球底像素位置：
  $$bottom_v(t)$$

对当前篮球 case，默认半径为：

$$
R = 0.12 \text{ m}
$$

当前 `sharedcam` 代码也预留了按帧半径估计的可选入口，但本文档对应的主线仍然是：除非显式提供额外估计，否则使用固定半径。这里应理解为篮球实例化，而不是整个框架对所有物体的统一模型。

投影模型为：

$$
u_t = f_x \frac{X_t}{Z_t} + c_x
$$

$$
v_t = f_y \frac{Y_t}{Z_t} + c_y
$$

$$
r_t = f_x \frac{R}{Z_t}
$$

$$
bottom_v(t) = v_t + r_t
$$

---

## 2. Sharedcam baseline

代码位置：
- `scripts/shared/sharedcam/run_basketball_pose6d_sharedcam.py`
- `scripts/shared/sharedcam/support_geometry.py`

当前 shared-camera baseline 的目标函数为：

$$
E_{sharedcam}
= \lambda_{mask} E_{mask}
+ \lambda_{center} E_{center}
+ \lambda_{size} E_{size}
+ \lambda_{contact} E_{contact}
+ \lambda_{temp} E_{temp}^{xyz}
+ \lambda_{ztemp} E_{temp}^{z}
+ \lambda_{zb} E_{z-boundary}
+ \lambda_{zs} E_{z-slope}
+ E_{aux}
$$

### 2.0 Shared support geometry 与 floor-aware 初始化

在正式优化之前，`sharedcam` 会先从球的 contact event 帧估计一个场景级的支撑定义：

$$
S = (support\_type, floor_v, source, confidence)
$$

当前实现里：

- `support_type = floor`
- `source = ball_contact_events`
- `floor_v` 由接触帧上观测到的球底像素位置取中位数得到

这个支撑定义会被写到：

```text
results/pose6d_sharedcam/support_geometry.json
```

之后初始化会显式使用这条共享支撑线，但在框架层面，更通用的写法应该是：

$$
T_t^{init} \leftarrow \operatorname{InitFromObservation}(\mathbf{o}_t, K, \Pi, S),
$$

其中：

- $\mathbf{o}_t$ 是通用物体观测向量，
- $K$ 是相机内参，
- $\Pi$ 是物体先验包（几何/尺度假设），
- $S$ 是共享支撑几何。

对**当前篮球实例**，物体先验是“已知半径的球体近似”，所以代码才具体写成：

$$
Z_t^{init} = \frac{f_x R}{r_t^{obs}}
$$

以及

$$
r_t^{init} = \frac{f_x R}{Z_t^{init}}.
$$

当前代码不是直接用观测球心的 $v_t^{obs}$ 去反投影 $Y_t$，而是先用地面线恢复一个“应该的球心高度投影”：

$$
v_t^{center,init} = floor_v - r_t^{init}
$$

$$
X_t^{init} = \frac{(u_t^{obs}-c_x)Z_t^{init}}{f_x}
$$

$$
Y_t^{init} = \frac{(v_t^{center,init}-c_y)Z_t^{init}}{f_y}
$$

也就是说，当前**篮球特例**下的 `sharedcam` 初始化逻辑是：

```text
radius -> z
u + z -> x
floor + z + radius -> y
```

对非球体物体，初始化仍然应该理解成 observation-driven，只是使用不同的先验包 $\Pi$，而不是固定球体半径。

### 2.1 Mask / contour 项

让投影出来的圆轮廓尽量贴近观测 mask 的轮廓：

$$
E_{mask} = \sum_t d_{chamfer}(C_t^{proj}, C_t^{obs})
$$

当前权重：

$$
\lambda_{mask} = 0.018
$$

### 2.2 Center reprojection 项

$$
E_{center} = \sum_t \left[(u_t-u_t^{obs})^2 + (v_t-v_t^{obs})^2\right]
$$

当前权重：

$$
\lambda_{center} = 0.04
$$

### 2.3 Size 项

让投影后的直径 / 面积贴近观测 mask 的宽、高、平均尺度和面积：

$$
E_{size} = \sum_t \Big[(d_t-w_t)^2 + (d_t-h_t)^2 + (d_t-s_t)^2 + \alpha (a_t-a_t^{obs})^2\Big]
$$

其中 $d_t = 2r_t$。

当前权重：

$$
\lambda_{size} = 0.02
$$

### 2.4 Contact / floor 项

在接触帧上，让投影后的球底尽量贴近共享地面线：

$$
E_{contact} = \sum_{t \in C} (bottom_v(t)-floor_v)^2
$$

当前实现中，对 audio 对齐出来的弱接触帧还会额外加一个半强度版本：

$$
E_{contact}^{weak} = \sum_{t \in C_{weak}} \frac{1}{2}(bottom_v(t)-floor_v)^2
$$

当前权重：

$$
\lambda_{contact} = 10.0
$$

### 2.5 三维时序平滑项

对平移使用二阶差分平滑：

$$
E_{temp}^{xyz} = \sum_t \|T_{t+1} - 2T_t + T_{t-1}\|^2
$$

当前权重：

$$
\lambda_{temp} = 0.08
$$

### 2.6 深度额外平滑项

$$
E_{temp}^{z} = \sum_t (Z_{t+1} - 2Z_t + Z_{t-1})^2
$$

当前权重：

$$
\lambda_{ztemp} = 0.22
$$

### 2.7 分段边界连续项

约束相邻分段之间的深度连续性：

$$
E_{z-boundary} = \sum_k (Z_{start(k+1)} - Z_{end(k)})^2
$$

当前权重：

$$
\lambda_{zb} = 3.5
$$

### 2.8 分段斜率连续项

$$
E_{z-slope} = \sum_k (a_{k+1} - a_k)^2
$$

当前权重：

$$
\lambda_{zs} = 0.35
$$

### 2.9 其他辅助正则项

当前实现里还有几项较小的辅助正则：

#### 2.9.1 最小深度惩罚

当深度过小时，会加一个软下界惩罚：

$$
E_{z-min} = \sum_t \max(0, 0.35 - Z_t)
$$

对应系数：

$$
0.30
$$

#### 2.9.2 分段斜率幅值正则

如果每个分段内的深度写成：

$$
Z_t = a_k \tau_t + b_k, \qquad t \in segment\ k,
$$

那么当前实现还会直接惩罚每段斜率本身的大小：

$$
E_{seg-slope-mag} = \sum_k a_k
$$

对应系数：

$$
0.12
$$

#### 2.9.3 首尾稳定项

序列开头和结尾会通过惩罚第一段和最后一段的帧间跳动来做弱稳定：

$$
E_{endpoints} = \|T_2 - T_1\|^2 + \|T_T - T_{T-1}\|^2
$$

对应系数：

$$
0.15
$$

---

## 3. Contactphase anchor interpolation

代码位置：
- `scripts/shared/human_ball/contact/run_human_ball_contact_phase_calibration_anchorinterp.py`

这是当前主线 refinement 版本。

这一阶段保持观测到的 2D 球心不变，只优化深度轨迹：

$$
Z_1, Z_2, \dots, Z_T
$$

然后再由 $(u_t^{obs}, v_t^{obs}, Z_t)$ 反推出 $(X_t, Y_t)$：

$$
X_t = \frac{(u_t^{obs}-c_x)Z_t}{f_x},
\qquad
Y_t = \frac{(v_t^{obs}-c_y)Z_t}{f_y}
$$

这一阶段会读取：

```text
results/pose6d_sharedcam/support_geometry.json
```

并把其中的 `floor_v` / support metadata 继承到输出里，但它**不会**在 `handball` 模块里重新估 floor。

### 3.1 右手掌 contact anchor

对当前篮球序列，active hand 固定为右手。

我们用 SMPL-X joints 构造一个粗略的右掌心 proxy：

$$
P_t^{right\text{-}palm}
= \frac{1}{5}\left(
P_t^{21}
+ P_t^{40}
+ P_t^{43}
+ P_t^{46}
+ P_t^{49}
\right)
$$

并把它的深度记为：

$$
Z_{hand}(t) = P_{t,z}^{right\text{-}palm}
$$

设 $C$ 为 hand-contact event 帧集合。在这些帧上，深度被当成精确锚点：

$$
Z_t = Z_{hand}(t), \qquad t \in C
$$

### 3.2 全局深度层校正

sharedcam baseline 的深度记为 $Z_t^{base}$。

我们先在 hand-contact event 上估一个鲁棒的全局深度偏移：

$$
\Delta Z = \operatorname{median}_{t \in C}\left(Z_{hand}(t) - Z_t^{base}\right)
$$

然后得到整体平移后的参考轨迹：

$$
Z_t^{ref} = Z_t^{base} + \Delta Z
$$

### 3.3 Anchor interpolation 目标函数

在 hand-event 锚点之间，轨迹形状不再由局部 hand window 驱动，而是由两部分决定：

1. 对全局平移参考轨迹的弱贴合
2. 对深度二阶差分的强平滑

目标函数为：

$$
E_{anchorinterp} = w_{ref} E_{ref} + w_{temp} E_{temp}
$$

约束为：

$$
Z_t = Z_{hand}(t), \qquad t \in C
$$

其中

$$
E_{ref} = \sum_t (Z_t - Z_t^{ref})^2
$$

以及

$$
E_{temp} = \sum_{t=2}^{T-1} (Z_{t+1} - 2Z_t + Z_{t-1})^2
$$

当前权重：

$$
w_{ref} = 0.70,
\qquad
w_{temp} = 5.00
$$

### 3.4 直观理解

这套写法把两件事明确拆开：

- **anchor 值**：hand-contact event 上的深度 $Z_{hand}(t)$
- **轨迹形状**：锚点之间的平滑插值

所以 hand event 只决定 contact 帧上的精确深度值，不再通过对称局部窗口去“拖动”前后很多帧。

---

## 4. 高层理解

### Sharedcam

`sharedcam` baseline 的作用是：
把球放进和 GVHMR human 一样的相机几何里，先定义场景级 support geometry，并用它来初始化球心高度；然后再通过 reprojection、size、floor contact 和 temporal smoothness，得到一条比较稳定的球轨迹。

### Contactphase anchor interpolation

当前 `contactphase` 主线的作用是：
先读取 `sharedcam` 输出的 support geometry，再做一次相对人体的全局深度层校正，把 hand-contact event 帧钉到右掌心深度，最后用平滑插值补出锚点之间的轨迹。
